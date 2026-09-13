"""Profile dispatch, degradation rules and environment checks.

Guide sections 4.2, 6.2 and 7. Every stage module is imported lazily so that a
``--mock`` run, ``--check-env`` and the unit tests work in a bare Python
installation with no inference dependencies at all.
"""

from __future__ import annotations

import os
import copy
import platform
import sys

from ._core import errors, job_schema, model_manifest, paths, preview, result_schema, skeleton

from . import export_result, mock_source, postprocess, preview_export

#: Python modules reported by ``--check-env``.
OPTIONAL_MODULES = (
    "numpy",
    "cv2",
    "mediapipe",
    "torch",
    "torchvision",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmpose",
)

#: Profiles served by the MediaPipe backend.
MEDIAPIPE_PROFILES = ("preview", "fallback_cpu")

#: Profiles served by the MMPose + MotionBERT backend.
MMPOSE_PROFILES = ("quality", "quality_plus", 'quality_feet')

#: Warning codes emitted by this module.
CODE_PROFILE_FALLBACK = "PROFILE_FALLBACK"
CODE_HANDS_DISABLED = "HANDS_DISABLED"
CODE_MOCK = "MOCK_RESULT"


# --------------------------------------------------------------------------------------
# Environment probes
# --------------------------------------------------------------------------------------


def module_report() -> dict:
    """Import-check every optional dependency without failing."""
    report = {}
    for name in OPTIONAL_MODULES:
        try:
            module = __import__(name)
        except Exception as exc:  # ImportError plus broken installs
            report[name] = {"available": False, "error": "{0}: {1}".format(type(exc).__name__, exc)}
            continue
        report[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "") or ""),
        }
    return report


def check_env(profile=None) -> dict:
    """Payload for ``--check-env``."""
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable or "",
        "platform": platform.platform(),
        "addon_root": paths.addon_root(),
        "skeleton": skeleton.SKELETON_ID,
        "result_version": result_schema.RESULT_VERSION,
    }
    if profile:
        from .environment import validate_environment
        report.update(validate_environment(profile))
    else:
        report["modules"] = module_report()
    return report


def check_cuda() -> dict:
    """Payload for ``--check-cuda``; never raises."""
    payload = {"available": False}
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        payload["error"] = errors.MocapError(
            errors.CUDA_UNAVAILABLE,
            "worker 环境未安装 PyTorch，无法检测 CUDA：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 安装 CUDA 版 PyTorch，或改用 preview / fallback_cpu profile。",
        ).to_dict()
        return payload

    payload["torch_version"] = str(getattr(torch, "__version__", ""))
    payload["cuda_build_version"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver level failure
        payload["error"] = errors.MocapError(
            errors.CUDA_UNAVAILABLE, "CUDA 检测失败：{0}".format(exc)
        ).to_dict()
        return payload

    payload["available"] = available
    if not available:
        payload["error"] = errors.MocapError(
            errors.CUDA_UNAVAILABLE,
            "PyTorch 已安装但 CUDA 不可用。",
            suggestion="确认安装的是 CUDA 版 PyTorch 且显卡驱动正常；或改用 preview / fallback_cpu profile。",
        ).to_dict()
        return payload

    try:
        # An availability flag alone does not exercise driver / binary compatibility.
        sample = torch.eye(2, device="cuda:0")
        if not torch.equal(sample @ sample, sample):
            raise RuntimeError("CUDA matrix multiplication returned incorrect values")
        from torchvision.ops import nms as tv_nms
        from mmcv.ops import nms as mmcv_nms
        boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device="cuda:0")
        scores = torch.tensor([0.9, 0.8], device="cuda:0")
        if tv_nms(boxes, scores, 0.5).tolist() != [0] or mmcv_nms(boxes, scores, 0.5)[1].tolist() != [0]:
            raise RuntimeError("CUDA NMS returned incorrect indices")
        torch.cuda.synchronize()
        payload["operations"] = ["torch.matmul", "torchvision.nms", "mmcv.nms"]
    except Exception as exc:
        payload["available"] = False
        payload["error"] = errors.MocapError(
            errors.CUDA_UNAVAILABLE, "CUDA 算子测试失败：{0}".format(exc)
        ).to_dict()
        return payload

    try:
        payload["device_count"] = int(torch.cuda.device_count())
        payload["device_name"] = str(torch.cuda.get_device_name(0))
        total = torch.cuda.get_device_properties(0).total_memory
        payload["total_memory_gb"] = round(float(total) / (1024 ** 3), 2)
    except Exception as exc:  # pragma: no cover - informational only
        payload["device_error"] = str(exc)
    return payload


# --------------------------------------------------------------------------------------
# Job execution
# --------------------------------------------------------------------------------------


def run_job(job: dict, reporter, cancel_token=None, mock: bool = False) -> str:
    """Execute ``job`` and return the written result path.

    Raises :class:`errors.MocapError` on failure. Cancellation raises
    ``MocapError(CANCELLED)`` so the CLI can emit the ``cancelled`` event and
    exit with code 130.
    """
    job_schema.validate_job(job)
    mode = str(job.get("mode") or job_schema.MODE_CAPTURE)
    model_section = job.get("model") or {}
    options = dict(job.get("options") or {})
    options.setdefault('processing_version', '0.3')
    options.setdefault('motion_type', 'general')
    options['coordinate_space'] = 'root_relative'
    options["_preview_rows"] = []
    input_section = job.get("input") or {}
    output_section = job.get("output") or {}

    profile = str(model_section.get("profile") or "preview")
    models_root = str(model_section.get("models_root") or "")

    if mode == job_schema.MODE_SELF_TEST:
        return _run_self_test(job, reporter)

    if mock:
        return _run_mock(job, reporter, cancel_token)

    source_identity = job.get('source_identity') or preview.fingerprint(input_section['path'])
    if not preview.source_matches(source_identity, input_section['path']):
        raise errors.MocapError(errors.MEDIA_OPEN_FAILED, '素材在任务开始前已改变，请重新生成。')

    manifest = model_manifest.load_manifest(models_root)
    report = model_manifest.check_profile_requirements(profile, models_root)
    for message in report.warnings:
        reporter.debug("preflight: {0}".format(message))
    if report.effective_profile != profile:
        reporter.warning(
            CODE_PROFILE_FALLBACK,
            "profile {0} 不可用，已回退到 {1}。".format(profile, report.effective_profile),
        )
        profile = report.effective_profile
    if report.missing_required:
        raise model_manifest.first_missing_error(report)

    from .environment import validate_environment
    environment = validate_environment(profile)
    if not environment["ok"]:
        raise errors.MocapError.from_dict(environment["error"])

    if profile in MEDIAPIPE_PROFILES:
        frames, fps, warnings = _run_mediapipe(
            job, profile, manifest, reporter, cancel_token, options
        )
    elif profile in MMPOSE_PROFILES:
        frames, fps, warnings, profile = _run_mmpose(
            job, profile, manifest, reporter, cancel_token, options
        )
    else:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "profile {0} 没有对应的推理链路。".format(profile),
            details={"profile": profile},
        )

    _raise_if_cancelled(cancel_token)

    if not frames:
        raise errors.MocapError(
            errors.NO_PERSON_DETECTED,
            "整段素材中都没有检测到可用的人体姿态。",
            details={"path": str(input_section.get("path") or "")},
        )

    raw_frames = copy.deepcopy(frames)
    frames, post_warnings = postprocess.postprocess(frames, fps, options, reporter)
    if profile in MMPOSE_PROFILES:
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            options.setdefault('_diagnostics', {})['cuda_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            options['_diagnostics']['cuda_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
    warnings = list(warnings) + list(post_warnings)
    if not preview.source_matches(source_identity, input_section['path']):
        raise errors.MocapError(errors.MEDIA_OPEN_FAILED, '素材在捕捉期间已改变，请重新生成。')

    result = export_result.build_result(
        frames,
        fps,
        source_path=str(input_section.get("path") or ""),
        source_type=str(input_section.get("type") or "video"),
        profile=profile,
        warnings=postprocess.summarise_warnings(warnings),
    )
    result_path = export_result.write_result(
        result, output_section.get("dir"), output_section.get("result_filename")
    )
    # Preserve processing flags only in the sidecar, keeping the v0.1 result compatible.
    flags = {frame["frame"]: frame.get("interpolated_joints", []) for frame in frames}
    for frame in result["frames"]:
        frame["interpolated_joints"] = flags.get(frame["frame"], [])
    preview_export.write(job, result, result_path, options["_preview_rows"],
                         options.get("_preview_meta", {}), profile, raw_frames=raw_frames,
                         diagnostics=options.get('_diagnostics', {}))
    return result_path


def _raise_if_cancelled(cancel_token) -> None:
    if cancel_token is not None and cancel_token.cancelled():
        raise errors.MocapError(errors.CANCELLED, "worker 收到取消请求。")


def _run_mock(job: dict, reporter, cancel_token) -> str:
    """Generate a synthetic result so the Blender side can be developed offline."""
    input_section = job.get("input") or {}
    output_section = job.get("output") or {}
    options = dict(job.get("options") or {})
    profile = str((job.get("model") or {}).get("profile") or "preview")

    fps = float(input_section.get("target_fps") or mock_source.DEFAULT_FPS)
    reporter.loading_model(profile=profile, model_id="mock", fraction=0.1)
    frames = mock_source.generate_frames(
        frame_start=int(input_section.get("frame_start") or 1),
        frame_end=int(input_section.get("frame_end") or 0),
        fps=fps,
        include_hands=bool(options.get("include_hands", False)),
    )
    total = len(frames)
    for index, frame in enumerate(frames, start=1):
        _raise_if_cancelled(cancel_token)
        if index == 1 or index == total or index % 10 == 0:
            reporter.processing_frame(frame["frame"], total, fraction=index / float(total))

    warnings = [
        {
            "code": CODE_MOCK,
            "message": "这是 --mock 生成的合成动作，不是真实动捕结果。",
        }
    ]
    # Contacts are recomputed from the geometry rather than trusting the
    # generator, so the mock run exercises the same detection and foot-lock code
    # path as a real capture.
    frames, post_warnings = postprocess.postprocess(frames, fps, options, reporter)
    warnings.extend(post_warnings)

    result = export_result.build_result(
        frames,
        fps,
        source_path=str(input_section.get("path") or "mock"),
        source_type=str(input_section.get("type") or "video"),
        profile=profile,
        warnings=postprocess.summarise_warnings(warnings),
    )
    return export_result.write_result(
        result, output_section.get("dir"), output_section.get("result_filename")
    )


def _run_self_test(job: dict, reporter) -> str:
    """Load the models a profile needs without running a capture."""
    model_section = job.get("model") or {}
    output_section = job.get("output") or {}
    profile = str(model_section.get("profile") or "preview")
    models_root = str(model_section.get("models_root") or "")
    device = str(model_section.get("device") or "cpu")

    report = model_manifest.check_profile_requirements(profile, models_root)
    if report.missing_required:
        raise model_manifest.first_missing_error(report)
    effective = report.effective_profile
    from .environment import validate_environment
    environment = validate_environment(effective)
    if not environment["ok"]:
        raise errors.MocapError.from_dict(environment["error"])
    if effective != profile:
        reporter.warning(
            CODE_PROFILE_FALLBACK,
            "profile {0} 不可用，自测使用 {1}。".format(profile, effective),
        )

    manifest = model_manifest.load_manifest(models_root)
    loaded = []
    if effective in MEDIAPIPE_PROFILES:
        from . import pose_mediapipe

        artifact_id, model_path = pose_mediapipe.select_body_model(models_root, effective, manifest)
        reporter.loading_model(profile=effective, model_id=artifact_id or "mediapipe_pose", fraction=0.3)
        estimator = pose_mediapipe.MediaPipeBodyEstimator(model_path, video_mode=False)
        estimator.close()
        loaded.append(artifact_id or model_path)
    else:
        from . import pose2d_mmpose, pose3d_motionbert

        resolved_device = pose2d_mmpose.resolve_device(device, reporter)
        reporter.loading_model(profile=effective, fraction=0.2)
        detector = pose2d_mmpose.PersonDetector(models_root, resolved_device, manifest, reporter)
        loaded.append(pose2d_mmpose.DETECTOR_WEIGHTS)
        reporter.loading_model(profile=effective, fraction=0.5)
        body2d = pose2d_mmpose.Body2DEstimator(models_root, effective, resolved_device, manifest, reporter)
        loaded.append(body2d.weights_id)
        reporter.loading_model(profile=effective, fraction=0.8)
        pose3d_motionbert.Body3DLifter(models_root, resolved_device, manifest, reporter)
        loaded.append(pose3d_motionbert.LIFTER_WEIGHTS)
        del detector

    payload = {
        "version": result_schema.RESULT_VERSION,
        "self_test": True,
        "profile": profile,
        "effective_profile": effective,
        "device": device,
        "loaded_models": loaded,
    }
    target_dir = paths.ensure_dir(str(output_section.get("dir") or ""))
    target = os.path.join(target_dir, "self_test.json")
    import json

    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


# --------------------------------------------------------------------------------------
# MediaPipe branch (Phase 3)
# --------------------------------------------------------------------------------------


def _run_mediapipe(job, profile, manifest, reporter, cancel_token, options) -> tuple:
    from . import media_decode, pose_mediapipe

    input_section = job.get("input") or {}
    model_section = job.get("model") or {}
    models_root = str(model_section.get("models_root") or "")
    warnings = []

    artifact_id, model_path = pose_mediapipe.select_body_model(models_root, profile, manifest)
    reporter.loading_model(profile=profile, model_id=artifact_id or "mediapipe_pose", fraction=0.05)

    include_hands = bool(options.get("include_hands", True))
    hand_status = manifest.artifact_status(pose_mediapipe.HAND_MODEL_ID)
    hand_path = hand_status.absolute_path if hand_status is not None and hand_status.exists else ""
    if include_hands and not hand_path:
        warnings.append(
            {
                "code": pose_mediapipe.CODE_HANDS_DISABLED,
                "message": "缺少 MediaPipe Hand 模型，已只做身体捕捉。",
            }
        )
        reporter.warning(warnings[-1]["code"], warnings[-1]["message"])
        include_hands = False

    source = media_decode.open_media(
        str(input_section.get("path") or ""),
        str(input_section.get("type") or "auto"),
        float(input_section.get("target_fps") or 30.0),
        int(input_section.get("frame_start") or 1),
        int(input_section.get("frame_end") or 0),
        cancel_token=cancel_token,
        reporter=reporter,
    )
    video_mode = source.media_type == "video"
    options["_preview_meta"] = preview_export.source_meta(source)
    body_estimator = pose_mediapipe.MediaPipeBodyEstimator(model_path, video_mode=video_mode)
    hand_estimator = (
        pose_mediapipe.MediaPipeHandEstimator(hand_path, video_mode=video_mode)
        if include_hands
        else None
    )

    frames = []
    frame_start = int(input_section.get("frame_start") or 1)
    world_warning_sent = False
    first_center = None
    reference_scale = 1.6  # metres per normalised unit, refined from the figure height

    try:
        for decoded in source:
            _raise_if_cancelled(cancel_token)
            row = preview_export.decoded_row(decoded, frame_start)
            options.setdefault("_preview_rows", []).append(row)
            timestamp_ms = int(round(decoded.timestamp * 1000.0)) if video_mode else 0
            detection = body_estimator.detect(decoded.image, timestamp_ms)
            world = getattr(detection, "pose_world_landmarks", None) or []
            normalized = getattr(detection, "pose_landmarks", None) or []
            if normalized:
                row["body2d"] = preview_export.mp_points(normalized[0])
                row['feet2d'] = {'L': [row['body2d'][31], row['body2d'][29]],
                                 'R': [row['body2d'][32], row['body2d'][30]]}
                row['pelvis2d'] = [(row['body2d'][23][axis] + row['body2d'][24][axis]) / 2 for axis in range(2)]
            if not world:
                reporter.warning(
                    errors.NO_PERSON_DETECTED,
                    "第 {0} 帧未检测到人体。".format(frame_start + decoded.index),
                    frame=frame_start + decoded.index,
                )
                continue

            body, confidence = pose_mediapipe.landmarks_to_standard(world[0])
            if not body:
                continue

            head = body.get("head")
            pelvis = body.get("pelvis")
            if head is not None and pelvis is not None:
                height = abs(head[2] - pelvis[2])
                if height > 0.2:
                    reference_scale = height / 0.35

            drift, first_center = pose_mediapipe.horizontal_offset(
                normalized[0] if normalized else None, reference_scale, first_center
            )
            total_offset = (drift[0], drift[1], drift[2])
            body = {name: _translate(value, total_offset) for name, value in body.items()}

            hands = {}
            if hand_estimator is not None:
                hand_result = hand_estimator.detect(decoded.image, timestamp_ms)
                hands = pose_mediapipe.hand_landmarks_to_standard(hand_result, body)
                for hand_index, landmarks in enumerate(getattr(hand_result, "hand_landmarks", None) or []):
                    handedness = getattr(hand_result, "handedness", None) or []
                    label = handedness[hand_index][0].category_name if hand_index < len(handedness) else ""
                    row["hands2d"].append({"side": label, "points": preview_export.mp_points(landmarks)})

            if not world_warning_sent:
                warnings.append(
                    {
                        "code": pose_mediapipe.CODE_WORLD_APPROXIMATE,
                        "message": "MediaPipe 世界坐标以髋部为原点，全局位移为近似值。",
                    }
                )
                reporter.warning(warnings[-1]["code"], warnings[-1]["message"])
                world_warning_sent = True

            frames.append(
                {
                    "frame": frame_start + decoded.index,
                    "time": decoded.timestamp,
                    "body3d": body,
                    "hands3d": hands,
                    "confidence": confidence,
                    "contacts": {},
                }
            )
            if source.total_frames:
                reporter.processing_frame(frame_start + decoded.index, source.total_frames)
            elif decoded.index % 10 == 0:
                reporter.processing_frame(frame_start + decoded.index)
    finally:
        body_estimator.close()
        if hand_estimator is not None:
            hand_estimator.close()
        source.close()

    return frames, source.fps, warnings


def _translate(value, offset):
    return (value[0] + offset[0], value[1] + offset[1], value[2] + offset[2])


# --------------------------------------------------------------------------------------
# MMPose branch (Phase 5)
# --------------------------------------------------------------------------------------


def _run_mmpose(job, profile, manifest, reporter, cancel_token, options) -> tuple:
    from . import media_decode, pose2d_mmpose, pose3d_motionbert

    input_section = job.get("input") or {}
    model_section = job.get("model") or {}
    models_root = str(model_section.get("models_root") or "")
    device = pose2d_mmpose.resolve_device(str(model_section.get("device") or "cuda:0"), reporter)
    warnings = []

    detector = pose2d_mmpose.PersonDetector(models_root, device, manifest, reporter)
    try:
        body2d = pose2d_mmpose.Body2DEstimator(models_root, profile, device, manifest, reporter)
    except errors.MocapError as exc:
        if profile == "quality_plus" and exc.code in (errors.CUDA_OOM, errors.MODEL_MISSING, errors.CONFIG_MISSING):
            warnings.append(
                {
                    "code": CODE_PROFILE_FALLBACK,
                    "message": "quality_plus 不可用（{0}），回退到 quality。".format(exc.code),
                }
            )
            reporter.warning(warnings[-1]["code"], warnings[-1]["message"])
            profile = "quality"
            body2d = pose2d_mmpose.Body2DEstimator(models_root, profile, device, manifest, reporter)
        else:
            raise

    requested_start = int(input_section.get('frame_start') or 1)
    requested_end = int(input_section.get('frame_end') or 0)
    video_context = media_decode.resolve_media_type(input_section.get('path', ''), input_section.get('type', 'auto')) == 'video'
    context_start = max(1, requested_start - pose3d_motionbert.WINDOW_SIZE // 2) if video_context else requested_start
    context_end = requested_end + pose3d_motionbert.WINDOW_SIZE // 2 if video_context and requested_end else requested_end
    source = media_decode.open_media(
        str(input_section.get("path") or ""),
        str(input_section.get("type") or "auto"),
        float(input_section.get("target_fps") or 30.0),
        context_start,
        context_end,
        cancel_token=cancel_token,
        reporter=reporter,
    )

    frame_start = context_start
    keypoints_seq = []
    options["_preview_meta"] = preview_export.source_meta(source)
    scores_seq = []
    bboxes_seq = []
    frame_meta = []
    image_size = None

    try:
        for decoded in source:
            _raise_if_cancelled(cancel_token)
            row = preview_export.decoded_row(decoded, frame_start)
            options.setdefault("_preview_rows", []).append(row)
            if image_size is None:
                image_size = (decoded.width, decoded.height)
            bbox = detector.detect(decoded.image)
            if bbox is None:
                keypoints_seq.append(None)
                scores_seq.append(None)
                bboxes_seq.append(None)
                reporter.warning(
                    errors.NO_PERSON_DETECTED,
                    "第 {0} 帧未检测到人体。".format(frame_start + decoded.index),
                    frame=frame_start + decoded.index,
                )
                continue
            keypoints, keypoint_scores = body2d.estimate(decoded.image, bbox)
            if keypoints is None:
                keypoints_seq.append(None)
                scores_seq.append(None)
                bboxes_seq.append(None)
                reporter.warning(
                    errors.NO_PERSON_DETECTED,
                    "第 {0} 帧 2D 姿态估计为空。".format(frame_start + decoded.index),
                    frame=frame_start + decoded.index,
                )
                continue
            h36m_points, h36m_scores = pose2d_mmpose.coco17_to_h36m17(keypoints[:17], keypoint_scores[:17])
            row["body2d"] = preview_export.coco_points(keypoints, keypoint_scores, decoded.width, decoded.height)
            row['pelvis2d'] = [(row['body2d'][11][axis] + row['body2d'][12][axis]) / 2 for axis in range(2)]
            if len(row['body2d']) >= 23:
                row['feet2d'] = {'L': row['body2d'][17:20], 'R': row['body2d'][20:23]}
            keypoints_seq.append(h36m_points)
            scores_seq.append(h36m_scores)
            bboxes_seq.append(bbox)
            frame_meta.append((frame_start + decoded.index, decoded.timestamp))
            if source.total_frames:
                reporter.processing_frame(frame_start + decoded.index, source.total_frames)
    finally:
        source.close()

    if not frame_meta:
        return [], source.fps, warnings, profile

    _raise_if_cancelled(cancel_token)
    reporter.loading_model(profile=profile, model_id=pose3d_motionbert.LIFTER_WEIGHTS, fraction=0.7)
    lifter = pose3d_motionbert.Body3DLifter(models_root, device, manifest, reporter)
    lifted = lifter.lift(keypoints_seq, scores_seq, image_size, bboxes_seq, cancel_token,
                         sample_fps=source.fps, max_gap_seconds=.1 if options.get('motion_type') == 'attack' else .2)
    valid_scores = [score for score in scores_seq if score is not None]
    if len(lifted) != len(frame_meta):
        raise errors.MocapError(errors.INTERNAL_ERROR, "MotionBERT 输出帧数与有效输入帧数不一致。")

    from statistics import median
    scale = median(pose3d_motionbert.estimate_metric_scale(pose) for pose in lifted) if len(lifted) else 1.0
    frames = []
    for index, (frame_number, timestamp) in enumerate(frame_meta):
        if frame_number < requested_start or (requested_end and frame_number > requested_end):
            continue
        body, confidence = pose3d_motionbert.h36m_to_standard(
            lifted[index], valid_scores[index], scale
        )
        if profile == 'quality_feet':
            observation = next(row for row in options['_preview_rows'] if row['sample_frame'] == frame_number)
            pose3d_motionbert.refine_feet_from_2d(body, observation['body2d'], image_size)
        frames.append(
            {
                "frame": frame_number,
                "time": timestamp,
                "body3d": body,
                "hands3d": {},
                "confidence": confidence,
                "contacts": {},
            }
        )

    options['_preview_rows'] = [row for row in options['_preview_rows'] if row['sample_frame'] >= requested_start
                               and (not requested_end or row['sample_frame'] <= requested_end)]
    if options.get("include_hands"):
        warnings.append(
            {
                "code": CODE_HANDS_DISABLED,
                "message": "当前 Quality 链路不恢复三维手指，已忽略 Include Hands。",
            }
        )
        reporter.warning(warnings[-1]["code"], warnings[-1]["message"])

    return frames, source.fps, warnings, profile
