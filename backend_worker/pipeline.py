"""Profile dispatch, degradation rules and environment checks.

Guide sections 4.2, 6.2 and 7. Every stage module is imported lazily so that a
``--mock`` run, ``--check-env`` and the unit tests work in a bare Python
installation with no inference dependencies at all.
"""

from __future__ import annotations

import os
import platform
import sys

from ._core import errors, job_schema, model_manifest, paths, result_schema, skeleton

from . import export_result, mock_source, postprocess

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
MMPOSE_PROFILES = ("quality", "quality_plus")

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


def check_env() -> dict:
    """Payload for ``--check-env``."""
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable or "",
        "platform": platform.platform(),
        "addon_root": paths.addon_root(),
        "modules": module_report(),
        "skeleton": skeleton.SKELETON_ID,
        "result_version": result_schema.RESULT_VERSION,
    }


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
    input_section = job.get("input") or {}
    output_section = job.get("output") or {}

    profile = str(model_section.get("profile") or "preview")
    models_root = str(model_section.get("models_root") or "")

    if mode == job_schema.MODE_SELF_TEST:
        return _run_self_test(job, reporter)

    if mock:
        return _run_mock(job, reporter, cancel_token)

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

    if profile in MEDIAPIPE_PROFILES:
        frames, fps, warnings = _run_mediapipe(
            job, profile, manifest, reporter, cancel_token, options
        )
    elif profile in MMPOSE_PROFILES:
        frames, fps, warnings = _run_mmpose(
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

    frames, post_warnings = postprocess.postprocess(frames, fps, options, reporter)
    warnings = list(warnings) + list(post_warnings)

    result = export_result.build_result(
        frames,
        fps,
        source_path=str(input_section.get("path") or ""),
        source_type=str(input_section.get("type") or "video"),
        profile=profile,
        warnings=postprocess.summarise_warnings(warnings),
    )
    return export_result.write_result(
        result, output_section.get("dir"), output_section.get("result_filename")
    )


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
            timestamp_ms = int(round(decoded.timestamp * 1000.0)) if video_mode else 0
            detection = body_estimator.detect(decoded.image, timestamp_ms)
            world = getattr(detection, "pose_world_landmarks", None) or []
            normalized = getattr(detection, "pose_landmarks", None) or []
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

            offset = pose_mediapipe.standing_offset(body)
            drift, first_center = pose_mediapipe.horizontal_offset(
                normalized[0] if normalized else None, reference_scale, first_center
            )
            total_offset = (offset[0] + drift[0], offset[1] + drift[1], offset[2] + drift[2])
            body = {name: _translate(value, total_offset) for name, value in body.items()}

            hands = {}
            if hand_estimator is not None:
                hand_result = hand_estimator.detect(decoded.image, timestamp_ms)
                hands = pose_mediapipe.hand_landmarks_to_standard(hand_result, body)

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

    source = media_decode.open_media(
        str(input_section.get("path") or ""),
        str(input_section.get("type") or "auto"),
        float(input_section.get("target_fps") or 30.0),
        int(input_section.get("frame_start") or 1),
        int(input_section.get("frame_end") or 0),
        cancel_token=cancel_token,
        reporter=reporter,
    )

    frame_start = int(input_section.get("frame_start") or 1)
    keypoints_seq = []
    scores_seq = []
    frame_meta = []
    image_size = None

    try:
        for decoded in source:
            _raise_if_cancelled(cancel_token)
            if image_size is None:
                image_size = (decoded.width, decoded.height)
            bbox = detector.detect(decoded.image)
            if bbox is None:
                reporter.warning(
                    errors.NO_PERSON_DETECTED,
                    "第 {0} 帧未检测到人体。".format(frame_start + decoded.index),
                    frame=frame_start + decoded.index,
                )
                continue
            keypoints, keypoint_scores = body2d.estimate(decoded.image, bbox)
            if keypoints is None:
                reporter.warning(
                    errors.NO_PERSON_DETECTED,
                    "第 {0} 帧 2D 姿态估计为空。".format(frame_start + decoded.index),
                    frame=frame_start + decoded.index,
                )
                continue
            h36m_points, h36m_scores = pose2d_mmpose.coco17_to_h36m17(keypoints, keypoint_scores)
            keypoints_seq.append(h36m_points)
            scores_seq.append(h36m_scores)
            frame_meta.append((frame_start + decoded.index, decoded.timestamp))
            if source.total_frames:
                reporter.processing_frame(frame_start + decoded.index, source.total_frames)
    finally:
        source.close()

    if not keypoints_seq:
        return [], source.fps, warnings

    _raise_if_cancelled(cancel_token)
    reporter.loading_model(profile=profile, model_id=pose3d_motionbert.LIFTER_WEIGHTS, fraction=0.7)
    lifter = pose3d_motionbert.Body3DLifter(models_root, device, manifest, reporter)
    lifted = lifter.lift(keypoints_seq, scores_seq, image_size)

    scale = pose3d_motionbert.estimate_metric_scale(lifted[0]) if len(lifted) else 1.0
    frames = []
    for index, (frame_number, timestamp) in enumerate(frame_meta):
        if index >= len(lifted):
            break
        body, confidence = pose3d_motionbert.h36m_to_standard(
            lifted[index], scores_seq[index], scale
        )
        body = pose3d_motionbert.ground_and_stand(body)
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

    if options.get("include_hands"):
        warnings.append(
            {
                "code": CODE_HANDS_DISABLED,
                "message": "quality 链路 v0.1 不做手部捕捉，已忽略 Include Hands。",
            }
        )
        reporter.warning(warnings[-1]["code"], warnings[-1]["message"])

    return frames, source.fps, warnings
