"""MotionBERT 3D pose lifting (guide section 7.5).

Phase 5 adapter. Consumes a sequence of H36M-17 2D keypoints and produces
per-frame 3D joints already converted into the standard skeleton and coordinate
system, so no model-specific axis convention ever reaches Blender.

Cannot be exercised without the OpenMMLab stack and the MotionBERT weights; see
``docs/ACCEPTANCE.md``.
"""

from __future__ import annotations


from ._core import errors, model_manifest, skeleton
from ._core import retarget_math as rm

from . import pose2d_mmpose as mm

#: Manifest artifact ids.
LIFTER_WEIGHTS = "motionbert_body3d"
LIFTER_CONFIG = "config_motionbert_body3d"
FALLBACK_WEIGHTS = "videopose3d_body3d"

#: MotionBERT temporal window (guide section 7.5).
WINDOW_SIZE = 243

#: Nominal hip-to-head height used to scale root-relative output into metres.
NOMINAL_PELVIS_TO_HEAD = 0.78

#: Forward offset used to synthesise the missing toe joints, in metres.
TOE_FORWARD_OFFSET = 0.16

#: Input-normalisation modes accepted by :meth:`Body3DLifter.lift`.
#:
#: ``current`` feeds raw video pixels. ``MotionBERTLabel.encode`` then divides
#: them by the *image* size, so the magnitude the network sees is decided by how
#: much of the frame the subject happens to fill. ``canonical`` rebases the
#: sequence into the H36M statistics below first, which is the distribution the
#: checkpoint was trained on.
NORMALISATION_CURRENT = "current"
NORMALISATION_CANONICAL = "canonical"

#: H36M mean bounding box shipped with MMPose (``_base_/datasets/h36m.py``,
#: ``stats_info``). MotionBERT was trained on keypoints rebased onto this box,
#: and the checkpoint assumes 2D input normalised to roughly ``[-1, 1]``, so a
#: shorter subject moves that input out of distribution.
CANONICAL_IMAGE_SIZE = (1000.0, 1000.0)
CANONICAL_BBOX_CENTER = (528.0, 427.0)
CANONICAL_BBOX_SCALE = 400.0

#: Minimum sequence-wide keypoint spread, in pixels, that can be rebased.
MIN_CANONICAL_SPREAD = 1.0


def require_lifter_api():
    """Import MMPose's pose-lifting API lazily."""
    try:
        from mmpose.apis import inference_pose_lifter_model, init_model  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmpose 的 3D 提升接口：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行 tools/bootstrap_worker_env.ps1 -Environment quality。",
            details={"module": "mmpose.apis.inference_pose_lifter_model"},
        )
    try:
        from mmengine.structures import InstanceData  # noqa: PLC0415
        from mmpose.structures import PoseDataSample  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmengine/mmpose 数据结构：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行 tools/bootstrap_worker_env.ps1 -Environment quality。",
            details={"module": "mmpose.structures"},
        )
    return init_model, inference_pose_lifter_model, PoseDataSample, InstanceData


def pad_sequence(sequence, window: int = WINDOW_SIZE) -> list:
    """Pad a short sequence to ``window`` frames by repeating the boundary frames."""
    if not sequence:
        return []
    if len(sequence) >= window:
        return list(sequence)
    missing = window - len(sequence)
    head = missing // 2
    tail = missing - head
    return [sequence[0]] * head + list(sequence) + [sequence[-1]] * tail


def canonical_rebase(keypoints_seq, bboxes_seq=None):
    """Rebase a 2D sequence onto the H36M bounding box the checkpoint expects.

    The mapping is affine and shape-preserving: every frame is translated so the
    subject sits on ``CANONICAL_BBOX_CENTER``, and the whole sequence is scaled
    by one factor so its largest keypoint spread equals ``2 * CANONICAL_BBOX_SCALE``.

    One factor for the entire sequence is deliberate. Dividing by a per-frame
    bounding box would cancel exactly the apparent-size change that carries the
    subject's depth, which is the only monocular depth cue available. This is
    also what the reference implementation does: ``crop_scale`` in MotionBERT's
    ``lib/utils/utils_data.py`` normalises a clip using its sequence-wide
    bounding box, not a per-frame one.

    Returns ``(keypoints, bboxes)``; both are rebased so the 2D evidence and the
    boxes stay in the same frame. Returns the inputs unchanged when the sequence
    has no usable spread, so a degenerate clip still produces a result object
    instead of raising.
    """
    numpy = mm.require_numpy()
    points = [numpy.asarray(item, dtype=numpy.float32) for item in keypoints_seq if item is not None]
    if not points:
        return keypoints_seq, bboxes_seq
    stacked = numpy.concatenate(points, axis=0)
    low = stacked[:, :2].min(axis=0)
    high = stacked[:, :2].max(axis=0)
    spread = float(max(high[0] - low[0], high[1] - low[1]))
    if not numpy.isfinite(spread) or spread < MIN_CANONICAL_SPREAD:
        return keypoints_seq, bboxes_seq
    scale = 2.0 * CANONICAL_BBOX_SCALE / spread
    center = numpy.asarray(CANONICAL_BBOX_CENTER, dtype=numpy.float32)

    def rebase_points(item):
        if item is None:
            return None
        array = numpy.asarray(item, dtype=numpy.float32).copy()
        array[:, :2] = (array[:, :2] - low) * scale + center - CANONICAL_BBOX_SCALE
        return array

    rebased = [rebase_points(item) for item in keypoints_seq]
    boxes = None
    if bboxes_seq is not None:
        boxes = []
        for bbox in bboxes_seq:
            if bbox is None:
                boxes.append(None)
                continue
            array = numpy.asarray(bbox, dtype=numpy.float32)[:4].copy()
            array[:2] = (array[:2] - low) * scale + center - CANONICAL_BBOX_SCALE
            array[2:4] = (array[2:4] - low) * scale + center - CANONICAL_BBOX_SCALE
            boxes.append(array)
    return rebased, boxes


def build_windows(count: int, window: int = WINDOW_SIZE) -> list:
    """Split ``count`` frames into ``(start, end)`` half-open windows."""
    if count <= 0:
        return []
    if count <= window:
        return [(0, count)]
    windows = []
    start = 0
    while start < count:
        end = min(count, start + window)
        windows.append((start, end))
        start = end
    return windows


def _align_motionbert_targets(data: dict) -> dict:
    """Match MMPose's dummy 3D targets to MotionBERT's temporal input.

    MMPose 1.3.2's generic lifter API supplies one target frame even for a
    243-frame input. MotionBERTLabel scales targets in place with a per-frame
    factor, so both targets and their visibility must cover the input window.
    These are inference placeholders, not ground-truth or predicted poses.
    """
    numpy = mm.require_numpy()
    aligned = dict(data)
    frames = data["keypoints"].shape[0]
    for name in ("lifting_target", "lifting_target_visible"):
        target = data.get(name)
        if target is None:
            continue
        if target.shape[0] not in (1, frames):
            raise ValueError("MotionBERT {0} has {1} frames; expected 1 or {2}".format(
                name, target.shape[0], frames))
        if target.shape[0] == 1 and frames != 1:
            aligned[name] = numpy.repeat(target, frames, axis=0)
    return aligned


def _prepare_motionbert_inference_pipeline(model) -> None:
    """Adapt only this model's MotionBERT encoder; leave package/config files intact."""
    dataset = model.cfg.test_dataloader.dataset
    pipeline = []
    def prepare(data):
        data = _align_motionbert_targets(data)
        scores = getattr(model, '_mocap_input_scores', None)
        if scores is not None:
            data['keypoints_visible'] = scores.copy()
        return data
    for transform in dataset.pipeline:
        if (isinstance(transform, dict) and transform.get("type") == "GenerateTarget"
                and isinstance(transform.get("encoder"), dict)
                and transform["encoder"].get("type") == "MotionBERTLabel"):
            if not pipeline or pipeline[-1] is not _align_motionbert_targets:
                pipeline.append(prepare)
        pipeline.append(transform)
    dataset.pipeline = pipeline


class Body3DLifter(object):
    """MotionBERT DSTFormer lifter."""

    def __init__(self, models_root: str, device: str = "cuda:0", manifest=None, reporter=None) -> None:
        init_model, inference_pose_lifter_model, pose_sample, instance_data = require_lifter_api()
        self._inference = inference_pose_lifter_model
        self._PoseDataSample = pose_sample
        self._InstanceData = instance_data
        #: Input normalisation actually used by the last :meth:`lift` call.
        self.last_normalisation = NORMALISATION_CURRENT
        active = manifest if manifest is not None else model_manifest.load_manifest(models_root)
        self.config = model_manifest.require_artifact(models_root, LIFTER_CONFIG, active)
        self.checkpoint = model_manifest.require_artifact(models_root, LIFTER_WEIGHTS, active)
        self.device = device
        if reporter is not None:
            reporter.loading_model(model_id=LIFTER_WEIGHTS)
        try:
            self.model = init_model(self.config, self.checkpoint, device=device)
            _prepare_motionbert_inference_pipeline(self.model)
        except Exception as exc:
            if mm.is_cuda_oom(exc):
                raise mm.oom_error("加载 MotionBERT", device)
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法初始化 MotionBERT：{0}".format(exc),
                details={"config": self.config, "checkpoint": self.checkpoint},
            )

    def _to_samples(self, keypoints_seq, scores_seq, bboxes_seq=None):
        numpy = mm.require_numpy()
        samples = []
        for index, (keypoints, scores) in enumerate(zip(keypoints_seq, scores_seq)):
            if keypoints is None:
                samples.append([])
                continue
            sample = self._PoseDataSample()
            instances = self._InstanceData()
            instances.keypoints = numpy.asarray(keypoints, dtype=numpy.float32)[None, ...]
            instances.keypoint_scores = numpy.asarray(scores, dtype=numpy.float32)[None, ...]
            points = instances.keypoints[0]
            if points.shape != (17, 2) or instances.keypoint_scores.shape != (1, 17):
                raise errors.MocapError(errors.INTERNAL_ERROR, "MotionBERT 输入必须为 17 个二维关节及置信度。")
            if not numpy.isfinite(points).all() or not numpy.isfinite(instances.keypoint_scores).all():
                raise errors.MocapError(errors.INTERNAL_ERROR, "MotionBERT 输入包含非有限值。")
            bbox = bboxes_seq[index] if bboxes_seq is not None else None
            if bbox is None:
                low, high = points.min(axis=0), points.max(axis=0)
                high = numpy.maximum(high, low + 1.0)
                bbox = (low[0], low[1], high[0], high[1])
            instances.bboxes = numpy.asarray(bbox[:4], dtype=numpy.float32)[None, ...]
            sample.pred_instances = instances
            sample.gt_instances = self._InstanceData()
            sample.track_id = 0
            samples.append([sample])
        return samples

    def lift(self, keypoints_seq, scores_seq, image_size=None, bboxes_seq=None, cancel_token=None,
             sample_fps=30., max_gap_seconds=.2, input_normalisation=NORMALISATION_CURRENT):
        """Lift a 2D sequence into ``(frames, 17, 3)`` root-relative coordinates.

        ``input_normalisation`` selects the 2D preprocessing:

        * ``current`` keeps the historical behaviour and feeds raw video pixels.
        * ``canonical`` rebases the sequence onto the H36M statistics first, so
          the network sees the input scale it was trained on.

        The returned coordinates are root-relative either way; the units are the
        model's own and are rescaled by :func:`estimate_metric_scale`.
        """
        numpy = mm.require_numpy()
        if len(keypoints_seq) == 0:
            return numpy.zeros((0, mm.H36M_KEYPOINT_COUNT, 3), dtype=numpy.float32)
        if input_normalisation not in (NORMALISATION_CURRENT, NORMALISATION_CANONICAL):
            raise errors.MocapError(
                errors.JOB_SCHEMA_INVALID,
                "未知的二维输入归一化模式：{0}".format(input_normalisation),
                details={"input_normalisation": str(input_normalisation)},
            )
        frame_count = len(keypoints_seq)
        if len(scores_seq) != frame_count or (bboxes_seq is not None and len(bboxes_seq) != frame_count):
            raise errors.MocapError(errors.INTERNAL_ERROR, "MotionBERT 输入序列长度不一致。")
        if image_size is None or len(image_size) != 2 or min(image_size) <= 0:
            raise errors.MocapError(errors.INTERNAL_ERROR, "MotionBERT 需要有效的图像宽高。")
        self.last_normalisation = input_normalisation
        if input_normalisation == NORMALISATION_CANONICAL:
            keypoints_seq, bboxes_seq = canonical_rebase(keypoints_seq, bboxes_seq)
            # The rebased keypoints no longer live in the video frame, so the
            # width/height that ``MotionBERTLabel.encode`` divides by must be the
            # canonical ones. Reusing the real video size here would scale the
            # input straight back out of distribution.
            image_size = CANONICAL_IMAGE_SIZE
        samples = self._to_samples(keypoints_seq, scores_seq, bboxes_seq)
        valid = [index for index, sample in enumerate(samples) if sample]
        if not valid:
            raise errors.MocapError(errors.NO_PERSON_DETECTED, "没有可提升的二维姿态。")
        # Keep the sampling grid, interpolate only bounded short gaps, and prevent
        # the model's receptive field from reaching across a long occlusion.
        bounds = {}
        segments = [[valid[0]]]
        for a, b in zip(valid, valid[1:]):
            if (b - a - 1) / sample_fps > max_gap_seconds + 1e-7:
                segments.append([])
            else:
                for missing in range(a + 1, b):
                    fraction = (missing - a) / (b - a)
                    points = numpy.asarray(keypoints_seq[a]) * (1 - fraction) + numpy.asarray(keypoints_seq[b]) * fraction
                    samples[missing] = self._to_samples([points], [numpy.zeros(17)])[0]
            segments[-1].append(b)
        for segment in segments:
            for index in segment:
                bounds[index] = (segment[0], segment[-1])
        window = int(self.model.cfg.model.backbone.get("seq_len", WINDOW_SIZE))
        if window <= 0 or window % 2 == 0:
            raise errors.MocapError(errors.CONFIG_MISSING, "MotionBERT 时间窗口必须为正奇数。")
        dataset = self.model.cfg.test_dataloader.dataset
        causal = bool(dataset.get("causal", False))
        step = int(dataset.get("seq_step", 1))
        if step <= 0:
            raise errors.MocapError(errors.CONFIG_MISSING, "MotionBERT seq_step 必须为正整数。")
        target_index = window - 1 if causal else window // 2
        output = []
        for index in valid:
            if cancel_token is not None and cancel_token.cancelled():
                raise errors.MocapError(errors.CANCELLED, "MotionBERT 推理已取消。")
            first, last = bounds[index]
            temporal = [samples[max(first, min(last, index + (offset - target_index) * step))]
                        for offset in range(window)]
            self.model._mocap_input_scores = numpy.stack([sample[0].pred_instances.keypoint_scores[0] for sample in temporal])
            try:
                results = self._inference(self.model, temporal, with_track_id=True,
                                          image_size=image_size, norm_pose_2d=False)
            except Exception as exc:
                if mm.is_cuda_oom(exc):
                    raise mm.oom_error("MotionBERT 推理", self.device)
                raise errors.MocapError(
                    errors.INTERNAL_ERROR, "MotionBERT 推理失败：{0}".format(exc)
                )

            lifted = _extract_keypoints_3d(numpy, results)
            if lifted is None or len(lifted) not in (1, window):
                raise errors.MocapError(
                    errors.INTERNAL_ERROR,
                    "MotionBERT 未返回可解析的 3D 关键点。",
                    details={"frame": index, "window_size": window},
                )
            output.append(lifted[0 if len(lifted) == 1 else target_index])
        return numpy.stack(output, axis=0)


def _extract_keypoints_3d(numpy, results):
    """Decode one person's output, retaining its temporal axis (not person count)."""
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        return None
    instances = getattr(results[0], "pred_instances", None)
    keypoints = getattr(instances, "keypoints", None)
    if keypoints is None:
        return None
    array = numpy.asarray(mm._to_numpy(numpy, keypoints), dtype=numpy.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (17, 3) or not numpy.isfinite(array).all():
        return None
    return array


def h36m_to_standard(joints_3d, confidences=None, scale: float = 1.0) -> tuple:
    """Convert one H36M-17 frame to standard-skeleton joints.

    Returns ``(body3d, confidence)``. Missing toes are synthesised from the ankle
    and the foot forward direction, and flagged with a low confidence.
    """
    body = {}
    confidence = {}
    for index, name in skeleton.H36M_17_MAP.items():
        if index >= len(joints_3d):
            continue
        x, y, z = (float(v) for v in joints_3d[index][:3])
        body[name] = rm.vec_scale(skeleton.h36m_to_blender(x, y, z), scale)
        if confidences is not None and index < len(confidences):
            confidence[name] = round(float(confidences[index]), 4)

    for side in ("L", "R"):
        ankle = body.get("ankle.{0}".format(side))
        if ankle is None:
            continue
        forward = (0.0, -TOE_FORWARD_OFFSET, 0.0)
        from ._core import orientations
        if all(n in body for n in ('shoulder.L', 'shoulder.R', 'chest', 'pelvis')):
            forward = rm.quat_rotate_vector(orientations.body_basis(body), forward)
        toe = rm.vec_add(ankle, forward)
        # Coordinates are still pelvis-relative: ankles normally have negative Z.
        # Grounding happens once for the whole body in ground_and_stand().
        body["toe.{0}".format(side)] = (toe[0], toe[1], ankle[2] - 0.04)
        confidence["toe.{0}".format(side)] = skeleton.SYNTHESISED_CONFIDENCE

    scores = [v for k, v in confidence.items() if k in skeleton.BODY_JOINTS]
    if scores:
        confidence["body_mean"] = round(sum(scores) / float(len(scores)), 4)
    return body, confidence


def estimate_metric_scale(joints_3d) -> float:
    """Scale factor mapping root-relative output onto a nominal human height."""
    try:
        pelvis = joints_3d[0][:3]
        head = joints_3d[10][:3]
    except (IndexError, TypeError):
        return 1.0
    distance = rm.vec_distance(
        (float(pelvis[0]), float(pelvis[1]), float(pelvis[2])),
        (float(head[0]), float(head[1]), float(head[2])),
    )
    if distance < 1e-6:
        return 1.0
    return NOMINAL_PELVIS_TO_HEAD / distance


def ground_and_stand(body3d: dict) -> dict:
    """Translate a root-relative pose so the lowest foot rests on ``z = 0``."""
    heights = [
        body3d[name][2]
        for name in ("toe.L", "toe.R", "ankle.L", "ankle.R", "heel.L", "heel.R")
        if name in body3d
    ]
    if not heights:
        return body3d
    offset = (0.0, 0.0, -min(heights))
    return {name: rm.vec_add(value, offset) for name, value in body3d.items()}
