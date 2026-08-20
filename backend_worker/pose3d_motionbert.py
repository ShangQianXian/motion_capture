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


def require_lifter_api():
    """Import MMPose's pose-lifting API lazily."""
    try:
        from mmpose.apis import inference_pose_lifter_model, init_model  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmpose 的 3D 提升接口：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 执行 mim install mmpose。",
            details={"module": "mmpose.apis.inference_pose_lifter_model"},
        )
    try:
        from mmengine.structures import InstanceData  # noqa: PLC0415
        from mmpose.structures import PoseDataSample  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmengine/mmpose 数据结构：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 执行 mim install mmengine mmpose。",
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


class Body3DLifter(object):
    """MotionBERT DSTFormer lifter."""

    def __init__(self, models_root: str, device: str = "cuda:0", manifest=None, reporter=None) -> None:
        init_model, inference_pose_lifter_model, pose_sample, instance_data = require_lifter_api()
        self._inference = inference_pose_lifter_model
        self._PoseDataSample = pose_sample
        self._InstanceData = instance_data
        active = manifest if manifest is not None else model_manifest.load_manifest(models_root)
        self.config = model_manifest.require_artifact(models_root, LIFTER_CONFIG, active)
        self.checkpoint = model_manifest.require_artifact(models_root, LIFTER_WEIGHTS, active)
        self.device = device
        if reporter is not None:
            reporter.loading_model(model_id=LIFTER_WEIGHTS)
        try:
            self.model = init_model(self.config, self.checkpoint, device=device)
        except Exception as exc:
            if mm.is_cuda_oom(exc):
                raise mm.oom_error("加载 MotionBERT", device)
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法初始化 MotionBERT：{0}".format(exc),
                details={"config": self.config, "checkpoint": self.checkpoint},
            )

    def _to_samples(self, keypoints_seq, scores_seq):
        numpy = mm.require_numpy()
        samples = []
        for keypoints, scores in zip(keypoints_seq, scores_seq):
            sample = self._PoseDataSample()
            instances = self._InstanceData()
            instances.keypoints = numpy.asarray(keypoints, dtype=numpy.float32)[None, ...]
            instances.keypoint_scores = numpy.asarray(scores, dtype=numpy.float32)[None, ...]
            sample.pred_instances = instances
            samples.append(sample)
        return samples

    def lift(self, keypoints_seq, scores_seq, image_size=None):
        """Lift a 2D sequence into ``(frames, 17, 3)`` root-relative coordinates."""
        numpy = mm.require_numpy()
        if not keypoints_seq:
            return numpy.zeros((0, mm.H36M_KEYPOINT_COUNT, 3), dtype=numpy.float32)

        frame_count = len(keypoints_seq)
        output = numpy.zeros((frame_count, mm.H36M_KEYPOINT_COUNT, 3), dtype=numpy.float32)

        for start, end in build_windows(frame_count):
            window_keypoints = list(keypoints_seq[start:end])
            window_scores = list(scores_seq[start:end])
            padded_keypoints = pad_sequence(window_keypoints)
            padded_scores = pad_sequence(window_scores)
            offset = (len(padded_keypoints) - len(window_keypoints)) // 2
            samples = self._to_samples(padded_keypoints, padded_scores)
            try:
                kwargs = {}
                if image_size is not None:
                    kwargs["image_size"] = image_size
                    kwargs["norm_pose_2d"] = True
                results = self._inference(self.model, samples, **kwargs)
            except TypeError:
                results = self._inference(self.model, samples)
            except Exception as exc:
                if mm.is_cuda_oom(exc):
                    raise mm.oom_error("MotionBERT 推理", self.device)
                raise errors.MocapError(
                    errors.INTERNAL_ERROR, "MotionBERT 推理失败：{0}".format(exc)
                )

            lifted = _extract_keypoints_3d(numpy, results)
            if lifted is None:
                raise errors.MocapError(
                    errors.INTERNAL_ERROR,
                    "MotionBERT 未返回可解析的 3D 关键点。",
                    details={"window": [start, end]},
                )
            for index in range(end - start):
                source = offset + index
                if source < len(lifted):
                    output[start + index] = lifted[source]
        return output


def _extract_keypoints_3d(numpy, results):
    """Pull ``(frames, 17, 3)`` out of whatever shape MMPose returned."""
    if results is None:
        return None
    frames = []
    sequence = results if isinstance(results, (list, tuple)) else [results]
    for item in sequence:
        instances = getattr(item, "pred_instances", None)
        if instances is None and isinstance(item, dict):
            instances = item.get("pred_instances") or item
        keypoints = None
        if instances is not None:
            keypoints = getattr(instances, "keypoints", None)
            if keypoints is None and isinstance(instances, dict):
                keypoints = instances.get("keypoints")
        if keypoints is None:
            continue
        array = mm._to_numpy(numpy, keypoints)
        array = numpy.asarray(array, dtype=numpy.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2 or array.shape[-1] < 3:
            continue
        frames.append(array[:, :3])
    if not frames:
        return None
    return numpy.stack(frames, axis=0)


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
        knee = body.get("knee.{0}".format(side))
        if ankle is None:
            continue
        forward = (0.0, -TOE_FORWARD_OFFSET, 0.0)
        if knee is not None:
            down = rm.vec_normalize(rm.vec_sub(ankle, knee))
            # Foot points roughly forward and perpendicular to the shin.
            side_axis = rm.vec_cross(down, (0.0, 0.0, 1.0))
            forward_dir = rm.vec_normalize(rm.vec_cross(side_axis, down)) if rm.vec_length(side_axis) > 1e-6 else (0.0, -1.0, 0.0)
            if forward_dir[1] > 0.0:
                forward_dir = rm.vec_neg(forward_dir)
            forward = rm.vec_scale(forward_dir, TOE_FORWARD_OFFSET)
        toe = rm.vec_add(ankle, forward)
        body["toe.{0}".format(side)] = (toe[0], toe[1], max(0.0, ankle[2] - 0.04))
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
