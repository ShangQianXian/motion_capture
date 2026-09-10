"""RTMDet person detection + RTMPose 2D body keypoints (guide section 7.5).

Phase 5 adapter. ``mmdet``/``mmpose``/``torch`` are imported lazily; missing
packages, missing weights, missing configs, an absent CUDA device and CUDA OOM
all surface as structured errors so the pipeline can degrade instead of crashing.

This module cannot be exercised on a machine without the OpenMMLab stack and the
downloaded weights; see ``docs/ACCEPTANCE.md`` for the manual validation steps.
"""

from __future__ import annotations

from ._core import errors, model_manifest

#: Manifest artifact ids for the detector.
DETECTOR_WEIGHTS = "rtmdet_m_person"
DETECTOR_CONFIG = "config_rtmdet_m_person"

#: Manifest artifact ids per profile for the 2D body model.
BODY2D_ARTIFACTS = {
    "quality": ("rtmpose_m_body", "config_rtmpose_m_body"),
    "quality_plus": ("rtmpose_x_body", "config_rtmpose_x_body"),
    "fallback_cpu": ("rtmpose_m_body", "config_rtmpose_m_body"),
}

#: COCO person class index in the RTMDet person config.
PERSON_CLASS_ID = 0

#: Minimum detection score for a usable person box.
DEFAULT_BBOX_SCORE = 0.3

#: Number of COCO keypoints produced by the RTMPose body models.
COCO_KEYPOINT_COUNT = 17

#: Number of H36M keypoints consumed by MotionBERT.
H36M_KEYPOINT_COUNT = 17


def require_torch():
    """Import torch lazily."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 PyTorch：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 在 worker 虚拟环境中安装 CUDA 版 PyTorch。",
            details={"module": "torch"},
        )
    return torch


def require_numpy():
    try:
        import numpy  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 numpy：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行锁定的 worker 环境安装脚本。",
            details={"module": "numpy"},
        )
    return numpy


def require_mmdet():
    try:
        from mmdet.apis import inference_detector, init_detector  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmdet：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行 tools/bootstrap_worker_env.ps1 -Environment quality。",
            details={"module": "mmdet"},
        )
    return init_detector, inference_detector


def require_mmpose():
    try:
        from mmpose.apis import inference_topdown, init_model  # noqa: PLC0415
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mmpose：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行 tools/bootstrap_worker_env.ps1 -Environment quality。",
            details={"module": "mmpose"},
        )
    return init_model, inference_topdown


def is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` looks like a CUDA out-of-memory failure."""
    text = "{0} {1}".format(type(exc).__name__, exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text or "cublas_status_alloc_failed" in text


def oom_error(stage: str, device: str = "") -> errors.MocapError:
    """Structured ``CUDA_OOM`` error with the documented remediation."""
    return errors.MocapError(
        errors.CUDA_OOM,
        "CUDA 显存不足（{0}）。".format(stage),
        suggestion="切换到 quality profile、降低输入分辨率，或减小 Max VRAM GB。",
        details={"stage": stage, "device": device},
    )


def resolve_device(requested: str, reporter=None) -> str:
    """Validate the requested device, raising ``CUDA_UNAVAILABLE`` when needed."""
    device = str(requested or "cpu")
    if device == "cpu":
        return device
    torch = require_torch()
    if not torch.cuda.is_available():
        raise errors.MocapError(
            errors.CUDA_UNAVAILABLE,
            "请求了 {0}，但当前 worker 环境的 PyTorch 无法使用 CUDA。".format(device),
            suggestion="安装 CUDA 版 PyTorch，或改用 preview / fallback_cpu profile。",
            details={"device": device, "torch_version": getattr(torch, "__version__", "")},
        )
    if reporter is not None:
        try:
            reporter.debug("CUDA device: {0}".format(torch.cuda.get_device_name(0)))
        except Exception:  # pragma: no cover - informational only
            pass
    return device


class PersonDetector(object):
    """RTMDet-m person detector; keeps the largest, most confident box."""

    def __init__(self, models_root: str, device: str = "cuda:0", manifest=None, reporter=None) -> None:
        init_detector, inference_detector = require_mmdet()
        self._inference = inference_detector
        active = manifest if manifest is not None else model_manifest.load_manifest(models_root)
        self.config = model_manifest.require_artifact(models_root, DETECTOR_CONFIG, active)
        self.checkpoint = model_manifest.require_artifact(models_root, DETECTOR_WEIGHTS, active)
        self.device = device
        if reporter is not None:
            reporter.loading_model(model_id=DETECTOR_WEIGHTS)
        try:
            self.model = init_detector(self.config, self.checkpoint, device=device)
            from mmpose.utils import adapt_mmdet_pipeline
            self.model.cfg = adapt_mmdet_pipeline(self.model.cfg)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise oom_error("加载 RTMDet-m", device)
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法初始化 RTMDet-m：{0}".format(exc),
                details={"config": self.config, "checkpoint": self.checkpoint},
            )

    def detect(self, image, score_threshold: float = DEFAULT_BBOX_SCORE):
        """Return the best person bbox as ``(x1, y1, x2, y2, score)`` or ``None``."""
        numpy = require_numpy()
        try:
            result = self._inference(self.model, image)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise oom_error("RTMDet-m 推理", self.device)
            raise errors.MocapError(
                errors.INTERNAL_ERROR, "RTMDet-m 推理失败：{0}".format(exc)
            )

        instances = getattr(result, "pred_instances", None)
        if instances is None:
            return None
        bboxes = _to_numpy(numpy, getattr(instances, "bboxes", None))
        scores = _to_numpy(numpy, getattr(instances, "scores", None))
        labels = _to_numpy(numpy, getattr(instances, "labels", None))
        if bboxes is None or scores is None or len(bboxes) == 0:
            return None

        best = None
        best_key = None
        for index in range(len(bboxes)):
            if labels is not None and int(labels[index]) != PERSON_CLASS_ID:
                continue
            score = float(scores[index])
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in bboxes[index][:4])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            # Largest area first, score breaks ties (guide section 7.5).
            key = (area, score)
            if best_key is None or key > best_key:
                best_key = key
                best = (x1, y1, x2, y2, score)
        return best


class Body2DEstimator(object):
    """RTMPose 2D body keypoints (``-m`` for quality, ``-x`` for quality_plus)."""

    def __init__(
        self,
        models_root: str,
        profile: str = "quality",
        device: str = "cuda:0",
        manifest=None,
        reporter=None,
    ) -> None:
        init_model, inference_topdown = require_mmpose()
        self._inference = inference_topdown
        active = manifest if manifest is not None else model_manifest.load_manifest(models_root)
        weights_id, config_id = BODY2D_ARTIFACTS.get(profile, BODY2D_ARTIFACTS["quality"])
        self.profile = profile
        self.weights_id = weights_id
        self.config = model_manifest.require_artifact(models_root, config_id, active)
        self.checkpoint = model_manifest.require_artifact(models_root, weights_id, active)
        self.device = device
        if reporter is not None:
            reporter.loading_model(profile=profile, model_id=weights_id)
        try:
            self.model = init_model(self.config, self.checkpoint, device=device)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise oom_error("加载 {0}".format(weights_id), device)
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法初始化 {0}：{1}".format(weights_id, exc),
                details={"config": self.config, "checkpoint": self.checkpoint},
            )

    def estimate(self, image, bbox):
        """Return ``(keypoints, scores)`` as ``(17, 2)`` and ``(17,)`` arrays."""
        numpy = require_numpy()
        boxes = numpy.array([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=numpy.float32)
        try:
            results = self._inference(self.model, image, boxes, bbox_format="xyxy")
        except Exception as exc:
            if is_cuda_oom(exc):
                raise oom_error("{0} 推理".format(self.weights_id), self.device)
            raise errors.MocapError(
                errors.INTERNAL_ERROR, "{0} 推理失败：{1}".format(self.weights_id, exc)
            )
        if not results:
            return None, None
        instances = getattr(results[0], "pred_instances", None)
        if instances is None:
            return None, None
        keypoints = _to_numpy(numpy, getattr(instances, "keypoints", None))
        scores = _to_numpy(numpy, getattr(instances, "keypoint_scores", None))
        if keypoints is None or len(keypoints) == 0:
            return None, None
        points = numpy.asarray(keypoints[0], dtype=numpy.float32)
        if scores is None or len(scores) == 0:
            confidences = numpy.ones((points.shape[0],), dtype=numpy.float32)
        else:
            confidences = numpy.asarray(scores[0], dtype=numpy.float32)
        return points, confidences


def _to_numpy(numpy, value):
    """Convert a torch tensor / list / array to numpy without importing torch."""
    if value is None:
        return None
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        try:
            value = cpu()
        except Exception:  # pragma: no cover - already on cpu
            pass
    detach = getattr(value, "detach", None)
    if detach is not None:
        try:
            value = detach()
        except Exception:  # pragma: no cover
            pass
    numpy_fn = getattr(value, "numpy", None)
    if numpy_fn is not None:
        try:
            return numpy_fn()
        except Exception:  # pragma: no cover
            pass
    return numpy.asarray(value)


def coco17_to_h36m17(keypoints, scores):
    """Convert COCO-17 keypoints to the H36M-17 layout MotionBERT expects.

    Implemented locally rather than relying on
    ``mmpose.apis.convert_keypoint_definition`` so the pipeline does not break
    when that private helper moves between MMPose releases.
    """
    numpy = require_numpy()
    points = numpy.asarray(keypoints, dtype=numpy.float32)
    confidence = numpy.asarray(scores, dtype=numpy.float32)
    if points.shape[0] < COCO_KEYPOINT_COUNT:
        raise errors.MocapError(
            errors.INTERNAL_ERROR,
            "2D 关键点数量不足：{0}（期望 {1}）。".format(points.shape[0], COCO_KEYPOINT_COUNT),
        )

    out = numpy.zeros((H36M_KEYPOINT_COUNT, points.shape[1]), dtype=numpy.float32)
    out_conf = numpy.zeros((H36M_KEYPOINT_COUNT,), dtype=numpy.float32)

    def blend(target, sources, weights=None):
        indices = list(sources)
        if weights is None:
            weights = [1.0 / len(indices)] * len(indices)
        out[target] = sum(points[i] * w for i, w in zip(indices, weights))
        out_conf[target] = min(confidence[i] for i in indices)

    blend(0, (11, 12))          # pelvis
    out[1], out_conf[1] = points[12], confidence[12]   # right hip
    out[2], out_conf[2] = points[14], confidence[14]   # right knee
    out[3], out_conf[3] = points[16], confidence[16]   # right ankle
    out[4], out_conf[4] = points[11], confidence[11]   # left hip
    out[5], out_conf[5] = points[13], confidence[13]   # left knee
    out[6], out_conf[6] = points[15], confidence[15]   # left ankle
    blend(8, (5, 6))            # thorax / chest
    out[7] = (out[0] + out[8]) * 0.5                   # spine
    out_conf[7] = min(out_conf[0], out_conf[8])
    out[9], out_conf[9] = points[0], confidence[0]     # neck / nose
    blend(10, (1, 2))          # H36M head: midpoint of the COCO eyes
    out[11], out_conf[11] = points[5], confidence[5]   # left shoulder
    out[12], out_conf[12] = points[7], confidence[7]   # left elbow
    out[13], out_conf[13] = points[9], confidence[9]   # left wrist
    out[14], out_conf[14] = points[6], confidence[6]   # right shoulder
    out[15], out_conf[15] = points[8], confidence[8]   # right elbow
    out[16], out_conf[16] = points[10], confidence[10]  # right wrist
    return out, out_conf
