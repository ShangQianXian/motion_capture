"""MediaPipe Pose / Hand backend for the ``preview`` and ``fallback_cpu`` profiles.

Guide section 7.4. MediaPipe is imported lazily; a missing package or model file
surfaces as a structured ``DEPENDENCY_MISSING`` / ``MODEL_MISSING`` error rather
than a traceback.

Known v0.1 limitation: MediaPipe world landmarks are hip-centred and carry no
reliable global translation. The estimator reconstructs a plausible standing
height and approximates horizontal travel from the normalised landmarks, and
always reports ``MEDIAPIPE_WORLD_APPROXIMATE`` so the user knows root motion is
an approximation.
"""

from __future__ import annotations

from ._core import errors, model_manifest, skeleton
from ._core import retarget_math as rm

#: Manifest artifact ids per requested quality, in preference order.
BODY_MODEL_PREFERENCE = {
    "preview": ("mediapipe_pose_full", "mediapipe_pose_heavy", "mediapipe_pose_lite"),
    "fallback_cpu": ("mediapipe_pose_lite", "mediapipe_pose_full", "mediapipe_pose_heavy"),
    "heavy": ("mediapipe_pose_heavy", "mediapipe_pose_full", "mediapipe_pose_lite"),
}

#: Hand model artifact id.
HAND_MODEL_ID = "mediapipe_hand"

#: Warning code emitted once per run about approximate world coordinates.
CODE_WORLD_APPROXIMATE = "MEDIAPIPE_WORLD_APPROXIMATE"

#: Warning code emitted when the optional hand model is unavailable.
CODE_HANDS_DISABLED = "HANDS_DISABLED"


def _require_mediapipe():
    try:
        import mediapipe  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 mediapipe：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 运行 tools/bootstrap_worker_env.ps1 -Environment preview。",
            details={"module": "mediapipe"},
        )
    return mediapipe


def select_body_model(models_root: str, profile: str, manifest=None) -> tuple:
    """Pick the best available body model for ``profile``.

    Returns ``(artifact_id, absolute_path)``. Raises ``MODEL_MISSING`` naming the
    profile's primary artifact when nothing is available.
    """
    active = manifest if manifest is not None else model_manifest.load_manifest(models_root)
    order = BODY_MODEL_PREFERENCE.get(profile) or BODY_MODEL_PREFERENCE["preview"]
    for artifact_id in order:
        status = active.artifact_status(artifact_id)
        if status is not None and status.exists:
            return artifact_id, status.absolute_path
    return None, model_manifest.require_artifact(models_root, order[0], active)


class MediaPipeBodyEstimator(object):
    """Thin wrapper around ``vision.PoseLandmarker``."""

    def __init__(self, model_path: str, video_mode: bool = True, num_poses: int = 1) -> None:
        mediapipe = _require_mediapipe()
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on mediapipe build
            raise errors.MocapError(
                errors.DEPENDENCY_MISSING,
                "mediapipe 版本过旧，缺少 tasks API：{0}".format(exc),
                suggestion="按 docs/INSTALL.md 重新安装锁定的 preview 环境。",
                details={"module": "mediapipe.tasks"},
            )
        self._mp = mediapipe
        self._vision = vision
        self.video_mode = bool(video_mode)
        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_poses=max(1, int(num_poses)),
                output_segmentation_masks=False,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法加载 MediaPipe Pose 模型 {0}：{1}".format(model_path, exc),
                details={"model_path": model_path},
            )

    def close(self) -> None:
        closer = getattr(self.landmarker, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:  # pragma: no cover - defensive
                pass

    def detect(self, image_bgr, timestamp_ms: int = 0):
        """Run the landmarker on a BGR image; returns the MediaPipe result."""
        rgb = image_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=_ascontiguous(rgb))
        if self.video_mode:
            return self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        return self.landmarker.detect(mp_image)


class MediaPipeHandEstimator(object):
    """Thin wrapper around ``vision.HandLandmarker`` (optional)."""

    def __init__(self, model_path: str, video_mode: bool = True) -> None:
        mediapipe = _require_mediapipe()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mediapipe
        self.video_mode = bool(video_mode)
        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_hands=2,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise errors.MocapError(
                errors.MODEL_MISSING,
                "无法加载 MediaPipe Hand 模型 {0}：{1}".format(model_path, exc),
                details={"model_path": model_path},
            )

    def close(self) -> None:
        closer = getattr(self.landmarker, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:  # pragma: no cover - defensive
                pass

    def detect(self, image_bgr, timestamp_ms: int = 0):
        rgb = image_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=_ascontiguous(rgb))
        if self.video_mode:
            return self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        return self.landmarker.detect(mp_image)


def _ascontiguous(array):
    """MediaPipe requires a contiguous uint8 buffer."""
    try:
        import numpy  # noqa: PLC0415 - lazy, only needed with mediapipe present
    except ImportError:  # pragma: no cover - mediapipe already depends on numpy
        return array
    return numpy.ascontiguousarray(array)


# --------------------------------------------------------------------------------------
# Landmark conversion
# --------------------------------------------------------------------------------------


def _landmark_confidence(landmark) -> float:
    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    values = [float(v) for v in (visibility, presence) if v is not None]
    if not values:
        return 1.0
    return rm.clamp(min(values), 0.0, 1.0)


def landmarks_to_standard(world_landmarks) -> tuple:
    """Convert MediaPipe world landmarks into standard-skeleton joints.

    Returns ``(body3d, confidence)`` in the standard coordinate system, still
    hip-centred (the caller applies the standing offset).
    """
    body = {}
    confidence = {}
    for index, joint in skeleton.MEDIAPIPE_POSE_MAP.items():
        if index >= len(world_landmarks):
            continue
        landmark = world_landmarks[index]
        body[joint] = skeleton.mediapipe_to_blender(landmark.x, landmark.y, landmark.z)
        confidence[joint] = _landmark_confidence(landmark)

    for joint, (indices, _offset) in skeleton.MEDIAPIPE_DERIVED.items():
        points = []
        scores = []
        for index in indices:
            if index >= len(world_landmarks):
                continue
            landmark = world_landmarks[index]
            points.append(skeleton.mediapipe_to_blender(landmark.x, landmark.y, landmark.z))
            scores.append(_landmark_confidence(landmark))
        if not points:
            continue
        body[joint] = points[0] if len(points) == 1 else rm.vec_midpoint(points[0], points[1])
        confidence[joint] = min(scores) if scores else 1.0

    pelvis = body.get("pelvis")
    neck = body.get("neck")
    if pelvis is not None and neck is not None:
        axis = rm.vec_sub(neck, pelvis)
        body["spine"] = rm.vec_add(pelvis, rm.vec_scale(axis, skeleton.MEDIAPIPE_SPINE_FRACTION))
        body["chest"] = rm.vec_add(pelvis, rm.vec_scale(axis, skeleton.MEDIAPIPE_CHEST_FRACTION))
        base = min(confidence.get("pelvis", 1.0), confidence.get("neck", 1.0))
        confidence["spine"] = base
        confidence["chest"] = base

    if body:
        scores = [v for k, v in confidence.items() if k in skeleton.BODY_JOINTS]
        confidence["body_mean"] = round(sum(scores) / float(len(scores)), 4) if scores else 1.0
    return body, confidence


def hand_landmarks_to_standard(hand_result, body3d) -> dict:
    """Convert MediaPipe hand world landmarks into standard finger joints.

    Handedness reported by MediaPipe is from the *viewer's* perspective, so it is
    inverted to obtain the subject's own side, then the fingers are anchored to
    the body wrist position.
    """
    hands = {}
    world = getattr(hand_result, "hand_world_landmarks", None) or []
    handedness = getattr(hand_result, "handedness", None) or []
    for index, landmarks in enumerate(world):
        label = ""
        if index < len(handedness) and handedness[index]:
            label = str(getattr(handedness[index][0], "category_name", "") or "")
        # MediaPipe "Left" means the left hand as seen in the image: the subject's right.
        side = "R" if label.lower().startswith("left") else "L"
        wrist_anchor = body3d.get("wrist.{0}".format(side))
        origin = None
        if len(landmarks) > 0:
            origin = skeleton.mediapipe_to_blender(landmarks[0].x, landmarks[0].y, landmarks[0].z)
        for landmark_index, suffix in skeleton.MEDIAPIPE_HAND_MAP.items():
            if landmark_index >= len(landmarks):
                continue
            landmark = landmarks[landmark_index]
            point = skeleton.mediapipe_to_blender(landmark.x, landmark.y, landmark.z)
            if origin is not None and wrist_anchor is not None:
                point = rm.vec_add(wrist_anchor, rm.vec_sub(point, origin))
            hands["{0}.{1}".format(suffix, side)] = point
    return hands


def standing_offset(body3d) -> tuple:
    """Offset that puts the lowest foot joint on ``z = 0``."""
    heights = [
        body3d[name][2]
        for name in ("toe.L", "toe.R", "heel.L", "heel.R", "ankle.L", "ankle.R")
        if name in body3d
    ]
    if not heights:
        return (0.0, 0.0, 0.0)
    return (0.0, 0.0, -min(heights))


def horizontal_offset(normalized_landmarks, reference_scale: float, first_center=None) -> tuple:
    """Approximate world translation from normalised landmark drift.

    MediaPipe world landmarks have no global translation, so lateral and depth
    travel are approximated from the normalised hip midpoint. ``reference_scale``
    converts normalised units into metres.
    """
    if not normalized_landmarks or len(normalized_landmarks) <= 24:
        return (0.0, 0.0, 0.0), first_center
    left = normalized_landmarks[23]
    right = normalized_landmarks[24]
    center = (0.5 * (left.x + right.x), 0.5 * (left.y + right.y))
    if first_center is None:
        return (0.0, 0.0, 0.0), center
    dx = (center[0] - first_center[0]) * reference_scale
    return (dx, 0.0, 0.0), first_center
