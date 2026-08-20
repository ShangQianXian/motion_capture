"""Temporal post-processing applied to every capture profile (guide section 9).

Order of operations:

1. per-joint low-confidence interpolation,
2. per-joint position low-pass smoothing,
3. foot contact detection,
4. foot locking (a whole-pose correction, so the upper body keeps its shape),
5. warning aggregation.

Standard library only; all numeric work lives in ``core.smoothing``.
"""

from __future__ import annotations

from ._core import retarget_math as rm
from ._core import skeleton, smoothing

#: Warning codes emitted by this stage.
CODE_LOW_CONFIDENCE = "LOW_CONFIDENCE"
CODE_LONG_GAP = "LONG_OCCLUSION"
CODE_FOOT_LOCK = "FOOT_LOCK_APPLIED"
CODE_NO_TEMPORAL = "SINGLE_FRAME_NO_SMOOTHING"

#: Confidence weight floor so low-confidence frames still contribute a little.
_MIN_WEIGHT = 0.2


def _joint_names(frames) -> tuple:
    body = []
    hands = []
    for frame in frames:
        for name in frame.get("body3d") or {}:
            if name not in body:
                body.append(name)
        for name in frame.get("hands3d") or {}:
            if name not in hands:
                hands.append(name)
    return tuple(body), tuple(hands)


def _confidence_of(frame, joint: str, default: float = 1.0) -> float:
    confidence = frame.get("confidence") or {}
    if joint in confidence:
        try:
            return float(confidence[joint])
        except (TypeError, ValueError):
            return default
    fallback = confidence.get("body_mean", default)
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return default


def _weight_from_confidence(value: float) -> float:
    return rm.clamp(value / skeleton.CONFIDENCE_GOOD, _MIN_WEIGHT, 1.0)


def postprocess(
    frames,
    fps: float,
    options=None,
    reporter=None,
    recompute_contacts: bool = True,
) -> tuple:
    """Smooth, interpolate and foot-lock ``frames`` in place.

    Returns ``(frames, warnings)``. Warnings are plain dicts matching the
    ``warning`` progress event payload, so callers can both emit them live and
    store them in the result file.
    """
    options = options or {}
    warnings = []

    if not frames:
        return frames, warnings

    if len(frames) == 1:
        warnings.append(
            {
                "code": CODE_NO_TEMPORAL,
                "message": "单帧输入，跳过平滑、插值与足底锁定。",
            }
        )
        for frame in frames:
            frame.setdefault("contacts", {})
        _report(reporter, warnings[-1])
        return frames, warnings

    smoothing_strength = float(options.get("smoothing_strength", 0.0) or 0.0)
    foot_lock_strength = float(options.get("foot_lock_strength", 0.0) or 0.0)
    alpha = smoothing.alpha_from_strength(smoothing_strength)

    body_names, hand_names = _joint_names(frames)
    all_names = body_names + hand_names

    tracks = {}
    confidences = {}
    for name in all_names:
        source_key = "body3d" if name in body_names else "hands3d"
        tracks[name] = [(frame.get(source_key) or {}).get(name) for frame in frames]
        confidences[name] = [_confidence_of(frame, name) for frame in frames]

    # 1) interpolation of missing / low-confidence samples
    for name in all_names:
        filled, gaps = smoothing.interpolate_track(
            tracks[name],
            confidences[name],
            low_confidence=skeleton.CONFIDENCE_LOW,
            max_gap=smoothing.DEFAULT_MAX_GAP,
        )
        tracks[name] = filled
        for gap in gaps:
            start_frame = frames[gap["start"]].get("frame", gap["start"])
            if gap["held"]:
                warning = {
                    "code": CODE_LONG_GAP,
                    "message": "{0} 连续 {1} 帧置信度过低，保持上一可信姿态。".format(name, gap["length"]),
                    "frame": start_frame,
                    "joint": name,
                }
            else:
                warning = {
                    "code": CODE_LOW_CONFIDENCE,
                    "message": "{0} 在 {1} 帧内置信度过低，已插值。".format(name, gap["length"]),
                    "frame": start_frame,
                    "joint": name,
                }
            warnings.append(warning)
            _report(reporter, warning)

    # 2) position smoothing, weighted by confidence
    if alpha < 1.0:
        for name in all_names:
            weights = [_weight_from_confidence(value) for value in confidences[name]]
            tracks[name] = smoothing.lowpass_track(tracks[name], alpha, weights)

    # 3) contact detection
    contacts = {}
    for side in ("L", "R"):
        ankle = tracks.get("ankle.{0}".format(side))
        toe = tracks.get("toe.{0}".format(side))
        if ankle is None:
            contacts[side] = [False] * len(frames)
            continue
        foot_confidence = [
            min(
                confidences.get("ankle.{0}".format(side), [1.0] * len(frames))[index],
                confidences.get("toe.{0}".format(side), [1.0] * len(frames))[index],
            )
            for index in range(len(frames))
        ]
        if recompute_contacts:
            contacts[side] = smoothing.detect_contacts(
                ankle,
                toe,
                foot_confidence,
                fps=fps,
                confidence_minimum=skeleton.CONFIDENCE_CONTACT,
            )
        else:
            contacts[side] = [
                bool((frame.get("contacts") or {}).get("foot.{0}".format(side), False))
                for frame in frames
            ]

    # 4) foot locking: per-foot horizontal corrections plus a whole-body ground fix
    #    (guide section 9.4: only the foot and the pelvis height may be touched).
    foot_offsets = {}
    ground_offsets = None
    if foot_lock_strength > 0.0:
        sides = []
        foot_tracks = []
        contact_tracks = []
        for side in ("L", "R"):
            ankle = tracks.get("ankle.{0}".format(side))
            if ankle is None:
                continue
            sides.append(side)
            foot_tracks.append(ankle)
            contact_tracks.append(contacts[side])
        if foot_tracks:
            per_foot = smoothing.foot_lock_offsets(
                foot_tracks, contact_tracks, strength=foot_lock_strength
            )
            foot_offsets = dict(zip(sides, per_foot))
            ground_offsets = smoothing.ground_correction(
                foot_tracks, contact_tracks, strength=foot_lock_strength
            )
            moved = 0
            for index in range(len(frames)):
                magnitude = max(
                    (rm.vec_length(foot_offsets[side][index]) for side in sides), default=0.0
                )
                if ground_offsets is not None:
                    magnitude = max(magnitude, abs(ground_offsets[index][2]))
                if magnitude > 1e-4:
                    moved += 1
            if moved:
                warning = {
                    "code": CODE_FOOT_LOCK,
                    "message": "足底锁定修正了 {0} 帧（强度 {1:.2f}）。".format(moved, foot_lock_strength),
                }
                warnings.append(warning)
                _report(reporter, warning)

    # 5) write results back
    for index, frame in enumerate(frames):
        ground = ground_offsets[index] if ground_offsets is not None else None
        body = {}
        for name in body_names:
            value = tracks[name][index]
            if value is None:
                continue
            offset = _foot_offset_for(name, foot_offsets, index)
            if offset is not None:
                value = rm.vec_add(value, offset)
            if ground:
                value = rm.vec_add(value, ground)
            body[name] = value
        if body:
            frame["body3d"] = body
        if hand_names:
            hands = {}
            for name in hand_names:
                value = tracks[name][index]
                if value is None:
                    continue
                hands[name] = rm.vec_add(value, ground) if ground else value
            frame["hands3d"] = hands
        frame["contacts"] = {
            "foot.L": bool(contacts["L"][index]),
            "foot.R": bool(contacts["R"][index]),
        }

    return frames, warnings


#: Weight applied to the knee so a locked foot does not leave a rigid straight leg.
_KNEE_LOCK_WEIGHT = 0.5

#: Joints that follow a foot-lock correction, with their weights.
_FOOT_CHAIN_WEIGHTS = {
    "ankle": 1.0,
    "heel": 1.0,
    "toe": 1.0,
    "knee": _KNEE_LOCK_WEIGHT,
}


def _foot_offset_for(joint: str, foot_offsets: dict, index: int):
    """Foot-lock offset for ``joint``, or ``None`` when it is not part of a foot chain."""
    if not foot_offsets or "." not in joint:
        return None
    base, _, side = joint.rpartition(".")
    weight = _FOOT_CHAIN_WEIGHTS.get(base)
    if weight is None:
        return None
    track = foot_offsets.get(side)
    if track is None or index >= len(track):
        return None
    offset = track[index]
    return offset if weight == 1.0 else rm.vec_scale(offset, weight)


def _report(reporter, warning) -> None:
    if reporter is None:
        return
    reporter.warning(
        warning.get("code", "WARNING"),
        warning.get("message", ""),
        frame=warning.get("frame"),
    )


def summarise_warnings(warnings, limit: int = 40) -> list:
    """Collapse repeated per-joint warnings so the result stays readable."""
    counts = {}
    order = []
    for warning in warnings:
        key = (warning.get("code"), warning.get("joint"))
        if key not in counts:
            counts[key] = dict(warning)
            counts[key]["occurrences"] = 0
            order.append(key)
        counts[key]["occurrences"] += 1
    summarised = [counts[key] for key in order[:limit]]
    if len(order) > limit:
        summarised.append(
            {
                "code": "WARNINGS_TRUNCATED",
                "message": "另有 {0} 类 warning 未列出。".format(len(order) - limit),
            }
        )
    return summarised
