"""Optional constant pitch calibration for standing/walking captures (radians)."""

from __future__ import annotations

import copy
import math
import statistics

from . import errors


def estimate_standing_pitch(result, sample_limit: int = 120) -> float:
    """Estimate one X rotation from the median ankle-midpoint to head direction.

    This assumes the subject is upright on average. It is explicitly requested
    by the user, never applied automatically to bending or floor exercises.
    """
    angles = []
    count = len(result.frames)
    indices = sorted({round(i * (count - 1) / max(1, min(count, sample_limit) - 1))
                      for i in range(min(count, sample_limit))})
    for index in indices:
        frame = result.frames[index]
        names = ("head", "ankle.L", "ankle.R")
        if any(frame.joint(name) is None or frame.confidence_of(name) < 0.4 for name in names):
            continue
        head, left, right = (frame.joint(name) for name in names)
        y = head[1] - (left[1] + right[1]) * 0.5
        z = head[2] - (left[2] + right[2]) * 0.5
        if z > 0.2:
            angles.append(math.atan2(y, z))
    if not angles or abs(statistics.median(angles)) > math.pi / 4:
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID,
                               "缺少可靠的站立姿态，无法估算倾斜；请手动设置 X 轴校正角度。")
    return float(statistics.median(angles))


def calibrated_result(result, pitch: float):
    """Return an independent result with the same rigid pitch applied every frame.

    Rotate about each pelvis, then restore that frame's lowest foot height.
    This retains horizontal root travel and jump/ground height while correcting
    orientation. Body and hands undergo exactly the same rigid transform.
    """
    if not math.isfinite(pitch):
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "倾斜校正角度必须是有限数值。")
    output = copy.deepcopy(result)
    cosine, sine = math.cos(pitch), math.sin(pitch)
    for frame in output.frames:
        pivot = frame.body3d["pelvis"]
        feet = [name for name in ("ankle.L", "ankle.R", "toe.L", "toe.R", "heel.L", "heel.R")
                if name in frame.body3d]
        old_floor = min((frame.body3d[name][2] for name in feet), default=pivot[2])
        for store in (frame.body3d, frame.hands3d):
            for name, point in store.items():
                y, z = point[1] - pivot[1], point[2] - pivot[2]
                store[name] = (point[0], pivot[1] + cosine * y - sine * z,
                               pivot[2] + sine * y + cosine * z)
        new_floor = min((frame.body3d[name][2] for name in feet), default=pivot[2])
        shift = old_floor - new_floor
        for store in (frame.body3d, frame.hands3d):
            for name, point in store.items():
                store[name] = (point[0], point[1], point[2] + shift)
    output.data["frames"] = [frame.to_dict() for frame in output.frames]
    return output
