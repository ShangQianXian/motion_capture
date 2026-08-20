"""Synthetic motion source for ``--mock`` runs (guide section 7.2).

Standard library only, which lets the Blender-side tests import this module
directly in Blender's bundled Python instead of spawning a worker process.

The generated clip is a walk-in-place with a waving right arm:

* 30 FPS, 60 frames by default (the job's frame range always wins),
* pelvis bobs up and down,
* both feet alternate contact,
* the right arm waves,
* every standard skeleton joint is present, including the optional heels.
"""

from __future__ import annotations

import math

from ._core import retarget_math as rm
from ._core import skeleton

#: Default frame count (guide section 7.2).
DEFAULT_FRAME_COUNT = 60

#: Default frame rate.
DEFAULT_FPS = 30

#: Vertical travel of the pelvis bob, in metres.
BOB_AMPLITUDE = 0.03

#: Step cycles per second for the walk.
STEP_FREQUENCY = 1.0

#: Wave cycles per second for the right arm.
WAVE_FREQUENCY = 1.5

#: Height a swinging foot lifts off the ground, in metres.
FOOT_LIFT = 0.12

#: Forward/backward travel of a swinging foot, in metres.
FOOT_SWING = 0.16

#: Frames (relative to the clip start) that carry a deliberately low confidence.
LOW_CONFIDENCE_FRAMES = (20, 21, 22, 23, 24)

#: Joint whose confidence drops during :data:`LOW_CONFIDENCE_FRAMES`.
LOW_CONFIDENCE_JOINT = "wrist.R"

#: Confidence written for those frames (below the 0.4 interpolate threshold).
LOW_CONFIDENCE_VALUE = 0.35

#: Baseline confidence for everything else.
BASE_CONFIDENCE = 0.95


def rest_pose() -> dict:
    """A-pose reference skeleton in the standard coordinate system.

    Z is up, +X is the character's own left (matching Rigify ``.L`` bones) and
    the ground plane sits at ``z = 0``.
    """
    pose = {
        "pelvis": (0.0, 0.0, 1.00),
        "spine": (0.0, 0.010, 1.16),
        "chest": (0.0, 0.005, 1.42),
        "neck": (0.0, 0.000, 1.60),
        "head": (0.0, -0.015, 1.75),
    }
    for side, sign in (("L", 1.0), ("R", -1.0)):
        pose["shoulder.{0}".format(side)] = (sign * 0.175, 0.0, 1.555)
        pose["elbow.{0}".format(side)] = (sign * 0.445, 0.015, 1.395)
        pose["wrist.{0}".format(side)] = (sign * 0.660, 0.025, 1.245)
        pose["hip.{0}".format(side)] = (sign * 0.100, 0.0, 0.98)
        pose["knee.{0}".format(side)] = (sign * 0.100, 0.015, 0.545)
        pose["ankle.{0}".format(side)] = (sign * 0.100, 0.0, 0.095)
        pose["heel.{0}".format(side)] = (sign * 0.100, 0.055, 0.025)
        pose["toe.{0}".format(side)] = (sign * 0.100, -0.145, 0.025)
    return pose


def _rotate_about(point, pivot, axis, angle):
    """Rotate ``point`` around ``pivot`` by ``angle`` radians about ``axis``."""
    rotation = rm.quat_from_axis_angle(axis, angle)
    return rm.vec_add(pivot, rm.quat_rotate_vector(rotation, rm.vec_sub(point, pivot)))


def _swing_leg(pose: dict, side: str, phase: float) -> None:
    """Lift and swing one leg; ``phase`` in ``[0, 1)`` where 0.5 is mid-swing.

    Both the lift and the swing return to zero at ``phase`` 0 and 1 so the foot
    matches its stance position exactly at the hand-off, which keeps the clip
    free of teleports and lets foot locking verify as a no-op on mock data.
    """
    hip = pose["hip.{0}".format(side)]
    lift = math.sin(math.pi * phase) * FOOT_LIFT
    swing = math.sin(2.0 * math.pi * phase) * FOOT_SWING
    knee_angle = math.sin(math.pi * phase) * 0.55

    knee = pose["knee.{0}".format(side)]
    pose["knee.{0}".format(side)] = _rotate_about(knee, hip, (1.0, 0.0, 0.0), -knee_angle)

    for joint in ("ankle", "heel", "toe"):
        name = "{0}.{1}".format(joint, side)
        base = pose[name]
        pose[name] = (base[0], base[1] + swing, base[2] + lift)


def _wave_arm(pose: dict, side: str, phase: float) -> None:
    """Raise the arm overhead and wave the forearm; ``phase`` in ``[0, 1)``.

    Rotating about +Y maps ``+X -> -Z`` and ``-X -> +Z``, so the sign has to be
    the opposite of the side's X sign for the hand to travel upwards.
    """
    shoulder = pose["shoulder.{0}".format(side)]
    sign = 1.0 if side == "L" else -1.0
    raise_angle = -sign * math.radians(105.0)
    for joint in ("elbow", "wrist"):
        name = "{0}.{1}".format(joint, side)
        pose[name] = _rotate_about(pose[name], shoulder, (0.0, 1.0, 0.0), raise_angle)
    # Wave the forearm side to side. With the arm raised the forearm points
    # roughly +Z, so rotating about +Y swings the hand across X and gives a
    # visible wave; rotating about +Z would only spin it about its own axis.
    elbow = pose["elbow.{0}".format(side)]
    wave_angle = math.sin(2.0 * math.pi * phase) * math.radians(28.0)
    pose["wrist.{0}".format(side)] = _rotate_about(
        pose["wrist.{0}".format(side)], elbow, (0.0, 1.0, 0.0), wave_angle
    )


def frame_pose(index: int, fps: float = DEFAULT_FPS) -> tuple:
    """Build one mock frame.

    Returns ``(body3d, contacts)`` where ``body3d`` maps joint names to
    positions and ``contacts`` maps ``foot.L`` / ``foot.R`` to booleans.
    """
    pose = rest_pose()
    time_s = float(index) / float(fps if fps else DEFAULT_FPS)

    cycle = (time_s * STEP_FREQUENCY) % 1.0
    left_swings = cycle < 0.5
    leg_phase = (cycle * 2.0) % 1.0

    if left_swings:
        _swing_leg(pose, "L", leg_phase)
    else:
        _swing_leg(pose, "R", leg_phase)

    _wave_arm(pose, "R", (time_s * WAVE_FREQUENCY) % 1.0)

    bob = math.sin(2.0 * math.pi * 2.0 * STEP_FREQUENCY * time_s) * BOB_AMPLITUDE
    if bob:
        upper = (
            "pelvis",
            "spine",
            "chest",
            "neck",
            "head",
            "shoulder.L",
            "elbow.L",
            "wrist.L",
            "shoulder.R",
            "elbow.R",
            "wrist.R",
            "hip.L",
            "hip.R",
            "knee.L",
            "knee.R",
        )
        for name in upper:
            base = pose[name]
            pose[name] = (base[0], base[1], base[2] + bob)

    contacts = {"foot.L": not left_swings, "foot.R": left_swings}
    return pose, contacts


def generate_frames(
    frame_start: int = 1,
    frame_end: int = 0,
    fps: float = DEFAULT_FPS,
    count: int = DEFAULT_FRAME_COUNT,
    include_hands: bool = False,
) -> list:
    """Generate mock frames honouring the job's frame range.

    ``frame_end == 0`` means "use ``count`` frames" (guide section 5.1).
    """
    start = max(1, int(frame_start or 1))
    if frame_end and int(frame_end) >= start:
        total = int(frame_end) - start + 1
    else:
        total = max(1, int(count or DEFAULT_FRAME_COUNT))
    rate = float(fps) if fps else float(DEFAULT_FPS)

    frames = []
    for offset in range(total):
        body, contacts = frame_pose(offset, rate)
        confidence = {"body_mean": BASE_CONFIDENCE}
        if offset in LOW_CONFIDENCE_FRAMES:
            confidence[LOW_CONFIDENCE_JOINT] = LOW_CONFIDENCE_VALUE
            confidence["body_mean"] = 0.82
        entry = {
            "frame": start + offset,
            "time": offset / rate,
            "body3d": body,
            "hands3d": _mock_hands(body) if include_hands else {},
            "confidence": confidence,
            "contacts": contacts,
        }
        frames.append(entry)
    return frames


def _mock_hands(body: dict) -> dict:
    """Very simple straight fingers, so hand retargeting has data to consume."""
    hands = {}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        wrist = body.get("wrist.{0}".format(side))
        elbow = body.get("elbow.{0}".format(side))
        if wrist is None or elbow is None:
            continue
        forward = rm.vec_normalize(rm.vec_sub(wrist, elbow))
        if rm.vec_length(forward) < rm.EPSILON:
            forward = (sign, 0.0, 0.0)
        spread = rm.vec_normalize(rm.vec_cross(forward, (0.0, 0.0, 1.0)))
        for finger_index, (finger, _rig) in enumerate(skeleton.FINGERS):
            lateral = rm.vec_scale(spread, (finger_index - 2) * 0.018)
            base = rm.vec_add(rm.vec_add(wrist, rm.vec_scale(forward, 0.02)), lateral)
            for segment_index, segment in enumerate(skeleton.FINGER_SEGMENTS):
                distance = 0.025 + segment_index * 0.022
                hands["{0}.{1}.{2}".format(finger, segment, side)] = rm.vec_add(
                    base, rm.vec_scale(forward, distance)
                )
    return hands
