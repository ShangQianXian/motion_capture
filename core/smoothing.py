"""Temporal smoothing, gap interpolation, contact detection and foot locking.

Implements ``docs/DEVELOPMENT_GUIDE.md`` section 9. Standard library only and
free of any framework dependency so every rule is unit-testable.

Tracks are lists that are index-aligned with the frame list. A track entry may be
``None`` to mark "no data for this frame".
"""

from __future__ import annotations

from . import retarget_math as rm

#: Maximum smoothing strength; 1.0 would freeze the signal completely.
MAX_STRENGTH = 0.95

#: Default confidence below which a sample is treated as missing.
DEFAULT_LOW_CONFIDENCE = 0.4

#: Longest run of low-confidence frames that is still interpolated (section 9.3).
DEFAULT_MAX_GAP = 10

#: Contact detection defaults (section 9.4).
DEFAULT_CONTACT_HEIGHT = 0.06      # metres above the estimated ground
DEFAULT_CONTACT_SPEED = 0.35       # metres per second
DEFAULT_CONTACT_CONFIDENCE = 0.5

#: Smoothing applied to the foot-lock correction itself, to avoid pops.
#: Measured on a foot drifting 0.01 m per frame: 0.35 removes 92% of the slide,
#: 0.5 removes 96% and still ramps in over only two or three frames.
FOOT_LOCK_SMOOTHING_ALPHA = 0.5


def alpha_from_strength(strength: float) -> float:
    """Convert the UI ``smoothing_strength`` into a low-pass alpha (section 9.1)."""
    try:
        value = float(strength)
    except (TypeError, ValueError):
        value = 0.0
    return 1.0 - rm.clamp(value, 0.0, MAX_STRENGTH)


# --------------------------------------------------------------------------------------
# Position smoothing
# --------------------------------------------------------------------------------------


def lowpass_track(track, alpha: float, weights=None) -> list:
    """First-order low-pass filter: ``out[t] = a * raw[t] + (1 - a) * out[t-1]``.

    ``weights`` optionally scales alpha per frame so low-confidence samples pull
    the filter less (section 9.2 applies the same idea to rotations).
    """
    alpha = rm.clamp(float(alpha), 0.0, 1.0)
    smoothed = []
    previous = None
    for index, value in enumerate(track):
        if value is None:
            smoothed.append(None)
            continue
        if previous is None:
            previous = tuple(float(c) for c in value)
            smoothed.append(previous)
            continue
        effective = alpha
        if weights is not None and index < len(weights) and weights[index] is not None:
            effective = rm.clamp(alpha * float(weights[index]), 0.0, 1.0)
        blended = rm.vec_lerp(previous, tuple(float(c) for c in value), effective)
        smoothed.append(blended)
        previous = blended
    return smoothed


def smooth_quaternion_track(track, alpha: float, weights=None) -> list:
    """Quaternion low-pass using slerp; never averages Euler angles."""
    alpha = rm.clamp(float(alpha), 0.0, 1.0)
    smoothed = []
    previous = None
    for index, value in enumerate(track):
        if value is None:
            smoothed.append(None)
            continue
        current = rm.quat_normalize(value)
        if previous is None:
            smoothed.append(current)
            previous = current
            continue
        effective = alpha
        if weights is not None and index < len(weights) and weights[index] is not None:
            effective = rm.clamp(alpha * float(weights[index]), 0.0, 1.0)
        blended = rm.slerp(previous, current, effective)
        smoothed.append(blended)
        previous = blended
    return smoothed


# --------------------------------------------------------------------------------------
# Low confidence interpolation
# --------------------------------------------------------------------------------------


def interpolate_track(
    track,
    confidences=None,
    low_confidence: float = DEFAULT_LOW_CONFIDENCE,
    max_gap: int = DEFAULT_MAX_GAP,
) -> tuple:
    """Fill missing / low-confidence samples (section 9.3).

    Rules:

    * runs of at most ``max_gap`` frames are linearly interpolated between the
      surrounding valid samples,
    * longer runs hold the last trusted sample and report a gap,
    * leading and trailing runs copy the nearest valid sample.

    Returns ``(filled_track, gaps)`` where each gap is
    ``{"start", "end", "length", "held"}``.
    """
    count = len(track)
    filled = list(track)
    gaps = []

    valid = []
    for index in range(count):
        value = track[index]
        if value is None:
            valid.append(False)
            continue
        if confidences is not None and index < len(confidences):
            confidence = confidences[index]
            if confidence is not None and float(confidence) < low_confidence:
                valid.append(False)
                continue
        valid.append(True)

    if not any(valid):
        return filled, gaps

    index = 0
    while index < count:
        if valid[index]:
            index += 1
            continue
        start = index
        while index < count and not valid[index]:
            index += 1
        end = index - 1
        length = end - start + 1
        before = start - 1 if start > 0 else None
        after = index if index < count else None

        if before is None and after is not None:
            for i in range(start, end + 1):
                filled[i] = track[after]
            gaps.append({"start": start, "end": end, "length": length, "held": False})
        elif after is None and before is not None:
            for i in range(start, end + 1):
                filled[i] = filled[before]
            gaps.append({"start": start, "end": end, "length": length, "held": True})
        elif before is not None and after is not None:
            if length <= max_gap:
                a = filled[before]
                b = track[after]
                for offset, i in enumerate(range(start, end + 1), start=1):
                    t = offset / float(length + 1)
                    filled[i] = rm.vec_lerp(a, b, t)
                gaps.append({"start": start, "end": end, "length": length, "held": False})
            else:
                held = filled[before]
                for i in range(start, end + 1):
                    filled[i] = held
                gaps.append({"start": start, "end": end, "length": length, "held": True})
    return filled, gaps


# --------------------------------------------------------------------------------------
# Contacts and foot locking
# --------------------------------------------------------------------------------------


def percentile(values, fraction: float) -> float:
    """Simple percentile over a list of floats (no numpy)."""
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = rm.clamp(fraction, 0.0, 1.0) * (len(data) - 1)
    low = int(position)
    high = min(low + 1, len(data) - 1)
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def estimate_ground_z(tracks, fraction: float = 0.05) -> float:
    """Estimate the ground height from one or more foot tracks."""
    heights = []
    for track in tracks:
        for value in track:
            if value is not None:
                heights.append(value[2])
    if not heights:
        return 0.0
    return percentile(heights, fraction)


def horizontal_speeds(track, fps: float) -> list:
    """Per-frame horizontal speed in metres per second."""
    rate = float(fps) if fps and fps > 0 else 30.0
    speeds = []
    for index, value in enumerate(track):
        if value is None:
            speeds.append(None)
            continue
        previous = None
        for back in range(index - 1, -1, -1):
            if track[back] is not None:
                previous = track[back]
                break
        if previous is None:
            speeds.append(0.0)
            continue
        dx = value[0] - previous[0]
        dy = value[1] - previous[1]
        speeds.append(((dx * dx + dy * dy) ** 0.5) * rate)
    return speeds


def detect_contacts(
    ankle_track,
    toe_track=None,
    confidences=None,
    ground_z=None,
    fps: float = 30.0,
    height_tolerance: float = DEFAULT_CONTACT_HEIGHT,
    speed_tolerance: float = DEFAULT_CONTACT_SPEED,
    confidence_minimum: float = DEFAULT_CONTACT_CONFIDENCE,
) -> list:
    """Per-frame foot contact flags (section 9.4).

    A foot is in contact when it sits near the local ground, moves slowly
    horizontally and has usable confidence.
    """
    tracks = [ankle_track] if toe_track is None else [ankle_track, toe_track]
    ground = estimate_ground_z(tracks) if ground_z is None else float(ground_z)
    ankle_speeds = horizontal_speeds(ankle_track, fps)
    toe_speeds = horizontal_speeds(toe_track, fps) if toe_track is not None else None

    contacts = []
    for index, ankle in enumerate(ankle_track):
        if ankle is None:
            contacts.append(False)
            continue
        if confidences is not None and index < len(confidences):
            confidence = confidences[index]
            if confidence is not None and float(confidence) < confidence_minimum:
                contacts.append(False)
                continue
        toe = toe_track[index] if toe_track is not None and index < len(toe_track) else None
        lowest = ankle[2] if toe is None else min(ankle[2], toe[2])
        if lowest - ground > height_tolerance:
            contacts.append(False)
            continue
        speed = ankle_speeds[index] if ankle_speeds[index] is not None else 0.0
        if toe_speeds is not None and toe_speeds[index] is not None:
            speed = max(speed, toe_speeds[index])
        contacts.append(speed <= speed_tolerance)
    return contacts


def contact_segments(contacts) -> list:
    """Convert a boolean contact track into ``(start, end)`` inclusive ranges."""
    segments = []
    start = None
    for index, flag in enumerate(contacts):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(contacts) - 1))
    return segments


def foot_lock_offsets(
    foot_tracks,
    contact_tracks,
    strength: float = 0.7,
) -> list:
    """Per-foot horizontal corrections that keep contacting feet planted.

    Guide section 9.4 requires that foot locking only nudges the foot and the
    pelvis height, never the upper body pose. This therefore returns one offset
    track *per foot* (X/Y only); ``ground_correction`` handles height separately.

    For every contact segment the foot's world position at the segment start
    becomes the anchor, and each frame inside the segment is pulled back towards
    it by ``strength``. The correction is low-pass filtered so contact hand-offs
    do not pop.

    Returns a list parallel to ``foot_tracks``, each entry a list of offsets.
    """
    amount = rm.clamp(float(strength), 0.0, 1.0)
    results = []
    for track, contacts in zip(foot_tracks, contact_tracks):
        count = len(track)
        raw = [(0.0, 0.0, 0.0)] * count
        if amount > 0.0:
            for start, end in contact_segments(contacts):
                anchor = track[start]
                if anchor is None:
                    continue
                for index in range(start, end + 1):
                    current = track[index]
                    if current is None:
                        continue
                    raw[index] = (
                        (anchor[0] - current[0]) * amount,
                        (anchor[1] - current[1]) * amount,
                        0.0,
                    )
        smoothed = lowpass_track(raw, FOOT_LOCK_SMOOTHING_ALPHA)
        results.append([value if value is not None else (0.0, 0.0, 0.0) for value in smoothed])
    return results


def ground_correction(
    foot_tracks,
    contact_tracks,
    ground_z=None,
    strength: float = 0.7,
) -> list:
    """Z-only whole-body corrections so contacting feet reach the ground plane.

    This is the "adjust the pelvis height" half of guide section 9.4. Shifting
    every joint by the same Z keeps the captured pose intact while removing
    floating and ground penetration.
    """
    amount = rm.clamp(float(strength), 0.0, 1.0)
    count = max((len(track) for track in foot_tracks), default=0)
    if amount <= 0.0 or count == 0:
        return [(0.0, 0.0, 0.0)] * count

    ground = estimate_ground_z(foot_tracks) if ground_z is None else float(ground_z)
    raw = []
    for index in range(count):
        heights = []
        for track, contacts in zip(foot_tracks, contact_tracks):
            if index < len(contacts) and contacts[index] and index < len(track) and track[index] is not None:
                heights.append(track[index][2])
        if not heights:
            raw.append((0.0, 0.0, 0.0))
            continue
        raw.append((0.0, 0.0, (ground - min(heights)) * amount))

    smoothed = lowpass_track(raw, FOOT_LOCK_SMOOTHING_ALPHA)
    return [value if value is not None else (0.0, 0.0, 0.0) for value in smoothed]


def apply_offset(positions: dict, offset) -> dict:
    """Translate every joint in ``positions`` by ``offset``."""
    if offset == (0.0, 0.0, 0.0):
        return positions
    return {name: rm.vec_add(value, offset) for name, value in positions.items()}
