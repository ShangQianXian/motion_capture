"""``mocap_result.json`` validation and loading (guide sections 4.4 and 5.3).

Standard library only.
"""

from __future__ import annotations

import json
import os

from . import errors, paths, skeleton

#: Result schema version produced and accepted by v0.1.
RESULT_VERSION = "0.1"


class Frame(object):
    """One frame of a mocap result."""

    __slots__ = ("frame", "time", "body3d", "hands3d", "confidence", "contacts")

    def __init__(self, data: dict) -> None:
        self.frame = int(data.get("frame", 0))
        self.time = float(data.get("time", 0.0))
        self.body3d = {k: tuple(float(c) for c in v) for k, v in (data.get("body3d") or {}).items()}
        self.hands3d = {k: tuple(float(c) for c in v) for k, v in (data.get("hands3d") or {}).items()}
        self.confidence = dict(data.get("confidence") or {})
        self.contacts = dict(data.get("contacts") or {})

    def joint(self, name: str):
        """Position of ``name`` from body or hand data, or ``None``."""
        if name in self.body3d:
            return self.body3d[name]
        return self.hands3d.get(name)

    def confidence_of(self, name: str, default: float = 1.0) -> float:
        try:
            return float(self.confidence.get(name, self.confidence.get("body_mean", default)))
        except (TypeError, ValueError):
            return default

    def to_dict(self) -> dict:
        payload = {
            "frame": self.frame,
            "time": round(self.time, 6),
            "body3d": {k: [round(c, 6) for c in v] for k, v in self.body3d.items()},
            "hands3d": {k: [round(c, 6) for c in v] for k, v in self.hands3d.items()},
            "confidence": self.confidence,
            "contacts": self.contacts,
        }
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "Frame({0}, joints={1})".format(self.frame, len(self.body3d))


class MocapResult(object):
    """Validated mocap result."""

    __slots__ = ("data", "path", "frames", "warnings")

    def __init__(self, data: dict, path: str = "") -> None:
        self.data = data
        self.path = path
        self.frames = [Frame(entry) for entry in data.get("frames") or []]
        self.warnings = list(data.get("warnings") or [])

    @property
    def version(self) -> str:
        return str(self.data.get("version") or "")

    @property
    def fps(self) -> float:
        return float(self.data.get("fps") or 30.0)

    @property
    def source(self) -> dict:
        value = self.data.get("source")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def source_path(self) -> str:
        return str(self.source.get("path") or "")

    @property
    def profile(self) -> str:
        return str(self.source.get("profile") or "")

    @property
    def coordinate_system(self) -> str:
        return str(self.data.get("coordinate_system") or "")

    @property
    def unit(self) -> str:
        return str(self.data.get("unit") or "")

    @property
    def skeleton_id(self) -> str:
        return str(self.data.get("skeleton") or "")

    @property
    def frame_start(self) -> int:
        return self.frames[0].frame if self.frames else 0

    @property
    def frame_end(self) -> int:
        return self.frames[-1].frame if self.frames else 0

    @property
    def has_hands(self) -> bool:
        return any(frame.hands3d for frame in self.frames)

    def action_name(self) -> str:
        """Action name per guide section 10.4: ``Mocap_<source_name>_<profile>``."""
        stem = paths.source_stem(self.source_path) or "capture"
        profile = paths.safe_name(self.profile, "profile")
        return "Mocap_{0}_{1}".format(stem, profile)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "MocapResult(frames={0}, fps={1}, profile={2!r})".format(
            len(self.frames), self.fps, self.profile
        )


def _is_vec3(value) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return False
        # Reject NaN / infinity without importing math.isfinite on non-floats.
        if component != component or component in (float("inf"), float("-inf")):
            return False
    return True


def validate_result(data, max_reported: int = 8) -> dict:
    """Validate a result payload; raises ``RESULT_SCHEMA_INVALID`` on problems."""
    if not isinstance(data, dict):
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "result 必须是 JSON 对象。")

    problems = []

    version = str(data.get("version") or "")
    if version != RESULT_VERSION:
        problems.append("version 必须是 {0}，实际为 {1!r}".format(RESULT_VERSION, version))

    coordinate_system = str(data.get("coordinate_system") or "")
    if coordinate_system != skeleton.COORDINATE_SYSTEM:
        problems.append(
            "coordinate_system 必须是 {0}，实际为 {1!r}".format(skeleton.COORDINATE_SYSTEM, coordinate_system)
        )

    unit = str(data.get("unit") or "")
    if unit != skeleton.UNIT:
        problems.append("unit 必须是 {0}，实际为 {1!r}".format(skeleton.UNIT, unit))

    skeleton_id = str(data.get("skeleton") or "")
    if skeleton_id != skeleton.SKELETON_ID:
        problems.append("skeleton 必须是 {0}，实际为 {1!r}".format(skeleton.SKELETON_ID, skeleton_id))

    try:
        fps = float(data.get("fps", 0))
    except (TypeError, ValueError):
        fps = 0.0
    if fps <= 0.0:
        problems.append("fps 必须大于 0，实际为 {0!r}".format(data.get("fps")))

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        problems.append("frames 必须是非空数组")
        frames = []

    reported = 0
    seen_numbers = set()
    for index, entry in enumerate(frames):
        if reported >= max_reported:
            problems.append("... 其余帧的问题已省略")
            break
        if not isinstance(entry, dict):
            problems.append("frames[{0}] 不是对象".format(index))
            reported += 1
            continue
        number = entry.get("frame")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append("frames[{0}].frame 必须是整数".format(index))
            reported += 1
            continue
        if number in seen_numbers:
            problems.append("frames[{0}].frame 重复：{1}".format(index, number))
            reported += 1
        seen_numbers.add(number)

        body = entry.get("body3d")
        if not isinstance(body, dict):
            problems.append("frames[{0}].body3d 必须是对象".format(index))
            reported += 1
            continue
        missing = [joint for joint in skeleton.BODY_JOINTS if joint not in body]
        if missing:
            problems.append(
                "frames[{0}] 缺少必需关节：{1}".format(index, "、".join(missing[:6]))
            )
            reported += 1
            continue
        bad = [name for name, value in body.items() if not _is_vec3(value)]
        if bad:
            problems.append(
                "frames[{0}] 关节坐标非法：{1}".format(index, "、".join(sorted(bad)[:6]))
            )
            reported += 1
            continue
        hands = entry.get("hands3d")
        if hands is not None and not isinstance(hands, dict):
            problems.append("frames[{0}].hands3d 必须是对象".format(index))
            reported += 1
        elif isinstance(hands, dict):
            bad_hand = [name for name, value in hands.items() if not _is_vec3(value)]
            if bad_hand:
                problems.append(
                    "frames[{0}] 手部坐标非法：{1}".format(index, "、".join(sorted(bad_hand)[:6]))
                )
                reported += 1

    if problems:
        raise errors.MocapError(
            errors.RESULT_SCHEMA_INVALID,
            "mocap_result 校验失败：{0}".format("；".join(problems)),
            details={"problems": problems},
        )
    return data


def load_mocap_result(path: str) -> MocapResult:
    """Validate and load ``mocap_result.json``."""
    resolved = paths.normalize(path)
    if not resolved or not os.path.isfile(resolved):
        raise errors.MocapError(
            errors.RESULT_SCHEMA_INVALID,
            "结果文件不存在：{0}".format(path or "<empty>"),
            details={"result_path": str(path or "")},
        )
    try:
        # utf-8-sig tolerates the BOM that Notepad and PowerShell add on Windows.
        with open(resolved, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise errors.MocapError(
            errors.RESULT_SCHEMA_INVALID,
            "无法读取结果文件 {0}：{1}".format(resolved, exc),
            details={"result_path": resolved},
        )
    except ValueError as exc:
        raise errors.MocapError(
            errors.RESULT_SCHEMA_INVALID,
            "结果文件不是合法 JSON：{0}（{1}）".format(resolved, exc),
            details={"result_path": resolved},
        )
    validate_result(data)
    return MocapResult(data, resolved)


def detect_mirror(result: MocapResult, sample_limit: int = 120) -> bool:
    """True when the data looks left/right mirrored.

    In the standard coordinate system the character's own left is +X, so
    ``hip.L.x`` must be greater than ``hip.R.x``. A majority vote over the first
    ``sample_limit`` frames keeps single noisy frames from flipping the verdict.
    """
    votes_mirrored = 0
    votes_total = 0
    for frame in result.frames[:sample_limit]:
        left = frame.body3d.get("hip.L")
        right = frame.body3d.get("hip.R")
        if left is None or right is None:
            continue
        if abs(left[0] - right[0]) < 1e-6:
            continue
        votes_total += 1
        if left[0] < right[0]:
            votes_mirrored += 1
    if votes_total == 0:
        return False
    return votes_mirrored * 2 > votes_total


def looks_y_up(result: MocapResult, sample_limit: int = 120) -> bool:
    """Heuristic for Y-up data reaching the Z-up retarget layer.

    Guide section 5.3's sample payload is Y-up; real results must be Z-up. When a
    result declares the right coordinate system but has a flat Z range and a tall
    Y range, something upstream ignored the convention.
    """
    z_values = []
    y_values = []
    for frame in result.frames[:sample_limit]:
        for name in ("pelvis", "head", "ankle.L", "ankle.R"):
            position = frame.body3d.get(name)
            if position is None:
                continue
            y_values.append(position[1])
            z_values.append(position[2])
    if len(z_values) < 4:
        return False
    z_range = max(z_values) - min(z_values)
    y_range = max(y_values) - min(y_values)
    return z_range < 0.2 and y_range > 0.8


def mirror_result_x(result: MocapResult) -> None:
    """Negate X on every joint in place and swap left/right joint names."""
    for frame in result.frames:
        for store in (frame.body3d, frame.hands3d):
            flipped = {}
            for name, position in store.items():
                flipped[skeleton.mirror_joint(name)] = (-position[0], position[1], position[2])
            store.clear()
            store.update(flipped)
        for store in (frame.confidence, frame.contacts):
            swapped = {}
            for name, value in store.items():
                swapped[skeleton.mirror_joint(name)] = value
            store.clear()
            store.update(swapped)
