"""Result assembly and serialisation (guide section 5.3).

Standard library only. Every result is validated against
``core.result_schema`` before it reaches disk so a malformed payload can never
be handed to Blender.
"""

from __future__ import annotations

import json
import os

from ._core import errors, paths, result_schema, skeleton


def build_result(
    frames,
    fps: float,
    source_path: str = "",
    source_type: str = "video",
    profile: str = "",
    warnings=None,
) -> dict:
    """Assemble the ``mocap_result.json`` payload."""
    return {
        "version": result_schema.RESULT_VERSION,
        "source": {
            "path": source_path,
            "type": source_type,
            "profile": profile,
        },
        "fps": float(fps),
        "coordinate_system": skeleton.COORDINATE_SYSTEM,
        "unit": skeleton.UNIT,
        "skeleton": skeleton.SKELETON_ID,
        "frames": [_serialise_frame(frame) for frame in frames],
        "warnings": list(warnings or []),
    }


def _serialise_frame(frame) -> dict:
    if hasattr(frame, "to_dict"):
        return frame.to_dict()
    body = frame.get("body3d") or {}
    hands = frame.get("hands3d") or {}
    payload = {
        "frame": int(frame.get("frame", 0)),
        "time": round(float(frame.get("time", 0.0)), 6),
        "body3d": {name: _round_vec(value) for name, value in body.items()},
        "hands3d": {name: _round_vec(value) for name, value in hands.items()},
        "confidence": _round_map(frame.get("confidence") or {}),
        "contacts": {name: bool(value) for name, value in (frame.get("contacts") or {}).items()},
    }
    if frame.get('orientations'):
        payload['orientations'] = {name: [round(float(v), 8) for v in q] for name, q in frame['orientations'].items()}
    return payload


def _round_vec(value) -> list:
    return [round(float(component), 6) for component in value]


def _round_map(mapping) -> dict:
    rounded = {}
    for name, value in mapping.items():
        try:
            rounded[name] = round(float(value), 4)
        except (TypeError, ValueError):
            rounded[name] = value
    return rounded


def write_result(result: dict, output_dir: str, filename: str = "") -> str:
    """Validate then write the result; returns the absolute file path."""
    result_schema.validate_result(result)
    target_dir = paths.ensure_dir(output_dir)
    target = os.path.join(target_dir, filename or paths.DEFAULT_RESULT_FILENAME)
    try:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
    except OSError as exc:
        raise errors.MocapError(
            errors.INTERNAL_ERROR,
            "无法写入结果文件 {0}：{1}".format(target, exc),
            details={"result_path": target},
        )
    return target
