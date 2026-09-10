"""Worker job construction and validation (guide sections 4.2 and 5.1).

Standard library only. ``build_job`` is duck-typed over the Blender
``PropertyGroup`` / ``AddonPreferences`` objects so this module never imports
``bpy`` and can be exercised from plain Python tests with simple namespaces.
"""

from __future__ import annotations

import json
import os
import uuid

from . import errors, model_manifest, paths

#: Job schema version written into every job file.
JOB_VERSION = "0.1"

#: Supported job modes.
MODE_CAPTURE = "capture"
MODE_SELF_TEST = "self_test"
JOB_MODES = (MODE_CAPTURE, MODE_SELF_TEST)

#: Supported ``input.type`` values.
INPUT_TYPES = ("image", "video", "auto")

#: Supported ``options.root_motion`` values.
ROOT_MOTION_MODES = ("world", "in_place")

#: Supported ``options.scale_mode`` values.
SCALE_MODES = ("target_rig_height", "raw")

#: Profiles that must run on CPU.
_CPU_PROFILES = frozenset({"preview", "fallback_cpu"})


def _get(source, name, default=None):
    """Read ``name`` from a PropertyGroup, namespace or mapping."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _get_int(source, name, default):
    """Read an integer without letting ``0`` fall back to the default.

    ``int(value or default)`` is wrong for numeric settings: a deliberate ``0``
    would be silently replaced, so an invalid ``target_fps`` of 0 would sail
    through validation as 30.
    """
    value = _get(source, name, None)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "{0} 必须是整数，当前为 {1!r}。".format(name, value),
            details={name: value},
        )


def _get_float(source, name, default):
    """Read a float without letting ``0.0`` fall back to the default."""
    value = _get(source, name, None)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "{0} 必须是数值，当前为 {1!r}。".format(name, value),
            details={name: value},
        )


def _clamp(value, low, high):
    return max(low, min(high, value))


def new_job_id() -> str:
    """Fresh job identifier."""
    return str(uuid.uuid4())


def build_job(
    scene_props,
    preferences,
    blend_path: str | None = None,
    job_id: str | None = None,
    mode: str = MODE_CAPTURE,
    resolve_path=None,
) -> dict:
    """Build a worker job from Blender scene properties and add-on preferences.

    ``resolve_path`` lets the add-on pass ``bpy.path.abspath`` so Blender's
    ``//relative`` paths are expanded before validation.

    Raises :class:`errors.MocapError` when the configuration cannot produce a
    runnable job.
    """
    resolver = resolve_path or (lambda value: value)

    def path_of(name):
        return paths.normalize(resolver(_get(scene_props, name, "") or ""))

    if mode not in JOB_MODES:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "未知的 job mode：{0}".format(mode),
            details={"known_modes": list(JOB_MODES)},
        )

    profile = str(_get(scene_props, "capture_profile", "") or _get(preferences, "default_profile", "") or "preview")
    if profile not in model_manifest.CAPTURE_PROFILES:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "不支持的 capture profile：{0}".format(profile),
            details={"known_profiles": list(model_manifest.CAPTURE_PROFILES)},
        )

    models_root = paths.normalize(resolver(_get(preferences, "models_root", "") or ""))
    if not models_root or not os.path.isdir(models_root):
        raise errors.MocapError(
            errors.MODELS_ROOT_NOT_FOUND,
            "Models Root 未配置或不存在：{0}".format(models_root or "<empty>"),
            details={"models_root": models_root},
        )

    source_media = path_of("source_media")
    source_type = str(_get(scene_props, "source_type", "auto") or "auto")
    if source_type not in INPUT_TYPES:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "不支持的 source type：{0}".format(source_type),
            details={"known_types": list(INPUT_TYPES)},
        )

    if mode == MODE_CAPTURE:
        if not source_media:
            raise errors.MocapError(
                errors.MEDIA_NOT_FOUND,
                "未选择 Source Media。",
                details={"source_media": ""},
            )
        if not os.path.isfile(source_media):
            raise errors.MocapError(
                errors.MEDIA_NOT_FOUND,
                "Source Media 不存在：{0}".format(source_media),
                details={"source_media": source_media},
            )
        resolved_type = source_type if source_type != "auto" else paths.guess_media_type(source_media)
        if resolved_type == "unknown":
            raise errors.MocapError(
                errors.UNSUPPORTED_MEDIA_TYPE,
                "无法识别媒体类型：{0}".format(os.path.basename(source_media)),
                details={
                    "source_media": source_media,
                    "image_extensions": list(paths.image_extensions()),
                    "video_extensions": list(paths.video_extensions()),
                },
            )
    else:
        resolved_type = source_type if source_type != "auto" else "image"

    target_fps = _get_int(scene_props, "target_fps", 30)
    if target_fps < 1 or target_fps > 120:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "Target FPS 必须在 1 到 120 之间，当前为 {0}。".format(target_fps),
            details={"target_fps": target_fps},
        )

    frame_start = _get_int(scene_props, "frame_start", 1)
    if frame_start < 1:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "Frame Start 必须 >= 1，当前为 {0}。".format(frame_start),
            details={"frame_start": frame_start},
        )
    frame_end = _get_int(scene_props, "frame_end", 0)
    if frame_end != 0 and frame_end < frame_start:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "Frame End（{0}）必须为 0 或不小于 Frame Start（{1}）。".format(frame_end, frame_start),
            details={"frame_start": frame_start, "frame_end": frame_end},
        )

    identifier = job_id or new_job_id()
    output_dir = paths.job_dir(identifier, blend_path)
    try:
        paths.ensure_dir(output_dir)
    except OSError as exc:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "无法创建输出目录 {0}：{1}".format(output_dir, exc),
            details={"output_dir": output_dir},
        )

    device = "cpu" if profile in _CPU_PROFILES else "cuda:0"
    max_vram_gb = _get_float(preferences, "max_vram_gb", 7.0)

    job = {
        "version": JOB_VERSION,
        "job_id": identifier,
        "mode": mode,
        "input": {
            "path": source_media,
            "type": resolved_type,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "target_fps": target_fps,
        },
        "model": {
            "profile": profile,
            "device": device,
            "max_vram_gb": max_vram_gb,
            "models_root": models_root,
        },
        "options": {
            "single_person": True,
            "include_hands": bool(_get(scene_props, "include_hands", True)),
            "smoothing_strength": _clamp(_get_float(scene_props, "smoothing_strength", 0.65), 0.0, 1.0),
            "foot_lock_strength": _clamp(_get_float(scene_props, "foot_lock_strength", 0.7), 0.0, 1.0),
            "root_motion": str(_get(scene_props, "root_motion", "world") or "world"),
            "scale_mode": "target_rig_height",
        },
        "output": {
            "dir": output_dir,
            "result_filename": paths.DEFAULT_RESULT_FILENAME,
        },
    }

    if job["options"]["root_motion"] not in ROOT_MOTION_MODES:
        job["options"]["root_motion"] = "world"

    validate_job(job)
    return job


def validate_job(job) -> dict:
    """Validate a job dict in place; raises ``JOB_SCHEMA_INVALID`` on problems."""
    if not isinstance(job, dict):
        raise errors.MocapError(errors.JOB_SCHEMA_INVALID, "job 必须是 JSON 对象。")

    problems = []

    if not job.get("job_id"):
        problems.append("缺少 job_id")
    if job.get("mode", MODE_CAPTURE) not in JOB_MODES:
        problems.append("mode 非法：{0}".format(job.get("mode")))

    section = job.get("input")
    if not isinstance(section, dict):
        problems.append("缺少 input 段")
    else:
        if section.get("type") not in INPUT_TYPES:
            problems.append("input.type 非法：{0}".format(section.get("type")))
        try:
            fps = int(section.get("target_fps", 0))
        except (TypeError, ValueError):
            fps = 0
        if fps < 1 or fps > 120:
            problems.append("input.target_fps 非法：{0}".format(section.get("target_fps")))
        try:
            start = int(section.get("frame_start", 1))
            end = int(section.get("frame_end", 0))
        except (TypeError, ValueError):
            start, end = 0, -1
        if start < 1:
            problems.append("input.frame_start 必须 >= 1")
        if end != 0 and end < start:
            problems.append("input.frame_end 必须为 0 或 >= frame_start")
        if job.get("mode", MODE_CAPTURE) == MODE_CAPTURE and not section.get("path"):
            problems.append("缺少 input.path")

    section = job.get("model")
    if not isinstance(section, dict):
        problems.append("缺少 model 段")
    else:
        if section.get("profile") not in model_manifest.ALL_PROFILES:
            problems.append("model.profile 非法：{0}".format(section.get("profile")))
        if not section.get("models_root"):
            problems.append("缺少 model.models_root")
        device = str(section.get("device") or "")
        if device != "cpu" and not device.startswith("cuda"):
            problems.append("model.device 非法：{0}".format(device))

    section = job.get("options")
    if not isinstance(section, dict):
        problems.append("缺少 options 段")
    else:
        if section.get("single_person") is not True:
            problems.append("v0.1 要求 options.single_person 为 true")
        if section.get("root_motion") not in ROOT_MOTION_MODES:
            problems.append("options.root_motion 非法：{0}".format(section.get("root_motion")))
        if section.get("scale_mode") not in SCALE_MODES:
            problems.append("options.scale_mode 非法：{0}".format(section.get("scale_mode")))
        for key in ("smoothing_strength", "foot_lock_strength"):
            try:
                value = float(section.get(key, 0.0))
            except (TypeError, ValueError):
                problems.append("options.{0} 不是数值".format(key))
                continue
            if value < 0.0 or value > 1.0:
                problems.append("options.{0} 必须在 0..1 之间".format(key))

    section = job.get("output")
    if not isinstance(section, dict):
        problems.append("缺少 output 段")
    elif not section.get("dir"):
        problems.append("缺少 output.dir")

    if problems:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "job 校验失败：{0}".format("；".join(problems)),
            details={"problems": problems},
        )
    return job


def write_job(job: dict, path: str | None = None) -> str:
    """Serialise ``job`` next to its output directory and return the file path."""
    validate_job(job)
    target = path or os.path.join(job["output"]["dir"], paths.JOB_FILENAME)
    paths.ensure_dir(os.path.dirname(target))
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)
    return target


def read_job(path: str) -> dict:
    """Load and validate a job file."""
    try:
        # utf-8-sig tolerates the BOM that Notepad and PowerShell add on Windows.
        with open(paths.normalize(path), "r", encoding="utf-8-sig") as handle:
            job = json.load(handle)
    except OSError as exc:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "无法读取 job 文件 {0}：{1}".format(path, exc),
            details={"job_path": str(path)},
        )
    except ValueError as exc:
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "job 文件不是合法 JSON：{0}（{1}）".format(path, exc),
            details={"job_path": str(path)},
        )
    return validate_job(job)


def result_path(job: dict) -> str:
    """Absolute path of the result file described by ``job``."""
    output = job.get("output") or {}
    filename = str(output.get("result_filename") or paths.DEFAULT_RESULT_FILENAME)
    return os.path.join(paths.normalize(str(output.get("dir") or "")), filename)


def validate_worker_python(worker_python: str | None) -> str:
    """Validate the configured interpreter; raises ``WORKER_PYTHON_NOT_FOUND``."""
    resolved = paths.normalize(worker_python)
    if not resolved or not os.path.isfile(resolved):
        raise errors.MocapError(
            errors.WORKER_PYTHON_NOT_FOUND,
            "Worker Python 未配置或不存在：{0}".format(worker_python or "<empty>"),
            details={"worker_python": worker_python or ""},
        )
    return resolved
