"""Path helpers shared by the add-on and the worker.

Standard library only. ``addon_root()`` resolves the package root regardless of
whether this module was imported as ``core.paths`` (worker) or
``motion_capture.core.paths`` (Blender).
"""

from __future__ import annotations

import os
import re
import tempfile

#: Directory name used for per-job scratch data next to the .blend file.
JOB_DIR_NAME = ".mocap_jobs"

#: File the add-on drops into a job directory to request cooperative cancellation.
CANCEL_FILENAME = "cancel.request"

#: Default result file name (see job schema ``output.result_filename``).
DEFAULT_RESULT_FILENAME = "mocap_result.json"

#: Worker debug log inside the job directory.
WORKER_LOG_FILENAME = "worker.log"

#: Job description file inside the job directory.
JOB_FILENAME = "job.json"

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})

_UNSAFE_NAME = re.compile(r"[^0-9A-Za-z_.-]+")


def addon_root() -> str:
    """Absolute path of the add-on package root (the directory holding ``__init__.py``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bundled_models_dir() -> str:
    """``<addon_root>/models`` - holds the bundled ``manifest.example.json``."""
    return os.path.join(addon_root(), "models")


def bundled_manifest_path() -> str:
    """Fallback manifest shipped with the add-on."""
    return os.path.join(bundled_models_dir(), "manifest.example.json")


def normalize(path: str | None) -> str:
    """Normalise a user supplied path; returns ``""`` for empty input.

    Blender stores relative paths as ``//foo``; those are resolved by the caller
    via ``bpy.path.abspath`` before reaching this function.
    """
    if not path:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path).strip())))


def exists(path: str | None) -> bool:
    """True when ``path`` is non-empty and present on disk."""
    return bool(path) and os.path.exists(normalize(path))


def is_dir(path: str | None) -> bool:
    """True when ``path`` is non-empty and an existing directory."""
    return bool(path) and os.path.isdir(normalize(path))


def is_file(path: str | None) -> bool:
    """True when ``path`` is non-empty and an existing file."""
    return bool(path) and os.path.isfile(normalize(path))


def guess_media_type(path: str) -> str:
    """Return ``"image"``, ``"video"`` or ``"unknown"`` from the file extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def image_extensions() -> tuple:
    """Supported still-image extensions."""
    return tuple(sorted(_IMAGE_EXTENSIONS))


def video_extensions() -> tuple:
    """Supported video extensions."""
    return tuple(sorted(_VIDEO_EXTENSIONS))


def safe_name(name: str, fallback: str = "source") -> str:
    """Sanitise ``name`` for use inside an Action name or a directory name."""
    cleaned = _UNSAFE_NAME.sub("_", str(name or "")).strip("._-")
    return cleaned or fallback


def source_stem(path: str) -> str:
    """File stem of ``path``, sanitised for Action naming."""
    return safe_name(os.path.splitext(os.path.basename(str(path or "")))[0])


def jobs_root(blend_path: str | None = None) -> str:
    """Return the directory that holds per-job scratch folders.

    Prefers ``<dir of .blend>/.mocap_jobs`` so job data travels with the project;
    falls back to the system temporary directory for unsaved files.
    """
    if blend_path:
        base = os.path.dirname(normalize(blend_path))
        if base and os.path.isdir(base):
            return os.path.join(base, JOB_DIR_NAME)
    return os.path.join(tempfile.gettempdir(), JOB_DIR_NAME)


def job_dir(job_id: str, blend_path: str | None = None) -> str:
    """Scratch directory for a single job."""
    return os.path.join(jobs_root(blend_path), safe_name(job_id, "job"))


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if needed and return the normalised path."""
    normalized = normalize(path)
    os.makedirs(normalized, exist_ok=True)
    return normalized


def cancel_flag_path(output_dir: str) -> str:
    """Path of the cooperative cancellation sentinel for a job directory."""
    return os.path.join(normalize(output_dir), CANCEL_FILENAME)


def worker_log_path(output_dir: str) -> str:
    """Path of the worker debug log for a job directory."""
    return os.path.join(normalize(output_dir), WORKER_LOG_FILENAME)


def tail_text_file(path: str, max_chars: int = 2000) -> str:
    """Return the last ``max_chars`` characters of a text file, or ``""``."""
    try:
        with open(normalize(path), "r", encoding="utf-8", errors="replace") as handle:
            data = handle.read()
    except OSError:
        return ""
    if len(data) <= max_chars:
        return data
    return "..." + data[-max_chars:]
