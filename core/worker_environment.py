"""Framework-free routing for the two external worker environments."""
from __future__ import annotations

import os
from . import paths


def environment_for_profile(profile):
    return "preview" if profile in ("preview", "fallback_cpu") else "quality"


def select_python(quality_python, preview_python="", profile="quality", addon_root=None):
    """Explicit paths win; discover project venvs, then support old single-path setups."""
    root = addon_root or paths.addon_root()
    quality = quality_python or os.path.join(root, ".venv", "Scripts", "python.exe")
    if environment_for_profile(profile) == "quality":
        return paths.normalize(quality) if quality_python or os.path.isfile(quality) else ""
    if preview_python:
        return paths.normalize(preview_python)
    preview = os.path.join(root, ".venv-preview", "Scripts", "python.exe")
    if os.path.isfile(preview):
        return paths.normalize(preview)
    return paths.normalize(quality) if quality_python or os.path.isfile(quality) else ""
