"""Shared helpers for the unit tests (standard library only)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

#: Add-on root: <repo>/tests/unit/_util.py -> <repo>
ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if ADDON_ROOT not in sys.path:
    sys.path.insert(0, ADDON_ROOT)

FIXTURES = os.path.join(ADDON_ROOT, "tests", "fixtures")


class TempDirCase(unittest.TestCase):
    """Test case with a scratch directory that is removed on teardown."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="mocap_unit_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def path(self, *parts) -> str:
        return os.path.join(self.tmp, *parts)

    def touch(self, relative: str, content: str = "x") -> str:
        """Create a placeholder file below the scratch directory."""
        target = self.path(*relative.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        return target

    def write_json(self, relative: str, payload, encoding: str = "utf-8") -> str:
        import json

        target = self.path(*relative.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=encoding) as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return target


class Namespace(object):
    """Duck-typed stand-in for a Blender ``PropertyGroup`` / ``AddonPreferences``."""

    def __init__(self, **values) -> None:
        for key, value in values.items():
            setattr(self, key, value)


def scene_props(**overrides) -> Namespace:
    """Scene property defaults matching ``addon/properties.py``."""
    values = {
        "source_media": "",
        "source_type": "auto",
        "capture_profile": "preview",
        "frame_start": 1,
        "frame_end": 0,
        "target_fps": 30,
        "include_hands": True,
        "smoothing_strength": 0.65,
        "foot_lock_strength": 0.7,
        "root_motion": "world",
        "camera_view": "unspecified",
        "align_initial_facing": False,
        "motion_type": "general",
        "input_normalisation": "current",
    }
    values.update(overrides)
    return Namespace(**values)


def preferences(**overrides) -> Namespace:
    """Add-on preference defaults matching ``addon/preferences.py``."""
    values = {
        "worker_python": sys.executable,
        "models_root": "",
        "max_vram_gb": 7.0,
        "default_profile": "quality",
        "allow_auto_download": False,
    }
    values.update(overrides)
    return Namespace(**values)
