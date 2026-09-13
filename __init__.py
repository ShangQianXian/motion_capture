"""Motion Capture for Rigify - Blender add-on entry point.

The repository root *is* the add-on package, so this file registers everything:
preferences, scene properties, operators and panels (guide section 3.1).

Import-time rules (guide sections 3.1 and 15):

* never start a worker, read a model or import torch / mmpose / mediapipe / cv2,
* never touch the filesystem beyond resolving this package's own path.
"""

from __future__ import annotations

bl_info = {
    "name": "Motion Capture for Rigify",
    "author": "Project Team",
    "version": (0, 3, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Mocap",
    "description": "Capture body motion from video or image and retarget it to Rigify rigs.",
    "category": "Animation",
}

import importlib
import sys

import bpy

#: Submodules reloaded when the add-on is re-enabled in the same session.
_RELOADABLE = (
    "core.errors",
    "core.paths",
    "core.model_manifest",
    "core.skeleton",
    "core.retarget_math",
    "core.smoothing",
    "core.motion_processing",
    "core.job_schema",
    "core.result_schema",
    "core.pose_calibration",
    "core.preview",
    "core.progress",
    "core.worker_client",
    "core.worker_environment",
    "blender.temp_data",
    "blender.rigify_adapter",
    "blender.action_baker",
    "addon.ui_text",
    "addon.preferences",
    "addon.properties",
    "addon.review",
    "addon.preview_view",
    "addon.operators",
    "addon.panels",
)


def _reload_submodules() -> None:
    """Reload already-imported submodules so a re-enable picks up source edits."""
    for suffix in _RELOADABLE:
        module = sys.modules.get("{0}.{1}".format(__name__, suffix))
        if module is not None:
            try:
                importlib.reload(module)
            except Exception:  # noqa: BLE001 - a stale module must not block enabling
                pass


# ``_ALREADY_LOADED`` survives in this module's namespace across a re-enable,
# which is how we detect that the submodules need reloading.
if "_ALREADY_LOADED" in locals():  # pragma: no cover - only on a reload
    _reload_submodules()
_ALREADY_LOADED = True

from .addon import operators, panels, preferences, properties, review, preview_view  # noqa: E402

#: Registration order matters: property groups before the PointerProperty that
#: uses them, operators before the panels that reference them.
_CLASS_MODULES = (preferences, properties, operators, preview_view, panels)

_registered_classes = []


def register() -> None:
    """Register every class and the ``Scene.mocap_props`` pointer.

    Rolls back on a partial failure so a raising ``register()`` cannot leave
    orphaned classes behind that would then fail with "already registered".
    """
    try:
        for module in _CLASS_MODULES:
            for cls in getattr(module, "classes", ()):
                bpy.utils.register_class(cls)
                _registered_classes.append(cls)
        properties.register_scene_properties()
        review.register()
    except Exception:
        _rollback()
        raise


def _rollback() -> None:
    """Unregister whatever the failed :func:`register` already registered."""
    review.unregister()
    properties.unregister_scene_properties()
    for cls in reversed(_registered_classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):  # pragma: no cover
            pass
    del _registered_classes[:]


def unregister() -> None:
    """Reverse of :func:`register`, leaving no residue behind.

    Also stops any running worker and removes temporary objects and constraints,
    so disabling the add-on mid-capture cannot leave the file or the process tree
    in a broken state (Phase 0 acceptance criteria).
    """
    try:
        operators.clear_caches()
    except Exception:  # noqa: BLE001 - teardown must not raise
        pass
    review.unregister()
    try:
        from .core import worker_client

        worker_client.shutdown_all()
    except Exception:  # noqa: BLE001
        pass
    try:
        from .blender import temp_data

        temp_data.cleanup_all()
    except Exception:  # noqa: BLE001 - bpy data may already be unavailable
        pass

    properties.unregister_scene_properties()

    for cls in reversed(_registered_classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):  # pragma: no cover - already unregistered
            pass
    del _registered_classes[:]


if __name__ == "__main__":  # pragma: no cover - convenience for Text Editor use
    register()
