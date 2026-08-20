"""Shared helpers for the Blender background tests.

These scripts run inside Blender, so they may import ``bpy``. They are launched
as::

    blender --background --factory-startup --python tests/blender/test_*.py

The harness makes the add-on importable whether or not it has been installed
into Blender's add-on directory, and can generate a real Rigify human rig.
"""

from __future__ import annotations

import os
import sys
import traceback

import bpy

#: Add-on package name (the repository directory name).
PACKAGE = "motion_capture"

_failures = []
_checks = 0


# --------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------


def check(condition, message: str) -> bool:
    """Record a check; returns the boolean so callers can branch."""
    global _checks
    _checks += 1
    if condition:
        print("  ok   - {0}".format(message))
        return True
    print("  FAIL - {0}".format(message))
    _failures.append(message)
    return False


def check_equal(actual, expected, message: str) -> bool:
    return check(actual == expected, "{0} (got {1!r}, expected {2!r})".format(message, actual, expected))


def check_almost_equal(actual, expected, message: str, tolerance: float = 1e-5) -> bool:
    """Compare floats with a tolerance.

    Blender stores ``FloatProperty`` values as 32-bit floats, so 0.65 reads back
    as 0.6499999761581421.
    """
    try:
        close = abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        close = False
    return check(close, "{0} (got {1!r}, expected ~{2!r})".format(message, actual, expected))


def section(title: str) -> None:
    print("\n== {0} ==".format(title))


def finish() -> None:
    """Print a summary and exit with a non-zero status on any failure."""
    print("\n{0} checks, {1} failures".format(_checks, len(_failures)))
    if _failures:
        for message in _failures:
            print("  FAILED: {0}".format(message))
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")
    sys.exit(0)


def fatal(message: str, exc=None) -> None:
    """Abort the test run with a clear message."""
    print("FATAL: {0}".format(message))
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    print("RESULT: FAIL")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Add-on loading
# --------------------------------------------------------------------------------------


def addon_root() -> str:
    """Repository root, derived from this file's location."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_importable() -> str:
    """Make ``motion_capture`` importable and return the resolved root."""
    root = addon_root()
    parent = os.path.dirname(root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return root


def enable_addon() -> tuple:
    """Enable the add-on, preferring the real operator path.

    Returns ``(module, used_operator)``. When the add-on is not installed into a
    Blender add-on directory it is imported directly and ``register()`` is called,
    which still exercises the same registration code.
    """
    ensure_importable()
    try:
        bpy.ops.preferences.addon_enable(module=PACKAGE)
        if PACKAGE in {a.module for a in bpy.context.preferences.addons}:
            module = sys.modules.get(PACKAGE)
            if module is not None:
                return module, True
    except Exception as exc:  # noqa: BLE001 - fall back to a direct import
        print("  note - addon_enable failed ({0}: {1}); importing directly".format(
            type(exc).__name__, exc
        ))

    import importlib

    module = importlib.import_module(PACKAGE)
    module.register()
    return module, False


def disable_addon(module, used_operator: bool) -> None:
    """Undo :func:`enable_addon`."""
    if used_operator:
        bpy.ops.preferences.addon_disable(module=PACKAGE)
        return
    module.unregister()


# --------------------------------------------------------------------------------------
# Rigify
# --------------------------------------------------------------------------------------


def enable_rigify() -> bool:
    """Enable Rigify through the operator.

    ``addon_utils.enable(default_set=False)`` leaves ``RigifyParameters``
    incompletely registered and rig generation then fails with
    ``'RigifyParameters' object has no attribute 'make_custom_pivot'``, so the
    operator path is required.
    """
    try:
        bpy.ops.preferences.addon_enable(module="rigify")
    except Exception as exc:  # noqa: BLE001
        print("  note - could not enable rigify: {0}".format(exc))
        return False
    return hasattr(bpy.types.PoseBone.bl_rna.properties.get("rigify_parameters"), "fixed_type")


def generate_rigify_human():
    """Add a human metarig and generate the rig; returns the generated object."""
    if not enable_rigify():
        return None
    bpy.ops.object.armature_human_metarig_add()
    metarig = bpy.context.object
    try:
        bpy.ops.pose.rigify_generate()
    except Exception as exc:  # noqa: BLE001
        print("  note - rigify_generate failed: {0}".format(exc))
        return None
    rig = bpy.context.object
    if rig is metarig:
        return None
    return rig


def reset_scene() -> None:
    """Start from an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
