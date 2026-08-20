"""Blender background test: enable / disable the add-on cleanly.

Phase 0 acceptance (guide section, Phase 0):

* enabling the add-on raises nothing,
* ``bpy.ops.preferences.addon_enable(module="motion_capture")`` succeeds,
* every class, the Scene property and any temporary handler are removed on
  disable.

Run::

    blender --background --factory-startup --python tests/blender/test_enable_addon.py
"""

from __future__ import annotations

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness as harness  # noqa: E402

OPERATOR_CLASSES = (
    "MOCAP_OT_preflight",
    "MOCAP_OT_copy_missing_model_links",
    "MOCAP_OT_open_models_dir",
    "MOCAP_OT_test_worker_python",
    "MOCAP_OT_test_cuda",
    "MOCAP_OT_test_preview_model",
    "MOCAP_OT_test_quality_model",
    "MOCAP_OT_run_capture",
    "MOCAP_OT_cancel_capture",
    "MOCAP_OT_import_result",
    "MOCAP_OT_apply_to_rigify",
    "MOCAP_OT_bake_action",
    "MOCAP_OT_clear_temp_data",
)

PANELS = (
    "MOCAP_PT_environment",
    "MOCAP_PT_capture",
    "MOCAP_PT_rigify",
    "MOCAP_PT_logs",
)


def main() -> None:
    print("Blender {0} / Python {1}".format(bpy.app.version_string, sys.version.split()[0]))

    harness.section("enable")
    try:
        module, used_operator = harness.enable_addon()
    except Exception as exc:  # noqa: BLE001
        harness.fatal("enabling the add-on raised", exc)
        return
    harness.check(module is not None, "add-on module imported")
    print("  note - enabled via {0}".format("addon_enable operator" if used_operator else "direct import"))

    harness.section("bl_info")
    info = getattr(module, "bl_info", {})
    harness.check_equal(info.get("version"), (0, 1, 0), "bl_info version")
    harness.check_equal(info.get("blender"), (4, 0, 0), "bl_info minimum Blender")
    harness.check_equal(info.get("category"), "Animation", "bl_info category")

    harness.section("registration")
    harness.check(hasattr(bpy.types.Scene, "mocap_props"), "Scene.mocap_props exists")
    props = bpy.context.scene.mocap_props
    harness.check(props is not None, "scene property group instantiated")
    harness.check_equal(props.job_status, "idle", "initial job_status")
    harness.check_equal(props.target_fps, 30, "default target_fps")
    harness.check_almost_equal(props.smoothing_strength, 0.65, "default smoothing_strength")
    harness.check_almost_equal(props.foot_lock_strength, 0.7, "default foot_lock_strength")
    harness.check_equal(props.root_motion, "world", "default root_motion")
    harness.check_equal(props.frame_end, 0, "default frame_end means auto")

    # bpy.ops.<module>.<name> resolves lazily and never raises, so hasattr on it is
    # meaningless; the registered classes in bpy.types are the real evidence.
    for name in OPERATOR_CLASSES:
        harness.check(hasattr(bpy.types, name), "operator class {0} registered".format(name))
    for name in PANELS:
        harness.check(hasattr(bpy.types, name), "panel {0} registered".format(name))

    harness.section("operator is callable")
    try:
        result = bpy.ops.mocap.clear_temp_data()
        harness.check("FINISHED" in result, "mocap.clear_temp_data executes")
    except Exception as exc:  # noqa: BLE001
        harness.check(False, "mocap.clear_temp_data executes ({0})".format(exc))

    harness.section("preferences")
    from motion_capture.addon import preferences as prefs_module

    # AddonPreferences subclasses register under their bl_idname, not their class
    # name, so they never appear as bpy.types.MotionCapturePreferences.
    prefs = prefs_module.get_preferences()
    harness.check(prefs is not None, "preferences instance reachable")
    if prefs is not None:
        harness.check_almost_equal(prefs.max_vram_gb, 7.0, "default max_vram_gb")
        harness.check_equal(prefs.allow_auto_download, False, "auto download off by default")
        harness.check_equal(prefs.default_profile, "quality", "default profile is quality")

    harness.section("logging")
    props.clear_log()  # the clear_temp_data call above already logged one line
    props.log("INFO", "hello")
    harness.check_equal(len(props.log_entries), 1, "log entry appended")
    for index in range(40):
        props.log("INFO", "line {0}".format(index))
    harness.check(len(props.log_entries) <= 20, "log capped at 20 entries")
    props.clear_log()
    harness.check_equal(len(props.log_entries), 0, "log cleared")

    harness.section("core imports do not pull heavy deps")
    heavy = [name for name in ("torch", "mmpose", "mmdet", "mediapipe", "cv2") if name in sys.modules]
    harness.check(not heavy, "no inference dependency imported at load time (found {0})".format(heavy))

    harness.section("disable")
    try:
        harness.disable_addon(module, used_operator)
    except Exception as exc:  # noqa: BLE001
        harness.fatal("disabling the add-on raised", exc)
        return
    harness.check(not hasattr(bpy.types.Scene, "mocap_props"), "Scene.mocap_props removed")
    for name in PANELS:
        harness.check(not hasattr(bpy.types, name), "panel {0} unregistered".format(name))
    remaining = [name for name in OPERATOR_CLASSES if hasattr(bpy.types, name)]
    harness.check(not remaining, "operator classes unregistered (remaining {0})".format(remaining))
    harness.check(
        prefs_module.get_preferences() is None,
        "preferences no longer reachable",
    )

    harness.section("re-enable")
    try:
        module, used_operator = harness.enable_addon()
        harness.check(hasattr(bpy.types.Scene, "mocap_props"), "re-enable restores properties")
        harness.disable_addon(module, used_operator)
    except Exception as exc:  # noqa: BLE001
        harness.fatal("re-enabling the add-on raised", exc)
        return

    harness.finish()


if __name__ == "__main__":
    main()
