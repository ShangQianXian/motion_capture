"""Scene level properties (guide section 3.3).

Beyond the properties the guide lists, v0.1 adds a few implementation
necessities, all documented in ``README.md``: ``effective_profile``,
``last_job_dir``, ``progress``, ``progress_text``, ``switch_limbs_to_fk``,
``flip_x``, ``clean_curves``, ``frame_step``, ``applied_action_name`` and the
``log_entries`` collection that backs the log panel.

Note: no ``from __future__ import annotations`` here. Blender resolves string
annotations through ``typing.get_type_hints``, which swaps globals and locals and
breaks property registration.
"""

import time

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import ui_text

#: ``job_status`` values (guide section 11.3).
JOB_STATUSES = ("idle", "checking", "ready", "running", "completed", "failed", "cancelled")


def _armature_poll(self, obj) -> bool:
    return getattr(obj, "type", None) == "ARMATURE"


class MOCAP_LogEntry(bpy.types.PropertyGroup):
    """One line in the log panel."""

    level: StringProperty(name="Level", default="INFO")
    code: StringProperty(name="Code", default="")
    message: StringProperty(name="Message", default="")
    frame: IntProperty(name="Frame", default=-1)
    timestamp: StringProperty(name="Time", default="")


class MotionCaptureSceneProperties(bpy.types.PropertyGroup):
    """Per-scene capture settings and job state."""

    # -- input -------------------------------------------------------------------------

    source_media: StringProperty(
        name=ui_text.PROP_SOURCE_MEDIA,
        description=ui_text.PROP_SOURCE_MEDIA_DESC,
        subtype="FILE_PATH",
        default="",
    )
    source_type: EnumProperty(
        name=ui_text.PROP_SOURCE_TYPE,
        description=ui_text.PROP_SOURCE_TYPE_DESC,
        items=(
            ("auto", "自动", "按扩展名判断"),
            ("image", "图片", "单帧图片"),
            ("video", "视频", "视频文件"),
        ),
        default="auto",
    )
    target_armature: PointerProperty(
        name=ui_text.PROP_TARGET_ARMATURE,
        description=ui_text.PROP_TARGET_ARMATURE_DESC,
        type=bpy.types.Object,
        poll=_armature_poll,
    )
    capture_profile: EnumProperty(
        name=ui_text.PROP_CAPTURE_PROFILE,
        description=ui_text.PROP_CAPTURE_PROFILE_DESC,
        items=ui_text.capture_profile_items(),
        default="preview",
    )

    # -- range -------------------------------------------------------------------------

    frame_start: IntProperty(
        name=ui_text.PROP_FRAME_START,
        description=ui_text.PROP_FRAME_START_DESC,
        default=1,
        min=1,
    )
    frame_end: IntProperty(
        name=ui_text.PROP_FRAME_END,
        description=ui_text.PROP_FRAME_END_DESC,
        default=0,
        min=0,
    )
    target_fps: IntProperty(
        name=ui_text.PROP_TARGET_FPS,
        description=ui_text.PROP_TARGET_FPS_DESC,
        default=30,
        min=1,
        max=120,
    )

    # -- options -----------------------------------------------------------------------

    include_hands: BoolProperty(
        name=ui_text.PROP_INCLUDE_HANDS,
        description=ui_text.PROP_INCLUDE_HANDS_DESC,
        default=True,
    )
    smoothing_strength: FloatProperty(
        name=ui_text.PROP_SMOOTHING,
        description=ui_text.PROP_SMOOTHING_DESC,
        default=0.65,
        min=0.0,
        max=1.0,
    )
    foot_lock_strength: FloatProperty(
        name=ui_text.PROP_FOOT_LOCK,
        description=ui_text.PROP_FOOT_LOCK_DESC,
        default=0.7,
        min=0.0,
        max=1.0,
    )
    root_motion: EnumProperty(
        name=ui_text.PROP_ROOT_MOTION,
        description=ui_text.PROP_ROOT_MOTION_DESC,
        items=(
            ("world", "世界位移", "保留角色在世界空间中的移动"),
            ("in_place", "固定原地", "去掉水平位移，只保留上下起伏与旋转"),
        ),
        default="world",
    )

    # -- retarget options (v0.1 additions) ---------------------------------------------

    switch_limbs_to_fk: BoolProperty(
        name=ui_text.PROP_SWITCH_FK,
        description=ui_text.PROP_SWITCH_FK_DESC,
        default=True,
    )
    flip_x: BoolProperty(
        name=ui_text.PROP_FLIP_X,
        description=ui_text.PROP_FLIP_X_DESC,
        default=False,
    )
    clean_curves: BoolProperty(
        name=ui_text.PROP_CLEAN_CURVES,
        description=ui_text.PROP_CLEAN_CURVES_DESC,
        default=False,
    )
    frame_step: IntProperty(
        name=ui_text.PROP_FRAME_STEP,
        description=ui_text.PROP_FRAME_STEP_DESC,
        default=1,
        min=1,
        max=10,
    )

    # -- job state ---------------------------------------------------------------------

    job_status: StringProperty(name="Job Status", default="idle")
    effective_profile: StringProperty(name="Effective Profile", default="")
    last_result_path: StringProperty(name="Last Result", subtype="FILE_PATH", default="")
    last_job_dir: StringProperty(name="Last Job Dir", subtype="DIR_PATH", default="")
    last_error: StringProperty(name="Last Error", default="")
    progress: FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0)
    progress_text: StringProperty(name="Progress Text", default="")
    applied_action_name: StringProperty(name="Applied Action", default="")
    result_frame_start: IntProperty(name="Result Frame Start", default=0)
    result_frame_end: IntProperty(name="Result Frame End", default=0)
    preflight_ok: BoolProperty(name="Preflight OK", default=False)
    preflight_done: BoolProperty(name="Preflight Done", default=False)
    preflight_summary: StringProperty(name="Preflight Summary", default="")

    log_entries: CollectionProperty(type=MOCAP_LogEntry)

    # -- helpers -------------------------------------------------------------------------

    def set_status(self, status: str) -> None:
        if status in JOB_STATUSES:
            self.job_status = status

    def log(self, level: str, message: str, code: str = "", frame=None) -> None:
        """Append one entry, keeping only the most recent ``LOG_LIMIT`` lines."""
        entry = self.log_entries.add()
        entry.level = str(level)
        entry.code = str(code or "")
        entry.message = str(message or "")
        entry.frame = -1 if frame is None else int(frame)
        entry.timestamp = time.strftime("%H:%M:%S")
        while len(self.log_entries) > ui_text.LOG_LIMIT:
            self.log_entries.remove(0)

    def clear_log(self) -> None:
        self.log_entries.clear()

    def reset_job_state(self) -> None:
        self.progress = 0.0
        self.progress_text = ""
        self.last_error = ""

    def recent_logs(self, limit: int = ui_text.LOG_LIMIT) -> list:
        entries = list(self.log_entries)
        return entries[-limit:]


classes = (MOCAP_LogEntry, MotionCaptureSceneProperties)


def register_scene_properties() -> None:
    bpy.types.Scene.mocap_props = PointerProperty(type=MotionCaptureSceneProperties)


def unregister_scene_properties() -> None:
    if hasattr(bpy.types.Scene, "mocap_props"):
        del bpy.types.Scene.mocap_props


def get_props(context=None):
    """Return the scene property group, or ``None`` when not registered."""
    ctx = context or bpy.context
    scene = getattr(ctx, "scene", None)
    return getattr(scene, "mocap_props", None) if scene is not None else None
