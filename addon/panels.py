"""View3D sidebar panels (guide section 3.4).

UI rules:

* the capture button is disabled while the environment check has not passed,
* ``preview`` may run without the MMPose models,
* ``quality`` refuses to start while required models are missing,
* the log panel shows the most recent 20 progress / warning / error lines.
"""

from __future__ import annotations

import bpy

from ..core import model_manifest
from . import operators, preferences, properties, ui_text

#: Sidebar category.
CATEGORY = ui_text.TAB_CATEGORY


class _MocapPanel(bpy.types.Panel):
    """Shared panel configuration."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class MOCAP_PT_environment(_MocapPanel):
    """Preferences summary, preflight results and environment tests."""

    bl_idname = "MOCAP_PT_environment"
    bl_label = ui_text.PANEL_ENVIRONMENT

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        prefs = preferences.get_preferences(context)

        if prefs is None:
            layout.label(text="无法读取插件偏好设置。", icon="ERROR")
            return

        models_root = prefs.resolved_models_root()
        worker_python = prefs.resolved_worker_python()

        column = layout.column(align=True)
        column.label(
            text="{0}: {1}".format(
                ui_text.PREF_WORKER_PYTHON, _shorten(worker_python) or "未配置"
            ),
            icon="CONSOLE" if worker_python else "ERROR",
        )
        column.label(
            text="{0}: {1}".format(ui_text.PREF_MODELS_ROOT, _shorten(models_root) or "未配置"),
            icon="FILE_FOLDER" if models_root else "ERROR",
        )

        row = layout.row(align=True)
        row.operator("mocap.preflight", icon="CHECKMARK")
        row.operator("mocap.open_models_dir", text="", icon="FILE_FOLDER")

        row = layout.row(align=True)
        row.operator("mocap.test_worker_python", icon="CONSOLE")
        row.operator("mocap.test_cuda", icon="MEMORY")

        row = layout.row(align=True)
        row.operator("mocap.test_preview_model", icon="PLAY")
        row.operator("mocap.test_quality_model", icon="PLAY")

        if props is None:
            return

        report = operators.cached_preflight(props.capture_profile, models_root)
        if report is None:
            layout.label(text=ui_text.MSG_PREFLIGHT_FIRST, icon="INFO")
            return

        box = layout.box()
        header = box.row()
        header.label(
            text="Profile: {0}".format(ui_text.profile_label(report.effective_profile)),
            icon="CHECKMARK" if report.ok else "ERROR",
        )
        if report.effective_profile != report.profile:
            box.label(
                text="已从 {0} 回退".format(ui_text.profile_label(report.profile)), icon="LOOP_BACK"
            )
        box.label(text=props.preflight_summary or "")

        if report.missing_required:
            column = box.column(align=True)
            column.label(text="缺少必需文件：", icon="ERROR")
            for status in report.missing_required[:8]:
                _draw_artifact(column, status)
        if report.missing_optional:
            column = box.column(align=True)
            column.label(text="缺少可选文件：", icon="INFO")
            for status in report.missing_optional[:6]:
                _draw_artifact(column, status)

        if report.missing_required or report.missing_optional:
            box.operator("mocap.copy_missing_model_links", icon="COPYDOWN")


def _draw_artifact(layout, status) -> None:
    """One artifact status row: display name, relative path and download URL."""
    layout.label(text="  {0}".format(status.display_name))
    layout.label(text="    models/{0}".format(status.relative_path))
    if status.download_url:
        layout.label(text="    {0}".format(_shorten(status.download_url, 58)))
    if status.fallback and not status.required:
        layout.label(text="    降级：{0}".format(_shorten(status.fallback, 52)))


class MOCAP_PT_capture(_MocapPanel):
    """Source selection, capture settings and the run / cancel buttons."""

    bl_idname = "MOCAP_PT_capture"
    bl_label = ui_text.PANEL_CAPTURE

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        prefs = preferences.get_preferences(context)
        if props is None or prefs is None:
            layout.label(text="属性未注册。", icon="ERROR")
            return

        column = layout.column(align=True)
        column.prop(props, "source_media")
        column.prop(props, "source_type")
        column.prop(props, "capture_profile")

        row = layout.row(align=True)
        row.prop(props, "frame_start")
        row.prop(props, "frame_end")
        layout.prop(props, "target_fps")

        box = layout.box()
        box.label(text="处理选项", icon="MODIFIER")
        box.prop(props, "include_hands")
        box.prop(props, "smoothing_strength", slider=True)
        box.prop(props, "foot_lock_strength", slider=True)
        box.prop(props, "root_motion")

        running = props.job_status == "running"
        worker = operators.active_worker()
        blocked, reason = _capture_blocked(props, prefs)

        if props.capture_profile == "preview":
            layout.label(text=ui_text.MSG_PREVIEW_HINT, icon="INFO")
        if blocked and reason:
            layout.label(text=reason, icon="ERROR")

        row = layout.row(align=True)
        run = row.row(align=True)
        run.enabled = not running and not blocked
        run.operator("mocap.run_capture", icon="PLAY").mock = False
        mock = row.row(align=True)
        mock.enabled = not running
        mock.operator("mocap.run_capture", text=ui_text.OP_RUN_MOCK, icon="GHOST_ENABLED").mock = True

        cancel = layout.row()
        cancel.enabled = running and worker is not None
        cancel.operator("mocap.cancel_capture", icon="CANCEL")

        status = layout.box()
        status.label(
            text="状态：{0}".format(ui_text.status_label(props.job_status)),
            icon=_status_icon(props.job_status),
        )
        if running:
            # UILayout.progress() only exists in Blender 4.2+; 4.0 gets a label.
            if hasattr(status, "progress"):
                status.progress(factor=props.progress, text=props.progress_text or "...")
            else:
                status.label(text="{0:.0f}%  {1}".format(props.progress * 100.0, props.progress_text))
        if props.last_error:
            status.label(text=_shorten(props.last_error, 60), icon="ERROR")


def _capture_blocked(props, prefs) -> tuple:
    """``(blocked, reason)`` for the Run Capture button (guide section 11.3)."""
    if not prefs.resolved_worker_python():
        return True, ui_text.MSG_NO_WORKER_PYTHON
    if not props.source_media:
        return True, "请先选择源文件。"
    report = operators.cached_preflight(props.capture_profile, prefs.resolved_models_root())
    if report is None:
        return True, ui_text.MSG_PREFLIGHT_FIRST
    if not report.ok:
        return True, ui_text.MSG_ENV_BLOCKED
    return False, ""


class MOCAP_PT_rigify(_MocapPanel):
    """Target rig, import, apply and bake."""

    bl_idname = "MOCAP_PT_rigify"
    bl_label = ui_text.PANEL_RIGIFY

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        if props is None:
            layout.label(text="属性未注册。", icon="ERROR")
            return

        layout.prop(props, "target_armature")
        armature = props.target_armature
        if armature is not None:
            from ..blender import rigify_adapter

            detection = rigify_adapter.detect_rigify_human(armature)
            if detection.ok:
                layout.label(
                    text="Rigify rig 已识别（rig_id={0}）".format(detection.rig_id or "无"),
                    icon="CHECKMARK",
                )
            else:
                layout.label(
                    text=_shorten(detection.error.message if detection.error else "未识别", 58),
                    icon="ERROR",
                )

        box = layout.box()
        box.label(text="重定向选项", icon="CON_ROTLIKE")
        box.prop(props, "switch_limbs_to_fk")
        box.prop(props, "flip_x")
        box.prop(props, "frame_step")

        has_result = bool(props.last_result_path)
        loaded = operators.loaded_result() is not None
        has_action = bool(props.applied_action_name)

        row = layout.row()
        row.enabled = has_result
        row.operator("mocap.import_result", icon="IMPORT")

        row = layout.row()
        row.enabled = loaded and armature is not None
        row.operator("mocap.apply_to_rigify", icon="ARMATURE_DATA")

        column = layout.column(align=True)
        column.enabled = has_action and armature is not None
        column.prop(props, "clean_curves")
        column.operator("mocap.bake_action", icon="RENDER_ANIMATION")

        layout.operator("mocap.clear_temp_data", icon="TRASH")

        if props.applied_action_name:
            layout.label(text="Action: {0}".format(props.applied_action_name), icon="ACTION")
        if props.last_result_path:
            layout.label(text=_shorten(props.last_result_path, 58), icon="FILE")


class MOCAP_PT_logs(_MocapPanel):
    """Recent progress, warning and error lines."""

    bl_idname = "MOCAP_PT_logs"
    bl_label = ui_text.PANEL_LOGS
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        if props is None:
            layout.label(text="属性未注册。", icon="ERROR")
            return
        entries = props.recent_logs()
        if not entries:
            layout.label(text=ui_text.MSG_LOG_EMPTY, icon="INFO")
            return
        column = layout.column(align=True)
        for entry in reversed(entries):
            icon = {"ERROR": "ERROR", "WARNING": "ERROR"}.get(entry.level, "INFO")
            prefix = "{0} ".format(entry.timestamp) if entry.timestamp else ""
            text = "{0}{1}".format(prefix, entry.message)
            if entry.frame >= 0:
                text = "{0} (帧 {1})".format(text, entry.frame)
            column.label(text=_shorten(text, 64), icon=icon)


def _status_icon(status: str) -> str:
    return {
        "idle": "RADIOBUT_OFF",
        "checking": "SORTTIME",
        "ready": "CHECKMARK",
        "running": "PLAY",
        "completed": "CHECKMARK",
        "failed": "ERROR",
        "cancelled": "CANCEL",
    }.get(str(status), "INFO")


def _shorten(text, limit: int = 44) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return "..." + value[-(limit - 3):]


classes = (
    MOCAP_PT_environment,
    MOCAP_PT_capture,
    MOCAP_PT_rigify,
    MOCAP_PT_logs,
)


def profile_is_supported(profile: str) -> bool:
    """True when ``profile`` is a selectable capture profile."""
    return profile in model_manifest.CAPTURE_PROFILES
