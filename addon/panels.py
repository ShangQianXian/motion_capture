"""View3D sidebar panels (guide section 3.4).

UI rules:

* the capture button is disabled while the environment check has not passed,
* ``preview`` may run without the MMPose models,
* ``quality`` refuses to start while required models are missing,
* the log panel shows the most recent 20 progress / warning / error lines.
"""

from __future__ import annotations

import bpy

from ..core import model_manifest, paths
from . import operators, preferences, properties, ui_text, review

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
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        prefs = preferences.get_preferences(context)

        if prefs is None:
            layout.label(text="无法读取插件偏好设置。", icon="ERROR")
            return

        models_root = prefs.resolved_models_root()
        worker_python = prefs.resolved_worker_python(props.capture_profile)
        report = operators.cached_preflight(props.capture_profile, models_root)
        if report is not None and report.ok:
            layout.label(text="当前捕捉环境已就绪", icon="CHECKMARK")
            layout.prop(props, "show_environment", icon="TRIA_DOWN" if props.show_environment else "TRIA_RIGHT", emboss=False)
            if not props.show_environment:
                return

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
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        prefs = preferences.get_preferences(context)
        if props is None or prefs is None:
            layout.label(text="属性未注册。", icon="ERROR")
            return

        column = layout.column(align=True)
        column.enabled = props.job_status not in ('running', 'applying')
        column.prop(props, "source_media")
        column.prop(props, "source_type")
        value = review.session(context.scene, create=False)
        if value:
            if value.icon_id:
                layout.template_icon(icon_value=value.icon_id, scale=6.0)
            if value.info:
                import os
                layout.label(text=os.path.basename(props.source_media), icon='IMAGE_DATA')
                layout.label(text="{width} × {height}".format(**value.info))
                if value.info['type'] == 'video':
                    layout.label(text="{duration:.2f} 秒 · {fps:.3f} FPS".format(**value.info))
            if value.error:
                layout.label(text=_shorten(value.error, 55), icon='ERROR')
                row = layout.row(align=True)
                row.operator('mocap.relocate_source', icon='FILE_FOLDER')
                row.operator('mocap.preview_control', text='重新加载', icon='FILE_REFRESH').command = 'retry'
        row = layout.row()
        row.enabled = bool(props.source_media) and props.job_status != 'applying'
        row.operator('mocap.open_preview', text='查看素材 / 对照预览', icon='IMAGE_DATA')
        column = layout.column()
        column.enabled = props.job_status not in ('running', 'applying')
        column.prop(props, "capture_profile")
        column.prop(props, 'camera_view')
        if props.camera_view == 'left_front_45':
            column.prop(props, 'align_initial_facing')
        is_image = props.source_type == 'image' or (props.source_type == 'auto' and paths.guess_media_type(props.source_media) == 'image')

        if not is_image:
            column.prop(props, 'motion_type')
            row = column.row(align=True)
            row.prop(props, "frame_start")
            row.prop(props, "frame_end")
            column.prop(props, "target_fps")

        box = column.box()
        box.label(text="处理选项", icon="MODIFIER")
        if props.capture_profile in ('preview', 'fallback_cpu'):
            box.prop(props, "include_hands")
        if not is_image:
            box.prop(props, "smoothing_strength", slider=True)
            box.prop(props, "foot_lock_strength", slider=True)
        box.prop(props, "root_motion")

        running = props.job_status in ("running", "applying")
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
        if props.job_status == 'running':
            cancel = layout.row()
            cancel.enabled = worker is not None
            cancel.operator("mocap.cancel_capture", icon="CANCEL")
        if props.job_status == 'completed':
            layout.label(text='下一步：打开对照预览并核对动作', icon='INFO')

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
        layout.prop(props, 'show_developer', icon='TRIA_DOWN' if props.show_developer else 'TRIA_RIGHT', emboss=False)
        if props.show_developer:
            row = layout.row()
            row.enabled = not running
            row.operator('mocap.run_capture', text=ui_text.OP_RUN_MOCK, icon='GHOST_ENABLED').mock = True
            layout.operator('mocap.import_result', text='加载已有结果', icon='IMPORT')


class MOCAP_PT_preview(_MocapPanel):
    bl_idname = 'MOCAP_PT_preview'
    bl_label = '2 · 预览与校正'
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        props = properties.get_props(context)
        value = review.session(context.scene, create=False)
        if not props or not value or not value.state:
            layout.label(text='生成结果后，在这里核对动作。', icon='INFO')
            return
        layout.operator('mocap.open_preview', icon='IMAGE_DATA')
        if value.raw_result:
            layout.prop(props, 'preview_stage', expand=True)
        if value.state.manifest and value.state.manifest.get('diagnostics'):
            diagnostics = value.state.manifest['diagnostics']
            counts = diagnostics.get('contact_counts', {})
            if counts:
                layout.label(text='脚部接触帧 L {0} / R {1}'.format(counts.get('L', {}).get('contact', 0), counts.get('R', {}).get('contact', 0)))
            if diagnostics.get('support_events'):
                layout.label(text='支撑事件 L {0} / R {1}'.format(*(diagnostics['support_events'].get(s, 0) for s in ('L', 'R'))))
            if not diagnostics.get('root_trajectory_available', True):
                layout.label(text='根相对结果：无真实水平轨迹', icon='INFO')
            for name in ('ankle.L', 'ankle.R', 'wrist.L', 'wrist.R'):
                joint = diagnostics.get('joints', {}).get(name, {})
                if joint.get('amplitude_ratio') is not None:
                    layout.label(text='{0} 幅度比 {1:.2f} / 偏移 {2} 帧'.format(name, joint['amplitude_ratio'], joint.get('lag_frames')))
            if diagnostics.get('joints'):
                layout.label(text='幅度比比较处理前后，不是识别准确率', icon='INFO')
        layout.prop(props, 'preview_overlay')
        if value.info.get('type') != 'image':
            row = layout.row(align=True)
            row.prop(props, 'preview_frame')
            row.label(text='/ {0}'.format(value.total()))
            row = layout.row(align=True)
            row.enabled = value.view is not None
            row.operator('mocap.preview_control', text='', icon='REW').command = 'previous'
            row.operator('mocap.preview_control', text='暂停' if props.preview_playing else '播放',
                         icon='PAUSE' if props.preview_playing else 'PLAY').command = 'play'
            row.operator('mocap.preview_control', text='', icon='FF').command = 'next'
            layout.prop(props, 'preview_speed')
            layout.prop(props, 'preview_loop')
            row = layout.row(align=True)
            row.operator('mocap.preview_control', text='上个问题帧', icon='PREV_KEYFRAME').command = 'problem_previous'
            row.operator('mocap.preview_control', text='下个问题帧', icon='NEXT_KEYFRAME').command = 'problem_next'
        if value.state.manifest is None:
            layout.label(text='旧版 / Mock 结果没有二维检测数据。', icon='INFO')
        layout.prop(props, 'flip_x')
        layout.prop(props, 'pitch_correction')
        layout.operator('mocap.calibrate_pitch', icon='ORIENTATION_GLOBAL')
        if value.state.stale:
            layout.label(text=value.state.stale_reason, icon='ERROR')
        elif value.state.viewed:
            layout.label(text='当前预览已显示，可选择 Rigify 应用。', icon='CHECKMARK')


def _capture_blocked(props, prefs) -> tuple:
    """``(blocked, reason)`` for the Run Capture button (guide section 11.3)."""
    if not prefs.resolved_worker_python(props.capture_profile):
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
    bl_order = 2

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

        has_action = bool(props.applied_action_name)
        reason = review.apply_block_reason(context.scene)
        row = layout.row()
        row.enabled = not reason
        row.scale_y = 1.4
        row.operator("mocap.apply_to_rigify", icon="ARMATURE_DATA")
        if reason:
            layout.label(text=_shorten(reason, 56), icon='INFO')
        layout.label(text='新 Action 从第 1 帧开始，保留原动画。', icon='ACTION')
        layout.prop(props, 'show_advanced', icon='TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT', emboss=False)
        if props.show_advanced:
            box = layout.box()
            box.prop(props, 'switch_limbs_to_fk')
            box.prop(props, 'frame_step')
            column = box.column(align=True)
            column.enabled = has_action and armature is not None and props.job_status != 'applying'
            column.prop(props, "clean_curves")
            column.operator("mocap.bake_action", icon="RENDER_ANIMATION")
            box.operator("mocap.clear_temp_data", icon="TRASH")

        if props.applied_action_name:
            layout.label(text="Action: {0}".format(props.applied_action_name), icon="ACTION")
        if props.last_result_path:
            layout.label(text=_shorten(props.last_result_path, 58), icon="FILE")


class MOCAP_PT_logs(_MocapPanel):
    """Recent progress, warning and error lines."""

    bl_idname = "MOCAP_PT_logs"
    bl_label = ui_text.PANEL_LOGS
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

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
        "applying": "ACTION",
    }.get(str(status), "INFO")


def _shorten(text, limit: int = 44) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return "..." + value[-(limit - 3):]


classes = (
    MOCAP_PT_environment,
    MOCAP_PT_capture,
    MOCAP_PT_preview,
    MOCAP_PT_rigify,
    MOCAP_PT_logs,
)


def profile_is_supported(profile: str) -> bool:
    """True when ``profile`` is a selectable capture profile."""
    return profile in model_manifest.CAPTURE_PROFILES
