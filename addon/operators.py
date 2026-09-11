"""Operators (guide section 3.5).

Rules enforced here:

* no ``torch`` / ``mmpose`` / ``mediapipe`` / ``cv2`` imports,
* every long task runs in the external worker,
* Blender's main thread only builds jobs, polls progress and applies results,
* every failure is reported through ``self.report`` **and** ``props.last_error``.

Note: no ``from __future__ import annotations`` here either - Blender resolves
string annotations through ``typing.get_type_hints``, which breaks operator
property registration.
"""

import os

import bpy
from bpy.props import BoolProperty, StringProperty

from ..core import errors, job_schema, model_manifest, paths, progress as progress_mod
from ..core import result_schema, worker_client, preview
from . import preferences, properties, ui_text, review

#: Modal timer interval in seconds.
TIMER_INTERVAL = 0.15

#: Cache of the most recent preflight report, keyed by ``(profile, models_root)``.
_preflight_cache = {}

#: The running worker, if any. Only one capture at a time in v0.1.
_active_job = {"worker": None, "job": None, "scene": None, "operator": None}
_active_application = None

# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _fail(operator, props, error) -> set:
    """Report an error consistently and return ``{'CANCELLED'}``."""
    if isinstance(error, errors.MocapError):
        text = error.user_text()
        code = error.code
    else:
        text = str(error)
        code = ""
    if props is not None:
        props.last_error = text
        props.log("ERROR", text, code)
    operator.report({"ERROR"}, text)
    return {"CANCELLED"}


def _info(operator, props, message, code: str = "") -> None:
    if props is not None:
        props.log("INFO", message, code)
    operator.report({"INFO"}, message)


def _warn(operator, props, message, code: str = "") -> None:
    if props is not None:
        props.log("WARNING", message, code)
    operator.report({"WARNING"}, message)


def _tag_redraw() -> None:
    window_manager = bpy.context.window_manager
    for window in getattr(window_manager, "windows", []) or []:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()


def get_preflight(profile: str, models_root: str, refresh: bool = False):
    """Cached :func:`model_manifest.check_profile_requirements`."""
    key = (str(profile), str(models_root))
    if refresh or key not in _preflight_cache:
        _preflight_cache[key] = model_manifest.check_profile_requirements(profile, models_root)
    return _preflight_cache[key]


def cached_preflight(profile: str, models_root: str):
    """Preflight report if it was already computed, otherwise ``None``."""
    return _preflight_cache.get((str(profile), str(models_root)))


def clear_caches() -> None:
    """Drop every module-level cache; called from ``unregister``."""
    _preflight_cache.clear()
    capture_operator = _active_job.get('operator')
    if capture_operator is not None:
        capture_operator._remove_timer(bpy.context)
    worker = _active_job.get("worker")
    if worker is not None:
        try:
            worker.close()
        except Exception:  # pragma: no cover - defensive teardown
            pass
    _active_job["worker"] = None
    _active_job["job"] = None
    _active_job["scene"] = None
    _active_job["operator"] = None
    global _active_application
    if _active_application is not None:
        _active_application.abort()
        _active_application = None


def active_worker():
    """The running worker, or ``None``."""
    return _active_job.get("worker")


def loaded_result(context=None):
    """The imported :class:`result_schema.MocapResult`, or ``None``."""
    value = review.session((context or bpy.context).scene, create=False)
    return value.state.result if value and value.state else None


def _blend_path() -> str:
    return bpy.data.filepath or ""


def _require_prefs(operator, props):
    prefs = preferences.get_preferences()
    if prefs is None:
        _fail(operator, props, errors.MocapError(
            errors.INTERNAL_ERROR, "无法读取插件偏好设置。"
        ))
        return None
    return prefs


# --------------------------------------------------------------------------------------
# Environment operators
# --------------------------------------------------------------------------------------


class MOCAP_OT_preflight(bpy.types.Operator):
    """Check that the models the selected profile needs are present."""

    bl_idname = "mocap.preflight"
    bl_label = ui_text.OP_PREFLIGHT
    bl_description = ui_text.OP_PREFLIGHT_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}

        profile = props.capture_profile if props is not None else prefs.default_profile
        models_root = prefs.resolved_models_root()
        if props is not None:
            props.set_status("checking")

        report = get_preflight(profile, models_root, refresh=True)
        summary = ui_text.MSG_ENV_MISSING.format(
            len(report.missing_required), len(report.missing_optional)
        )
        if props is not None:
            props.preflight_done = True
            props.preflight_ok = report.ok
            props.effective_profile = report.effective_profile
            props.preflight_summary = summary
            for message in report.warnings[: ui_text.LOG_LIMIT]:
                props.log("WARNING", message)
            props.set_status("ready" if report.ok else "failed")

        _tag_redraw()
        if report.ok:
            _info(self, props, ui_text.MSG_ENV_OK.format(report.effective_profile))
            return {"FINISHED"}
        if not models_root:
            return _fail(self, props, errors.MocapError(
                errors.MODELS_ROOT_NOT_FOUND, ui_text.MSG_NO_MODELS_ROOT
            ))
        error = model_manifest.first_missing_error(report)
        if error is not None:
            return _fail(self, props, error)
        _warn(self, props, summary)
        return {"FINISHED"}


class MOCAP_OT_copy_missing_model_links(bpy.types.Operator):
    """Copy the missing models' save paths and download URLs to the clipboard."""

    bl_idname = "mocap.copy_missing_model_links"
    bl_label = ui_text.OP_COPY_LINKS
    bl_description = ui_text.OP_COPY_LINKS_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}
        profile = props.capture_profile if props is not None else prefs.default_profile
        report = get_preflight(profile, prefs.resolved_models_root())
        text = model_manifest.format_missing_links(report)
        context.window_manager.clipboard = text
        _info(self, props, ui_text.MSG_COPIED.format(len(text.splitlines())))
        return {"FINISHED"}


class MOCAP_OT_open_models_dir(bpy.types.Operator):
    """Open the models root in the system file browser."""

    bl_idname = "mocap.open_models_dir"
    bl_label = ui_text.OP_OPEN_MODELS_DIR
    bl_description = ui_text.OP_OPEN_MODELS_DIR_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}
        target = prefs.resolved_models_root() or paths.bundled_models_dir()
        if not os.path.isdir(target):
            return _fail(self, props, errors.MocapError(
                errors.MODELS_ROOT_NOT_FOUND,
                "目录不存在：{0}".format(target),
                details={"models_root": target},
            ))
        bpy.ops.wm.path_open(filepath=target)
        _info(self, props, "已打开：{0}".format(target))
        return {"FINISHED"}


class MOCAP_OT_test_worker_python(bpy.types.Operator):
    """Run the worker's ``--check-env`` and report the result."""

    bl_idname = "mocap.test_worker_python"
    bl_label = ui_text.OP_TEST_WORKER
    bl_description = ui_text.OP_TEST_WORKER_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}
        try:
            profile = props.capture_profile if props is not None else prefs.default_profile
            report = worker_client.check_env(prefs.resolved_worker_python(profile), profile=profile)
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        payload = report.get("report") or {}
        if report.get("event") == "failed" or report.get("returncode", 0) != 0 or not payload.get("ok", True):
            return _fail(self, props, errors.MocapError.from_dict(report.get("error") or payload.get("error") or {
                "code": errors.DEPENDENCY_MISSING, "message": "Worker 环境检查失败。"
            }))
        modules = payload.get("modules") or {}
        available = sorted(name for name, info in modules.items() if info.get("available"))
        missing = sorted(name for name, info in modules.items() if not info.get("available"))
        message = "Python {0}；可用：{1}；缺失：{2}".format(
            payload.get("python", "?"),
            "、".join(available) or "无",
            "、".join(missing) or "无",
        )
        _info(self, props, message)
        return {"FINISHED"}


class MOCAP_OT_test_cuda(bpy.types.Operator):
    """Run the worker's ``--check-cuda`` and report the result."""

    bl_idname = "mocap.test_cuda"
    bl_label = ui_text.OP_TEST_CUDA
    bl_description = ui_text.OP_TEST_CUDA_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}
        try:
            report = worker_client.check_cuda(prefs.resolved_worker_python())
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        payload = report.get("report") or {}
        if payload.get("available"):
            _info(self, props, "CUDA 可用：{0}（{1} GB）".format(
                payload.get("device_name", "?"), payload.get("total_memory_gb", "?")
            ))
            return {"FINISHED"}
        error = errors.MocapError.from_dict(
            report.get("error") or payload.get("error") or {
                "code": errors.CUDA_UNAVAILABLE, "message": "CUDA 不可用。"
            }
        )
        _warn(self, props, error.user_text(), error.code)
        return {"FINISHED"}


class _SelfTestBase(bpy.types.Operator):
    """Shared implementation of the profile self-test operators."""

    bl_options = {"REGISTER"}
    profile = "preview"

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None:
            return {"CANCELLED"}
        if props is None:
            return {"CANCELLED"}

        original = props.capture_profile
        try:
            props.capture_profile = self.profile
            job = job_schema.build_job(
                props,
                prefs,
                blend_path=_blend_path(),
                mode=job_schema.MODE_SELF_TEST,
                resolve_path=bpy.path.abspath,
            )
        except errors.MocapError as exc:
            props.capture_profile = original
            return _fail(self, props, exc)
        finally:
            props.capture_profile = original

        try:
            effective = model_manifest.check_profile_requirements(self.profile, prefs.resolved_models_root()).effective_profile
            job_path = job_schema.write_job(job)
            result_path, events = worker_client.run_blocking(
                prefs.resolved_worker_python(effective),
                job_path,
                output_dir=job["output"]["dir"],
                on_event=lambda event: _log_event(props, event),
                timeout=600.0,
            )
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        _info(self, props, ui_text.MSG_SELF_TEST_OK.format(result_path or self.profile))
        return {"FINISHED"}


class MOCAP_OT_test_preview_model(_SelfTestBase):
    """Load the MediaPipe models to validate the preview pipeline."""

    bl_idname = "mocap.test_preview_model"
    bl_label = ui_text.OP_TEST_PREVIEW
    bl_description = ui_text.OP_TEST_PREVIEW_DESC
    profile = "preview"


class MOCAP_OT_test_quality_model(_SelfTestBase):
    """Load RTMDet / RTMPose / MotionBERT to validate the quality pipeline."""

    bl_idname = "mocap.test_quality_model"
    bl_label = ui_text.OP_TEST_QUALITY
    bl_description = ui_text.OP_TEST_QUALITY_DESC
    profile = "quality"


# --------------------------------------------------------------------------------------
# Capture operators
# --------------------------------------------------------------------------------------


def _log_event(props, event) -> None:
    """Mirror a worker progress event into the scene log and progress fields."""
    if props is None:
        return
    effective = event.fields.get("profile")
    if effective in model_manifest.CAPTURE_PROFILES:
        props.effective_profile = effective
    if event.event == progress_mod.EVENT_PROCESSING_FRAME:
        fraction = event.progress
        if fraction is not None:
            props.progress = fraction
        props.progress_text = event.message
        return
    if event.event == progress_mod.EVENT_LOADING_MODEL:
        fraction = event.progress
        if fraction is not None:
            props.progress = fraction
        props.progress_text = event.message
        props.log("INFO", event.message, event.code)
        return
    props.log(event.level, event.message, event.code, event.frame)


class MOCAP_OT_run_capture(bpy.types.Operator):
    """Start the external worker and follow its progress without blocking the UI."""

    bl_idname = "mocap.run_capture"
    bl_label = ui_text.OP_RUN_CAPTURE
    bl_description = ui_text.OP_RUN_CAPTURE_DESC
    bl_options = {"REGISTER"}

    mock: BoolProperty(
        name="Mock",
        description=ui_text.OP_RUN_MOCK_DESC,
        default=False,
        options={"SKIP_SAVE"},
    )

    _timer = None

    # -- lifecycle -----------------------------------------------------------------------

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        props = properties.get_props(context)
        prefs = _require_prefs(self, props)
        if prefs is None or props is None:
            return {"CANCELLED"}

        if _active_application is not None or (_active_job.get("worker") is not None and _active_job["worker"].is_running()):
            return _fail(self, props, errors.MocapError(
                errors.JOB_SCHEMA_INVALID, "已有捕捉任务在运行，请先取消。"
            ))

        if not self.mock:
            report = get_preflight(props.capture_profile, prefs.resolved_models_root(), refresh=True)
            props.preflight_done = True
            props.preflight_ok = report.ok
            props.effective_profile = report.effective_profile
            if not report.ok:
                error = model_manifest.first_missing_error(report) or errors.MocapError(
                    errors.MODEL_MISSING, ui_text.MSG_ENV_BLOCKED
                )
                return _fail(self, props, error)

        try:
            job = job_schema.build_job(
                props, prefs, blend_path=_blend_path(), resolve_path=bpy.path.abspath
            )
            job["review_settings"] = preview.settings_snapshot(props)
            job["source_identity"] = preview.fingerprint(job['input']['path'])
            job_path = job_schema.write_job(job)
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        props.reset_job_state()
        review.begin_capture(context.scene)
        props.clear_log()
        props.last_result_path = ""
        props.last_job_dir = job["output"]["dir"]
        props.set_status("running")

        extra_args = ["--mock"] if self.mock else []
        try:
            worker = worker_client.run_worker(
                prefs.resolved_worker_python(props.capture_profile if self.mock else report.effective_profile),
                job_path,
                output_dir=job["output"]["dir"],
                extra_args=extra_args,
            )
        except errors.MocapError as exc:
            props.set_status("failed")
            return _fail(self, props, exc)

        _active_job["worker"] = worker
        _active_job["job"] = job
        _active_job["scene"] = context.scene
        _active_job["operator"] = self
        self._scene = context.scene
        _info(self, props, ui_text.MSG_CAPTURE_STARTED.format(job["job_id"]))

        window = context.window
        if window is None:  # background mode: fall back to a blocking run
            return self._run_blocking(context, props, worker)

        self._timer = context.window_manager.event_timer_add(TIMER_INTERVAL, window=window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _run_blocking(self, context, props, worker) -> set:
        """Drain the worker synchronously (used when there is no window)."""
        import time

        try:
            while True:
                self._drain(props, worker)
                if worker.drain_finished():
                    break
                worker.tick_cancel()
                time.sleep(0.02)
            worker.wait(timeout=5.0)
            self._drain(props, worker)
            return self._finish(context, props, worker)
        finally:
            pass

    def modal(self, context, event):
        try:
            props = self._scene.mocap_props
        except ReferenceError:
            clear_caches()
            self._remove_timer(context)
            return {"CANCELLED"}
        worker = _active_job.get("worker")
        if worker is None:
            self._remove_timer(context)
            return {"CANCELLED"}

        if event.type == "ESC" and context.scene == self._scene:
            worker.cancel()
            if props is not None:
                props.log("WARNING", ui_text.MSG_CAPTURE_CANCELLED)

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        self._drain(props, worker)
        worker.tick_cancel()
        _tag_redraw()

        if worker.drain_finished():
            worker.wait(timeout=1.0)
            self._drain(props, worker)
            self._remove_timer(context)
            return self._finish(context, props, worker)
        return {"RUNNING_MODAL"}

    # -- internals -----------------------------------------------------------------------

    def _drain(self, props, worker) -> None:
        for event in worker.poll_events():
            _log_event(props, event)
            if event.event == progress_mod.EVENT_COMPLETED:
                if props is not None:
                    props.last_result_path = event.result_path
                    props.progress = 1.0
            elif event.event == progress_mod.EVENT_FAILED:
                if props is not None:
                    error = event.error
                    props.last_error = error.user_text() if error else event.message

    def _finish(self, context, props, worker) -> set:
        returncode = worker.returncode
        result_path = props.last_result_path if props is not None else ""
        cancelled = worker.cancel_requested or returncode == worker_client.EXIT_CANCELLED
        error_text = props.last_error if props is not None else ""

        _active_job["worker"] = None
        _active_job["job"] = None
        _active_job["scene"] = None
        _active_job["operator"] = None
        try:
            worker.close()
        except Exception:  # pragma: no cover
            pass
        _tag_redraw()

        if props is None:
            return {"FINISHED"}
        if cancelled:
            props.set_status("cancelled")
            self.report({"WARNING"}, ui_text.MSG_CAPTURE_CANCELLED)
            return {"CANCELLED"}
        if error_text:
            props.set_status("failed")
            self.report({"ERROR"}, error_text)
            return {"CANCELLED"}
        if returncode not in (0, None):
            return _fail(self, props, worker.exit_error())
        if not result_path:
            return _fail(self, props, errors.MocapError(
                errors.WORKER_EXIT_ERROR, "worker 结束但没有返回结果路径。"
            ))
        props.set_status("completed")
        try:
            review.load_result(props.id_data, result_path)
        except errors.MocapError as exc:
            props.set_status("failed")
            return _fail(self, props, exc)
        _info(self, props, ui_text.MSG_CAPTURE_DONE.format(result_path))
        return {"FINISHED"}

    def _remove_timer(self, context) -> None:
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except (RuntimeError, ValueError):  # pragma: no cover
                pass
            self._timer = None


class MOCAP_OT_cancel_capture(bpy.types.Operator):
    """Ask the running worker to stop."""

    bl_idname = "mocap.cancel_capture"
    bl_label = ui_text.OP_CANCEL_CAPTURE
    bl_description = ui_text.OP_CANCEL_CAPTURE_DESC
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = properties.get_props(context)
        worker = _active_job.get("worker")
        if worker is None or not worker.is_running():
            _warn(self, props, ui_text.MSG_NO_RUNNING_JOB)
            return {"CANCELLED"}
        worker.cancel()
        if props is not None:
            props.log("WARNING", ui_text.MSG_CAPTURE_CANCELLED)
        _info(self, props, ui_text.MSG_CAPTURE_CANCELLED)
        return {"FINISHED"}


# --------------------------------------------------------------------------------------
# Result and Rigify operators
# --------------------------------------------------------------------------------------


class MOCAP_OT_import_result(bpy.types.Operator):
    """Load and validate ``mocap_result.json``."""

    bl_idname = "mocap.import_result"
    bl_label = ui_text.OP_IMPORT_RESULT
    bl_description = ui_text.OP_IMPORT_RESULT_DESC
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH", default="", options={"SKIP_SAVE"})
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = properties.get_props(context)
        if props is None:
            return {"CANCELLED"}
        if props.job_status in ('running', 'applying'):
            self.report({'WARNING'}, '请等待当前任务完成后再加载结果。')
            return {'CANCELLED'}
        path = self.filepath or props.last_result_path
        if not path:
            return _fail(self, props, errors.MocapError(
                errors.RESULT_SCHEMA_INVALID, ui_text.MSG_NO_RESULT
            ))
        try:
            result = review.load_result(context.scene, path, restore_settings=True)
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        props.last_result_path = result.path
        props.result_frame_start = result.frame_start
        props.result_frame_end = result.frame_end
        for warning in result.warnings[: ui_text.LOG_LIMIT]:
            message = warning.get("message") if isinstance(warning, dict) else str(warning)
            code = warning.get("code", "") if isinstance(warning, dict) else ""
            props.log("WARNING", message, code)
        _info(
            self,
            props,
            ui_text.MSG_RESULT_IMPORTED.format(len(result.frames), result.fps, result.profile),
        )
        _tag_redraw()
        return {"FINISHED"}


class MOCAP_OT_calibrate_pitch(bpy.types.Operator):
    """Estimate a constant orientation correction for an upright capture."""

    bl_idname = "mocap.calibrate_pitch"
    bl_label = ui_text.OP_CALIBRATE_PITCH
    bl_description = ui_text.OP_CALIBRATE_PITCH_DESC
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import math
        from ..core import pose_calibration

        props = properties.get_props(context)
        if props is None:
            return {"CANCELLED"}
        result = loaded_result(context)
        if result is None:
            return _fail(self, props, errors.MocapError(
                errors.RESULT_SCHEMA_INVALID, "请先点击“导入结果”。"))
        try:
            props.pitch_correction = pose_calibration.estimate_standing_pitch(result)
        except errors.MocapError as exc:
            return _fail(self, props, exc)
        _info(self, props, "已设置 X 轴倾斜校正 {0:.1f}°，请在对照预览核对后确认应用。".format(
            math.degrees(props.pitch_correction)))
        _tag_redraw()
        return {"FINISHED"}


class MOCAP_OT_apply_to_rigify(bpy.types.Operator):
    """Write the imported motion onto the Rigify control bones."""

    bl_idname = "mocap.apply_to_rigify"
    bl_label = ui_text.OP_APPLY_RIGIFY
    bl_description = ui_text.OP_APPLY_RIGIFY_DESC
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _steps = None

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        global _active_application
        from ..blender import rigify_adapter

        props = properties.get_props(context)
        reason = review.apply_block_reason(context.scene)
        if _active_application is not None:
            reason = "已有应用操作在运行。"
        if reason:
            return _fail(self, props, errors.MocapError(errors.RESULT_SCHEMA_INVALID, reason))
        value = review.session(context.scene)
        state = value.state
        if value.view:
            value.view.finish()
        self._scene = context.scene
        self._view_layer = context.view_layer
        self._window = context.window
        self._wm = context.window_manager
        self._wm.progress_begin(0, 100)
        self._rig = props.target_armature
        self._options = rigify_adapter.RetargetOptions(
            root_motion=props.root_motion, include_hands=props.include_hands,
            switch_limbs_to_fk=props.switch_limbs_to_fk, flip_x=props.flip_x,
            frame_step=props.frame_step, pitch_correction=props.pitch_correction,
            scene_fps=context.scene.render.fps / context.scene.render.fps_base,
            start_time=value.rows[0]["time"] if value.rows else state.result.frames[0].time,
        )
        self._steps = rigify_adapter.retarget_steps(state.result, self._rig, self._options)
        props.set_status("applying")
        props.progress = 0
        props.last_error = ""
        _active_application = self
        if bpy.app.background or context.window is None:
            return self.advance(blocking=True)
        self._timer = self._wm.event_timer_add(0.015, window=self._window)
        self._wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def advance(self, blocking=False):
        import math
        import time
        from ..blender import action_baker
        props = self._scene.mocap_props
        deadline = time.monotonic() + .035
        try:
            with bpy.context.temp_override(scene=self._scene, view_layer=self._view_layer):
                while True:
                    done, total = next(self._steps)
                    props.progress = done / max(1, total)
                    props.progress_text = "应用动作 {0}/{1} · Esc 取消".format(done, total)
                    self._wm.progress_update(int(props.progress * 100))
                    if not blocking and time.monotonic() >= deadline:
                        break
        except StopIteration as finished:
            self._steps = None
            action = finished.value
            props.applied_action_name = action.name
            props.result_frame_start = 1
            props.result_frame_end = math.ceil(action["mocap_frame_end"])
            props.progress = 1
            summary = action_baker.action_summary(action)
            for warning in summary.get("warnings") or []:
                props.log("WARNING", str(warning))
            self.finish()
            _info(self, props, ui_text.MSG_APPLIED.format(summary["name"], summary["fcurves"], summary["keyframes"]))
            return {'FINISHED'}
        except Exception as exc:
            self.abort()
            return _fail(self, props, exc if isinstance(exc, errors.MocapError) else errors.wrap_unexpected(exc, "retarget"))
        _tag_redraw()
        return {'RUNNING_MODAL'}

    def finish(self):
        global _active_application
        if self._timer is not None:
            self._wm.event_timer_remove(self._timer)
            self._timer = None
        self._wm.progress_end()
        self._scene.mocap_props.set_status("completed")
        _active_application = None
        _tag_redraw()

    def abort(self):
        if self._steps is not None:
            with bpy.context.temp_override(scene=self._scene, view_layer=self._view_layer):
                self._steps.close()
            self._steps = None
        self.finish()

    def cancel(self, context):
        self.abort()

    def modal(self, context, event):
        if event.type == 'ESC' or context.scene != self._scene:
            self.abort()
            self.report({'INFO'}, "已取消应用，原 Action 和姿态已恢复。")
            return {'CANCELLED'}
        if event.type == 'TIMER':
            return self.advance()
        return {'RUNNING_MODAL'}


class MOCAP_OT_bake_action(bpy.types.Operator):
    """Bake visual transforms into an editable Action and clean up."""

    bl_idname = "mocap.bake_action"
    bl_label = ui_text.OP_BAKE_ACTION
    bl_description = ui_text.OP_BAKE_ACTION_DESC
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..blender import action_baker

        props = properties.get_props(context)
        if props is None:
            return {"CANCELLED"}
        armature = props.target_armature
        if armature is None:
            return _fail(self, props, errors.MocapError(
                errors.RIGIFY_NOT_FOUND, ui_text.MSG_SELECT_ARMATURE
            ))

        start = props.result_frame_start or context.scene.frame_start
        end = props.result_frame_end or context.scene.frame_end
        try:
            action = action_baker.bake_action(
                armature, start, end, clean_curves=props.clean_curves
            )
        except errors.MocapError as exc:
            return _fail(self, props, exc)
        except Exception as exc:  # noqa: BLE001
            return _fail(self, props, errors.wrap_unexpected(exc, "bake"))

        summary = action_baker.action_summary(action)
        props.applied_action_name = action.name
        _info(
            self,
            props,
            ui_text.MSG_BAKED.format(summary["name"], summary["fcurves"], summary["keyframes"]),
        )
        _tag_redraw()
        return {"FINISHED"}


class MOCAP_OT_clear_temp_data(bpy.types.Operator):
    """Delete the temporary objects and constraints this add-on created."""

    bl_idname = "mocap.clear_temp_data"
    bl_label = ui_text.OP_CLEAR_TEMP
    bl_description = ui_text.OP_CLEAR_TEMP_DESC
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..blender import temp_data

        props = properties.get_props(context)
        report = temp_data.cleanup_all()
        _info(
            self,
            props,
            ui_text.MSG_TEMP_CLEARED.format(report["objects"], report["constraints"]),
        )
        return {"FINISHED"}


classes = (
    MOCAP_OT_preflight,
    MOCAP_OT_copy_missing_model_links,
    MOCAP_OT_open_models_dir,
    MOCAP_OT_test_worker_python,
    MOCAP_OT_test_cuda,
    MOCAP_OT_test_preview_model,
    MOCAP_OT_test_quality_model,
    MOCAP_OT_run_capture,
    MOCAP_OT_cancel_capture,
    MOCAP_OT_import_result,
    MOCAP_OT_calibrate_pitch,
    MOCAP_OT_apply_to_rigify,
    MOCAP_OT_bake_action,
    MOCAP_OT_clear_temp_data,
)
