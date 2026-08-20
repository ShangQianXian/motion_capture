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
from ..core import result_schema, worker_client
from . import preferences, properties, ui_text

#: Modal timer interval in seconds.
TIMER_INTERVAL = 0.15

#: Cache of the most recent preflight report, keyed by ``(profile, models_root)``.
_preflight_cache = {}

#: The running worker, if any. Only one capture at a time in v0.1.
_active_job = {"worker": None, "job": None}

#: Result loaded by ``mocap.import_result``, consumed by ``mocap.apply_to_rigify``.
_loaded_result = {"result": None, "path": ""}


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
    _loaded_result["result"] = None
    _loaded_result["path"] = ""
    worker = _active_job.get("worker")
    if worker is not None:
        try:
            worker.close()
        except Exception:  # pragma: no cover - defensive teardown
            pass
    _active_job["worker"] = None
    _active_job["job"] = None


def active_worker():
    """The running worker, or ``None``."""
    return _active_job.get("worker")


def loaded_result():
    """The imported :class:`result_schema.MocapResult`, or ``None``."""
    return _loaded_result.get("result")


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
            report = worker_client.check_env(prefs.resolved_worker_python())
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        payload = report.get("report") or {}
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
            job_path = job_schema.write_job(job)
            result_path, events = worker_client.run_blocking(
                prefs.resolved_worker_python(),
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

        if _active_job.get("worker") is not None and _active_job["worker"].is_running():
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
            job_path = job_schema.write_job(job)
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        props.reset_job_state()
        props.clear_log()
        props.last_result_path = ""
        props.last_job_dir = job["output"]["dir"]
        props.set_status("running")

        extra_args = ["--mock"] if self.mock else []
        try:
            worker = worker_client.run_worker(
                prefs.resolved_worker_python(),
                job_path,
                output_dir=job["output"]["dir"],
                extra_args=extra_args,
            )
        except errors.MocapError as exc:
            props.set_status("failed")
            return _fail(self, props, exc)

        _active_job["worker"] = worker
        _active_job["job"] = job
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
        props = properties.get_props(context)
        worker = _active_job.get("worker")
        if worker is None:
            self._remove_timer(context)
            return {"CANCELLED"}

        if event.type == "ESC":
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

    def execute(self, context):
        props = properties.get_props(context)
        if props is None:
            return {"CANCELLED"}
        path = self.filepath or props.last_result_path
        if not path:
            return _fail(self, props, errors.MocapError(
                errors.RESULT_SCHEMA_INVALID, ui_text.MSG_NO_RESULT
            ))
        try:
            result = result_schema.load_mocap_result(bpy.path.abspath(path))
        except errors.MocapError as exc:
            return _fail(self, props, exc)

        _loaded_result["result"] = result
        _loaded_result["path"] = result.path
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


class MOCAP_OT_apply_to_rigify(bpy.types.Operator):
    """Write the imported motion onto the Rigify control bones."""

    bl_idname = "mocap.apply_to_rigify"
    bl_label = ui_text.OP_APPLY_RIGIFY
    bl_description = ui_text.OP_APPLY_RIGIFY_DESC
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..blender import action_baker, rigify_adapter

        props = properties.get_props(context)
        if props is None:
            return {"CANCELLED"}
        result = _loaded_result.get("result")
        if result is None:
            return _fail(self, props, errors.MocapError(
                errors.RESULT_SCHEMA_INVALID, "请先点击“导入结果”。"
            ))
        armature = props.target_armature
        if armature is None:
            return _fail(self, props, errors.MocapError(
                errors.RIGIFY_NOT_FOUND, ui_text.MSG_SELECT_ARMATURE
            ))

        window_manager = context.window_manager
        window_manager.progress_begin(0, 100)

        def report_progress(done, total):
            window_manager.progress_update(int(100.0 * done / max(1, total)))
            props.progress = float(done) / max(1, total)

        options = rigify_adapter.RetargetOptions(
            root_motion=props.root_motion,
            include_hands=props.include_hands,
            switch_limbs_to_fk=props.switch_limbs_to_fk,
            flip_x=props.flip_x,
            frame_step=props.frame_step,
            progress=report_progress,
        )
        try:
            action = rigify_adapter.retarget_to_rigify(result, armature, options)
        except errors.MocapError as exc:
            return _fail(self, props, exc)
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures cleanly
            return _fail(self, props, errors.wrap_unexpected(exc, "retarget"))
        finally:
            window_manager.progress_end()

        summary = action_baker.action_summary(action)
        props.applied_action_name = action.name
        for warning in summary.get("warnings") or []:
            props.log("WARNING", str(warning))
        _info(
            self,
            props,
            ui_text.MSG_APPLIED.format(summary["name"], summary["fcurves"], summary["keyframes"]),
        )
        _tag_redraw()
        return {"FINISHED"}


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
    MOCAP_OT_apply_to_rigify,
    MOCAP_OT_bake_action,
    MOCAP_OT_clear_temp_data,
)
