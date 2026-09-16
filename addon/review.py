"""Per-scene review sessions and asynchronous, bounded Blender image ownership."""

import math
import os
import time
from collections import OrderedDict

import bpy
import bpy.utils.previews
from bpy.app.handlers import persistent

from ..core import errors, job_schema, paths, preview, result_schema, worker_client
from . import preferences

_sessions = {}
_closing = False


def scene_key(scene):
    return scene.as_pointer()


def session(scene=None, create=True):
    scene = scene or bpy.context.scene
    key = scene_key(scene)
    value = _sessions.get(key)
    if value is None and create:
        value = Session(scene)
        _sessions[key] = value
    return value


class Session:
    def __init__(self, scene):
        self.scene = scene
        self.state = None
        self.media = None
        self.media_key = None
        self.info = {}
        self.error = ""
        self.images = OrderedDict()
        self.image_bytes = 0
        self.image = None
        self.icons = None
        self.icon_id = 0
        self.request_id = 0
        self.pending_id = None
        self.pending_sample = 0
        self.pending_source = -1
        self.displayed_sample = -1
        self.displayed_source = -1
        self.last_clock = time.monotonic()
        self.rows = []
        self.pose_frames = {}
        self.raw_result = None
        self.pose_bounds = ((0, 0, .9), 1.5, 1.8)
        self.view = None
        self.suspend = False
        self.auto_loaded = ""
        self.media_started = 0.0
        self.settings_key = None

    @property
    def props(self):
        return self.scene.mocap_props

    def source_path(self):
        return paths.normalize(bpy.path.abspath(self.props.source_media)) if self.props.source_media else ""

    def invalidate(self):
        if self.suspend:
            return
        if self.state:
            reason = self.state.mismatch_reason(preview.settings_snapshot(self.props), self.source_path())
            if reason:
                self.state.invalidate(reason)
                self.props.preview_playing = False

    def stop_media(self):
        if self.media:
            self.media.send({"command": "close", "request_id": self.request_id + 1})
            self.media.close()
            self.media = None
        self.pending_id = None
        self.info = {}
        self.image = None
        self.displayed_sample = -1
        self.displayed_source = -1
        for image in list(self.images.values()):
            try:
                bpy.data.images.remove(image)
            except (ReferenceError, RuntimeError):
                pass
        self.images.clear()
        self.image_bytes = 0
        if self.icons is not None:
            bpy.utils.previews.remove(self.icons)
            self.icons = None
        self.icon_id = 0

    def close(self):
        if self.view:
            self.view.finish()
        self.stop_media()

    def ensure_media(self):
        prefs = preferences.get_preferences()
        source = self.source_path()
        python = prefs.resolved_worker_python("preview") if prefs else ""
        identity = preview.fingerprint(source)
        key = (source, self.props.source_type, python, identity["size"], identity["mtime_ns"])
        if key == self.media_key:
            return
        self.stop_media()
        self.media_key = key
        self.error = ""
        if not source:
            return
        if self.state and not self.state.media_matches():
            self.state.viewed = False
        try:
            if identity["size"] < 0:
                raise errors.MocapError(errors.MEDIA_NOT_FOUND, "素材不存在，请重新定位文件。")
            python = job_schema.validate_worker_python(python)
            identifier = job_schema.new_job_id()
            folder = paths.job_dir(identifier, bpy.data.filepath)
            cache_root = os.path.join(os.path.dirname(folder), "media_cache")
            job = {"version": "0.2", "mode": job_schema.MODE_MEDIA_PREVIEW, "job_id": identifier,
                   "input": {"path": source, "type": self.props.source_type},
                   "output": {"dir": folder, "cache_dir": cache_root}}
            job_path = job_schema.write_job(job)
            self.media = worker_client.WorkerProcess(python, job_path, folder, duplex=True)
            self.media_started = time.monotonic()
        except (errors.MocapError, OSError) as exc:
            self.error = exc.user_text() if isinstance(exc, errors.MocapError) else str(exc)

    def retry_media(self):
        self.media_key = None
        self.ensure_media()

    def refresh_rows(self):
        if self.state:
            if self.state.manifest:
                self.rows = self.state.manifest["frames"]
            elif self.info:
                self.rows = preview.legacy_rows(self.state.result, self.info["fps"])
        else:
            self.rows = []

    def total(self):
        return len(self.rows) if self.rows else self.info.get("total_frames", 1)

    def row(self, sample):
        if self.rows:
            return self.rows[max(0, min(sample, len(self.rows) - 1))]
        return {"source_index": sample, "time": sample / max(1.0, self.info.get("fps", 30)),
                "result_frame": None, "body2d": [], "status": "source"}

    def seek(self, sample):
        if not self.media or not self.info:
            return
        sample = max(0, min(int(sample), self.total() - 1))
        row = self.row(sample)
        source_index = row["source_index"]
        if self.pending_id is not None and sample == self.pending_sample and source_index == self.pending_source:
            return
        if self.pending_id is None and sample == self.displayed_sample and source_index == self.displayed_source:
            return
        self.request_id += 1
        self.pending_id = self.request_id
        self.pending_sample = sample
        self.pending_source = source_index
        if source_index in self.images:
            self.image = self.images.pop(source_index)
            self.images[source_index] = self.image
            self.displayed_sample, self.displayed_source = sample, source_index
            self.pending_id = None
            if not self.props.preview_playing:
                self.last_clock = time.monotonic()
        elif not self.media.send({"command": "frame", "request_id": self.request_id, "source_index": source_index}):
            self.pending_id = None
            self.error = "媒体预览进程已退出，请重新加载。"

    def accept_frame(self, fields):
        # Both the request and exact source frame must match; stale replies cannot advance either pane.
        if fields.get("request_id") != self.pending_id or fields.get("source_index") != self.pending_source:
            return False
        try:
            image = bpy.data.images.load(fields["path"], check_existing=False)
            image.colorspace_settings.name = 'Non-Color'
            image.name = "Mocap Preview {0}".format(self.pending_id)
            size = image.size[0] * image.size[1] * 32  # CPU float pixels + GPU + conservative overhead
            if size > preview.IMAGE_MEMORY_LIMIT:
                bpy.data.images.remove(image)
                raise ValueError("预览图像超过内存限制。")
            while self.images and self.image_bytes + size > preview.IMAGE_MEMORY_LIMIT:
                _, old = self.images.popitem(last=False)
                self.image_bytes -= old.size[0] * old.size[1] * 32
                bpy.data.images.remove(old)
            old = self.images.pop(self.pending_source, None)
            if old:
                self.image_bytes -= old.size[0] * old.size[1] * 32
                bpy.data.images.remove(old)
            self.images[self.pending_source] = image
            self.image_bytes += size
            self.image = image
            self.displayed_sample, self.displayed_source = self.pending_sample, self.pending_source
            self.pending_id = None
            self.error = ""
            if not self.props.preview_playing:
                self.last_clock = time.monotonic()
            if self.icons is None:
                self.icons = bpy.utils.previews.new()
                self.icon_id = self.icons.load("source", fields["path"], "IMAGE").icon_id
            return True
        except (RuntimeError, OSError, ValueError) as exc:
            self.error = "无法显示预览：{0}".format(exc)
            self.pending_id = None
            return False

    def poll(self):
        self.ensure_media()
        if self.media:
            for event in self.media.poll_events():
                fields = event.fields
                if event.event == "media_info":
                    self.info = fields["info"]
                    self.refresh_rows()
                    self.seek(max(0, self.props.preview_frame - 1))
                elif event.event == "media_frame":
                    self.accept_frame(fields)
                elif event.event in ("failed", "media_error"):
                    if event.event == "failed" or fields.get("request_id") == self.pending_id:
                        self.error = event.error.user_text() if event.error else event.message
                        self.pending_id = None
                        self.props.preview_playing = False
                elif event.event == "completed":
                    self.pending_id = None
            if self.media.drain_finished() and not self.error:
                self.error = "媒体预览进程已退出，请重新加载。"
                self.pending_id = None
            if not self.info and not self.error and time.monotonic() - self.media_started > 20:
                self.error = "媒体读取超时，请重新加载或更换素材。"
                self.stop_media()
            if not self.view and self.image is not None and self.pending_id is None and not self.error:
                # A sidebar thumbnail needs no persistent decoder or playback process.
                self.media.send({"command": "close", "request_id": self.request_id + 1})
                self.media.close()
                self.media = None
        if self.state:
            self.invalidate()
            result = self.state.corrected(self.props.pitch_correction, self.props.flip_x)
            key = (self.state.correction, self.props.preview_stage)
            if self.settings_key != key or not self.pose_frames:
                if self.props.preview_stage == 'raw' and self.raw_result:
                    result = preview.corrected_result(self.raw_result, self.props.pitch_correction, self.props.flip_x)
                self.pose_frames = {frame.frame: frame for frame in result.frames}
                first = result.frames[0].body3d['pelvis']
                low, high = [float('inf')] * 3, [float('-inf')] * 3
                for frame in result.frames:
                    pelvis = frame.body3d['pelvis']
                    for point in frame.body3d.values():
                        for axis in range(3):
                            offset = pelvis[axis] - first[axis] if axis < 2 and self.props.root_motion == 'in_place' else 0
                            coordinate = point[axis] - offset
                            low[axis], high[axis] = min(low[axis], coordinate), max(high[axis], coordinate)
                self.pose_bounds = (tuple((a + b) / 2 for a, b in zip(low, high)),
                                    max(.2, math.hypot(high[0] - low[0], high[1] - low[1])),
                                    max(.2, high[2] - low[2]))
                self.settings_key = key
        if self.view and self.props.preview_playing and self.pending_id is None and self.displayed_sample >= 0:
            index = self.displayed_sample
            row = self.row(index)
            next_index = index + 1
            delta = (self.row(next_index)["time"] - row["time"]) if next_index < self.total() else 1 / self.info.get("fps", 30)
            if time.monotonic() - self.last_clock >= max(0.001, delta) / float(self.props.preview_speed):
                if next_index >= self.total():
                    if self.props.preview_loop:
                        next_index = 0
                    else:
                        self.props.preview_playing = False
                        return
                self.props.preview_frame = next_index + 1
                self.last_clock = time.monotonic()
                self.seek(next_index)


def capture_changed(props, context):
    scene = props.id_data
    value = session(scene, create=False)
    if value:
        value.invalidate()


def correction_changed(props, context):
    value = session(props.id_data, create=False)
    if value and value.state:
        value.state.viewed = False


def frame_changed(props, context):
    value = session(props.id_data, create=False)
    if value:
        value.seek(props.preview_frame - 1)


def load_result(scene, path, restore_settings=False):
    result = result_schema.load_mocap_result(bpy.path.abspath(path))
    manifest = preview.load_manifest(result)
    raw_result = None
    if manifest and manifest.get('stages', {}).get('raw'):
        import hashlib
        stage = manifest['stages']['raw']
        filename = stage['file']
        if filename != os.path.basename(filename) or filename != 'mocap_raw.json':
            raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, '原始三维文件引用无效。')
        stage_path = os.path.join(os.path.dirname(result.path), filename)
        try:
            with open(stage_path, 'rb') as handle:
                if hashlib.sha256(handle.read()).hexdigest() != stage['sha256']:
                    raise ValueError('checksum')
            raw_result = result_schema.load_mocap_result(stage_path)
        except (OSError, ValueError) as exc:
            raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, '原始三维阶段文件丢失或损坏。') from exc
    value = session(scene)
    props = scene.mocap_props
    if value.view:
        value.view.finish()
    value.suspend = True
    try:
        if restore_settings:
            props.motion_type = 'general'
            props.camera_view = 'unspecified'
            props.align_initial_facing = False
            for name, setting in (manifest or {}).get("capture_settings", {}).items():
                if name in preview.CAPTURE_FIELDS:
                    setattr(props, name, setting)
            props.source_media = result.source_path
        elif not props.source_media:
            props.source_media = result.source_path
        props.pitch_correction = 0.0
        props.preview_frame = 1
        props.preview_playing = False
        props.preview_stage = 'processed'
    finally:
        value.suspend = False
    settings = (manifest or {}).get("capture_settings") or preview.settings_snapshot(props)
    value.state = preview.ReviewState(result, manifest, settings, value.source_path())
    value.raw_result = raw_result
    if paths.normalize(result.source_path) != value.source_path():
        if preview.source_matches(value.state.source, value.source_path(), relocated=True):
            value.state.relocated = True
        else:
            value.state.invalidate()
    value.rows = manifest["frames"] if manifest else []
    value.pose_frames = {}
    value.settings_key = None
    value.auto_loaded = result.path
    value.displayed_sample = -1
    value.pending_id = None
    props.last_result_path = result.path
    props.result_frame_start = result.frame_start
    props.result_frame_end = result.frame_end
    if value.info:
        value.refresh_rows()
        value.seek(0)
    return result


def begin_capture(scene):
    value = session(scene)
    if value.view:
        value.view.finish()
    value.state = None
    value.raw_result = None
    value.rows = []
    value.pose_frames = {}
    value.auto_loaded = ""
    value.props.applied_action_name = ""
    value.props.preview_playing = False


def apply_block_reason(scene):
    value = session(scene, create=False)
    state = value.state if value else None
    props = scene.mocap_props
    if props.job_status in ("running", "applying"):
        return "请等待当前任务完成。"
    if state is None:
        return "请先生成或加载动捕结果。"
    if props.preview_stage != 'processed':
        return '请切换到处理后结果，核对最终动作再应用。'
    reason = state.mismatch_reason(preview.settings_snapshot(props), value.source_path())
    if reason:
        return reason
    if not state.media_matches():
        return "素材丢失或已改变，请重新定位匹配的文件。"
    if state.correction != (float(props.pitch_correction), bool(props.flip_x)) or not state.viewed:
        return "请打开对照预览，核对当前动作后再应用。"
    if props.target_armature is None:
        return "请选择目标 Rigify 骨架。"
    from ..blender import rigify_adapter
    detection = rigify_adapter.detect_rigify_human(props.target_armature)
    if not detection.ok:
        return detection.error.message if detection.error else "请选择有效的 Rigify Human 骨架。"
    if props.target_armature.name not in scene.objects:
        return "目标骨架不属于当前场景。"
    return ""


def _tick():
    if _closing:
        return None
    active = set()
    for window in bpy.context.window_manager.windows:
        scene = window.scene
        if not hasattr(scene, "mocap_props"):
            continue
        active.add(scene_key(scene))
        props = scene.mocap_props
        if not props.source_media and not props.last_result_path:
            continue
        value = session(scene)
        try:
            if props.last_result_path and value.state is None and value.auto_loaded != props.last_result_path and props.job_status != "running":
                value.auto_loaded = props.last_result_path
                load_result(scene, props.last_result_path)
            value.poll()
        except Exception as exc:
            value.error = "预览异常：{0}".format(exc)
            props.preview_playing = False
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    for key, value in list(_sessions.items()):
        if key not in active and value.media:
            value.close()
            value.media_key = None
    return 0.015 if any(value.view for value in _sessions.values()) else 0.15


@persistent
def _load_pre(_):
    from . import operators
    operators.clear_caches()
    cleanup()


@persistent
def _load_post(_):
    for scene in bpy.data.scenes:
        if hasattr(scene, "mocap_props"):
            scene.mocap_props.preview_playing = False
            if scene.mocap_props.job_status in ("running", "applying"):
                scene.mocap_props.job_status = "idle"
    register_timer()


def register_timer():
    global _closing
    _closing = False
    if not bpy.app.background and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.3)


def register():
    register_timer()
    bpy.app.handlers.load_pre.append(_load_pre)
    bpy.app.handlers.load_post.append(_load_post)
    bpy.app.handlers.undo_pre.append(_load_pre)
    bpy.app.handlers.undo_post.append(_load_post)
    bpy.app.handlers.redo_pre.append(_load_pre)
    bpy.app.handlers.redo_post.append(_load_post)


def cleanup():
    global _closing
    _closing = True
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    for value in list(_sessions.values()):
        value.close()
    _sessions.clear()


def unregister():
    cleanup()
    for handlers, callback in ((bpy.app.handlers.load_pre, _load_pre), (bpy.app.handlers.load_post, _load_post),
                               (bpy.app.handlers.undo_pre, _load_pre), (bpy.app.handlers.undo_post, _load_post),
                               (bpy.app.handlers.redo_pre, _load_pre), (bpy.app.handlers.redo_post, _load_post)):
        if callback in handlers:
            handlers.remove(callback)
