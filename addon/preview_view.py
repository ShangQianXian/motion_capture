"""A scene-independent two-pane review canvas in an existing View3D region."""

import math
import time

import bpy
import blf
import gpu
from bpy.props import StringProperty
from gpu_extras.batch import batch_for_shader

from ..core import preview, paths, retarget_math as rm
from . import review

BG = (0.035, 0.045, 0.065, 1)
PANE = (0.060, 0.075, 0.100, 1)
TEXT = (0.90, 0.93, 0.98, 1)
MUTED = (0.53, 0.61, 0.70, 1)
LEFT = (0.15, 0.78, 0.90, 1)
RIGHT = (0.97, 0.53, 0.25, 1)
LOW = (1.0, 0.25, 0.27, 1)
ACCENT = (0.15, 0.68, 0.49, 1)
_draw_scale = 1.0


def scaled(points):
    return [(point[0] * _draw_scale, point[1] * _draw_scale) for point in points]


def box(rect, color):
    x, y, w, h = rect
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": scaled(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))},
                             indices=((0, 1, 2), (0, 2, 3)))
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def line(a, b, color, width=2):
    # Geometry rather than driver-dependent wide GL lines.
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max(0.001, math.hypot(dx, dy))
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": scaled(((a[0]+nx, a[1]+ny), (a[0]-nx, a[1]-ny),
                                                       (b[0]-nx, b[1]-ny), (b[0]+nx, b[1]+ny)))},
                             indices=((0, 1, 2), (0, 2, 3)))
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def dot(point, color, radius=3):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    vertices = [point] + [(point[0] + math.cos(i * math.tau / 10) * radius,
                          point[1] + math.sin(i * math.tau / 10) * radius) for i in range(11)]
    batch = batch_for_shader(shader, "TRIS", {"pos": scaled(vertices)}, indices=[(0, i, i + 1) for i in range(1, 11)])
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def label(text, x, y, size=14, color=TEXT):
    blf.size(0, size * _draw_scale)
    blf.color(0, *color)
    blf.position(0, x * _draw_scale, y * _draw_scale, 0)
    blf.draw(0, str(text))


def inside(rect, x, y):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


def fit_image(rect, width, height):
    x, y, w, h = rect
    scale = min(w / max(1, width), h / max(1, height))
    nw, nh = width * scale, height * scale
    return x + (w - nw) / 2, y + (h - nh) / 2, nw, nh


def draw_image(image, rect):
    texture = gpu.texture.from_image(image)
    # Media textures use Non-Color so the UI receives the original sRGB values.
    shader = gpu.shader.from_builtin("IMAGE")
    x, y, w, h = rect
    batch = batch_for_shader(shader, "TRI_FAN", {
        "pos": scaled(((x, y), (x + w, y), (x + w, y + h), (x, y + h))),
        "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
    })
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


class MOCAP_OT_open_preview(bpy.types.Operator):
    bl_idname = "mocap.open_preview"
    bl_label = "打开对照预览"
    bl_description = "原素材与动捕结果同步核对；预览不改变场景动画"

    _handler = None
    _timer = None
    _closed = False

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        if context.area is None or context.area.type != "VIEW_3D" or bpy.app.background:
            self.report({'ERROR'}, "请从三维视图的 Mocap 侧栏打开预览。")
            return {'CANCELLED'}
        self.value = review.session(context.scene)
        if self.value.view:
            self.value.view.finish()
        self.window = context.window
        self.area = context.area
        self.region = next(region for region in self.area.regions if region.type == "WINDOW")
        self.scene = context.scene
        self.space = self.area.spaces.active
        self.saved_space = {name: getattr(self.space, name) for name in
                            ('show_region_toolbar', 'show_region_tool_header', 'show_gizmo')}
        for name in self.saved_space:
            setattr(self.space, name, False)
        self.scale = max(1.0, context.preferences.system.ui_scale, context.preferences.system.pixel_size)
        self.value.view = self
        self._closed = False
        self.yaw, self.pitch, self.zoom = 0.0, 0.18, 1.0
        self.drag = None
        self.buttons = []
        self.timeline = (0, 0, 0, 0)
        self.right_rect = (0, 0, 0, 0)
        if self.value.media is None:
            self.value.media_key = None
        self.value.ensure_media()
        self.value.seek(self.scene.mocap_props.preview_frame - 1)
        self._handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback, (), 'WINDOW', 'POST_PIXEL')
        self._timer = context.window_manager.event_timer_add(0.025, window=self.window)
        context.window_manager.modal_handler_add(self)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def finish(self):
        if self._closed:
            return
        self._closed = True
        if self._handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
            self._handler = None
        if self._timer:
            try:
                bpy.context.window_manager.event_timer_remove(self._timer)
            except (ReferenceError, RuntimeError):
                pass
            self._timer = None
        try:
            self.scene.mocap_props.preview_playing = False
            for name, value in self.saved_space.items():
                setattr(self.space, name, value)
            self.value.view = None
            # Closing releases the decoder and image resources; the verified state survives.
            self.value.stop_media()
            self.value.media_key = None
            self.area.tag_redraw()
        except (ReferenceError, RuntimeError):
            pass

    def cancel(self, context):
        self.finish()

    def button(self, text, rect, command, color=PANE):
        box(rect, color)
        label(text, rect[0] + 8, rect[1] + 8, 13)
        self.buttons.append((rect, command))

    def draw_callback(self):
        global _draw_scale
        if self._closed or bpy.context.area != self.area or bpy.context.region != self.region:
            return
        if bpy.context.scene != self.scene:
            return
        blend = gpu.state.blend_get()
        depth = gpu.state.depth_test_get()
        _draw_scale = self.scale
        try:
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('NONE')
            self.draw_canvas()
        except Exception as exc:
            self.value.error = "预览绘制失败：{0}".format(exc)
            if self.value.state:
                self.value.state.viewed = False
            label(self.value.error, 25, 110, color=LOW)
        finally:
            gpu.state.blend_set(blend)
            gpu.state.depth_test_set(depth)

    def canvas_size(self):
        width = self.region.width
        if self.space.show_region_ui:
            ui = next((r for r in self.area.regions if r.type == 'UI' and r.width > 1), None)
            if ui and ui.x >= self.region.x:
                width = min(width, ui.x - self.region.x)
        return width / self.scale, self.region.height / self.scale

    def draw_canvas(self):
        width, height = self.canvas_size()
        box((0, 0, width, height), BG)
        # Blender's overlapping native header occupies the upper 26 UI pixels.
        height -= 28
        if width < 420 or height < 280:
            label("请扩大三维视图区，按 Esc 返回。", 20, max(30, height / 2))
            return
        props, value = self.scene.mocap_props, self.value
        margin, gap = 18, 14
        pane_width = (width - margin * 2 - gap) / 2
        left_rect = (margin, 128, pane_width, max(50, height - 202))
        self.right_rect = (margin + pane_width + gap, 128, pane_width, max(50, height - 202))
        box(left_rect, PANE)
        box(self.right_rect, PANE)
        label("素材与动作核对", margin, height - 28, 18)
        label("原素材 / 二维识别关键点", margin, height - 57, 14)
        label("原始三维（诊断）" if props.preview_stage == 'raw' else "三维待应用动作", self.right_rect[0], height - 57, 14)
        self.buttons = []
        self.button("关闭 · Esc", (width - 108, height - 38, 90, 27), "close")
        row = value.row(value.displayed_sample) if value.displayed_sample >= 0 else None
        if value.image is not None:
            fit = fit_image(left_rect, *value.image.size)
            draw_image(value.image, fit)
            if row and props.preview_overlay:
                self.draw_2d(row, fit)
        else:
            label("正在读取素材…" if not value.error else "素材不可用", left_rect[0] + 15, height / 2, color=MUTED)
        frame = value.pose_frames.get(row.get("result_frame")) if row else None
        if frame:
            self.draw_3d(frame, self.right_rect)
            state = value.state
            if (props.preview_stage == 'processed' and value.image is not None and state and not state.stale and state.media_matches()
                    and state.matches(preview.settings_snapshot(props), value.source_path())
                    and state.correction == (float(props.pitch_correction), bool(props.flip_x))):
                state.viewed = True
        else:
            message = "此帧未检测到可用姿态" if value.state and row else "生成完成后，在这里核对三维动作"
            label(message, self.right_rect[0] + 12, height / 2, 13, MUTED)
        rx = self.right_rect[0]
        label("L 左侧", rx + 12, 137, 12, LEFT)
        label("R 右侧", rx + 78, 137, 12, RIGHT)
        label("中键旋转 / 滚轮缩放", rx + 146, 137, 12, MUTED)
        if row and row.get('estimated_joints'):
            label("短线显示头 / 脚朝向；灰色为估算", rx + 12, 158, 11, MUTED)
        self.button("正面", (rx + 8, height - 107, 52, 26), "front")
        self.button("侧面", (rx + 64, height - 107, 52, 26), "side")
        self.button("复位", (rx + 120, height - 107, 52, 26), "reset")
        if value.raw_result:
            self.button('处理后' if props.preview_stage == 'raw' else '原始', (rx + 176, height - 107, 65, 26), 'stage')
        if row and row.get('contact_states'):
            names = {'contact': '支撑', 'air': '离地', 'unknown': '未知'}
            label('L {0} / R {1}'.format(*(names.get(row['contact_states'].get(s), '未知') for s in ('L', 'R'))), rx, 111, 12, MUTED)
        if value.state and not value.state.manifest:
            label("旧版 / Mock 结果：无二维检测数据", margin, 111, 12, MUTED)
        elif row:
            status = {'missing': '漏检', 'interpolated': '含插值或保持姿态的关节',
                      'low_confidence': '含低置信度关节', 'detected': '已检测'}.get(row.get('status'), '')
            label(status, margin, 111, 12, LOW if preview.is_problem(row) else MUTED)
        if value.state and value.state.stale:
            label("素材或参数已改变，请重新生成", rx, 111, 12, LOW)
        if value.error:
            label(value.error[:100], margin, 94, 12, LOW)
        elif value.pending_id is not None:
            label("缓冲中 · 两侧保持同一帧", margin, 94, 12, MUTED)
        elif row:
            label("源帧 {0}  ·  {1:.3f} 秒".format(row['source_index'] + 1, row['time']), margin, 94, 12, MUTED)
        self.timeline = (margin, 71, width - margin * 2, 9)
        box(self.timeline, PANE)
        total = value.total()
        if total > 1:
            for index, candidate in enumerate(value.rows):
                x = margin + index / max(1, total - 1) * self.timeline[2]
                for lane, side in enumerate(('L', 'R')):
                    contact = candidate.get('contact_states', {}).get(side, 'unknown')
                    colors = {'contact': ACCENT, 'air': LEFT, 'unknown': MUTED}
                    box((x, 62 - lane * 4, max(2, self.timeline[2] / total), 3), colors.get(contact, MUTED))
                if preview.is_problem(candidate):
                    box((x, 71, 2, 9), LOW)
        fraction = max(0, value.displayed_sample) / max(1, total - 1)
        dot((margin + fraction * self.timeline[2], 75), LEFT, 6)
        if value.info.get("type") != "image":
            self.button("暂停" if props.preview_playing else "播放", (margin, 24, 55, 30), "play")
            self.button("上帧", (margin + 60, 24, 49, 30), "previous")
            self.button("下帧", (margin + 114, 24, 49, 30), "next")
            self.button(props.preview_speed + "x", (margin + 168, 24, 48, 30), "speed")
            self.button("循环" if props.preview_loop else "单次", (margin + 221, 24, 48, 30), "loop")
        reason = review.apply_block_reason(self.scene)
        if value.state:
            button_width = 134
            self.button("确认并应用", (width - button_width - margin, 24, button_width, 30),
                        "apply" if not reason else "blocked", ACCENT if not reason else PANE)

    def draw_2d(self, row, fit):
        points = row.get("body2d", [])
        topology = (self.value.state.manifest or {}).get("topology")
        if topology == 'coco_wholebody133':
            points = points[:91]  # Body, feet and face; hand capture remains separate.
        if not points:
            return
        x, y, width, height = fit
        coords = [(x + p[0] * width, y + (1 - p[1]) * height) for p in points]
        left_ids = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19} if topology in ('coco17', 'coco_wholebody133') else {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
        def color(index):
            return LOW if points[index][2] < 0.4 else (LEFT if index in left_ids else RIGHT)
        for a, b in self.value.state.manifest.get("edges", []):
            if a < len(points) and b < len(points) and inside(fit, *coords[a]) and inside(fit, *coords[b]):
                line(coords[a], coords[b], LOW if min(points[a][2], points[b][2]) < .4 else color(b), 2.5)
        for index, point in enumerate(coords):
            if inside(fit, *point):
                dot(point, color(index), 3)
        for hand in row.get('hands2d', []):
            hand_points = [(x + p[0] * width, y + (1 - p[1]) * height) for p in hand['points']]
            hand_color = LEFT if hand.get('side') == 'Left' else RIGHT
            for a, b in preview.HAND_EDGES:
                if a < len(hand_points) and b < len(hand_points) and inside(fit, *hand_points[a]) and inside(fit, *hand_points[b]):
                    line(hand_points[a], hand_points[b], hand_color, 1)

    def draw_3d(self, frame, rect):
        x, y, width, height = rect
        result = self.value.state.transformed
        first = result.frames[0].body3d['pelvis']
        pelvis = frame.body3d['pelvis']
        center, horizontal_span, vertical_span = self.value.pose_bounds
        scale = min((width - 40) / (horizontal_span * 1.15),
                    (height - 80) / (vertical_span * 1.15)) * self.zoom
        yaw, tilt = self.yaw, self.pitch
        def project(point, pose=True):
            px, py, pz = (point[i] - center[i] for i in range(3))
            if pose and self.scene.mocap_props.root_motion == 'in_place':
                px -= pelvis[0] - first[0]
                py -= pelvis[1] - first[1]
            horizontal = math.cos(yaw) * px + math.sin(yaw) * py
            depth = -math.sin(yaw) * px + math.cos(yaw) * py
            vertical = math.cos(tilt) * pz - math.sin(tilt) * depth
            return (x + width / 2 + horizontal * scale, y + height / 2 + vertical * scale)
        for n in range(-4, 5):
            # Grid stays fixed in space, while motion remains visible.
            a, b = project((first[0] + n / 2, first[1] - 2, 0), False), project((first[0] + n / 2, first[1] + 2, 0), False)
            if inside(rect, *a) and inside(rect, *b):
                line(a, b, (.15, .18, .23, 1), 1)
        joints = dict(frame.body3d, **frame.hands3d)
        row = self.value.row(self.value.displayed_sample)
        estimated = row.get('estimated_joints', [])
        coords = {name: project(point) for name, point in joints.items()}
        edges = list(preview.BODY_EDGES)
        for side in ('L', 'R'):
            for finger, _ in preview.skeleton.FINGERS:
                chain = ['wrist.' + side] + [finger + '.' + segment + '.' + side for segment in preview.skeleton.FINGER_SEGMENTS]
                edges.extend(zip(chain[:-1], chain[1:]))
        for a, b in edges:
            if a not in coords or b not in coords or not inside(rect, *coords[a]) or not inside(rect, *coords[b]):
                continue
            color = LEFT if b.endswith('.L') else (RIGHT if b.endswith('.R') else TEXT)
            if b in estimated:
                color = MUTED
            elif min(frame.confidence_of(a), frame.confidence_of(b)) < preview.skeleton.CONFIDENCE_LOW:
                color = LOW
            line(coords[a], coords[b], color, 3)
        for name, point in coords.items():
            if inside(rect, *point):
                color = MUTED if name in estimated else LOW if frame.confidence_of(name) < preview.skeleton.CONFIDENCE_LOW else (LEFT if name.endswith('.L') else RIGHT if name.endswith('.R') else TEXT)
                dot(point, color, 4 if name in frame.body3d else 2)
        for name, q in frame.orientations.items():
            anchor = frame.body3d.get('head' if name == 'head' else 'ankle.' + name[-1])
            if anchor is None:
                continue
            quality_key = 'raw_orientation_quality' if self.scene.mocap_props.preview_stage == 'raw' else 'orientation_quality'
            info = row.get(quality_key, {}).get(name, {})
            color = MUTED if info.get('estimated', True) else (TEXT if name == 'head' else LEFT if name.endswith('.L') else RIGHT)
            for vector, size in (((0, -1, 0), .13), ((0, 0, 1), .065)):
                end = rm.vec_add(anchor, rm.vec_scale(rm.quat_rotate_vector(q, vector), size))
                a, b = project(anchor), project(end)
                if inside(rect, *a) and inside(rect, *b):
                    line(a, b, color, 2)
                    dot(b, color, 3)

    def dispatch(self, command):
        props = self.scene.mocap_props
        if command == 'close':
            self.finish()
        elif command == 'play':
            props.preview_playing = not props.preview_playing
            self.value.last_clock = time.monotonic()
        elif command == 'previous':
            props.preview_playing = False
            props.preview_frame = max(1, props.preview_frame - 1)
        elif command == 'next':
            props.preview_playing = False
            props.preview_frame = min(self.value.total(), props.preview_frame + 1)
        elif command == 'speed':
            speeds = ['1', '0.5', '0.25']
            props.preview_speed = speeds[(speeds.index(props.preview_speed) + 1) % 3]
        elif command == 'loop':
            props.preview_loop = not props.preview_loop
        elif command == 'stage':
            props.preview_stage = 'raw' if props.preview_stage == 'processed' else 'processed'
        elif command in ('front', 'side', 'reset'):
            self.yaw = math.pi / 2 if command == 'side' else 0.0
            self.pitch = 0.18 if command == 'reset' else 0.0
            self.zoom = 1.0
        elif command == 'apply':
            self.finish()
            bpy.ops.mocap.apply_to_rigify('INVOKE_DEFAULT')
        elif command == 'blocked':
            self.report({'WARNING'}, review.apply_block_reason(self.scene))

    def modal(self, context, event):
        if self._closed:
            return {'FINISHED'}
        try:
            if self.window.scene != self.scene or self.area.type != 'VIEW_3D':
                self.finish()
                return {'CANCELLED'}
        except ReferenceError:
            self.finish()
            return {'CANCELLED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            self.finish()
            return {'FINISHED'}
        if event.type == 'TIMER':
            self.area.tag_redraw()
            return {'PASS_THROUGH'}
        mx, my = (event.mouse_x - self.region.x) / self.scale, (event.mouse_y - self.region.y) / self.scale
        width, height = self.canvas_size()
        in_canvas = 0 <= mx <= width and 0 <= my <= height
        if not in_canvas and self.drag is None:
            return {'PASS_THROUGH'}
        if event.type == 'SPACE' and event.value == 'PRESS':
            self.dispatch('play')
            return {'RUNNING_MODAL'}
        if event.type in ('LEFT_ARROW', 'RIGHT_ARROW') and event.value == 'PRESS':
            self.dispatch('previous' if event.type == 'LEFT_ARROW' else 'next')
            return {'RUNNING_MODAL'}
        if event.type == 'MIDDLEMOUSE':
            self.drag = ('orbit', mx, my) if event.value == 'PRESS' and inside(self.right_rect, mx, my) else None
        elif event.type in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE') and inside(self.right_rect, mx, my):
            self.zoom = max(.2, min(6, self.zoom * (1.1 if event.type == 'WHEELUPMOUSE' else 1 / 1.1)))
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                for rect, command in self.buttons:
                    if inside(rect, mx, my):
                        self.dispatch(command)
                        return {'FINISHED'} if self._closed else {'RUNNING_MODAL'}
                if abs(my - (self.timeline[1] + 4)) < 15:
                    self.drag = ('seek', mx, my)
                    self.scene.mocap_props.preview_playing = False
            else:
                self.drag = None
        if self.drag and (event.type == 'MOUSEMOVE' or self.drag[0] == 'seek'):
            kind, px, py = self.drag
            if kind == 'orbit':
                self.yaw += (mx - px) * .01
                self.pitch = max(-1.3, min(1.3, self.pitch + (my - py) * .01))
            else:
                fraction = max(0, min(1, (mx - self.timeline[0]) / max(1, self.timeline[2])))
                self.scene.mocap_props.preview_frame = 1 + round(fraction * (self.value.total() - 1))
            self.drag = (kind, mx, my)
        self.area.tag_redraw()
        # Do not let review gestures transform scene objects or start Blender playback.
        return {'RUNNING_MODAL'}


class MOCAP_OT_preview_control(bpy.types.Operator):
    bl_idname = "mocap.preview_control"
    bl_label = "预览控制"
    command: StringProperty(default="next")

    def execute(self, context):
        value = review.session(context.scene)
        props = context.scene.mocap_props
        if self.command == 'retry':
            value.retry_media()
        elif self.command in ('problem_next', 'problem_previous'):
            props.preview_playing = False
            props.preview_frame = 1 + preview.next_problem(value.rows, props.preview_frame - 1,
                                                          1 if self.command == 'problem_next' else -1)
        elif value.view:
            value.view.dispatch(self.command)
        return {'FINISHED'}


class MOCAP_OT_relocate_source(bpy.types.Operator):
    bl_idname = 'mocap.relocate_source'
    bl_label = '重新定位素材'
    filepath: StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        value = review.session(context.scene)
        source = paths.normalize(bpy.path.abspath(self.filepath))
        if value.state and not preview.source_matches(value.state.source, source, relocated=True):
            self.report({'ERROR'}, '文件大小或修改时间与捕捉素材不匹配，请选择原素材。')
            return {'CANCELLED'}
        value.suspend = True
        try:
            context.scene.mocap_props.source_media = source
        finally:
            value.suspend = False
        if value.state:
            value.state.source_path = source
            value.state.relocated = True
            value.state.viewed = False
        value.retry_media()
        return {'FINISHED'}


classes = (MOCAP_OT_open_preview, MOCAP_OT_preview_control, MOCAP_OT_relocate_source)
