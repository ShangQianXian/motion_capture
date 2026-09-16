"""Framework-free review metadata, timing and immutable capture identities."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os

from . import errors, paths, pose_calibration, result_schema, skeleton

VERSION = "0.2"
MANIFEST_FILENAME = "preview_manifest.json"
IMAGE_MEMORY_LIMIT = 64 * 1024 * 1024
DISK_CACHE_LIMIT = 512 * 1024 * 1024
PREVIEW_LONG_SIDE = 960
CAPTURE_FIELDS = (
    "source_type", "capture_profile", "frame_start", "frame_end", "target_fps",
    "include_hands", "smoothing_strength", "foot_lock_strength", "root_motion",
    'motion_type', 'camera_view', 'align_initial_facing', 'input_normalisation',
)

# Actual detector topologies; synthetic standard-skeleton joints are not detections.
COCO_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
              (13, 15), (12, 14), (14, 16))
MP_EDGES = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6),
            (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17),
            (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18),
            (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32))
HAND_EDGES = tuple((start, start + 1) for start in range(21)
                   if start not in (0, 4, 8, 12, 16, 20)) + (
                       (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
BODY_EDGES = (('pelvis', 'spine'), ('spine', 'chest'), ('chest', 'neck'),
              ('neck', 'head')) + tuple(edge for side in ('L', 'R') for edge in (
                  ('chest', 'shoulder.' + side), ('shoulder.' + side, 'elbow.' + side),
                  ('elbow.' + side, 'wrist.' + side), ('pelvis', 'hip.' + side),
                  ('hip.' + side, 'knee.' + side), ('knee.' + side, 'ankle.' + side),
                  ('ankle.' + side, 'toe.' + side)))


def fingerprint(path):
    resolved = paths.normalize(path)
    try:
        stat = os.stat(resolved)
        return {"path": resolved, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": resolved, "size": -1, "mtime_ns": -1}


def source_matches(expected, path, relocated=False):
    actual = fingerprint(path)
    if actual["size"] < 0:
        return False
    keys = ("size", "mtime_ns") if relocated else ("path", "size", "mtime_ns")
    return all((os.path.normcase(str(actual[key])) == os.path.normcase(str(expected.get(key))))
               for key in keys)


def settings_snapshot(props):
    defaults = dict(motion_type='general', camera_view='unspecified',
                    align_initial_facing=False, input_normalisation='current')
    return {key: getattr(props, key, defaults.get(key)) for key in CAPTURE_FIELDS}


def signature(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def settings_equal(left, right):
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, float) and isinstance(other, (float, int)):
            if not math.isclose(value, other, rel_tol=0.0, abs_tol=1e-6):
                return False
        elif value != other:
            return False
    return True


def corrected_result(result, pitch=0.0, flip_x=False):
    """The exact correction order shared by the renderer and retargeter."""
    output = pose_calibration.calibrated_result(result, pitch) if pitch else copy.deepcopy(result)
    if flip_x:
        result_schema.mirror_result_x(output)
    return output


def action_frame(timestamp, start_time, scene_fps, image=False):
    if not all(math.isfinite(float(v)) for v in (timestamp, start_time, scene_fps)) or scene_fps <= 0:
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "动作时间或场景帧率无效。")
    return 1.0 if image else 1.0 + max(0.0, timestamp - start_time) * scene_fps


def is_problem(row):
    if row.get("status") in ("missing", "interpolated", "low_confidence"):
        return True
    points = row.get('body2d', [])
    if len(points) == 133:
        points = points[:23]  # Face/finger output is outside this capture profile's scope.
    weak_orientation = any(info.get('confidence', 0) < .4 for info in row.get('orientation_quality', {}).values())
    return weak_orientation or bool(row.get('unreliable') or row.get('unreliable_joints')) or any(point[2] < skeleton.CONFIDENCE_LOW for point in points)


def next_problem(rows, current, direction):
    if not rows:
        return current
    for offset in range(1, len(rows) + 1):
        index = (current + direction * offset) % len(rows)
        if is_problem(rows[index]):
            return index
    return current


def validate_manifest(data, result=None):
    try:
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise ValueError("预览文件版本不支持")
        if not isinstance(data.get("source"), dict) or not isinstance(data.get("frames"), list):
            raise ValueError("预览文件缺少素材或帧数据")
        if not data["frames"]:
            raise ValueError("预览没有帧")
        diagnostics = data.get('diagnostics', {})
        if not isinstance(diagnostics, dict) or any(not isinstance(diagnostics.get(k, {}), dict)
                                                   for k in ('joints', 'contact_counts', 'contact_intervals')):
            raise ValueError('诊断数据结构无效')
        stage = data.get('stages', {}).get('raw')
        if stage is not None:
            if not isinstance(stage, dict) or stage.get('file') != 'mocap_raw.json':
                raise ValueError('原始阶段引用无效')
            digest = stage.get('sha256', '')
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('原始阶段校验值无效')
        previous_time, previous_source = -1.0, -1
        numbers = {f.frame for f in result.frames} if result else None
        for row in data["frames"]:
            if any(state not in ('contact', 'air', 'unknown') for state in row.get('contact_states', {}).values()):
                raise ValueError('接触状态无效')
            source_index, timestamp = row["source_index"], row["time"]
            if type(source_index) is not int or source_index < 0 or source_index <= previous_source:
                raise ValueError("源帧编号必须递增")
            if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < previous_time:
                raise ValueError("预览时间无效")
            previous_time, previous_source = timestamp, source_index
            number = row.get("result_frame")
            if number is not None and numbers is not None and number not in numbers:
                raise ValueError("预览与三维结果不匹配")
            for points in [row.get("body2d", [])] + [hand["points"] for hand in row.get("hands2d", [])]:
                if not isinstance(points, list):
                    raise ValueError("二维坐标无效")
                for point in points:
                    if len(point) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                        raise ValueError("二维坐标必须是有限数值")
                    if not 0 <= point[2] <= 1:
                        raise ValueError("置信度超出范围")
            for edge in data.get("edges", []):
                if len(edge) != 2 or any(type(i) is not int or i < 0 for i in edge):
                    raise ValueError("二维骨架连线无效")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "无法读取预览数据：{0}".format(exc))
    return data


def load_manifest(result):
    """A missing sidecar is a supported v0.1 result, a corrupt sidecar is an error."""
    path = os.path.join(os.path.dirname(result.path), MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "预览文件损坏：{0}".format(exc))
    if data.get("result_sha256"):
        with open(result.path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != data["result_sha256"]:
            raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "预览文件与动捕结果不属于同一版本。")
    return validate_manifest(data, result)


def legacy_rows(result, native_fps):
    """Legacy timestamps still preserve crop/decimation offsets."""
    return [{"source_index": 0 if result.source.get('type') == 'image' else max(0, round(f.time * native_fps)), "time": f.time,
             "result_frame": f.frame, "body2d": [], "hands2d": [], "status": "legacy"}
            for f in result.frames]


class ReviewState:
    """Per-scene state; confirmation is only awarded after a matching rendered pair."""

    def __init__(self, result, manifest, settings, source_path):
        self.result = result
        self.manifest = manifest
        self.settings = dict(settings)
        self.settings.setdefault('motion_type', 'general')
        self.settings.setdefault('camera_view', 'unspecified')
        self.settings.setdefault('align_initial_facing', False)
        self.settings.setdefault('input_normalisation', 'current')
        self.source = dict(manifest["source"]) if manifest else fingerprint(source_path)
        self.source_path = source_path
        self.relocated = False
        self.viewed = False
        self.stale = False
        self.stale_reason = ""
        self.correction = None
        self.transformed = None
        self.revision = 0

    def invalidate(self, reason="素材或捕捉参数已改变，请重新生成。"):
        if self.stale:
            return
        self.stale = True
        self.stale_reason = reason
        self.viewed = False
        self.revision += 1

    def matches(self, settings, path):
        return not self.mismatch_reason(settings, path)

    def mismatch_reason(self, settings, path):
        if self.manifest and self.manifest.get('processing_version'):
            from .orientations import VERSION as processing_version
            if self.manifest['processing_version'] != processing_version:
                return "处理版本不一致：重启后加载，仍不匹配则重新生成。"
        if self.stale:
            return self.stale_reason
        current = dict(settings)
        current.setdefault('motion_type', 'general')
        current.setdefault('camera_view', 'unspecified')
        current.setdefault('align_initial_facing', False)
        current.setdefault('input_normalisation', 'current')
        if not settings_equal(self.settings, current):
            return "捕捉参数已改变，请重新生成。"
        if os.path.normcase(paths.normalize(path)) != os.path.normcase(paths.normalize(self.source_path)):
            return "素材路径已改变，请重新生成或重新定位匹配的文件。"
        return ""

    def media_matches(self):
        return source_matches(self.source, self.source_path, self.relocated)

    def corrected(self, pitch, flip_x):
        key = (float(pitch), bool(flip_x))
        if self.correction != key:
            self.transformed = corrected_result(self.result, *key)
            self.correction = key
            self.viewed = False
        return self.transformed
