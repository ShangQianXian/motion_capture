"""Media decoding for images and videos (guide section 7.3).

OpenCV is imported lazily so the add-on, the mock pipeline and the unit tests
never require it. Failures are reported as structured
``MEDIA_OPEN_FAILED`` / ``DEPENDENCY_MISSING`` errors.
"""

from __future__ import annotations

import os
import math

from ._core import errors, paths

#: Long-side limit for RTX4060-class hardware (technical design section 8).
DEFAULT_MAX_LONG_SIDE = 1280

#: Lower bound of the recommended long-side range.
MIN_MAX_LONG_SIDE = 960


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise errors.MocapError(
            errors.DEPENDENCY_MISSING,
            "worker 环境缺少 opencv-python：{0}".format(exc),
            suggestion="按 docs/INSTALL.md 安装对应 profile 的锁定环境；每个环境只保留一种 OpenCV 包。",
            details={"module": "cv2"},
        )
    return cv2


class DecodedFrame(object):
    """One decoded frame handed to a pose estimator."""

    __slots__ = ("index", "source_index", "timestamp", "image", "width", "height", "scale")

    def __init__(self, index, source_index, timestamp, image, scale=1.0) -> None:
        self.index = int(index)
        self.source_index = int(source_index)
        self.timestamp = float(timestamp)
        self.image = image
        self.scale = float(scale)
        try:
            self.height, self.width = image.shape[0], image.shape[1]
        except (AttributeError, IndexError):  # pragma: no cover - defensive
            self.height, self.width = 0, 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "DecodedFrame(index={0}, {1}x{2})".format(self.index, self.width, self.height)


class MediaSource(object):
    """Iterable frame source with a known effective frame rate."""

    def __init__(self, path, media_type, fps, total_frames, frames_iter, native_fps=None,
                 width=0, height=0, native_total=0):
        self.path = path
        self.media_type = media_type
        self.fps = float(fps)
        self.native_fps = float(native_fps) if native_fps else float(fps)
        self.total_frames = int(total_frames)
        self._frames_iter = frames_iter
        self._closed = False
        self.width, self.height = int(width), int(height)
        self.native_total = int(native_total or total_frames)

    def __iter__(self):
        return self._frames_iter

    def close(self) -> None:
        self._closed = True
        closer = getattr(self._frames_iter, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:  # pragma: no cover - generator already finished
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "MediaSource({0!r}, {1}, fps={2}, frames={3})".format(
            os.path.basename(self.path), self.media_type, self.fps, self.total_frames
        )


def _resize_limit(cv2, image, max_long_side):
    """Downscale so the long side fits ``max_long_side``; returns (image, scale)."""
    if not max_long_side or max_long_side <= 0:
        return image, 1.0
    height, width = image.shape[0], image.shape[1]
    longest = max(width, height)
    if longest <= max_long_side:
        return image, 1.0
    scale = float(max_long_side) / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def resolve_media_type(path: str, declared: str = "auto") -> str:
    """Resolve ``input.type``; ``auto`` uses the file extension."""
    if declared and declared != "auto":
        return declared
    guessed = paths.guess_media_type(path)
    if guessed == "unknown":
        raise errors.MocapError(
            errors.UNSUPPORTED_MEDIA_TYPE,
            "无法根据扩展名判断媒体类型：{0}".format(os.path.basename(path)),
            details={
                "path": path,
                "image_extensions": list(paths.image_extensions()),
                "video_extensions": list(paths.video_extensions()),
            },
        )
    return guessed


def read_image(cv2, path):
    """OpenCV's Windows imread does not reliably accept Unicode paths."""
    import numpy
    try:
        with open(path, "rb") as handle:
            return cv2.imdecode(numpy.frombuffer(handle.read(), dtype=numpy.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def open_media(
    path: str,
    media_type: str = "auto",
    target_fps: float = 30.0,
    frame_start: int = 1,
    frame_end: int = 0,
    max_long_side: int = DEFAULT_MAX_LONG_SIDE,
    cancel_token=None,
    reporter=None,
) -> MediaSource:
    """Open ``path`` and return a :class:`MediaSource`.

    Videos are decimated to ``target_fps`` rather than processing every native
    frame (guide section 7.3). ``frame_end == 0`` means "until the end".
    """
    resolved = paths.normalize(path)
    if not resolved or not os.path.isfile(resolved):
        raise errors.MocapError(
            errors.MEDIA_NOT_FOUND,
            "媒体文件不存在：{0}".format(path),
            details={"path": str(path or "")},
        )
    kind = resolve_media_type(resolved, media_type)
    cv2 = _require_cv2()
    limit = max(MIN_MAX_LONG_SIDE, int(max_long_side)) if max_long_side else 0

    if kind == "image":
        image = read_image(cv2, resolved)
        if image is None:
            raise errors.MocapError(
                errors.MEDIA_OPEN_FAILED,
                "无法解码图片：{0}".format(resolved),
                details={"path": resolved},
            )
        height, width = image.shape[:2]
        image, scale = _resize_limit(cv2, image, limit)

        def image_frames():
            yield DecodedFrame(0, 0, 0.0, image, scale)

        return MediaSource(resolved, "image", float(target_fps or 30.0), 1, image_frames(),
                           width=width, height=height, native_total=1)

    capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        capture.release()
        raise errors.MocapError(
            errors.MEDIA_OPEN_FAILED,
            "无法打开视频：{0}".format(resolved),
            details={"path": resolved},
        )

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(native_fps) or native_fps <= 0.0:
        native_fps = float(target_fps or 30.0)
        if reporter is not None:
            reporter.warning(
                "MEDIA_FPS_UNKNOWN",
                "无法读取视频帧率，按 {0:.2f} FPS 处理。".format(native_fps),
            )
    native_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    wanted_fps = float(target_fps) if target_fps and target_fps > 0 else native_fps
    step = max(1, int(round(native_fps / wanted_fps))) if wanted_fps > 0 else 1
    effective_fps = native_fps / float(step)

    start_index = max(0, int(frame_start or 1) - 1)
    if frame_end and int(frame_end) >= int(frame_start or 1):
        wanted_count = int(frame_end) - int(frame_start or 1) + 1
    else:
        wanted_count = 0

    if native_total > 0:
        available = max(0, (native_total - start_index * step + step - 1) // step)
        total = available if wanted_count == 0 else min(wanted_count, available)
    else:
        total = wanted_count

    def video_frames():
        try:
            source_index = 0
            emitted = 0
            previous_timestamp = -1.0
            if start_index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_index * step)
                source_index = start_index * step
            while True:
                if cancel_token is not None and cancel_token.cancelled():
                    return
                ok, image = capture.read()
                if not ok or image is None:
                    return
                if (source_index - start_index * step) % step == 0:
                    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    if not math.isfinite(timestamp) or timestamp <= previous_timestamp or (source_index > 0 and timestamp == 0):
                        timestamp = max(previous_timestamp + 1.0 / effective_fps, source_index / native_fps)
                    previous_timestamp = timestamp
                    resized, scale = _resize_limit(cv2, image, limit)
                    yield DecodedFrame(
                        emitted,
                        source_index,
                        timestamp,
                        resized,
                        scale,
                    )
                    emitted += 1
                    if wanted_count and emitted >= wanted_count:
                        return
                source_index += 1
        finally:
            capture.release()

    return MediaSource(resolved, "video", effective_fps, total, video_frames(), native_fps,
                       width=width, height=height, native_total=native_total)
