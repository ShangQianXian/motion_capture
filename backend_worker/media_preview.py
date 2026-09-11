"""Model-free, latest-request-first media decoder for the Blender review canvas."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
from pathlib import Path

from ._core import errors, paths, preview
from . import media_decode


def trim_cache(root, keep, limit=preview.DISK_CACHE_LIMIT):
    """Only the dedicated, rebuildable JPEG cache is eligible for eviction."""
    root = Path(root).resolve()
    files = []
    for item in root.glob("*/*.jpg"):
        try:
            resolved = item.resolve()
            if root not in resolved.parents or item.is_symlink():
                continue
            stat = item.stat()
            files.append((stat.st_mtime_ns, stat.st_size, item))
        except OSError:
            continue
    total = sum(item[1] for item in files)
    for _, size, item in sorted(files):
        if total <= limit:
            break
        if str(item.resolve()) == str(Path(keep).resolve()):
            continue
        try:
            item.unlink()
            total -= size
        except OSError:
            pass
    return total


class Decoder:
    def __init__(self, path, kind, cache_root):
        self.cv2 = media_decode._require_cv2()
        self.path = paths.normalize(path)
        self.identity = preview.fingerprint(self.path)
        if self.identity["size"] < 0:
            raise errors.MocapError(errors.MEDIA_NOT_FOUND, "素材不存在，请重新选择或定位。")
        self.kind = media_decode.resolve_media_type(path, kind)
        self.cache_root = paths.ensure_dir(cache_root)
        digest = preview.signature(self.identity)
        self.cache_dir = paths.ensure_dir(os.path.join(self.cache_root, digest))
        self.capture = None
        self.image = None
        self.next_index = 0
        self.writes = 0
        if self.kind == "image":
            self.image = media_decode.read_image(self.cv2, self.path)
            if self.image is None:
                raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "图片无法解码，请检查文件格式。")
            height, width = self.image.shape[:2]
            # Retain only a bounded preview, not a full resolution image.
            self.image, _ = media_decode._resize_limit(self.cv2, self.image, preview.PREVIEW_LONG_SIDE)
            fps, total = 1.0, 1
        else:
            self.capture = self.cv2.VideoCapture(self.path)
            if not self.capture.isOpened():
                self.close()
                raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "视频无法解码，请检查文件格式或编码。")
            width = int(self.capture.get(self.cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.capture.get(self.cv2.CAP_PROP_FPS))
            total = int(self.capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
        self.info = dict(self.identity, type=self.kind, width=width, height=height,
                         fps=fps, total_frames=max(1, total), duration=0.0 if self.kind == "image" else total / fps)

    def frame(self, index):
        index = max(0, min(int(index), self.info["total_frames"] - 1))
        if not preview.source_matches(self.identity, self.path):
            raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "素材在预览期间发生变化，请重新加载。")
        target = os.path.join(self.cache_dir, "{0:09d}.jpg".format(index))
        if os.path.isfile(target):
            os.utime(target, None)
            return target, index
        if self.kind == "image":
            frame = self.image
        else:
            if index != self.next_index:
                if index > self.next_index and index - self.next_index <= 8:
                    while self.next_index < index:
                        if not self.capture.grab():
                            raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "无法读取指定视频帧。")
                        self.next_index += 1
                else:
                    self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.capture.read()
            if not ok or frame is None:
                raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "无法读取视频第 {0} 帧。".format(index + 1))
            position = int(round(self.capture.get(self.cv2.CAP_PROP_POS_FRAMES)))
            if position != index + 1:
                raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "解码器无法精确定位该帧，请换用 MP4/H.264 素材。")
            self.next_index = index + 1
            frame, _ = media_decode._resize_limit(self.cv2, frame, preview.PREVIEW_LONG_SIDE)
        ok, encoded = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise errors.MocapError(errors.MEDIA_OPEN_FAILED, "预览图片编码失败。")
        with open(target + ".tmp", "wb") as handle:
            handle.write(encoded.tobytes())
        os.replace(target + ".tmp", target)
        # The cache cap is strict; no full-video extraction or image accumulation.
        trim_cache(self.cache_root, target)
        return target, index

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.image = None


def serve(job, reporter, cancel_token, stdin):
    """The reader coalesces seeks before decoding; close and EOF always terminate."""
    requests = queue.Queue(maxsize=32)

    def read_requests():
        for line in stdin:
            try:
                command = json.loads(line) if len(line) < 8192 else {}
                if not isinstance(command, dict):
                    continue
            except ValueError:
                continue
            while True:
                try:
                    requests.put_nowait(command)
                    break
                except queue.Full:
                    try:
                        requests.get_nowait()
                    except queue.Empty:
                        pass
            if command.get("command") == "close":
                return
        requests.put({"command": "close", "request_id": -1})

    thread = threading.Thread(target=read_requests, daemon=True)
    decoder = None
    try:
        decoder = Decoder(job["input"]["path"], job["input"].get("type", "auto"),
                          job["output"].get("cache_dir") or os.path.join(job["output"]["dir"], "media_cache"))
        reporter.emit("media_info", request_id=0, info=decoder.info)
        thread.start()
        while not cancel_token.cancelled():
            try:
                command = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [command]
            while True:
                try:
                    batch.append(requests.get_nowait())
                except queue.Empty:
                    break
            if any(item.get("command") == "close" for item in batch):
                return
            latest = None
            for item in batch:
                if item.get("command") == "probe":
                    reporter.emit("media_info", request_id=item.get("request_id", 0), info=decoder.info)
                elif item.get("command") == "frame":
                    latest = item
            if latest:
                request_id = latest.get("request_id", 0)
                try:
                    path, index = decoder.frame(latest.get("source_index", 0))
                    reporter.emit("media_frame", request_id=request_id, source_index=index, path=path)
                except (errors.MocapError, ValueError, TypeError, OSError) as exc:
                    error = exc if isinstance(exc, errors.MocapError) else errors.wrap_unexpected(exc, "media preview")
                    reporter.emit("media_error", request_id=request_id, error=error.to_dict())
    finally:
        if decoder is not None:
            decoder.close()
