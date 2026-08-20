"""JSONL progress reporting for the worker (guide section 5.2).

Contract: **stdout carries nothing but one JSON object per line.** Every human
readable message goes to stderr and to ``<output.dir>/worker.log``.
"""

from __future__ import annotations

import json
import os
import sys
import time

from ._core import errors, paths, progress


def _reconfigure_utf8(stream):
    """Force UTF-8 on a text stream when the platform default differs."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - already configured
            pass
    return stream


class Reporter(object):
    """Emits progress events and keeps a debug log."""

    def __init__(self, job_id: str = "", output_dir: str = "", stream=None) -> None:
        self.job_id = job_id or ""
        self.output_dir = paths.normalize(output_dir) if output_dir else ""
        self.stream = _reconfigure_utf8(stream or sys.stdout)
        self.log_handle = None
        self.emitted = []
        self.warning_count = 0
        self._terminal = False
        if self.output_dir:
            try:
                paths.ensure_dir(self.output_dir)
                self.log_handle = open(
                    paths.worker_log_path(self.output_dir), "a", encoding="utf-8", errors="replace"
                )
            except OSError:
                self.log_handle = None
        _reconfigure_utf8(sys.stderr)

    # -- low level ---------------------------------------------------------------------

    def emit(self, event: str, **fields) -> dict:
        """Write one JSONL progress line."""
        payload = {"event": event}
        if self.job_id and "job_id" not in fields:
            payload["job_id"] = self.job_id
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        line = json.dumps(payload, ensure_ascii=False)
        try:
            self.stream.write(line + "\n")
            self.stream.flush()
        except (OSError, ValueError):  # pragma: no cover - stdout closed
            pass
        self.emitted.append(payload)
        del self.emitted[:-200]
        self.debug("EVENT {0}".format(line))
        if event in progress.TERMINAL_EVENTS:
            self._terminal = True
        return payload

    def debug(self, message: str) -> None:
        """Write a human readable line to stderr and the worker log."""
        text = "[{0}] {1}".format(time.strftime("%H:%M:%S"), message)
        try:
            sys.stderr.write(text + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):  # pragma: no cover
            pass
        if self.log_handle is not None:
            try:
                self.log_handle.write(text + "\n")
                self.log_handle.flush()
            except (OSError, ValueError):
                pass

    @property
    def finished(self) -> bool:
        """True once a terminal event was emitted."""
        return self._terminal

    # -- typed events ------------------------------------------------------------------

    def started(self, message: str = "Worker started") -> dict:
        return self.emit(progress.EVENT_STARTED, message=message)

    def loading_model(self, profile: str = "", model_id: str = "", fraction=None) -> dict:
        return self.emit(
            progress.EVENT_LOADING_MODEL,
            profile=profile or None,
            model_id=model_id or None,
            progress=None if fraction is None else round(float(fraction), 4),
        )

    def processing_frame(self, frame: int, total_frames=None, fraction=None) -> dict:
        if fraction is None and total_frames:
            fraction = float(frame) / float(total_frames)
        return self.emit(
            progress.EVENT_PROCESSING_FRAME,
            frame=int(frame),
            total_frames=None if total_frames is None else int(total_frames),
            progress=None if fraction is None else round(min(1.0, max(0.0, float(fraction))), 4),
        )

    def warning(self, code: str, message: str, frame=None, **extra) -> dict:
        self.warning_count += 1
        return self.emit(
            progress.EVENT_WARNING,
            code=code,
            message=message,
            frame=None if frame is None else int(frame),
            **extra
        )

    def completed(self, result_path: str, frames=None) -> dict:
        return self.emit(
            progress.EVENT_COMPLETED,
            result_path=result_path,
            frames=None if frames is None else int(frames),
            progress=1.0,
        )

    def failed(self, error) -> dict:
        payload = error.to_dict() if isinstance(error, errors.MocapError) else errors.wrap_unexpected(error).to_dict()
        return self.emit(progress.EVENT_FAILED, error=payload)

    def cancelled(self, message: str = "Worker cancelled by user") -> dict:
        return self.emit(progress.EVENT_CANCELLED, message=message)

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except (OSError, ValueError):
                pass
            self.log_handle = None


class CancellationToken(object):
    """Cooperative cancellation via a sentinel file in the job directory."""

    def __init__(self, output_dir: str = "", poll_interval: float = 0.25) -> None:
        self.path = paths.cancel_flag_path(output_dir) if output_dir else ""
        self.poll_interval = float(poll_interval)
        self._last_check = 0.0
        self._cancelled = False

    def cancelled(self) -> bool:
        """True once the add-on requested cancellation (rate-limited stat call)."""
        if self._cancelled:
            return True
        if not self.path:
            return False
        now = time.time()
        if now - self._last_check < self.poll_interval:
            return False
        self._last_check = now
        if os.path.exists(self.path):
            self._cancelled = True
        return self._cancelled

    def clear(self) -> None:
        """Remove a stale sentinel before a run starts."""
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self._cancelled = False
