"""Worker progress JSONL parsing (guide sections 4.3 and 5.2).

Standard library only. A malformed line must never crash the add-on: it is
downgraded to a warning event that keeps the original text for triage.
"""

from __future__ import annotations

import json

from . import errors

EVENT_STARTED = "started"
EVENT_LOADING_MODEL = "loading_model"
EVENT_PROCESSING_FRAME = "processing_frame"
EVENT_WARNING = "warning"
EVENT_COMPLETED = "completed"
EVENT_FAILED = "failed"
EVENT_CANCELLED = "cancelled"

#: Events the worker is allowed to emit (guide section 5.2).
ALLOWED_EVENTS = (
    EVENT_STARTED,
    EVENT_LOADING_MODEL,
    EVENT_PROCESSING_FRAME,
    EVENT_WARNING,
    EVENT_COMPLETED,
    EVENT_FAILED,
    EVENT_CANCELLED,
)

#: Events after which no further progress is expected.
TERMINAL_EVENTS = (EVENT_COMPLETED, EVENT_FAILED, EVENT_CANCELLED)

#: Warning code used when a stdout line could not be parsed as JSON.
CODE_OUTPUT_UNPARSED = "WORKER_OUTPUT_UNPARSED"

#: Warning code used when a parsed line carries an unknown ``event``.
CODE_UNKNOWN_EVENT = "WORKER_UNKNOWN_EVENT"


class ProgressEvent(object):
    """One parsed JSONL progress line."""

    __slots__ = ("event", "fields", "raw", "parsed")

    def __init__(self, event: str, fields: dict | None = None, raw: str = "", parsed: bool = True) -> None:
        self.event = event
        self.fields = dict(fields) if fields else {}
        self.raw = raw
        self.parsed = bool(parsed)

    # -- convenience accessors ---------------------------------------------------------

    @property
    def message(self) -> str:
        value = self.fields.get("message")
        if value:
            return str(value)
        error = self.error
        if error is not None:
            return error.message
        if self.event == EVENT_PROCESSING_FRAME:
            total = self.fields.get("total_frames")
            frame = self.fields.get("frame")
            if total:
                return "处理帧 {0}/{1}".format(frame, total)
            return "处理帧 {0}".format(frame)
        if self.event == EVENT_LOADING_MODEL:
            return "加载模型 {0}".format(self.fields.get("model_id") or self.fields.get("profile") or "")
        return self.event

    @property
    def code(self) -> str:
        value = self.fields.get("code")
        if value:
            return str(value)
        error = self.error
        return error.code if error is not None else ""

    @property
    def progress(self):
        value = self.fields.get("progress")
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @property
    def frame(self):
        value = self.fields.get("frame")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def result_path(self) -> str:
        return str(self.fields.get("result_path") or "")

    @property
    def error(self):
        payload = self.fields.get("error")
        if isinstance(payload, dict):
            return errors.MocapError.from_dict(payload)
        return None

    @property
    def is_terminal(self) -> bool:
        return self.event in TERMINAL_EVENTS

    @property
    def level(self) -> str:
        """Log level for the UI log panel."""
        if self.event == EVENT_FAILED:
            return "ERROR"
        if self.event == EVENT_WARNING or not self.parsed:
            return "WARNING"
        return "INFO"

    def to_dict(self) -> dict:
        payload = {"event": self.event}
        payload.update(self.fields)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "ProgressEvent({0!r}, parsed={1})".format(self.event, self.parsed)


def parse_progress_line(line: str) -> ProgressEvent:
    """Parse one JSONL progress line from worker stdout.

    Never raises. Unparseable or structurally unexpected lines become
    ``warning`` events that preserve the raw text.
    """
    raw = "" if line is None else str(line)
    text = raw.strip()
    if not text:
        return ProgressEvent(
            EVENT_WARNING,
            {"code": CODE_OUTPUT_UNPARSED, "message": "worker 输出空行。"},
            raw=raw,
            parsed=False,
        )
    try:
        data = json.loads(text)
    except ValueError:
        return ProgressEvent(
            EVENT_WARNING,
            {
                "code": CODE_OUTPUT_UNPARSED,
                "message": "worker 输出非 JSON 行：{0}".format(_truncate(text)),
            },
            raw=raw,
            parsed=False,
        )
    if not isinstance(data, dict):
        return ProgressEvent(
            EVENT_WARNING,
            {
                "code": CODE_OUTPUT_UNPARSED,
                "message": "worker 输出的 JSON 不是对象：{0}".format(_truncate(text)),
            },
            raw=raw,
            parsed=False,
        )

    event = str(data.get("event") or "")
    fields = {key: value for key, value in data.items() if key != "event"}
    if event not in ALLOWED_EVENTS:
        return ProgressEvent(
            EVENT_WARNING,
            {
                "code": CODE_UNKNOWN_EVENT,
                "message": "worker 输出未知 event {0!r}：{1}".format(event, _truncate(text)),
                "original_event": event,
            },
            raw=raw,
            parsed=False,
        )
    return ProgressEvent(event, fields, raw=raw, parsed=True)


def _truncate(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."
