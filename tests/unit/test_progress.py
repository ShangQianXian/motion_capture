"""Progress JSONL parsing (guide sections 4.3 and 5.2)."""

from __future__ import annotations

import json
import unittest

from core import errors, progress


class TestParseProgressLine(unittest.TestCase):
    def test_every_documented_event_parses(self):
        samples = {
            "started": '{"event":"started","job_id":"u","message":"Worker started"}',
            "loading_model": '{"event":"loading_model","profile":"quality","model_id":"rtmpose_m_body","progress":0.15}',
            "processing_frame": '{"event":"processing_frame","frame":42,"total_frames":300,"progress":0.42}',
            "warning": '{"event":"warning","code":"LOW_CONFIDENCE","message":"low","frame":88}',
            "completed": '{"event":"completed","result_path":"D:/r.json","progress":1.0}',
            "failed": '{"event":"failed","error":{"code":"CUDA_OOM","message":"oom"}}',
            "cancelled": '{"event":"cancelled","message":"stopped"}',
        }
        for name, line in samples.items():
            event = progress.parse_progress_line(line)
            self.assertTrue(event.parsed, name)
            self.assertEqual(event.event, name)

    def test_terminal_events(self):
        for name in ("completed", "failed", "cancelled"):
            event = progress.parse_progress_line(json.dumps({"event": name}))
            self.assertTrue(event.is_terminal, name)
        for name in ("started", "loading_model", "processing_frame", "warning"):
            event = progress.parse_progress_line(json.dumps({"event": name}))
            self.assertFalse(event.is_terminal, name)

    def test_frame_and_progress_accessors(self):
        event = progress.parse_progress_line(
            '{"event":"processing_frame","frame":42,"total_frames":300,"progress":0.42}'
        )
        self.assertEqual(event.frame, 42)
        self.assertAlmostEqual(event.progress, 0.42)
        self.assertIn("42", event.message)

    def test_progress_is_clamped(self):
        event = progress.parse_progress_line('{"event":"processing_frame","progress":7}')
        self.assertEqual(event.progress, 1.0)
        event = progress.parse_progress_line('{"event":"processing_frame","progress":-3}')
        self.assertEqual(event.progress, 0.0)

    def test_progress_is_none_when_absent_or_bad(self):
        self.assertIsNone(progress.parse_progress_line('{"event":"started"}').progress)
        self.assertIsNone(
            progress.parse_progress_line('{"event":"started","progress":"soon"}').progress
        )

    def test_failed_event_exposes_a_structured_error(self):
        line = json.dumps(
            {
                "event": "failed",
                "error": {
                    "code": "CUDA_OOM",
                    "message": "CUDA ran out of memory while running RTMPose-x.",
                    "suggestion": "Switch to quality profile or reduce input resolution.",
                    "recoverable": True,
                },
            }
        )
        event = progress.parse_progress_line(line)
        error = event.error
        self.assertIsInstance(error, errors.MocapError)
        self.assertEqual(error.code, "CUDA_OOM")
        self.assertTrue(error.recoverable)
        self.assertEqual(event.code, "CUDA_OOM")
        self.assertEqual(event.level, "ERROR")

    def test_result_path_accessor(self):
        event = progress.parse_progress_line(
            '{"event":"completed","result_path":"D:/out/mocap_result.json"}'
        )
        self.assertEqual(event.result_path, "D:/out/mocap_result.json")

    # -- robustness --------------------------------------------------------------------

    def test_non_json_line_becomes_a_warning_and_keeps_the_raw_text(self):
        raw = "Traceback (most recent call last):"
        event = progress.parse_progress_line(raw)
        self.assertFalse(event.parsed)
        self.assertEqual(event.event, "warning")
        self.assertEqual(event.code, progress.CODE_OUTPUT_UNPARSED)
        self.assertEqual(event.raw, raw)
        self.assertIn(raw, event.message)
        self.assertEqual(event.level, "WARNING")

    def test_blank_and_none_lines_are_tolerated(self):
        for raw in ("", "   ", "\n", None):
            event = progress.parse_progress_line(raw)
            self.assertFalse(event.parsed)
            self.assertEqual(event.event, "warning")

    def test_json_array_is_rejected_as_unparsed(self):
        event = progress.parse_progress_line("[1, 2, 3]")
        self.assertFalse(event.parsed)
        self.assertEqual(event.code, progress.CODE_OUTPUT_UNPARSED)

    def test_unknown_event_is_downgraded_to_warning(self):
        event = progress.parse_progress_line('{"event":"exploded","message":"boom"}')
        self.assertFalse(event.parsed)
        self.assertEqual(event.event, "warning")
        self.assertEqual(event.code, progress.CODE_UNKNOWN_EVENT)
        self.assertEqual(event.fields.get("original_event"), "exploded")

    def test_missing_event_key_is_downgraded(self):
        event = progress.parse_progress_line('{"message":"no event key"}')
        self.assertFalse(event.parsed)
        self.assertEqual(event.code, progress.CODE_UNKNOWN_EVENT)

    def test_very_long_line_is_truncated_in_the_message_but_kept_raw(self):
        raw = "x" * 5000
        event = progress.parse_progress_line(raw)
        self.assertLess(len(event.message), 400)
        self.assertEqual(len(event.raw), 5000)

    def test_to_dict_roundtrip(self):
        event = progress.parse_progress_line('{"event":"warning","code":"C","message":"m"}')
        payload = event.to_dict()
        self.assertEqual(payload["event"], "warning")
        self.assertEqual(payload["code"], "C")


if __name__ == "__main__":
    unittest.main()
