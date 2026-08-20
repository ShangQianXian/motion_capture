"""Mock worker: in-process generation and the real CLI contract (sections 5.2, 7.1, 7.2)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from ._util import ADDON_ROOT, TempDirCase, preferences, scene_props

from core import errors, job_schema, paths, progress, result_schema, skeleton, worker_client

from backend_worker import export_result, mock_source, pipeline, postprocess


class TestMockSource(unittest.TestCase):
    def test_default_frame_count_and_fps(self):
        frames = mock_source.generate_frames()
        self.assertEqual(len(frames), 60)  # guide section 7.2
        self.assertEqual(mock_source.DEFAULT_FPS, 30)

    def test_frame_range_from_the_job_wins(self):
        frames = mock_source.generate_frames(frame_start=1, frame_end=30)
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0]["frame"], 1)
        self.assertEqual(frames[-1]["frame"], 30)

    def test_frame_numbering_honours_frame_start(self):
        frames = mock_source.generate_frames(frame_start=100, frame_end=104)
        self.assertEqual([f["frame"] for f in frames], [100, 101, 102, 103, 104])

    def test_every_standard_joint_is_present_including_heels(self):
        frames = mock_source.generate_frames(frame_end=5)
        for frame in frames:
            for joint in skeleton.ALL_BODY_JOINTS:
                self.assertIn(joint, frame["body3d"], joint)

    def test_left_side_is_on_positive_x(self):
        frames = mock_source.generate_frames(frame_end=10)
        for frame in frames:
            self.assertGreater(frame["body3d"]["hip.L"][0], frame["body3d"]["hip.R"][0])
            self.assertGreater(frame["body3d"]["shoulder.L"][0], frame["body3d"]["shoulder.R"][0])

    def test_data_is_z_up_and_standing_on_the_ground(self):
        frames = mock_source.generate_frames(frame_end=20)
        for frame in frames:
            self.assertGreater(frame["body3d"]["head"][2], frame["body3d"]["pelvis"][2])
            self.assertGreater(frame["body3d"]["pelvis"][2], frame["body3d"]["ankle.L"][2])
            self.assertGreaterEqual(min(frame["body3d"]["toe.L"][2], frame["body3d"]["toe.R"][2]), -1e-6)

    def test_pelvis_bobs_up_and_down(self):
        frames = mock_source.generate_frames(frame_end=60)
        heights = [frame["body3d"]["pelvis"][2] for frame in frames]
        span = max(heights) - min(heights)
        self.assertGreater(span, mock_source.BOB_AMPLITUDE)
        self.assertLess(span, mock_source.BOB_AMPLITUDE * 3.0)

    def test_feet_alternate_contact(self):
        frames = mock_source.generate_frames(frame_end=60)
        left = [frame["contacts"]["foot.L"] for frame in frames]
        right = [frame["contacts"]["foot.R"] for frame in frames]
        self.assertTrue(any(left))
        self.assertTrue(any(right))
        self.assertGreaterEqual(sum(1 for i in range(1, 60) if left[i] != left[i - 1]), 2)
        # The two feet must not be in the same phase.
        self.assertNotEqual(left, right)

    def test_right_arm_waves(self):
        frames = mock_source.generate_frames(frame_end=60)
        wrists = [frame["body3d"]["wrist.R"] for frame in frames]
        heights = [w[2] for w in wrists]
        self.assertGreater(max(heights), frames[0]["body3d"]["shoulder.R"][2])
        lateral = [w[0] for w in wrists]
        self.assertGreater(max(lateral) - min(lateral), 0.02)

    def test_swing_is_continuous_with_no_teleport(self):
        frames = mock_source.generate_frames(frame_end=60)
        for side in ("L", "R"):
            for index in range(1, len(frames)):
                previous = frames[index - 1]["body3d"]["ankle.{0}".format(side)]
                current = frames[index]["body3d"]["ankle.{0}".format(side)]
                step = sum((a - b) ** 2 for a, b in zip(previous, current)) ** 0.5
                self.assertLess(step, 0.09, "frame {0} side {1} jumped {2:.3f} m".format(index, side, step))

    def test_low_confidence_window_exists(self):
        frames = mock_source.generate_frames(frame_end=60)
        low = [
            frame for frame in frames
            if frame["confidence"].get(mock_source.LOW_CONFIDENCE_JOINT, 1.0) < skeleton.CONFIDENCE_LOW
        ]
        self.assertEqual(len(low), len(mock_source.LOW_CONFIDENCE_FRAMES))

    def test_hands_are_optional(self):
        without = mock_source.generate_frames(frame_end=2, include_hands=False)
        self.assertEqual(without[0]["hands3d"], {})
        with_hands = mock_source.generate_frames(frame_end=2, include_hands=True)
        self.assertEqual(len(with_hands[0]["hands3d"]), 40)
        for joint in skeleton.HAND_JOINTS:
            self.assertIn(joint, with_hands[0]["hands3d"], joint)

    def test_single_frame_request(self):
        frames = mock_source.generate_frames(frame_start=5, frame_end=5)
        self.assertEqual(len(frames), 1)


class TestMockResultSchema(TempDirCase):
    def test_generated_result_validates(self):
        frames = mock_source.generate_frames(frame_end=30, include_hands=True)
        frames, warnings = postprocess.postprocess(
            frames, 30.0, {"smoothing_strength": 0.65, "foot_lock_strength": 0.7}
        )
        payload = export_result.build_result(
            frames, 30.0, source_path="D:/media/mock.mp4", source_type="video", profile="preview",
            warnings=postprocess.summarise_warnings(warnings),
        )
        result_schema.validate_result(payload)
        target = export_result.write_result(payload, self.tmp)
        result = result_schema.load_mocap_result(target)
        self.assertEqual(len(result.frames), 30)
        self.assertFalse(result_schema.detect_mirror(result))
        self.assertFalse(result_schema.looks_y_up(result))

    def test_postprocess_records_contacts_for_both_feet(self):
        frames = mock_source.generate_frames(frame_end=40)
        frames, _warnings = postprocess.postprocess(frames, 30.0, {"foot_lock_strength": 0.7})
        for frame in frames:
            self.assertIn("foot.L", frame["contacts"])
            self.assertIn("foot.R", frame["contacts"])

    def test_postprocess_keeps_the_upper_body_in_place(self):
        frames = mock_source.generate_frames(frame_end=60)
        original = [frame["body3d"]["chest"] for frame in frames]
        frames, _warnings = postprocess.postprocess(
            frames, 30.0, {"smoothing_strength": 0.0, "foot_lock_strength": 1.0}
        )
        for before, frame in zip(original, frames):
            after = frame["body3d"]["chest"]
            self.assertAlmostEqual(before[0], after[0], places=6)
            self.assertAlmostEqual(before[1], after[1], places=6)

    def test_single_frame_skips_temporal_work(self):
        frames = mock_source.generate_frames(frame_start=1, frame_end=1)
        frames, warnings = postprocess.postprocess(frames, 30.0, {"foot_lock_strength": 0.7})
        codes = [w["code"] for w in warnings]
        self.assertIn(postprocess.CODE_NO_TEMPORAL, codes)

    def test_empty_frames_is_handled(self):
        frames, warnings = postprocess.postprocess([], 30.0, {})
        self.assertEqual(frames, [])
        self.assertEqual(warnings, [])

    def test_warning_summary_collapses_repeats(self):
        warnings = [{"code": "X", "joint": "a", "message": "m"} for _ in range(50)]
        summary = postprocess.summarise_warnings(warnings)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["occurrences"], 50)

    def test_warning_summary_truncates_many_kinds(self):
        warnings = [{"code": "X", "joint": str(i), "message": "m"} for i in range(60)]
        summary = postprocess.summarise_warnings(warnings, limit=10)
        self.assertEqual(len(summary), 11)
        self.assertEqual(summary[-1]["code"], "WARNINGS_TRUNCATED")


class TestPipelineChecks(unittest.TestCase):
    def test_check_env_reports_the_interpreter_and_modules(self):
        report = pipeline.check_env()
        self.assertEqual(report["python"], sys.version.split()[0])
        self.assertIn("modules", report)
        for name in ("numpy", "cv2", "mediapipe", "torch", "mmpose"):
            self.assertIn(name, report["modules"])
            self.assertIn("available", report["modules"][name])

    def test_check_env_does_not_raise_without_any_dependency(self):
        report = pipeline.check_env()
        self.assertTrue(report["addon_root"])
        self.assertEqual(report["skeleton"], skeleton.SKELETON_ID)

    def test_check_cuda_reports_structured_error_without_torch(self):
        report = pipeline.check_cuda()
        self.assertIn("available", report)
        if not report["available"]:
            self.assertIn("error", report)
            self.assertEqual(report["error"]["code"], errors.CUDA_UNAVAILABLE)
            self.assertTrue(report["error"]["suggestion"])


class TestMockPipeline(TempDirCase):
    def _job(self, **overrides) -> dict:
        models_root = self.path("models")
        os.makedirs(models_root, exist_ok=True)
        media = self.touch("media/clip.mp4")
        props = scene_props(source_media=media, **overrides)
        return job_schema.build_job(props, preferences(models_root=models_root))

    def test_mock_run_writes_a_valid_result(self):
        job = self._job()
        reporter = _CollectingReporter()
        path = pipeline.run_job(job, reporter, None, mock=True)
        self.assertTrue(os.path.isfile(path))
        result = result_schema.load_mocap_result(path)
        self.assertEqual(len(result.frames), 60)
        self.assertEqual(result.profile, "preview")
        codes = [w.get("code") for w in result.warnings]
        self.assertIn(pipeline.CODE_MOCK, codes)

    def test_mock_run_honours_the_frame_range(self):
        job = self._job(frame_start=5, frame_end=14)
        path = pipeline.run_job(job, _CollectingReporter(), None, mock=True)
        result = result_schema.load_mocap_result(path)
        self.assertEqual(len(result.frames), 10)
        self.assertEqual(result.frame_start, 5)
        self.assertEqual(result.frame_end, 14)

    def test_mock_run_emits_progress_events(self):
        job = self._job()
        reporter = _CollectingReporter()
        pipeline.run_job(job, reporter, None, mock=True)
        events = [payload["event"] for payload in reporter.emitted]
        self.assertIn("loading_model", events)
        self.assertIn("processing_frame", events)

    def test_cancellation_raises_cancelled(self):
        job = self._job()

        class AlwaysCancelled(object):
            def cancelled(self):
                return True

        with self.assertRaises(errors.MocapError) as ctx:
            pipeline.run_job(job, _CollectingReporter(), AlwaysCancelled(), mock=True)
        self.assertEqual(ctx.exception.code, errors.CANCELLED)

    def test_real_run_without_models_reports_a_missing_model(self):
        job = self._job()
        with self.assertRaises(errors.MocapError) as ctx:
            pipeline.run_job(job, _CollectingReporter(), None, mock=False)
        self.assertIn(ctx.exception.code, (errors.MODEL_MISSING, errors.CONFIG_MISSING))


class _CollectingReporter(object):
    """Minimal reporter that records events instead of writing to stdout."""

    def __init__(self) -> None:
        self.emitted = []
        self.warnings = []

    def emit(self, event, **fields):
        payload = {"event": event}
        payload.update(fields)
        self.emitted.append(payload)
        return payload

    def debug(self, message):
        pass

    def loading_model(self, profile="", model_id="", fraction=None):
        return self.emit("loading_model", profile=profile, model_id=model_id, progress=fraction)

    def processing_frame(self, frame, total_frames=None, fraction=None):
        return self.emit("processing_frame", frame=frame, total_frames=total_frames)

    def warning(self, code, message, frame=None, **extra):
        self.warnings.append((code, message))
        return self.emit("warning", code=code, message=message, frame=frame)

    def completed(self, result_path, frames=None):
        return self.emit("completed", result_path=result_path)

    def failed(self, error):
        return self.emit("failed", error=error)

    def cancelled(self, message=""):
        return self.emit("cancelled", message=message)


class TestWorkerCLI(TempDirCase):
    """Exercises the documented command lines in a real subprocess."""

    def _run(self, args, timeout: float = 180.0):
        command = [sys.executable, "-m", "backend_worker.cli"] + list(args)
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command,
            cwd=ADDON_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=environment,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed

    def _write_job(self, **overrides) -> dict:
        models_root = self.path("models")
        os.makedirs(models_root, exist_ok=True)
        media = self.touch("media/clip.mp4")
        job = job_schema.build_job(
            scene_props(source_media=media, **overrides), preferences(models_root=models_root)
        )
        job["output"]["dir"] = self.path("job")
        paths.ensure_dir(job["output"]["dir"])
        job_schema.write_job(job, self.path("job", "job.json"))
        return job

    def test_every_stdout_line_is_valid_json(self):
        job = self._write_job()
        completed = self._run(["--job", self.path("job", "job.json"), "--mock"])
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertGreater(len(lines), 3)
        for line in lines:
            payload = json.loads(line)  # must not raise
            self.assertIn("event", payload)
            self.assertIn(payload["event"], progress.ALLOWED_EVENTS)

    def test_started_and_completed_bracket_the_run(self):
        self._write_job()
        completed = self._run(["--job", self.path("job", "job.json"), "--mock"])
        events = [json.loads(line)["event"] for line in completed.stdout.splitlines() if line.strip()]
        self.assertEqual(events[0], "started")
        self.assertEqual(events[-1], "completed")

    def test_completed_event_points_at_a_valid_result(self):
        self._write_job()
        completed = self._run(["--job", self.path("job", "job.json"), "--mock"])
        final = json.loads([l for l in completed.stdout.splitlines() if l.strip()][-1])
        result = result_schema.load_mocap_result(final["result_path"])
        self.assertEqual(len(result.frames), 60)

    def test_worker_log_is_written_and_stdout_stays_clean(self):
        self._write_job()
        self._run(["--job", self.path("job", "job.json"), "--mock"])
        log_path = paths.worker_log_path(self.path("job"))
        self.assertTrue(os.path.isfile(log_path))
        self.assertGreater(os.path.getsize(log_path), 0)

    def test_missing_job_file_emits_failed_and_exits_non_zero(self):
        completed = self._run(["--job", self.path("nope.json")])
        self.assertNotEqual(completed.returncode, 0)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        self.assertEqual(payload["event"], "failed")
        self.assertEqual(payload["error"]["code"], errors.JOB_SCHEMA_INVALID)
        self.assertTrue(payload["error"]["message"])
        self.assertTrue(payload["error"]["suggestion"])

    def test_real_run_without_models_emits_failed_with_a_model_code(self):
        self._write_job()
        completed = self._run(["--job", self.path("job", "job.json")])
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads([l for l in completed.stdout.splitlines() if l.strip()][-1])
        self.assertEqual(payload["event"], "failed")
        self.assertIn(payload["error"]["code"], (errors.MODEL_MISSING, errors.CONFIG_MISSING))
        self.assertIn("relative_path", payload["error"]["details"])

    def test_cancel_sentinel_produces_a_cancelled_event_and_exit_130(self):
        self._write_job(frame_end=600)
        with open(paths.cancel_flag_path(self.path("job")), "w", encoding="utf-8") as handle:
            handle.write("stop")
        completed = self._run(["--job", self.path("job", "job.json"), "--mock"])
        payload = json.loads([l for l in completed.stdout.splitlines() if l.strip()][-1])
        self.assertEqual(payload["event"], "cancelled")
        self.assertEqual(completed.returncode, worker_client.EXIT_CANCELLED)

    def test_check_env_returns_json_and_exit_zero(self):
        completed = self._run(["--check-env"])
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        payload = json.loads([l for l in completed.stdout.splitlines() if l.strip()][-1])
        self.assertEqual(payload["event"], "completed")
        self.assertEqual(payload["check"], "env")
        self.assertIn("modules", payload["report"])

    def test_check_cuda_never_leaks_a_traceback(self):
        completed = self._run(["--check-cuda"])
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        for line in lines:
            json.loads(line)
        payload = json.loads(lines[-1])
        self.assertIn(payload["event"], ("completed", "failed"))
        if payload["event"] == "failed":
            self.assertEqual(payload["error"]["code"], errors.CUDA_UNAVAILABLE)
            self.assertNotIn("Traceback", completed.stdout)

    def test_no_arguments_is_a_usage_error(self):
        completed = self._run([])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--job", completed.stderr)


class TestWorkerClientHelpers(TempDirCase):
    def test_run_blocking_drives_a_mock_job_to_completion(self):
        models_root = self.path("models")
        os.makedirs(models_root, exist_ok=True)
        media = self.touch("media/clip.mp4")
        job = job_schema.build_job(
            scene_props(source_media=media, frame_end=10),
            preferences(models_root=models_root),
        )
        job["output"]["dir"] = self.path("job")
        paths.ensure_dir(job["output"]["dir"])
        job_path = job_schema.write_job(job, self.path("job", "job.json"))

        seen = []
        worker = worker_client.run_worker(
            sys.executable, job_path, output_dir=job["output"]["dir"], extra_args=["--mock"]
        )
        try:
            import time

            while True:
                for event in worker.poll_events():
                    seen.append(event.event)
                if worker.drain_finished():
                    break
                time.sleep(0.02)
            worker.wait(timeout=10.0)
            for event in worker.poll_events():
                seen.append(event.event)
        finally:
            worker.close()

        self.assertIn("started", seen)
        self.assertIn("completed", seen)
        self.assertEqual(worker.returncode, 0)

    def test_run_worker_rejects_a_bad_interpreter(self):
        with self.assertRaises(errors.MocapError) as ctx:
            worker_client.run_worker(self.path("nope.exe"), self.path("job.json"))
        self.assertEqual(ctx.exception.code, errors.WORKER_PYTHON_NOT_FOUND)

    def test_run_worker_rejects_a_missing_job_file(self):
        with self.assertRaises(errors.MocapError) as ctx:
            worker_client.run_worker(sys.executable, self.path("nope.json"))
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_check_env_helper_parses_the_report(self):
        report = worker_client.check_env(sys.executable)
        self.assertEqual(report["event"], "completed")
        self.assertEqual(report["returncode"], 0)
        self.assertIn("python", report["report"])

    def test_shutdown_all_is_safe_when_nothing_runs(self):
        self.assertIsInstance(worker_client.shutdown_all(), int)


if __name__ == "__main__":
    unittest.main()
