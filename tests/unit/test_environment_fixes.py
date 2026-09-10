"""Regression tests for routing, process isolation and invalid result metadata."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core import errors, result_schema, worker_client, worker_environment
from backend_worker import export_result, mock_source


class EnvironmentFixes(unittest.TestCase):
    def test_explicit_paths_route_by_profile(self):
        for profile in ("preview", "fallback_cpu"):
            self.assertEqual(worker_environment.select_python("quality.exe", "preview.exe", profile),
                             os.path.abspath("preview.exe"))
        for profile in ("quality", "quality_plus"):
            self.assertEqual(worker_environment.select_python("quality.exe", "preview.exe", profile),
                             os.path.abspath("quality.exe"))

    def test_discovery_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(worker_environment.select_python("", "", "preview", directory), "")
            self.assertEqual(worker_environment.select_python(sys.executable, "", "preview", directory), sys.executable)
            preview = Path(directory) / ".venv-preview/Scripts/python.exe"
            preview.parent.mkdir(parents=True)
            preview.touch()
            self.assertEqual(worker_environment.select_python(sys.executable, "", "preview", directory), str(preview))

    def test_parent_python_paths_are_not_inherited(self):
        with patch.dict(os.environ, {"PYTHONHOME": "wrong-home", "PYTHONPATH": "wrong-modules"}):
            env = worker_client._popen_kwargs()["env"]
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("PYTHONPATH", env)
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(os.environ["PYTHONHOME"], "wrong-home")

    def test_non_finite_fps_is_rejected(self):
        for fps in (float("nan"), float("inf"), -float("inf"), True):
            data = export_result.build_result(mock_source.generate_frames(frame_start=1, frame_end=1), 30.0)
            data["fps"] = fps
            with self.subTest(fps=fps), self.assertRaises(errors.MocapError):
                result_schema.validate_result(data)

    def test_direct_cli_from_another_directory(self):
        script = Path(__file__).resolve().parents[2] / "backend_worker/cli.py"
        with tempfile.TemporaryDirectory() as directory:
            proc = subprocess.run([sys.executable, str(script), "--check-env"], cwd=directory,
                                  capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(events[-1]["event"], "completed")

    def test_profile_probe_failure_is_nonzero_and_json(self):
        from backend_worker import cli
        from io import StringIO
        failure = errors.MocapError(errors.DEPENDENCY_MISSING, "missing").to_dict()
        output = StringIO()
        with patch("sys.stdout", output), patch.object(cli.pipeline, "check_env", return_value={"ok": False, "error": failure}):
            self.assertEqual(cli.main(["--check-env", "--profile", "quality"]), 1)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["event"], "failed")

    def test_runtime_fallback_exports_actual_profile(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from core import job_schema
        from backend_worker import pipeline
        from tests.unit._util import scene_props, preferences
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "image.png"
            media.touch()
            job = job_schema.build_job(scene_props(source_media=str(media), capture_profile="quality_plus"),
                                       preferences(models_root=directory))
            job["output"]["dir"] = directory
            report = SimpleNamespace(warnings=[], effective_profile="quality_plus", missing_required=[])
            frames = mock_source.generate_frames(frame_start=1, frame_end=2)
            with patch.object(pipeline.model_manifest, "load_manifest", return_value={}), \
                    patch.object(pipeline.model_manifest, "check_profile_requirements", return_value=report), \
                    patch("backend_worker.environment.validate_environment", return_value={"ok": True}), \
                    patch.object(pipeline, "_run_mmpose", return_value=(frames, 30., [], "quality")):
                output = pipeline.run_job(job, Mock())
            self.assertEqual(result_schema.load_mocap_result(output).profile, "quality")


if __name__ == "__main__":
    unittest.main()
