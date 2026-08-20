"""Job construction and validation (guide sections 4.2 and 5.1)."""

from __future__ import annotations

import json
import os
import unittest

from ._util import TempDirCase, preferences, scene_props

from core import errors, job_schema, paths


class TestBuildJob(TempDirCase):
    def _prefs(self, **overrides):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        values = {"models_root": root}
        values.update(overrides)
        return preferences(**values)

    def _media(self, name: str = "clip.mp4") -> str:
        return self.touch("media/" + name)

    def test_minimal_job_has_every_documented_section(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media()), self._prefs(), blend_path=self.path("scene.blend")
        )
        for section in ("job_id", "mode", "input", "model", "options", "output"):
            self.assertIn(section, job)
        self.assertEqual(job["mode"], "capture")
        self.assertEqual(job["input"]["type"], "video")
        self.assertEqual(job["input"]["frame_start"], 1)
        self.assertEqual(job["input"]["frame_end"], 0)
        self.assertEqual(job["input"]["target_fps"], 30)
        self.assertEqual(job["options"]["single_person"], True)
        self.assertEqual(job["options"]["scale_mode"], "target_rig_height")
        self.assertEqual(job["output"]["result_filename"], "mocap_result.json")
        self.assertTrue(os.path.isdir(job["output"]["dir"]))

    def test_job_dir_lives_next_to_the_blend_file(self):
        blend = self.path("project", "scene.blend")
        os.makedirs(os.path.dirname(blend), exist_ok=True)
        job = job_schema.build_job(
            scene_props(source_media=self._media()), self._prefs(), blend_path=blend
        )
        self.assertIn(paths.JOB_DIR_NAME, job["output"]["dir"])
        self.assertTrue(job["output"]["dir"].startswith(os.path.dirname(blend)))

    def test_unsaved_file_falls_back_to_temp(self):
        job = job_schema.build_job(scene_props(source_media=self._media()), self._prefs())
        self.assertIn(paths.JOB_DIR_NAME, job["output"]["dir"])

    def test_auto_type_uses_the_extension(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media("still.png")), self._prefs()
        )
        self.assertEqual(job["input"]["type"], "image")

    def test_unknown_extension_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(scene_props(source_media=self._media("clip.xyz")), self._prefs())
        self.assertEqual(ctx.exception.code, errors.UNSUPPORTED_MEDIA_TYPE)

    def test_explicit_type_overrides_the_extension(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media("clip.xyz"), source_type="video"), self._prefs()
        )
        self.assertEqual(job["input"]["type"], "video")

    def test_missing_source_media_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(scene_props(), self._prefs())
        self.assertEqual(ctx.exception.code, errors.MEDIA_NOT_FOUND)

    def test_nonexistent_source_media_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(
                scene_props(source_media=self.path("missing.mp4")), self._prefs()
            )
        self.assertEqual(ctx.exception.code, errors.MEDIA_NOT_FOUND)

    def test_missing_models_root_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(
                scene_props(source_media=self._media()), preferences(models_root="")
            )
        self.assertEqual(ctx.exception.code, errors.MODELS_ROOT_NOT_FOUND)

    def test_invalid_target_fps_is_rejected(self):
        for fps in (0, -1, 121):
            with self.assertRaises(errors.MocapError) as ctx:
                job_schema.build_job(
                    scene_props(source_media=self._media(), target_fps=fps), self._prefs()
                )
            self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID, fps)

    def test_frame_end_before_frame_start_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(
                scene_props(source_media=self._media(), frame_start=10, frame_end=5), self._prefs()
            )
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_frame_end_zero_means_until_the_end(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media(), frame_start=10, frame_end=0), self._prefs()
        )
        self.assertEqual(job["input"]["frame_end"], 0)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(
                scene_props(source_media=self._media(), capture_profile="nope"), self._prefs()
            )
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_hand_enhanced_is_not_a_capture_profile(self):
        with self.assertRaises(errors.MocapError):
            job_schema.build_job(
                scene_props(source_media=self._media(), capture_profile="hand_enhanced"),
                self._prefs(),
            )

    def test_device_is_cpu_only_for_fallback_profile(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media(), capture_profile="quality"), self._prefs()
        )
        self.assertEqual(job["model"]["device"], "cuda:0")
        job = job_schema.build_job(
            scene_props(source_media=self._media(), capture_profile="fallback_cpu"), self._prefs()
        )
        self.assertEqual(job["model"]["device"], "cpu")

    def test_strengths_are_clamped(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media(), smoothing_strength=5.0, foot_lock_strength=-2.0),
            self._prefs(),
        )
        self.assertEqual(job["options"]["smoothing_strength"], 1.0)
        self.assertEqual(job["options"]["foot_lock_strength"], 0.0)

    def test_invalid_root_motion_falls_back_to_world(self):
        job = job_schema.build_job(
            scene_props(source_media=self._media(), root_motion="sideways"), self._prefs()
        )
        self.assertEqual(job["options"]["root_motion"], "world")

    def test_self_test_mode_does_not_need_media(self):
        job = job_schema.build_job(
            scene_props(), self._prefs(), mode=job_schema.MODE_SELF_TEST
        )
        self.assertEqual(job["mode"], "self_test")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.build_job(scene_props(), self._prefs(), mode="bogus")
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_resolve_path_hook_is_applied(self):
        media = self._media()
        calls = []

        def resolver(value):
            calls.append(value)
            return value

        job_schema.build_job(
            scene_props(source_media=media), self._prefs(), resolve_path=resolver
        )
        self.assertTrue(calls)

    def test_job_ids_are_unique(self):
        media = self._media()
        first = job_schema.build_job(scene_props(source_media=media), self._prefs())
        second = job_schema.build_job(scene_props(source_media=media), self._prefs())
        self.assertNotEqual(first["job_id"], second["job_id"])


class TestValidateJob(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "version": "0.1",
            "job_id": "abc",
            "mode": "capture",
            "input": {
                "path": "D:/x.mp4",
                "type": "video",
                "frame_start": 1,
                "frame_end": 0,
                "target_fps": 30,
            },
            "model": {
                "profile": "quality",
                "device": "cuda:0",
                "max_vram_gb": 7.0,
                "models_root": "D:/models",
            },
            "options": {
                "single_person": True,
                "include_hands": True,
                "smoothing_strength": 0.65,
                "foot_lock_strength": 0.7,
                "root_motion": "world",
                "scale_mode": "target_rig_height",
            },
            "output": {"dir": "D:/out", "result_filename": "mocap_result.json"},
        }

    def test_valid_job_passes(self):
        job_schema.validate_job(self._valid())

    def test_not_a_dict_is_rejected(self):
        with self.assertRaises(errors.MocapError):
            job_schema.validate_job(["not", "a", "dict"])

    def test_single_person_must_be_true_in_v0_1(self):
        job = self._valid()
        job["options"]["single_person"] = False
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.validate_job(job)
        self.assertIn("single_person", ctx.exception.message)

    def test_each_missing_section_is_reported(self):
        for section in ("input", "model", "options", "output"):
            job = self._valid()
            del job[section]
            with self.assertRaises(errors.MocapError) as ctx:
                job_schema.validate_job(job)
            self.assertIn(section, ctx.exception.message)

    def test_bad_device_is_rejected(self):
        job = self._valid()
        job["model"]["device"] = "tpu"
        with self.assertRaises(errors.MocapError):
            job_schema.validate_job(job)

    def test_missing_job_id_is_rejected(self):
        job = self._valid()
        job["job_id"] = ""
        with self.assertRaises(errors.MocapError):
            job_schema.validate_job(job)

    def test_problems_are_listed_in_details(self):
        job = self._valid()
        job["model"]["device"] = "tpu"
        job["options"]["root_motion"] = "nope"
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.validate_job(job)
        self.assertGreaterEqual(len(ctx.exception.details["problems"]), 2)


class TestJobIO(TempDirCase):
    def test_write_then_read_roundtrip(self):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        media = self.touch("media/clip.mp4")
        job = job_schema.build_job(
            scene_props(source_media=media), preferences(models_root=root)
        )
        job_path = job_schema.write_job(job)
        self.assertTrue(os.path.isfile(job_path))
        loaded = job_schema.read_job(job_path)
        self.assertEqual(loaded["job_id"], job["job_id"])
        self.assertEqual(job_schema.result_path(loaded), os.path.join(job["output"]["dir"], "mocap_result.json"))

    def test_read_job_with_bom(self):
        payload = {
            "job_id": "x",
            "mode": "capture",
            "input": {"path": "a.mp4", "type": "video", "frame_start": 1, "frame_end": 0, "target_fps": 30},
            "model": {"profile": "preview", "device": "cpu", "models_root": "m"},
            "options": {
                "single_person": True,
                "include_hands": False,
                "smoothing_strength": 0.5,
                "foot_lock_strength": 0.5,
                "root_motion": "world",
                "scale_mode": "target_rig_height",
            },
            "output": {"dir": "o"},
        }
        target = self.write_json("job.json", payload, encoding="utf-8-sig")
        loaded = job_schema.read_job(target)
        self.assertEqual(loaded["job_id"], "x")

    def test_read_missing_file_reports_job_schema_invalid(self):
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.read_job(self.path("nope.json"))
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_read_malformed_json_reports_job_schema_invalid(self):
        target = self.path("bad.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(errors.MocapError) as ctx:
            job_schema.read_job(target)
        self.assertEqual(ctx.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_written_job_is_utf8_json(self):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        media = self.touch("media/clip.mp4")
        job = job_schema.build_job(
            scene_props(source_media=media), preferences(models_root=root)
        )
        job_path = job_schema.write_job(job)
        with open(job_path, "r", encoding="utf-8") as handle:
            json.load(handle)


class TestWorkerPython(TempDirCase):
    def test_missing_worker_python_is_reported(self):
        for value in ("", self.path("nope.exe")):
            with self.assertRaises(errors.MocapError) as ctx:
                job_schema.validate_worker_python(value)
            self.assertEqual(ctx.exception.code, errors.WORKER_PYTHON_NOT_FOUND)

    def test_existing_interpreter_is_accepted(self):
        import sys

        self.assertTrue(job_schema.validate_worker_python(sys.executable))


if __name__ == "__main__":
    unittest.main()
