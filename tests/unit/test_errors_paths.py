"""Error objects and path helpers."""

from __future__ import annotations

import os
import unittest

from ._util import TempDirCase

from core import errors, paths


class TestErrorCodes(unittest.TestCase):
    def test_all_documented_codes_exist(self):
        # guide section 5.4
        required = (
            "WORKER_PYTHON_NOT_FOUND",
            "MODELS_ROOT_NOT_FOUND",
            "MANIFEST_INVALID",
            "MODEL_MISSING",
            "CONFIG_MISSING",
            "CUDA_UNAVAILABLE",
            "CUDA_OOM",
            "MEDIA_OPEN_FAILED",
            "NO_PERSON_DETECTED",
            "RIGIFY_NOT_FOUND",
            "RIGIFY_MAPPING_FAILED",
            "RESULT_SCHEMA_INVALID",
        )
        for name in required:
            self.assertTrue(hasattr(errors, name), name)
            self.assertEqual(getattr(errors, name), name)
            self.assertIn(name, errors.REQUIRED_ERROR_CODES)
        self.assertEqual(len(errors.REQUIRED_ERROR_CODES), 12)

    def test_every_code_has_a_default_suggestion(self):
        for code in errors.ALL_ERROR_CODES:
            self.assertTrue(errors.default_suggestion(code), code)

    def test_unknown_code_has_an_empty_suggestion(self):
        self.assertEqual(errors.default_suggestion("NOT_A_CODE"), "")


class TestMocapError(unittest.TestCase):
    def test_serialised_shape_matches_the_guide(self):
        error = errors.MocapError(
            errors.MODEL_MISSING,
            "Required model is missing: RTMPose-m Body.",
            details={"artifact_id": "rtmpose_m_body", "relative_path": "a/b.pth"},
        )
        payload = error.to_dict()
        self.assertEqual(
            sorted(payload), ["code", "details", "message", "recoverable", "suggestion"]
        )
        self.assertEqual(payload["code"], "MODEL_MISSING")
        self.assertTrue(payload["recoverable"])
        self.assertEqual(payload["details"]["artifact_id"], "rtmpose_m_body")

    def test_details_are_omitted_when_empty(self):
        payload = errors.MocapError(errors.CUDA_OOM, "oom").to_dict()
        self.assertNotIn("details", payload)

    def test_default_suggestion_is_filled_in(self):
        error = errors.MocapError(errors.CUDA_OOM, "oom")
        self.assertIn("quality", error.suggestion)

    def test_explicit_suggestion_wins(self):
        error = errors.MocapError(errors.CUDA_OOM, "oom", suggestion="custom")
        self.assertEqual(error.suggestion, "custom")

    def test_recoverability_defaults(self):
        self.assertTrue(errors.MocapError(errors.MODEL_MISSING, "m").recoverable)
        self.assertFalse(errors.MocapError(errors.INTERNAL_ERROR, "m").recoverable)
        self.assertFalse(errors.MocapError(errors.RESULT_SCHEMA_INVALID, "m").recoverable)

    def test_explicit_recoverability_wins(self):
        self.assertTrue(errors.MocapError(errors.INTERNAL_ERROR, "m", recoverable=True).recoverable)

    def test_round_trip_through_from_dict(self):
        original = errors.MocapError(
            errors.CUDA_OOM, "oom", suggestion="s", recoverable=True, details={"k": "v"}
        )
        restored = errors.MocapError.from_dict(original.to_dict())
        self.assertEqual(restored.code, original.code)
        self.assertEqual(restored.message, original.message)
        self.assertEqual(restored.suggestion, original.suggestion)
        self.assertEqual(restored.details, original.details)

    def test_from_dict_tolerates_garbage(self):
        error = errors.MocapError.from_dict("not a dict")
        self.assertEqual(error.code, errors.INTERNAL_ERROR)
        error = errors.MocapError.from_dict({})
        self.assertEqual(error.code, errors.INTERNAL_ERROR)

    def test_user_text_includes_code_and_suggestion(self):
        text = errors.MocapError(errors.CUDA_OOM, "oom").user_text()
        self.assertIn("CUDA_OOM", text)
        self.assertIn("oom", text)

    def test_it_is_a_real_exception(self):
        with self.assertRaises(errors.MocapError):
            raise errors.MocapError(errors.CUDA_OOM, "oom")

    def test_wrap_unexpected_passes_through_mocap_errors(self):
        original = errors.MocapError(errors.CUDA_OOM, "oom")
        self.assertIs(errors.wrap_unexpected(original), original)

    def test_wrap_unexpected_converts_other_exceptions(self):
        error = errors.wrap_unexpected(ValueError("bad value"), "stage")
        self.assertEqual(error.code, errors.INTERNAL_ERROR)
        self.assertIn("ValueError", error.message)
        self.assertIn("stage", error.message)
        self.assertEqual(error.details["context"], "stage")


class TestPaths(TempDirCase):
    def test_addon_root_contains_the_package_marker(self):
        root = paths.addon_root()
        self.assertTrue(os.path.isfile(os.path.join(root, "__init__.py")))
        self.assertEqual(os.path.basename(root), "motion_capture")

    def test_bundled_manifest_exists(self):
        self.assertTrue(os.path.isfile(paths.bundled_manifest_path()))

    def test_normalize_handles_empty_and_relative(self):
        self.assertEqual(paths.normalize(""), "")
        self.assertEqual(paths.normalize(None), "")
        self.assertTrue(os.path.isabs(paths.normalize(".")))

    def test_exists_helpers(self):
        target = self.touch("a/b.txt")
        self.assertTrue(paths.exists(target))
        self.assertTrue(paths.is_file(target))
        self.assertFalse(paths.is_dir(target))
        self.assertTrue(paths.is_dir(os.path.dirname(target)))
        self.assertFalse(paths.exists(""))
        self.assertFalse(paths.is_file(self.path("nope")))

    def test_media_type_detection(self):
        self.assertEqual(paths.guess_media_type("a.PNG"), "image")
        self.assertEqual(paths.guess_media_type("a.jpeg"), "image")
        self.assertEqual(paths.guess_media_type("a.webp"), "image")
        self.assertEqual(paths.guess_media_type("a.MP4"), "video")
        self.assertEqual(paths.guess_media_type("a.mkv"), "video")
        self.assertEqual(paths.guess_media_type("a.gif"), "unknown")
        self.assertEqual(paths.guess_media_type("a"), "unknown")

    def test_documented_extension_lists(self):
        self.assertEqual(set(paths.image_extensions()), {".png", ".jpg", ".jpeg", ".webp"})
        self.assertEqual(set(paths.video_extensions()), {".mp4", ".mov", ".avi", ".mkv"})

    def test_safe_name_strips_unsafe_characters(self):
        self.assertEqual(paths.safe_name("my clip (final)!"), "my_clip_final")
        self.assertEqual(paths.safe_name(""), "source")
        self.assertEqual(paths.safe_name("***"), "source")

    def test_source_stem(self):
        self.assertEqual(paths.source_stem("D:/media/My Walk 01.mp4"), "My_Walk_01")
        self.assertEqual(paths.source_stem(""), "source")

    def test_job_dir_layout(self):
        blend = self.path("project", "scene.blend")
        os.makedirs(os.path.dirname(blend), exist_ok=True)
        directory = paths.job_dir("abc-123", blend)
        self.assertTrue(directory.endswith(os.path.join(paths.JOB_DIR_NAME, "abc-123")))
        self.assertTrue(directory.startswith(os.path.dirname(blend)))

    def test_job_dir_falls_back_to_temp_when_unsaved(self):
        directory = paths.job_dir("abc-123", "")
        self.assertIn(paths.JOB_DIR_NAME, directory)

    def test_job_id_is_sanitised_into_the_directory_name(self):
        directory = paths.job_dir("../../evil", "")
        self.assertNotIn("..", os.path.basename(directory))

    def test_ensure_dir_is_idempotent(self):
        target = self.path("deep", "nested", "dir")
        self.assertEqual(paths.ensure_dir(target), paths.normalize(target))
        self.assertEqual(paths.ensure_dir(target), paths.normalize(target))
        self.assertTrue(os.path.isdir(target))

    def test_cancel_and_log_paths(self):
        directory = self.path("job")
        self.assertTrue(paths.cancel_flag_path(directory).endswith(paths.CANCEL_FILENAME))
        self.assertTrue(paths.worker_log_path(directory).endswith(paths.WORKER_LOG_FILENAME))

    def test_tail_text_file(self):
        target = self.touch("log.txt", "abcdefghij")
        self.assertEqual(paths.tail_text_file(target), "abcdefghij")
        self.assertTrue(paths.tail_text_file(target, 4).endswith("ghij"))
        self.assertEqual(paths.tail_text_file(self.path("nope.txt")), "")


if __name__ == "__main__":
    unittest.main()
