"""Manifest loading and per-profile preflight rules (guide sections 4.1 and 6)."""

from __future__ import annotations

import os
import unittest

from ._util import FIXTURES, TempDirCase

from core import errors, model_manifest, paths

#: Files a complete ``quality`` install needs, relative to the models root.
QUALITY_FILES = (
    "openmmlab/detectors/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    "openmmlab/configs/rtmdet_m_640-8xb32_coco-person.py",
    "openmmlab/body2d/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
    "openmmlab/configs/rtmpose-m_8xb256-420e_body8-256x192.py",
    "openmmlab/body3d/motionbert_ft_h36m-d80af323_20230531.pth",
    "openmmlab/configs/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py",
)

PREVIEW_FILE = "mediapipe/pose_landmarker_full.task"


class TestBundledManifest(unittest.TestCase):
    def test_bundled_example_exists_and_parses(self):
        manifest = model_manifest.load_manifest("")
        self.assertEqual(manifest.source_path, paths.bundled_manifest_path())
        self.assertEqual(manifest.schema_version, "0.1")
        self.assertTrue(manifest.manual_download_only)
        self.assertEqual(len(manifest.artifacts), 21)

    def test_every_artifact_has_the_required_fields(self):
        manifest = model_manifest.load_manifest("")
        for artifact in manifest.artifacts:
            for field in ("id", "display_name", "relative_path", "download_url", "required"):
                self.assertIn(field, artifact, "artifact {0} lacks {1}".format(artifact, field))

    def test_all_profiles_are_declared(self):
        manifest = model_manifest.load_manifest("")
        for profile in model_manifest.ALL_PROFILES:
            self.assertIn(profile, manifest.profiles, profile)

    def test_config_artifacts_report_config_missing(self):
        manifest = model_manifest.load_manifest("")
        status = manifest.artifact_status("config_rtmpose_m_body")
        self.assertEqual(status.kind, "config")
        self.assertEqual(status.error_code, errors.CONFIG_MISSING)
        status = manifest.artifact_status("rtmpose_m_body")
        self.assertEqual(status.error_code, errors.MODEL_MISSING)


class TestManifestErrors(TempDirCase):
    def test_broken_manifest_raises_manifest_invalid(self):
        os.makedirs(self.path("models"), exist_ok=True)
        with open(os.path.join(FIXTURES, "manifest_broken.json"), "r", encoding="utf-8") as src:
            broken = src.read()
        target = self.path("models", "manifest.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(broken)
        with self.assertRaises(errors.MocapError) as ctx:
            model_manifest.load_manifest(self.path("models"))
        self.assertEqual(ctx.exception.code, errors.MANIFEST_INVALID)

    def test_manifest_without_artifacts_is_invalid(self):
        self.write_json("models/manifest.json", {"schema_version": "0.1"})
        with self.assertRaises(errors.MocapError) as ctx:
            model_manifest.load_manifest(self.path("models"))
        self.assertEqual(ctx.exception.code, errors.MANIFEST_INVALID)

    def test_user_manifest_wins_over_bundled_example(self):
        self.write_json(
            "models/manifest.json",
            {
                "schema_version": "0.1",
                "artifacts": [
                    {
                        "id": "custom_model",
                        "kind": "weights",
                        "display_name": "Custom",
                        "required": True,
                        "profiles": ["preview"],
                        "relative_path": "custom/model.bin",
                        "download_url": "https://example.invalid/model.bin",
                    }
                ],
                "preflight_rules": {"preview": {"required_artifact_ids": ["custom_model"]}},
            },
        )
        manifest = model_manifest.load_manifest(self.path("models"))
        self.assertEqual(manifest.source_path, self.path("models", "manifest.json"))
        self.assertIsNotNone(manifest.artifact("custom_model"))

        report = model_manifest.check_profile_requirements("preview", self.path("models"))
        self.assertFalse(report.ok)
        self.assertEqual([s.id for s in report.missing_required], ["custom_model"])

        self.touch("models/custom/model.bin")
        report = model_manifest.check_profile_requirements("preview", self.path("models"))
        self.assertTrue(report.ok, report.warnings)

    def test_manifest_with_bom_is_accepted(self):
        # Notepad and PowerShell add a UTF-8 BOM; that must not break loading.
        self.write_json(
            "models/manifest.json",
            {"schema_version": "0.1", "artifacts": []},
            encoding="utf-8-sig",
        )
        manifest = model_manifest.load_manifest(self.path("models"))
        self.assertEqual(manifest.artifacts, [])


class TestPreflight(TempDirCase):
    def _models_root(self) -> str:
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        return root

    def test_empty_models_root_reports_models_root_not_found(self):
        report = model_manifest.check_profile_requirements("preview", "")
        self.assertFalse(report.ok)
        self.assertIsNotNone(report.error)
        self.assertEqual(report.error.code, errors.MODELS_ROOT_NOT_FOUND)

    def test_missing_models_root_directory_reports_models_root_not_found(self):
        report = model_manifest.check_profile_requirements("preview", self.path("nope"))
        self.assertEqual(report.error.code, errors.MODELS_ROOT_NOT_FOUND)

    def test_unknown_profile_is_rejected(self):
        report = model_manifest.check_profile_requirements("bogus", self._models_root())
        self.assertFalse(report.ok)
        self.assertEqual(report.error.code, errors.JOB_SCHEMA_INVALID)

    def test_preview_requires_only_mediapipe_pose_full(self):
        root = self._models_root()
        report = model_manifest.check_profile_requirements("preview", root)
        self.assertEqual([s.id for s in report.missing_required], ["mediapipe_pose_full"])

        self.touch("models/" + PREVIEW_FILE)
        report = model_manifest.check_profile_requirements("preview", root)
        self.assertTrue(report.ok, report.warnings)
        # MMPose models stay absent and must not block preview.
        self.assertNotIn("rtmpose_m_body", [s.id for s in report.missing_required])

    def test_preview_reports_optional_hand_model(self):
        root = self._models_root()
        self.touch("models/" + PREVIEW_FILE)
        report = model_manifest.check_profile_requirements("preview", root)
        self.assertIn("mediapipe_hand", [s.id for s in report.missing_optional])
        self.assertTrue(report.ok)

    def test_quality_requires_six_artifacts(self):
        root = self._models_root()
        report = model_manifest.check_profile_requirements("quality", root)
        missing = [s.id for s in report.missing_required]
        self.assertEqual(len(missing), 6, missing)
        self.assertEqual(
            set(missing),
            {
                "rtmdet_m_person",
                "config_rtmdet_m_person",
                "rtmpose_m_body",
                "config_rtmpose_m_body",
                "motionbert_body3d",
                "config_motionbert_body3d",
            },
        )
        for relative in QUALITY_FILES:
            self.touch("models/" + relative)
        report = model_manifest.check_profile_requirements("quality", root)
        self.assertTrue(report.ok, report.warnings)
        self.assertEqual(report.effective_profile, "quality")

    def test_quality_plus_falls_back_to_quality_when_rtmpose_x_missing(self):
        root = self._models_root()
        for relative in QUALITY_FILES:
            self.touch("models/" + relative)
        report = model_manifest.check_profile_requirements("quality_plus", root)
        self.assertEqual(report.effective_profile, "quality")
        self.assertTrue(report.ok, report.warnings)
        self.assertTrue(any("回退" in w for w in report.warnings), report.warnings)

    def test_quality_plus_stays_when_rtmpose_x_present(self):
        root = self._models_root()
        for relative in QUALITY_FILES:
            self.touch("models/" + relative)
        self.touch(
            "models/openmmlab/body2d/"
            "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth"
        )
        self.touch("models/openmmlab/configs/rtmpose-x_8xb256-700e_coco-384x288.py")
        report = model_manifest.check_profile_requirements("quality_plus", root)
        self.assertEqual(report.effective_profile, "quality_plus")
        self.assertTrue(report.ok, report.warnings)

    def test_fallback_cpu_accepts_either_lite_or_full(self):
        root = self._models_root()
        report = model_manifest.check_profile_requirements("fallback_cpu", root)
        self.assertFalse(report.ok)
        missing = [s.id for s in report.missing_required]
        self.assertEqual(set(missing), {"mediapipe_pose_lite", "mediapipe_pose_full"})

        # Guide section 6.2: prefer Lite, otherwise Full - either alone suffices.
        self.touch("models/mediapipe/pose_landmarker_lite.task")
        report = model_manifest.check_profile_requirements("fallback_cpu", root)
        self.assertTrue(report.ok, report.warnings)

    def test_fallback_cpu_accepts_full_alone(self):
        root = self._models_root()
        self.touch("models/" + PREVIEW_FILE)
        report = model_manifest.check_profile_requirements("fallback_cpu", root)
        self.assertTrue(report.ok, report.warnings)

    def test_hand_enhanced_has_no_required_artifacts(self):
        root = self._models_root()
        report = model_manifest.check_profile_requirements("hand_enhanced", root)
        self.assertEqual(report.missing_required, [])
        self.assertTrue(report.ok)
        self.assertTrue(report.missing_optional)


class TestMissingReport(TempDirCase):
    def test_missing_report_carries_every_documented_field(self):
        report = model_manifest.check_profile_requirements("quality", self.path("models") + "x")
        os.makedirs(self.path("models"), exist_ok=True)
        report = model_manifest.check_profile_requirements("quality", self.path("models"))
        self.assertTrue(report.missing_required)
        for status in report.missing_required:
            payload = status.to_dict()
            for field in ("id", "display_name", "relative_path", "download_url", "required"):
                self.assertIn(field, payload)
            self.assertTrue(payload["id"])
            self.assertTrue(payload["display_name"])
            self.assertTrue(payload["relative_path"])
            self.assertTrue(payload["download_url"].startswith("https://"))
            self.assertTrue(payload["required"])

    def test_clipboard_format_matches_guide_section_6_3(self):
        os.makedirs(self.path("models"), exist_ok=True)
        report = model_manifest.check_profile_requirements("quality", self.path("models"))
        text = model_manifest.format_missing_links(report)
        lines = text.splitlines()
        self.assertEqual(lines[0], "Missing required models:")
        self.assertTrue(lines[1].startswith("- "))
        self.assertTrue(lines[2].startswith("  Save to: models/"))
        self.assertTrue(lines[3].startswith("  URL: https://"))

    def test_clipboard_text_when_nothing_is_missing(self):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        self.touch("models/" + PREVIEW_FILE)
        report = model_manifest.check_profile_requirements("preview", root)
        text = model_manifest.format_missing_links(report)
        self.assertIn("Missing optional models:", text)

    def test_first_missing_error_uses_the_right_code(self):
        os.makedirs(self.path("models"), exist_ok=True)
        report = model_manifest.check_profile_requirements("quality", self.path("models"))
        error = model_manifest.first_missing_error(report)
        self.assertIn(error.code, (errors.MODEL_MISSING, errors.CONFIG_MISSING))
        self.assertIn("artifact_id", error.details)
        self.assertIn("relative_path", error.details)

    def test_require_artifact_raises_for_absent_file(self):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        with self.assertRaises(errors.MocapError) as ctx:
            model_manifest.require_artifact(root, "rtmpose_m_body")
        self.assertEqual(ctx.exception.code, errors.MODEL_MISSING)

    def test_require_artifact_returns_path_when_present(self):
        root = self.path("models")
        os.makedirs(root, exist_ok=True)
        expected = self.touch("models/" + PREVIEW_FILE)
        actual = model_manifest.require_artifact(root, "mediapipe_pose_full")
        self.assertEqual(os.path.normcase(actual), os.path.normcase(expected))


if __name__ == "__main__":
    unittest.main()
