"""Result schema validation, loading and coordinate sanity checks (section 5.3)."""

from __future__ import annotations

import copy
import unittest

from ._util import TempDirCase

from core import errors, result_schema, skeleton


def build_frame(number: int = 1, height: float = 1.0) -> dict:
    """A minimal valid frame: every required body joint, Z-up, ``.L`` at +X."""
    body = {
        "pelvis": [0.0, 0.0, height],
        "spine": [0.0, 0.01, height + 0.16],
        "chest": [0.0, 0.0, height + 0.42],
        "neck": [0.0, 0.0, height + 0.60],
        "head": [0.0, 0.0, height + 0.75],
    }
    for side, sign in (("L", 1.0), ("R", -1.0)):
        body["shoulder.{0}".format(side)] = [sign * 0.18, 0.0, height + 0.55]
        body["elbow.{0}".format(side)] = [sign * 0.45, 0.0, height + 0.39]
        body["wrist.{0}".format(side)] = [sign * 0.66, 0.0, height + 0.24]
        body["hip.{0}".format(side)] = [sign * 0.10, 0.0, height - 0.02]
        body["knee.{0}".format(side)] = [sign * 0.10, 0.0, height - 0.45]
        body["ankle.{0}".format(side)] = [sign * 0.10, 0.0, height - 0.90]
        body["toe.{0}".format(side)] = [sign * 0.10, -0.15, height - 0.97]
    return {
        "frame": number,
        "time": (number - 1) / 30.0,
        "body3d": body,
        "hands3d": {},
        "confidence": {"body_mean": 0.93},
        "contacts": {"foot.L": True, "foot.R": False},
    }


def build_result(frames: int = 2) -> dict:
    return {
        "version": "0.1",
        "source": {"path": "D:/media/walk.mp4", "type": "video", "profile": "quality"},
        "fps": 30,
        "coordinate_system": "blender_world",
        "unit": "meter",
        "skeleton": "mocap_standard_v0",
        "frames": [build_frame(i + 1) for i in range(frames)],
        "warnings": [],
    }


class TestValidateResult(unittest.TestCase):
    def test_valid_result_passes(self):
        result_schema.validate_result(build_result())

    def test_not_a_dict_is_rejected(self):
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result([1, 2, 3])
        self.assertEqual(ctx.exception.code, errors.RESULT_SCHEMA_INVALID)

    def test_wrong_version_is_rejected(self):
        payload = build_result()
        payload["version"] = "0.2"
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result(payload)
        self.assertIn("version", ctx.exception.message)

    def test_y_up_coordinate_system_string_is_rejected(self):
        payload = build_result()
        payload["coordinate_system"] = "y_up"
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result(payload)
        self.assertIn("coordinate_system", ctx.exception.message)

    def test_wrong_unit_is_rejected(self):
        payload = build_result()
        payload["unit"] = "centimeter"
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_wrong_skeleton_is_rejected(self):
        payload = build_result()
        payload["skeleton"] = "smpl"
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_non_positive_fps_is_rejected(self):
        for fps in (0, -30):
            payload = build_result()
            payload["fps"] = fps
            with self.assertRaises(errors.MocapError):
                result_schema.validate_result(payload)

    def test_empty_frames_is_rejected(self):
        payload = build_result()
        payload["frames"] = []
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_missing_required_joint_is_rejected(self):
        payload = build_result()
        del payload["frames"][0]["body3d"]["wrist.L"]
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result(payload)
        self.assertIn("wrist.L", ctx.exception.message)

    def test_optional_heel_joints_may_be_absent(self):
        payload = build_result()
        for joint in skeleton.OPTIONAL_BODY_JOINTS:
            self.assertNotIn(joint, payload["frames"][0]["body3d"])
        result_schema.validate_result(payload)

    def test_bad_vector_length_is_rejected(self):
        payload = build_result()
        payload["frames"][0]["body3d"]["pelvis"] = [0.0, 1.0]
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_non_numeric_component_is_rejected(self):
        payload = build_result()
        payload["frames"][0]["body3d"]["pelvis"] = [0.0, "up", 1.0]
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_nan_and_infinity_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            payload = build_result()
            payload["frames"][0]["body3d"]["pelvis"] = [0.0, bad, 1.0]
            with self.assertRaises(errors.MocapError):
                result_schema.validate_result(payload)

    def test_duplicate_frame_numbers_are_rejected(self):
        payload = build_result()
        payload["frames"][1]["frame"] = payload["frames"][0]["frame"]
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result(payload)
        self.assertIn("重复", ctx.exception.message)

    def test_non_integer_frame_number_is_rejected(self):
        payload = build_result()
        payload["frames"][0]["frame"] = 1.5
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_bad_hand_coordinates_are_rejected(self):
        payload = build_result()
        payload["frames"][0]["hands3d"] = {"index.01.L": [0.0, 0.0]}
        with self.assertRaises(errors.MocapError):
            result_schema.validate_result(payload)

    def test_problem_list_is_capped(self):
        payload = build_result(frames=40)
        for frame in payload["frames"]:
            del frame["body3d"]["head"]
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.validate_result(payload)
        self.assertLessEqual(len(ctx.exception.details["problems"]), 12)


class TestLoadResult(TempDirCase):
    def test_load_roundtrip(self):
        target = self.write_json("mocap_result.json", build_result(frames=3))
        result = result_schema.load_mocap_result(target)
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(result.fps, 30.0)
        self.assertEqual(result.profile, "quality")
        self.assertEqual(result.frame_start, 1)
        self.assertEqual(result.frame_end, 3)
        self.assertFalse(result.has_hands)

    def test_action_name_pattern(self):
        target = self.write_json("mocap_result.json", build_result())
        result = result_schema.load_mocap_result(target)
        self.assertEqual(result.action_name(), "Mocap_walk_quality")

    def test_action_name_sanitises_odd_source_names(self):
        payload = build_result()
        payload["source"]["path"] = "D:/media/my clip (final)!.mp4"
        target = self.write_json("mocap_result.json", payload)
        result = result_schema.load_mocap_result(target)
        self.assertEqual(result.action_name(), "Mocap_my_clip_final_quality")

    def test_missing_file_reports_result_schema_invalid(self):
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.load_mocap_result(self.path("nope.json"))
        self.assertEqual(ctx.exception.code, errors.RESULT_SCHEMA_INVALID)

    def test_malformed_json_reports_result_schema_invalid(self):
        target = self.path("bad.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("{oops")
        with self.assertRaises(errors.MocapError) as ctx:
            result_schema.load_mocap_result(target)
        self.assertEqual(ctx.exception.code, errors.RESULT_SCHEMA_INVALID)

    def test_bom_is_tolerated(self):
        target = self.write_json("mocap_result.json", build_result(), encoding="utf-8-sig")
        result = result_schema.load_mocap_result(target)
        self.assertEqual(len(result.frames), 2)

    def test_frame_accessors(self):
        payload = build_result()
        payload["frames"][0]["hands3d"] = {"index.01.L": [0.7, 0.0, 1.2]}
        payload["frames"][0]["confidence"]["wrist.L"] = 0.4
        target = self.write_json("mocap_result.json", payload)
        result = result_schema.load_mocap_result(target)
        frame = result.frames[0]
        self.assertIsNotNone(frame.joint("pelvis"))
        self.assertIsNotNone(frame.joint("index.01.L"))
        self.assertIsNone(frame.joint("nose"))
        self.assertAlmostEqual(frame.confidence_of("wrist.L"), 0.4)
        self.assertAlmostEqual(frame.confidence_of("elbow.L"), 0.93)  # falls back to body_mean
        self.assertTrue(result.has_hands)


class TestMirrorAndCoordinateChecks(TempDirCase):
    def _result(self, payload):
        target = self.write_json("mocap_result.json", payload)
        return result_schema.load_mocap_result(target)

    def test_correct_handedness_is_not_flagged(self):
        result = self._result(build_result(frames=5))
        self.assertFalse(result_schema.detect_mirror(result))

    def test_mirrored_data_is_detected(self):
        payload = build_result(frames=5)
        for frame in payload["frames"]:
            for name, value in frame["body3d"].items():
                frame["body3d"][name] = [-value[0], value[1], value[2]]
        result = self._result(payload)
        self.assertTrue(result_schema.detect_mirror(result))

    def test_single_noisy_frame_does_not_flip_the_verdict(self):
        payload = build_result(frames=9)
        frame = payload["frames"][3]
        frame["body3d"]["hip.L"] = [-0.1, 0.0, 0.98]
        frame["body3d"]["hip.R"] = [0.1, 0.0, 0.98]
        result = self._result(payload)
        self.assertFalse(result_schema.detect_mirror(result))

    def test_y_up_payload_is_detected(self):
        # Guide section 5.3's illustrative sample is Y-up; real results must be Z-up.
        payload = build_result(frames=4)
        for frame in payload["frames"]:
            for name, value in frame["body3d"].items():
                frame["body3d"][name] = [value[0], value[2], 0.0]
        result = self._result(payload)
        self.assertTrue(result_schema.looks_y_up(result))

    def test_z_up_payload_is_not_flagged(self):
        result = self._result(build_result(frames=4))
        self.assertFalse(result_schema.looks_y_up(result))

    def test_mirror_result_x_swaps_sides_and_negates_x(self):
        payload = build_result(frames=2)
        payload["frames"][0]["hands3d"] = {"index.01.L": [0.7, 0.0, 1.2]}
        payload["frames"][0]["confidence"]["wrist.L"] = 0.5
        original = copy.deepcopy(payload)
        result = self._result(payload)
        result_schema.mirror_result_x(result)
        frame = result.frames[0]
        expected_left = original["frames"][0]["body3d"]["hip.R"]
        self.assertAlmostEqual(frame.body3d["hip.L"][0], -expected_left[0])
        self.assertIn("index.01.R", frame.hands3d)
        self.assertIn("wrist.R", frame.confidence)
        self.assertFalse(result_schema.detect_mirror(result))


if __name__ == "__main__":
    unittest.main()
