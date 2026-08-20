"""Standard skeleton definition and source-model mappings (guide section 8)."""

from __future__ import annotations

import unittest

from core import skeleton
from core import retarget_math as rm


class TestJointNames(unittest.TestCase):
    def test_required_body_joints_match_the_documented_set(self):
        self.assertEqual(len(skeleton.BODY_JOINTS), 19)
        expected = {
            "pelvis", "spine", "chest", "neck", "head",
            "shoulder.L", "elbow.L", "wrist.L",
            "shoulder.R", "elbow.R", "wrist.R",
            "hip.L", "knee.L", "ankle.L", "toe.L",
            "hip.R", "knee.R", "ankle.R", "toe.R",
        }
        self.assertEqual(set(skeleton.BODY_JOINTS), expected)

    def test_heels_are_optional(self):
        self.assertEqual(set(skeleton.OPTIONAL_BODY_JOINTS), {"heel.L", "heel.R"})
        for joint in skeleton.OPTIONAL_BODY_JOINTS:
            self.assertNotIn(joint, skeleton.BODY_JOINTS)

    def test_no_duplicate_joint_names(self):
        self.assertEqual(len(set(skeleton.ALL_BODY_JOINTS)), len(skeleton.ALL_BODY_JOINTS))
        self.assertEqual(len(set(skeleton.HAND_JOINTS)), len(skeleton.HAND_JOINTS))

    def test_hand_joints_cover_five_fingers_per_side(self):
        self.assertEqual(len(skeleton.HAND_JOINTS), 40)
        for side in ("L", "R"):
            joints = skeleton.hand_joints(side)
            self.assertEqual(len(joints), 20)
            for finger, _rig in skeleton.FINGERS:
                for segment in skeleton.FINGER_SEGMENTS:
                    self.assertIn("{0}.{1}.{2}".format(finger, segment, side), joints)

    def test_side_helpers(self):
        self.assertTrue(skeleton.is_left("wrist.L"))
        self.assertFalse(skeleton.is_left("wrist.R"))
        self.assertTrue(skeleton.is_right("wrist.R"))
        self.assertEqual(skeleton.mirror_joint("wrist.L"), "wrist.R")
        self.assertEqual(skeleton.mirror_joint("wrist.R"), "wrist.L")
        self.assertEqual(skeleton.mirror_joint("pelvis"), "pelvis")


class TestConstants(unittest.TestCase):
    def test_documented_identifiers(self):
        self.assertEqual(skeleton.SKELETON_ID, "mocap_standard_v0")
        self.assertEqual(skeleton.COORDINATE_SYSTEM, "blender_world")
        self.assertEqual(skeleton.UNIT, "meter")

    def test_confidence_thresholds_match_guide_section_8_3(self):
        self.assertEqual(skeleton.CONFIDENCE_GOOD, 0.7)
        self.assertEqual(skeleton.CONFIDENCE_LOW, 0.4)
        self.assertEqual(skeleton.CONFIDENCE_CONTACT, 0.5)


class TestRetargetChains(unittest.TestCase):
    def test_every_chain_has_a_unique_target(self):
        targets = [chain.target for chain in skeleton.RETARGET_CHAINS]
        self.assertEqual(len(targets), len(set(targets)))

    def test_chain_sources_reference_known_joints(self):
        known = set(skeleton.ALL_BODY_JOINTS) | set(skeleton.HAND_JOINTS)
        for chain in skeleton.RETARGET_CHAINS:
            if chain.source is None:
                continue
            start, end = chain.source
            self.assertIn(start, known, chain.target)
            if end is not None:
                self.assertIn(end, known, chain.target)

    def test_left_chains_use_left_sources_only(self):
        for chain in skeleton.RETARGET_CHAINS:
            if not chain.target.endswith(".L") or chain.source is None:
                continue
            for joint in chain.source:
                if joint is None or "." not in joint:
                    continue
                self.assertFalse(
                    joint.endswith(".R"),
                    "left target {0} pulls from right joint {1}".format(chain.target, joint),
                )

    def test_right_chains_use_right_sources_only(self):
        for chain in skeleton.RETARGET_CHAINS:
            if not chain.target.endswith(".R") or chain.source is None:
                continue
            for joint in chain.source:
                if joint is None or "." not in joint:
                    continue
                self.assertFalse(
                    joint.endswith(".L"),
                    "right target {0} pulls from left joint {1}".format(chain.target, joint),
                )

    def test_left_and_right_chains_are_symmetric(self):
        left = {c.target for c in skeleton.RETARGET_CHAINS if c.target.endswith(".L")}
        right = {c.target for c in skeleton.RETARGET_CHAINS if c.target.endswith(".R")}
        self.assertEqual({name[:-2] for name in left}, {name[:-2] for name in right})

    def test_body_and_hand_groups(self):
        body = [c for c in skeleton.RETARGET_CHAINS if c.group == "body"]
        hands = [c for c in skeleton.RETARGET_CHAINS if c.group == "hand"]
        self.assertEqual(len(hands), 30)  # 5 fingers x 3 segments x 2 sides
        self.assertEqual(len(body) + len(hands), len(skeleton.RETARGET_CHAINS))
        self.assertTrue(all(not c.required for c in hands))

    def test_reference_modes_name_a_reference_axis(self):
        for chain in skeleton.RETARGET_CHAINS:
            if chain.mode == skeleton.MODE_AIM_REF:
                self.assertIn(chain.ref, (skeleton.REF_HIP_LINE, skeleton.REF_SHOULDER_LINE))

    def test_hips_and_chest_are_not_aim_targets(self):
        # Their rest direction is +Y horizontal, so aiming them would be wrong.
        primary_targets = {chain.target for chain in skeleton.RETARGET_CHAINS}
        self.assertNotIn("hips", primary_targets)

    def test_toe_alias_covers_older_rigify_naming(self):
        chain = next(c for c in skeleton.RETARGET_CHAINS if c.target == "toe_fk.L")
        self.assertIn("toe.L", chain.candidates)
        self.assertEqual(chain.mode, skeleton.MODE_IDENTITY)

    def test_root_chain_drives_the_torso(self):
        chain = next(c for c in skeleton.RETARGET_CHAINS if c.mode == skeleton.MODE_ROOT)
        self.assertEqual(chain.target, "torso")
        self.assertIn("root", chain.candidates)

    def test_optional_chains_are_marked(self):
        optional = {c.target for c in skeleton.RETARGET_CHAINS if not c.required}
        self.assertIn("hand_fk.L", optional)
        self.assertIn("hand_fk.R", optional)

    def test_required_body_chains_cover_the_documented_mapping(self):
        required = {c.target for c in skeleton.RETARGET_CHAINS if c.required}
        for name in (
            "torso", "spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003",
            "neck", "head", "shoulder.L", "upper_arm_fk.L", "forearm_fk.L",
            "thigh_fk.L", "shin_fk.L", "foot_fk.L", "toe_fk.L",
        ):
            self.assertIn(name, required, name)


class TestMediaPipeMapping(unittest.TestCase):
    def test_landmark_indices_are_in_range(self):
        for index in skeleton.MEDIAPIPE_POSE_MAP:
            self.assertTrue(0 <= index <= 32, index)
        for indices, _offset in skeleton.MEDIAPIPE_DERIVED.values():
            for index in indices:
                self.assertTrue(0 <= index <= 32, index)

    def test_mapped_joints_exist_in_the_standard_skeleton(self):
        for joint in skeleton.MEDIAPIPE_POSE_MAP.values():
            self.assertIn(joint, skeleton.ALL_BODY_JOINTS, joint)
        for joint in skeleton.MEDIAPIPE_DERIVED:
            self.assertIn(joint, skeleton.ALL_BODY_JOINTS, joint)

    def test_left_landmarks_map_to_left_joints(self):
        # MediaPipe indices 11/23/25/27 are the subject's left.
        for index, joint in ((11, "shoulder.L"), (23, "hip.L"), (25, "knee.L"), (27, "ankle.L")):
            self.assertEqual(skeleton.MEDIAPIPE_POSE_MAP[index], joint)
        for index, joint in ((12, "shoulder.R"), (24, "hip.R"), (26, "knee.R"), (28, "ankle.R")):
            self.assertEqual(skeleton.MEDIAPIPE_POSE_MAP[index], joint)

    def test_axis_conversion_puts_up_on_z(self):
        # MediaPipe: x right, y DOWN, z towards camera.
        self.assertEqual(skeleton.mediapipe_to_blender(0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(skeleton.mediapipe_to_blender(1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(skeleton.mediapipe_to_blender(0.0, 0.0, 1.0), (0.0, 1.0, 0.0))

    def test_subject_left_ends_up_on_positive_x(self):
        # A subject facing the camera has their own left on the image right (+x),
        # which must map to +X so Rigify ".L" bones line up.
        left = skeleton.mediapipe_to_blender(0.2, -1.0, 0.0)
        right = skeleton.mediapipe_to_blender(-0.2, -1.0, 0.0)
        self.assertGreater(left[0], right[0])

    def test_hand_landmark_map_covers_twenty_joints(self):
        self.assertEqual(len(skeleton.MEDIAPIPE_HAND_MAP), 20)
        suffixes = set(skeleton.MEDIAPIPE_HAND_MAP.values())
        for finger, _rig in skeleton.FINGERS:
            for segment in skeleton.FINGER_SEGMENTS:
                self.assertIn("{0}.{1}".format(finger, segment), suffixes)

    def test_spine_fractions_are_ordered(self):
        self.assertLess(skeleton.MEDIAPIPE_SPINE_FRACTION, skeleton.MEDIAPIPE_CHEST_FRACTION)
        self.assertTrue(0.0 < skeleton.MEDIAPIPE_SPINE_FRACTION < 1.0)
        self.assertTrue(0.0 < skeleton.MEDIAPIPE_CHEST_FRACTION < 1.0)


class TestH36MMapping(unittest.TestCase):
    def test_seventeen_joints_are_mapped(self):
        self.assertEqual(len(skeleton.H36M_17_MAP), 17)
        self.assertEqual(sorted(skeleton.H36M_17_MAP), list(range(17)))

    def test_mapped_joints_exist_in_the_standard_skeleton(self):
        for joint in skeleton.H36M_17_MAP.values():
            self.assertIn(joint, skeleton.ALL_BODY_JOINTS, joint)

    def test_no_duplicate_targets(self):
        values = list(skeleton.H36M_17_MAP.values())
        self.assertEqual(len(values), len(set(values)))

    def test_left_right_indices_are_not_swapped(self):
        self.assertEqual(skeleton.H36M_17_MAP[1], "hip.R")
        self.assertEqual(skeleton.H36M_17_MAP[4], "hip.L")
        self.assertEqual(skeleton.H36M_17_MAP[11], "shoulder.L")
        self.assertEqual(skeleton.H36M_17_MAP[14], "shoulder.R")

    def test_toes_are_declared_as_synthesised(self):
        for joint in skeleton.H36M_SYNTHESISED_JOINTS:
            self.assertNotIn(joint, skeleton.H36M_17_MAP.values())
            self.assertIn(joint, skeleton.BODY_JOINTS)
        self.assertLess(skeleton.SYNTHESISED_CONFIDENCE, skeleton.CONFIDENCE_LOW)

    def test_axis_conversion_puts_up_on_z(self):
        self.assertEqual(skeleton.h36m_to_blender(0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(skeleton.h36m_to_blender(1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


class TestRigifyConstants(unittest.TestCase):
    def test_ik_fk_switch_bones(self):
        self.assertEqual(
            set(skeleton.IK_FK_SWITCH_BONES),
            {"upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R"},
        )
        self.assertEqual(skeleton.IK_FK_PROPERTY, "IK_FK")

    def test_signature_bones_are_control_bones_not_deform_bones(self):
        for name in skeleton.RIGIFY_SIGNATURE_BONES:
            self.assertFalse(name.startswith("DEF-"), name)
            self.assertFalse(name.startswith("ORG-"), name)
            self.assertFalse(name.startswith("MCH-"), name)

    def test_no_chain_targets_a_deform_bone(self):
        for chain in skeleton.RETARGET_CHAINS:
            for candidate in chain.candidates:
                self.assertFalse(candidate.startswith("DEF-"), candidate)
                self.assertFalse(candidate.startswith("ORG-"), candidate)
                self.assertFalse(candidate.startswith("MCH-"), candidate)

    def test_bone_axis_is_y(self):
        self.assertEqual(rm.BONE_AXIS, rm.AXIS_Y)


if __name__ == "__main__":
    unittest.main()
