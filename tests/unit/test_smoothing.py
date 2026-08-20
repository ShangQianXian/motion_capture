"""Smoothing, gap interpolation, contact detection and foot locking (section 9)."""

from __future__ import annotations

import math
import unittest

from core import smoothing
from core import retarget_math as rm


class TestAlpha(unittest.TestCase):
    def test_documented_formula(self):
        # alpha = 1.0 - clamp(strength, 0.0, 0.95)   (guide section 9.1)
        self.assertAlmostEqual(smoothing.alpha_from_strength(0.0), 1.0)
        self.assertAlmostEqual(smoothing.alpha_from_strength(0.65), 0.35)
        self.assertAlmostEqual(smoothing.alpha_from_strength(0.95), 0.05)

    def test_strength_is_clamped_so_the_signal_never_freezes(self):
        self.assertAlmostEqual(smoothing.alpha_from_strength(1.0), 0.05)
        self.assertAlmostEqual(smoothing.alpha_from_strength(7.0), 0.05)
        self.assertAlmostEqual(smoothing.alpha_from_strength(-3.0), 1.0)

    def test_garbage_strength_is_treated_as_zero(self):
        self.assertAlmostEqual(smoothing.alpha_from_strength("nonsense"), 1.0)
        self.assertAlmostEqual(smoothing.alpha_from_strength(None), 1.0)


class TestLowpass(unittest.TestCase):
    def test_alpha_one_is_a_passthrough(self):
        track = [(float(i), 0.0, 0.0) for i in range(5)]
        self.assertEqual(smoothing.lowpass_track(track, 1.0), track)

    def test_recurrence_matches_the_documented_formula(self):
        track = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        alpha = 0.5
        out = smoothing.lowpass_track(track, alpha)
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[1][0], 0.5)
        self.assertAlmostEqual(out[2][0], 0.75)

    def test_smoothing_reduces_high_frequency_noise(self):
        raw = [(math.sin(i) * (1.0 if i % 2 else -1.0), 0.0, 0.0) for i in range(60)]
        smoothed = smoothing.lowpass_track(raw, 0.2)
        raw_energy = sum(abs(raw[i][0] - raw[i - 1][0]) for i in range(1, len(raw)))
        smooth_energy = sum(abs(smoothed[i][0] - smoothed[i - 1][0]) for i in range(1, len(smoothed)))
        self.assertLess(smooth_energy, raw_energy * 0.5)

    def test_none_entries_pass_through(self):
        track = [(0.0, 0.0, 0.0), None, (2.0, 0.0, 0.0)]
        out = smoothing.lowpass_track(track, 0.5)
        self.assertIsNone(out[1])

    def test_confidence_weights_reduce_the_pull_of_bad_samples(self):
        track = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        strong = smoothing.lowpass_track(track, 0.5, weights=[1.0, 1.0])
        weak = smoothing.lowpass_track(track, 0.5, weights=[1.0, 0.1])
        self.assertLess(weak[1][0], strong[1][0])

    def test_empty_track(self):
        self.assertEqual(smoothing.lowpass_track([], 0.5), [])


class TestQuaternionSmoothing(unittest.TestCase):
    def test_output_stays_normalised(self):
        track = [rm.quat_from_axis_angle((0, 0, 1), i * 0.2) for i in range(10)]
        for quaternion in smoothing.smooth_quaternion_track(track, 0.4):
            norm = sum(component * component for component in quaternion) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_sign_flipped_input_does_not_take_the_long_way(self):
        a = rm.quat_from_axis_angle((0, 0, 1), 0.2)
        b = tuple(-component for component in rm.quat_from_axis_angle((0, 0, 1), 0.3))
        smoothed = smoothing.smooth_quaternion_track([a, b], 0.5)
        # The blend must stay near the 0.2 - 0.3 rad range, not swing to ~pi.
        self.assertLess(rm.quat_angle(smoothed[1]), 0.6)

    def test_alpha_one_passes_rotations_through(self):
        track = [rm.quat_from_axis_angle((0, 1, 0), 0.4)]
        out = smoothing.smooth_quaternion_track(track, 1.0)
        self.assertAlmostEqual(rm.quat_angle(out[0]), 0.4, places=9)

    def test_none_entries_pass_through(self):
        out = smoothing.smooth_quaternion_track([rm.IDENTITY_QUAT, None], 0.5)
        self.assertIsNone(out[1])


class TestInterpolateTrack(unittest.TestCase):
    def test_short_gap_is_linearly_interpolated(self):
        track = [(0.0, 0.0, 0.0), None, None, (3.0, 0.0, 0.0)]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertAlmostEqual(filled[1][0], 1.0)
        self.assertAlmostEqual(filled[2][0], 2.0)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["length"], 2)
        self.assertFalse(gaps[0]["held"])

    def test_gap_of_exactly_ten_frames_is_interpolated(self):
        track = [(0.0, 0.0, 0.0)] + [None] * 10 + [(11.0, 0.0, 0.0)]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertFalse(gaps[0]["held"])
        self.assertAlmostEqual(filled[5][0], 5.0)

    def test_gap_longer_than_ten_frames_holds_the_last_trusted_pose(self):
        track = [(0.0, 0.0, 0.0)] + [None] * 11 + [(12.0, 0.0, 0.0)]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertTrue(gaps[0]["held"])
        self.assertEqual(gaps[0]["length"], 11)
        for index in range(1, 12):
            self.assertEqual(filled[index], (0.0, 0.0, 0.0))

    def test_low_confidence_counts_as_missing(self):
        track = [(0.0, 0.0, 0.0), (99.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        confidences = [0.9, 0.1, 0.9]
        filled, gaps = smoothing.interpolate_track(track, confidences, 0.4, 10)
        self.assertAlmostEqual(filled[1][0], 1.0)
        self.assertEqual(len(gaps), 1)

    def test_confidence_at_the_threshold_is_kept(self):
        track = [(0.0, 0.0, 0.0), (99.0, 0.0, 0.0)]
        filled, gaps = smoothing.interpolate_track(track, [0.9, 0.4], 0.4, 10)
        self.assertEqual(filled[1][0], 99.0)
        self.assertEqual(gaps, [])

    def test_leading_gap_copies_the_first_valid_sample(self):
        track = [None, None, (5.0, 0.0, 0.0)]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertEqual(filled[0], (5.0, 0.0, 0.0))
        self.assertEqual(filled[1], (5.0, 0.0, 0.0))
        self.assertEqual(len(gaps), 1)

    def test_trailing_gap_copies_the_last_valid_sample(self):
        track = [(5.0, 0.0, 0.0), None, None]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertEqual(filled[2], (5.0, 0.0, 0.0))
        self.assertTrue(gaps[0]["held"])

    def test_all_missing_track_is_returned_unchanged(self):
        track = [None, None, None]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertEqual(filled, track)
        self.assertEqual(gaps, [])

    def test_no_gaps_reports_nothing(self):
        track = [(float(i), 0.0, 0.0) for i in range(4)]
        filled, gaps = smoothing.interpolate_track(track, None, 0.4, 10)
        self.assertEqual(filled, track)
        self.assertEqual(gaps, [])


class TestPercentileAndGround(unittest.TestCase):
    def test_percentile_bounds(self):
        data = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(smoothing.percentile(data, 0.0), 0.0)
        self.assertAlmostEqual(smoothing.percentile(data, 1.0), 4.0)
        self.assertAlmostEqual(smoothing.percentile(data, 0.5), 2.0)

    def test_percentile_of_empty_and_single(self):
        self.assertEqual(smoothing.percentile([], 0.5), 0.0)
        self.assertEqual(smoothing.percentile([7.0], 0.5), 7.0)

    def test_ground_estimate_uses_the_low_percentile(self):
        track = [(0.0, 0.0, 0.02)] * 10 + [(0.0, 0.0, 0.5)] * 10
        self.assertLess(smoothing.estimate_ground_z([track]), 0.1)

    def test_ground_estimate_of_empty_tracks(self):
        self.assertEqual(smoothing.estimate_ground_z([[]]), 0.0)


class TestContactDetection(unittest.TestCase):
    def test_planted_slow_foot_is_in_contact(self):
        track = [(0.0, 0.0, 0.0)] * 10
        contacts = smoothing.detect_contacts(track, fps=30.0)
        self.assertTrue(all(contacts))

    def test_lifted_foot_is_not_in_contact(self):
        track = [(0.0, 0.0, 0.0)] * 5 + [(0.0, 0.0, 0.4)] * 5
        contacts = smoothing.detect_contacts(track, fps=30.0)
        self.assertEqual(contacts[:5], [True] * 5)
        self.assertEqual(contacts[5:], [False] * 5)

    def test_fast_horizontal_movement_breaks_contact(self):
        track = [(i * 0.1, 0.0, 0.0) for i in range(10)]  # 3 m/s at 30 fps
        contacts = smoothing.detect_contacts(track, fps=30.0)
        self.assertFalse(any(contacts[1:]))

    def test_low_confidence_breaks_contact(self):
        track = [(0.0, 0.0, 0.0)] * 4
        contacts = smoothing.detect_contacts(
            track, confidences=[0.9, 0.2, 0.9, 0.9], fps=30.0
        )
        self.assertEqual(contacts, [True, False, True, True])

    def test_toe_track_is_used_for_the_lowest_point(self):
        # Ankle sits above the tolerance, but the toe is planted.
        ankle = [(0.0, 0.0, 0.10)] * 6
        toe = [(0.0, -0.15, 0.0)] * 6
        self.assertTrue(all(smoothing.detect_contacts(ankle, toe, fps=30.0)))

    def test_missing_samples_are_not_in_contact(self):
        track = [(0.0, 0.0, 0.0), None, (0.0, 0.0, 0.0)]
        contacts = smoothing.detect_contacts(track, fps=30.0)
        self.assertFalse(contacts[1])

    def test_contact_segments(self):
        self.assertEqual(
            smoothing.contact_segments([False, True, True, False, True]),
            [(1, 2), (4, 4)],
        )
        self.assertEqual(smoothing.contact_segments([True, True]), [(0, 1)])
        self.assertEqual(smoothing.contact_segments([False, False]), [])
        self.assertEqual(smoothing.contact_segments([]), [])


class TestFootLock(unittest.TestCase):
    def _sliding(self, count: int = 30, speed: float = 0.01):
        track = [(0.0, i * speed, 0.0) for i in range(count)]
        contacts = [4 <= i <= 27 for i in range(count)]
        return track, contacts

    def test_offsets_are_horizontal_only(self):
        track, contacts = self._sliding()
        offsets = smoothing.foot_lock_offsets([track], [contacts], strength=1.0)[0]
        for offset in offsets:
            self.assertEqual(offset[2], 0.0)

    def test_locking_removes_most_of_the_slide(self):
        track, contacts = self._sliding()
        offsets = smoothing.foot_lock_offsets([track], [contacts], strength=1.0)[0]
        corrected = [rm.vec_add(track[i], offsets[i]) for i in range(len(track))]

        def mean_slide(values):
            steps = [
                abs(values[i][1] - values[i - 1][1]) for i in range(5, 28)
            ]
            return sum(steps) / len(steps)

        self.assertLess(mean_slide(corrected), mean_slide(track) * 0.25)

    def test_steady_state_slide_is_essentially_zero(self):
        track, contacts = self._sliding()
        offsets = smoothing.foot_lock_offsets([track], [contacts], strength=1.0)[0]
        corrected = [rm.vec_add(track[i], offsets[i]) for i in range(len(track))]
        tail = [abs(corrected[i][1] - corrected[i - 1][1]) for i in range(16, 28)]
        self.assertLess(max(tail), 1e-3)

    def test_zero_strength_is_a_no_op(self):
        track, contacts = self._sliding()
        offsets = smoothing.foot_lock_offsets([track], [contacts], strength=0.0)[0]
        self.assertTrue(all(offset == (0.0, 0.0, 0.0) for offset in offsets))

    def test_no_correction_outside_contact(self):
        track, contacts = self._sliding()
        offsets = smoothing.foot_lock_offsets([track], [contacts], strength=1.0)[0]
        self.assertEqual(offsets[0], (0.0, 0.0, 0.0))

    def test_each_foot_gets_its_own_track(self):
        left = [(0.0, i * 0.01, 0.0) for i in range(10)]
        right = [(0.0, 0.0, 0.0)] * 10
        results = smoothing.foot_lock_offsets(
            [left, right], [[True] * 10, [True] * 10], strength=1.0
        )
        self.assertEqual(len(results), 2)
        self.assertGreater(abs(results[0][5][1]), 1e-4)
        self.assertAlmostEqual(results[1][5][1], 0.0)

    def test_strength_scales_the_correction(self):
        track, contacts = self._sliding()
        full = smoothing.foot_lock_offsets([track], [contacts], strength=1.0)[0]
        half = smoothing.foot_lock_offsets([track], [contacts], strength=0.5)[0]
        self.assertLess(abs(half[20][1]), abs(full[20][1]))


class TestGroundCorrection(unittest.TestCase):
    def test_penetrating_foot_is_lifted(self):
        track = [(0.0, 0.0, -0.05)] * 20
        offsets = smoothing.ground_correction([track], [[True] * 20], ground_z=0.0, strength=1.0)
        self.assertGreater(offsets[-1][2], 0.04)
        for offset in offsets:
            self.assertEqual(offset[0], 0.0)
            self.assertEqual(offset[1], 0.0)

    def test_floating_foot_is_lowered(self):
        track = [(0.0, 0.0, 0.05)] * 20
        offsets = smoothing.ground_correction([track], [[True] * 20], ground_z=0.0, strength=1.0)
        self.assertLess(offsets[-1][2], -0.04)

    def test_no_contact_means_no_correction(self):
        track = [(0.0, 0.0, -0.05)] * 5
        offsets = smoothing.ground_correction([track], [[False] * 5], ground_z=0.0, strength=1.0)
        self.assertTrue(all(offset == (0.0, 0.0, 0.0) for offset in offsets))

    def test_zero_strength_is_a_no_op(self):
        track = [(0.0, 0.0, -0.05)] * 5
        offsets = smoothing.ground_correction([track], [[True] * 5], strength=0.0)
        self.assertTrue(all(offset == (0.0, 0.0, 0.0) for offset in offsets))


class TestApplyOffset(unittest.TestCase):
    def test_offset_is_added_to_every_joint(self):
        positions = {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)}
        moved = smoothing.apply_offset(positions, (1.0, 2.0, 3.0))
        self.assertEqual(moved["a"], (1.0, 2.0, 3.0))
        self.assertEqual(moved["b"], (2.0, 3.0, 4.0))

    def test_zero_offset_returns_the_same_mapping(self):
        positions = {"a": (0.0, 0.0, 0.0)}
        self.assertIs(smoothing.apply_offset(positions, (0.0, 0.0, 0.0)), positions)


if __name__ == "__main__":
    unittest.main()
