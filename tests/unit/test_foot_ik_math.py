"""The two-bone solver used by the retargeting foot correction.

The correction exists because a rig whose thigh and shin lengths differ from the
captured subject's puts its ankle somewhere else. The solver is the part that has
to be exactly right -- an ankle that lands short, or a knee that flips, is worse
than no correction at all -- so it is tested on its own, without a rig.
"""
import math
import unittest

from core import retarget_math as rm


def distance(a, b):
    return rm.vec_distance(a, b)


class ReachableTarget(unittest.TestCase):
    def test_ankle_lands_on_the_target_and_bones_keep_their_lengths(self):
        hip = (0.0, 0.0, 1.0)
        upper, lower = 0.4, 0.38
        # Every target is inside the reachable shell (|upper-lower|, upper+lower).
        for target in ((0.0, 0.0, 0.3), (0.2, 0.1, 0.35), (-0.15, -0.3, 0.5), (0.0, 0.45, 0.6)):
            with self.subTest(target=target):
                reached_distance = distance(hip, target)
                self.assertLess(abs(upper - lower), reached_distance)
                self.assertLess(reached_distance, upper + lower)
                knee, ankle, reached = rm.three_bone_ik(hip, target, upper, lower)
                self.assertTrue(reached)
                self.assertAlmostEqual(distance(hip, ankle), reached_distance, places=6)
                self.assertAlmostEqual(distance(hip, knee), upper, places=6)
                self.assertAlmostEqual(distance(knee, ankle), lower, places=6)

    def test_an_out_of_reach_target_is_clamped_to_full_extension(self):
        hip = (0.0, 0.0, 1.0)
        upper, lower = 0.4, 0.38
        # 0.7826 away, just past the 0.78 the chain can span.
        knee, ankle, reached = rm.three_bone_ik(hip, (0.2, 0.1, 0.25), upper, lower)
        self.assertFalse(reached)
        self.assertAlmostEqual(distance(hip, ankle), upper + lower, places=6)
        self.assertAlmostEqual(distance(hip, knee), upper, places=6)

    def test_knee_bends_towards_the_previous_knee(self):
        """The bend plane must follow the hint, or a knee crosses the other leg."""
        hip = (0.0, 0.0, 1.0)
        upper, lower = 0.45, 0.42
        target = (0.0, 0.0, 0.4)
        forward, backward = (0.0, -0.2, 0.7), (0.0, 0.2, 0.7)
        knee_forward, _, _ = rm.three_bone_ik(hip, target, upper, lower, previous_knee=forward)
        knee_backward, _, _ = rm.three_bone_ik(hip, target, upper, lower, previous_knee=backward)
        self.assertLess(knee_forward[1], hip[1])
        self.assertGreater(knee_backward[1], hip[1])
        self.assertNotAlmostEqual(knee_forward[1], knee_backward[1], places=3)

    def test_previous_knee_on_the_axis_still_produces_a_defined_bend(self):
        """A straight leg carries no plane; the solver must not return NaN."""
        hip = (0.0, 0.0, 1.0)
        target = (0.0, 0.0, 0.3)
        knee, ankle, reached = rm.three_bone_ik(hip, target, 0.45, 0.42,
                                                previous_knee=(0.0, 0.0, 0.7))
        self.assertTrue(reached)
        for value in knee + ankle:
            self.assertTrue(math.isfinite(value))


class OutOfRangeTarget(unittest.TestCase):
    def test_a_target_beyond_reach_is_reported_not_silently_shortened(self):
        hip = (0.0, 0.0, 1.0)
        upper, lower = 0.4, 0.38
        knee, ankle, reached = rm.three_bone_ik(hip, (0.0, 0.0, -1.0), upper, lower)
        self.assertFalse(reached)
        # Straightened along the aim, at full length: the closest it can get.
        self.assertAlmostEqual(distance(hip, ankle), upper + lower, places=6)
        self.assertAlmostEqual(distance(hip, knee), upper, places=6)
        self.assertLess(ankle[2], hip[2])

    def test_a_target_closer_than_the_bones_allow_is_reported_too(self):
        hip = (0.0, 0.0, 1.0)
        knee, ankle, reached = rm.three_bone_ik(hip, (0.0, 0.0, 1.001), 0.45, 0.30)
        self.assertFalse(reached)
        self.assertAlmostEqual(distance(hip, ankle), abs(0.45 - 0.30), places=6)

    def test_degenerate_target_does_not_produce_nan(self):
        hip = (0.0, 0.0, 1.0)
        knee, ankle, reached = rm.three_bone_ik(hip, hip, 0.45, 0.42)
        self.assertFalse(reached)
        for value in knee + ankle:
            self.assertTrue(math.isfinite(value), value)


class LengthsComeFromTheCaller(unittest.TestCase):
    def test_the_same_direction_at_any_reachable_distance_stays_on_that_direction(self):
        """The rig supplies distance, the capture supplies direction.

        This is what makes the correction scale-free: shortening or lengthening
        the rig's leg moves the ankle along the line the video shows and never
        bends the limb off it.
        """
        hip = (0.0, 0.0, 1.0)
        direction = rm.vec_normalize((0.3, -0.2, -1.0))
        for reach, fraction in ((0.64, 0.9), (0.99, 0.85), (0.64, 1.0)):
            with self.subTest(reach=reach):
                target = rm.vec_add(hip, rm.vec_scale(direction, reach * fraction))
                upper = reach * 0.55
                lower = reach * 0.45
                _, ankle, reached = rm.three_bone_ik(hip, target, upper, lower)
                self.assertTrue(reached)
                offset = rm.vec_sub(ankle, hip)
                self.assertAlmostEqual(rm.vec_length(offset), reach * fraction, places=6)
                self.assertAlmostEqual(rm.vec_dot(rm.vec_normalize(offset), direction), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
