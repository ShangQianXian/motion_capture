"""Framework-free retarget maths (guide section 10.3)."""

from __future__ import annotations

import itertools
import math
import unittest

from core import retarget_math as rm


def almost(a, b, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


class TestVectors(unittest.TestCase):
    def test_basic_algebra(self):
        a, b = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
        self.assertEqual(rm.vec_add(a, b), (5.0, 7.0, 9.0))
        self.assertEqual(rm.vec_sub(b, a), (3.0, 3.0, 3.0))
        self.assertEqual(rm.vec_scale(a, 2.0), (2.0, 4.0, 6.0))
        self.assertEqual(rm.vec_neg(a), (-1.0, -2.0, -3.0))
        self.assertEqual(rm.vec_dot(a, b), 32.0)
        self.assertEqual(rm.vec_cross((1, 0, 0), (0, 1, 0)), (0.0, 0.0, 1.0))
        self.assertEqual(rm.vec_midpoint((0, 0, 0), (2, 4, 6)), (1.0, 2.0, 3.0))

    def test_length_and_normalise(self):
        self.assertAlmostEqual(rm.vec_length((3.0, 4.0, 0.0)), 5.0)
        self.assertAlmostEqual(rm.vec_length(rm.vec_normalize((3.0, 4.0, 12.0))), 1.0)
        self.assertEqual(rm.vec_normalize((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))

    def test_lerp_endpoints(self):
        a, b = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
        self.assertEqual(rm.vec_lerp(a, b, 0.0), a)
        self.assertEqual(rm.vec_lerp(a, b, 1.0), b)
        self.assertEqual(rm.vec_lerp(a, b, 0.5), (5.0, 0.0, 0.0))

    def test_any_perpendicular_is_perpendicular_and_unit(self):
        for vector in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0.001, 0, 5)):
            perpendicular = rm.any_perpendicular(vector)
            self.assertAlmostEqual(rm.vec_length(perpendicular), 1.0, places=9)
            self.assertAlmostEqual(rm.vec_dot(perpendicular, rm.vec_normalize(vector)), 0.0, places=9)

    def test_any_perpendicular_of_zero_is_still_unit(self):
        self.assertAlmostEqual(rm.vec_length(rm.any_perpendicular((0, 0, 0))), 1.0)


class TestQuaternions(unittest.TestCase):
    def test_identity_rotates_nothing(self):
        vector = (0.3, -1.2, 4.0)
        rotated = rm.quat_rotate_vector(rm.IDENTITY_QUAT, vector)
        for a, b in zip(rotated, vector):
            self.assertAlmostEqual(a, b)

    def test_axis_angle_round_trip(self):
        quaternion = rm.quat_from_axis_angle((0, 0, 1), math.pi / 2.0)
        rotated = rm.quat_rotate_vector(quaternion, (1, 0, 0))
        self.assertAlmostEqual(rotated[0], 0.0, places=9)
        self.assertAlmostEqual(rotated[1], 1.0, places=9)
        self.assertAlmostEqual(rm.quat_angle(quaternion), math.pi / 2.0, places=9)

    def test_multiplication_composes_in_the_expected_order(self):
        first = rm.quat_from_axis_angle((0, 0, 1), math.pi / 2.0)
        second = rm.quat_from_axis_angle((1, 0, 0), math.pi / 2.0)
        combined = rm.quat_mul(second, first)  # apply first, then second
        step = rm.quat_rotate_vector(second, rm.quat_rotate_vector(first, (1, 0, 0)))
        once = rm.quat_rotate_vector(combined, (1, 0, 0))
        for a, b in zip(step, once):
            self.assertAlmostEqual(a, b, places=9)

    def test_conjugate_undoes_the_rotation(self):
        quaternion = rm.quat_from_axis_angle((1, 2, 3), 1.1)
        vector = (0.5, -2.0, 3.5)
        rotated = rm.quat_rotate_vector(quaternion, vector)
        restored = rm.quat_rotate_vector(rm.quat_conjugate(quaternion), rotated)
        for a, b in zip(restored, vector):
            self.assertAlmostEqual(a, b, places=9)

    def test_normalise_of_degenerate_quaternion(self):
        self.assertEqual(rm.quat_normalize((0.0, 0.0, 0.0, 0.0)), rm.IDENTITY_QUAT)


class TestRotationDifference(unittest.TestCase):
    def test_maps_source_onto_target(self):
        cases = [
            ((0, 1, 0), (1, 0, 0)),
            ((0, 1, 0), (0, 0, 1)),
            ((0.247, 0.062, -0.136), (0.5, -0.4, -0.3)),
            ((1, 1, 1), (-1, 2, 0.5)),
        ]
        for source, target in cases:
            quaternion = rm.quat_from_two_vectors(source, target)
            rotated = rm.quat_rotate_vector(quaternion, rm.vec_normalize(source))
            expected = rm.vec_normalize(target)
            for a, b in zip(rotated, expected):
                self.assertAlmostEqual(a, b, places=9, msg="{0} -> {1}".format(source, target))

    def test_parallel_vectors_give_identity(self):
        quaternion = rm.quat_from_two_vectors((0, 2, 0), (0, 5, 0))
        self.assertAlmostEqual(rm.quat_angle(quaternion), 0.0, places=9)

    def test_antiparallel_vectors_give_a_half_turn(self):
        source = (0.0, 1.0, 0.0)
        quaternion = rm.quat_from_two_vectors(source, (0.0, -1.0, 0.0))
        self.assertAlmostEqual(rm.quat_angle(quaternion), math.pi, places=7)
        rotated = rm.quat_rotate_vector(quaternion, source)
        for a, b in zip(rotated, (0.0, -1.0, 0.0)):
            self.assertAlmostEqual(a, b, places=7)

    def test_zero_length_input_gives_identity(self):
        self.assertEqual(rm.quat_from_two_vectors((0, 0, 0), (1, 0, 0)), rm.IDENTITY_QUAT)
        self.assertEqual(rm.quat_from_two_vectors((1, 0, 0), (0, 0, 0)), rm.IDENTITY_QUAT)

    def test_result_is_always_a_unit_quaternion(self):
        for source in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (-4, 0.2, 9)):
            for target in ((0, 1, 0), (-1, 0, 0), (0.1, -0.2, 0.3)):
                quaternion = rm.quat_from_two_vectors(source, target)
                norm = sum(component * component for component in quaternion) ** 0.5
                self.assertAlmostEqual(norm, 1.0, places=9)


class TestSlerp(unittest.TestCase):
    def test_endpoints(self):
        a = rm.IDENTITY_QUAT
        b = rm.quat_from_axis_angle((0, 0, 1), math.pi / 2.0)
        for component_a, component_b in zip(rm.slerp(a, b, 0.0), a):
            self.assertAlmostEqual(component_a, component_b, places=9)
        for component_a, component_b in zip(rm.slerp(a, b, 1.0), b):
            self.assertAlmostEqual(component_a, component_b, places=9)

    def test_midpoint_is_half_the_angle(self):
        a = rm.IDENTITY_QUAT
        b = rm.quat_from_axis_angle((0, 0, 1), math.pi / 2.0)
        mid = rm.slerp(a, b, 0.5)
        self.assertAlmostEqual(rm.quat_angle(mid), math.pi / 4.0, places=7)

    def test_takes_the_short_arc_when_the_dot_product_is_negative(self):
        a = rm.quat_from_axis_angle((0, 0, 1), 0.1)
        b = rm.quat_from_axis_angle((0, 0, 1), 0.3)
        flipped = tuple(-component for component in b)  # same rotation, opposite sign
        direct = rm.slerp(a, b, 0.5)
        via_flipped = rm.slerp(a, flipped, 0.5)
        # Both must describe the same rotation, i.e. differ at most by a global sign.
        same = all(almost(x, y, 1e-9) for x, y in zip(direct, via_flipped))
        negated = all(almost(x, -y, 1e-9) for x, y in zip(direct, via_flipped))
        self.assertTrue(same or negated)
        self.assertLess(rm.quat_angle(via_flipped), math.pi / 2.0)

    def test_nearly_identical_inputs_stay_stable(self):
        a = rm.quat_from_axis_angle((0, 0, 1), 0.1)
        b = rm.quat_from_axis_angle((0, 0, 1), 0.1 + 1e-9)
        result = rm.slerp(a, b, 0.5)
        norm = sum(component * component for component in result) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=9)


class TestMatrices(unittest.TestCase):
    def test_identity_helpers(self):
        self.assertEqual(rm.mat_column(rm.IDENTITY_MATRIX, 0), (1.0, 0.0, 0.0))
        self.assertEqual(rm.mat_mul_vec(rm.IDENTITY_MATRIX, (1, 2, 3)), (1.0, 2.0, 3.0))
        self.assertEqual(rm.mat_transpose(rm.IDENTITY_MATRIX), rm.IDENTITY_MATRIX)

    def test_quat_matrix_round_trip(self):
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0.3, -0.7, 0.2)):
            for angle in (0.0, 0.3, math.pi / 2.0, 2.0, math.pi - 0.01):
                quaternion = rm.quat_from_axis_angle(axis, angle)
                matrix = rm.matrix3_from_quat(quaternion)
                restored = rm.quat_from_matrix3(matrix)
                # A quaternion and its negation describe the same rotation.
                same = all(almost(a, b, 1e-7) for a, b in zip(quaternion, restored))
                negated = all(almost(a, -b, 1e-7) for a, b in zip(quaternion, restored))
                self.assertTrue(same or negated, "{0} {1}".format(axis, angle))

    def test_matrix_from_quat_is_orthonormal(self):
        quaternion = rm.quat_from_axis_angle((0.3, -0.7, 0.2), 1.3)
        matrix = rm.matrix3_from_quat(quaternion)
        columns = [rm.mat_column(matrix, i) for i in range(3)]
        for column in columns:
            self.assertAlmostEqual(rm.vec_length(column), 1.0, places=9)
        for i, j in itertools.combinations(range(3), 2):
            self.assertAlmostEqual(rm.vec_dot(columns[i], columns[j]), 0.0, places=9)
        determinant = rm.vec_dot(rm.vec_cross(columns[0], columns[1]), columns[2])
        self.assertAlmostEqual(determinant, 1.0, places=9)

    def test_quat_from_matrix_covers_every_shepperd_branch(self):
        # Rotations chosen so the trace and each diagonal element dominate in turn.
        for axis, angle in (
            ((0, 0, 1), 0.2),            # positive trace
            ((1, 0, 0), math.pi - 0.05),  # m00 dominates
            ((0, 1, 0), math.pi - 0.05),  # m11 dominates
            ((0, 0, 1), math.pi - 0.05),  # m22 dominates
        ):
            quaternion = rm.quat_from_axis_angle(axis, angle)
            restored = rm.quat_from_matrix3(rm.matrix3_from_quat(quaternion))
            self.assertAlmostEqual(rm.quat_angle(restored), rm.quat_angle(quaternion), places=6)


class TestBasisFromAxes(unittest.TestCase):
    def _assert_right_handed(self, matrix):
        columns = [rm.mat_column(matrix, i) for i in range(3)]
        for column in columns:
            self.assertAlmostEqual(rm.vec_length(column), 1.0, places=9)
        for i, j in itertools.combinations(range(3), 2):
            self.assertAlmostEqual(rm.vec_dot(columns[i], columns[j]), 0.0, places=9)
        determinant = rm.vec_dot(rm.vec_cross(columns[0], columns[1]), columns[2])
        self.assertAlmostEqual(determinant, 1.0, places=9)

    def test_primary_axis_is_exact(self):
        matrix = rm.basis_from_axes((0, 0, 2), rm.BONE_AXIS, (1, 0, 0), rm.AXIS_X)
        self._assert_right_handed(matrix)
        for a, b in zip(rm.mat_column(matrix, rm.BONE_AXIS), (0.0, 0.0, 1.0)):
            self.assertAlmostEqual(a, b, places=9)

    def test_reference_axis_is_orthogonalised_not_ignored(self):
        # Reference deliberately not perpendicular to the primary direction.
        matrix = rm.basis_from_axes((0, 0, 1), rm.BONE_AXIS, (1, 0, 0.9), rm.AXIS_X)
        self._assert_right_handed(matrix)
        reference = rm.mat_column(matrix, rm.AXIS_X)
        self.assertAlmostEqual(reference[2], 0.0, places=9)
        self.assertGreater(reference[0], 0.0)

    def test_every_axis_pairing_stays_right_handed(self):
        for primary in range(3):
            for secondary in range(3):
                if primary == secondary:
                    continue
                matrix = rm.basis_from_axes((0.2, 1.0, -0.3), primary, (1.0, 0.1, 0.4), secondary)
                self._assert_right_handed(matrix)

    def test_degenerate_reference_falls_back_gracefully(self):
        matrix = rm.basis_from_axes((0, 0, 1), rm.BONE_AXIS, (0, 0, 5), rm.AXIS_X)
        self._assert_right_handed(matrix)

    def test_zero_primary_returns_identity(self):
        self.assertEqual(
            rm.basis_from_axes((0, 0, 0), rm.BONE_AXIS, (1, 0, 0), rm.AXIS_X),
            rm.IDENTITY_MATRIX,
        )

    def test_missing_reference_still_builds_a_basis(self):
        matrix = rm.basis_from_axes((0.0, 1.0, 0.0), rm.BONE_AXIS)
        self._assert_right_handed(matrix)


class TestAimHelpers(unittest.TestCase):
    def test_aim_rotation_matches_rotation_difference(self):
        rest = (0.0, 0.152, 0.0)
        target = (0.3, -0.2, 0.9)
        quaternion = rm.aim_rotation(rest, target)
        rotated = rm.quat_rotate_vector(quaternion, rm.vec_normalize(rest))
        for a, b in zip(rotated, rm.vec_normalize(target)):
            self.assertAlmostEqual(a, b, places=9)

    def test_aim_with_reference_matches_both_axes(self):
        rest_direction = (0.0, 0.152, 0.0)
        rest_reference = (1.0, 0.0, 0.0)
        target_direction = (0.0, 0.0, 1.0)
        target_reference = (0.0, 1.0, 0.0)
        quaternion = rm.aim_rotation_with_reference(
            rest_direction, rest_reference, target_direction, target_reference
        )
        aimed = rm.quat_rotate_vector(quaternion, rm.vec_normalize(rest_direction))
        for a, b in zip(aimed, rm.vec_normalize(target_direction)):
            self.assertAlmostEqual(a, b, places=9)
        referenced = rm.quat_rotate_vector(quaternion, rest_reference)
        for a, b in zip(referenced, rm.vec_normalize(target_reference)):
            self.assertAlmostEqual(a, b, places=9)

    def test_aim_with_reference_removes_the_twist_ambiguity(self):
        # Same bone direction, two different body facings must give different spins.
        rest_direction = (0.0, 0.0, 0.2)
        rest_reference = (1.0, 0.0, 0.0)
        target_direction = (0.0, 0.0, 1.0)
        facing_a = rm.aim_rotation_with_reference(
            rest_direction, rest_reference, target_direction, (1.0, 0.0, 0.0)
        )
        facing_b = rm.aim_rotation_with_reference(
            rest_direction, rest_reference, target_direction, (0.0, 1.0, 0.0)
        )
        difference = rm.quat_mul(facing_b, rm.quat_conjugate(facing_a))
        self.assertAlmostEqual(rm.quat_angle(difference), math.pi / 2.0, places=7)

    def test_delta_rotation_between_bases(self):
        rest = rm.basis_from_axes((0.0, 1.0, 0.0), rm.BONE_AXIS, (1.0, 0.0, 0.0), rm.AXIS_X)
        target = rm.basis_from_axes((0.0, 0.0, 1.0), rm.BONE_AXIS, (1.0, 0.0, 0.0), rm.AXIS_X)
        quaternion = rm.delta_rotation(rest, target)
        rotated = rm.quat_rotate_vector(quaternion, (0.0, 1.0, 0.0))
        for a, b in zip(rotated, (0.0, 0.0, 1.0)):
            self.assertAlmostEqual(a, b, places=9)


class TestClamp(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(rm.clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(rm.clamp(2.0, 0.0, 1.0), 1.0)
        self.assertEqual(rm.clamp(0.5, 0.0, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
