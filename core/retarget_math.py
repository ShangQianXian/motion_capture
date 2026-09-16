"""Pure-Python vector / quaternion / matrix helpers for retargeting.

Standard library only, so the maths is unit-testable without Blender and can be
reused by the worker. Conventions:

* vectors are 3-tuples ``(x, y, z)``
* quaternions are 4-tuples ``(w, x, y, z)`` - same order as ``mathutils.Quaternion``
* matrices are row-major 3x3 tuples ``((m00, m01, m02), (m10, ...), (m20, ...))``

``blender/rigify_adapter.py`` converts to and from ``mathutils`` at the boundary
so the model maths never depends on Blender.
"""

from __future__ import annotations

import math

EPSILON = 1e-9

#: Local bone axis indices.
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

#: Blender bones point along their local +Y axis.
BONE_AXIS = AXIS_Y

_EVEN_PERMUTATIONS = frozenset({(0, 1, 2), (1, 2, 0), (2, 0, 1)})

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------


def vec_add(a, b) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a, b) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a, factor: float) -> tuple:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def vec_neg(a) -> tuple:
    return (-a[0], -a[1], -a[2])


def vec_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a, b) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_length(a) -> float:
    return math.sqrt(vec_dot(a, a))


def vec_distance(a, b) -> float:
    return vec_length(vec_sub(a, b))


def vec_normalize(a) -> tuple:
    length = vec_length(a)
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def vec_lerp(a, b, t: float) -> tuple:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def vec_midpoint(a, b) -> tuple:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))


def any_perpendicular(a) -> tuple:
    """Return an arbitrary unit vector perpendicular to ``a``."""
    direction = vec_normalize(a)
    if vec_length(direction) < EPSILON:
        return (1.0, 0.0, 0.0)
    # Cross with the world axis least aligned with ``direction``.
    axis = min(range(3), key=lambda i: abs(direction[i]))
    helper = [0.0, 0.0, 0.0]
    helper[axis] = 1.0
    return vec_normalize(vec_cross(direction, tuple(helper)))


# --------------------------------------------------------------------------------------
# Quaternions
# --------------------------------------------------------------------------------------


def quat_normalize(q) -> tuple:
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm < EPSILON:
        return IDENTITY_QUAT
    inv = 1.0 / norm
    return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


def quat_conjugate(q) -> tuple:
    return (q[0], -q[1], -q[2], -q[3])


def quat_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_mul(a, b) -> tuple:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate_vector(q, v) -> tuple:
    """Rotate ``v`` by unit quaternion ``q``."""
    w, x, y, z = quat_normalize(q)
    qv = (x, y, z)
    t = vec_scale(vec_cross(qv, v), 2.0)
    return vec_add(vec_add(v, vec_scale(t, w)), vec_cross(qv, t))


def quat_from_axis_angle(axis, angle: float) -> tuple:
    unit = vec_normalize(axis)
    if vec_length(unit) < EPSILON:
        return IDENTITY_QUAT
    half = 0.5 * angle
    s = math.sin(half)
    return (math.cos(half), unit[0] * s, unit[1] * s, unit[2] * s)


def quat_angle(q) -> float:
    """Rotation magnitude of ``q`` in radians (0..pi)."""
    w = abs(quat_normalize(q)[0])
    return 2.0 * math.acos(max(-1.0, min(1.0, w)))


def quat_from_two_vectors(source, target) -> tuple:
    """Minimal-arc rotation taking ``source`` onto ``target``.

    Equivalent to ``mathutils.Vector.rotation_difference``. Antiparallel inputs
    yield a 180 degree rotation around an arbitrary perpendicular axis.
    """
    a = vec_normalize(source)
    b = vec_normalize(target)
    if vec_length(a) < EPSILON or vec_length(b) < EPSILON:
        return IDENTITY_QUAT
    cos_angle = max(-1.0, min(1.0, vec_dot(a, b)))
    if cos_angle > 1.0 - 1e-12:
        return IDENTITY_QUAT
    if cos_angle < -1.0 + 1e-12:
        return quat_from_axis_angle(any_perpendicular(a), math.pi)
    axis = vec_cross(a, b)
    return quat_normalize((1.0 + cos_angle,) + axis)


def slerp(a, b, t: float) -> tuple:
    """Shortest-arc spherical interpolation between unit quaternions."""
    q0 = quat_normalize(a)
    q1 = quat_normalize(b)
    dot = quat_dot(q0, q1)
    if dot < 0.0:  # flip to avoid travelling the long way round
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    if dot > 0.9995:  # nearly identical, linear blend is stable and cheap
        blended = tuple(q0[i] + (q1[i] - q0[i]) * t for i in range(4))
        return quat_normalize(blended)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return quat_normalize(tuple(q0[i] * s0 + q1[i] * s1 for i in range(4)))


# --------------------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------------------


def mat_column(matrix, index: int) -> tuple:
    return (matrix[0][index], matrix[1][index], matrix[2][index])


def mat_from_columns(cx, cy, cz) -> tuple:
    return (
        (cx[0], cy[0], cz[0]),
        (cx[1], cy[1], cz[1]),
        (cx[2], cy[2], cz[2]),
    )


def mat_transpose(matrix) -> tuple:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def mat_mul(a, b) -> tuple:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def mat_mul_vec(matrix, v) -> tuple:
    return tuple(
        matrix[row][0] * v[0] + matrix[row][1] * v[1] + matrix[row][2] * v[2] for row in range(3)
    )


def basis_from_axes(primary, primary_axis: int, secondary=None, secondary_axis: int | None = None) -> tuple:
    """Build a right-handed orthonormal basis from one or two directions.

    ``primary`` is aligned exactly with local axis ``primary_axis``. When
    ``secondary`` is given, the component of it orthogonal to ``primary`` is
    aligned with ``secondary_axis``, which removes the twist ambiguity of a pure
    aim constraint. Degenerate inputs fall back to an arbitrary perpendicular.
    """
    u = vec_normalize(primary)
    if vec_length(u) < EPSILON:
        return IDENTITY_MATRIX

    if secondary is None or secondary_axis is None or secondary_axis == primary_axis:
        secondary_axis = (primary_axis + 1) % 3
        reference = any_perpendicular(u)
    else:
        reference = vec_sub(secondary, vec_scale(u, vec_dot(u, secondary)))
        if vec_length(reference) < 1e-6:
            reference = any_perpendicular(u)
        reference = vec_normalize(reference)

    third_axis = 3 - primary_axis - secondary_axis
    third = vec_cross(u, reference)
    if (primary_axis, secondary_axis, third_axis) not in _EVEN_PERMUTATIONS:
        third = vec_neg(third)

    columns = [None, None, None]
    columns[primary_axis] = u
    columns[secondary_axis] = reference
    columns[third_axis] = vec_normalize(third)
    return mat_from_columns(columns[0], columns[1], columns[2])


def quat_from_matrix3(matrix) -> tuple:
    """Convert an orthonormal rotation matrix to a quaternion (Shepperd's method)."""
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize((w, x, y, z))


def matrix3_from_quat(q) -> tuple:
    """Convert a quaternion to an orthonormal rotation matrix."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def delta_rotation(rest_matrix, target_matrix) -> tuple:
    """Rotation taking ``rest_matrix`` onto ``target_matrix`` (both orthonormal)."""
    return quat_from_matrix3(mat_mul(target_matrix, mat_transpose(rest_matrix)))


# --------------------------------------------------------------------------------------
# High level helpers used by the Rigify adapter
# --------------------------------------------------------------------------------------


def aim_rotation(rest_direction, target_direction) -> tuple:
    """World-space rotation aiming a bone's rest direction at ``target_direction``."""
    return quat_from_two_vectors(rest_direction, target_direction)


def aim_rotation_with_reference(
    rest_direction,
    rest_reference,
    target_direction,
    target_reference,
) -> tuple:
    """World-space rotation matching both a bone direction and a reference axis.

    Used for the torso chain where a pure aim constraint would leave the spin
    around the bone axis undefined and the character could face anywhere.
    """
    rest_basis = basis_from_axes(rest_direction, BONE_AXIS, rest_reference, AXIS_X)
    target_basis = basis_from_axes(target_direction, BONE_AXIS, target_reference, AXIS_X)
    return delta_rotation(rest_basis, target_basis)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


#: Below this distance between a two-bone chain's ends the bend plane is undefined.
FIXED_DIRECTION_EPSILON = 1e-6


def three_bone_ik(hip, target, upper: float, lower: float, previous_knee=None):
    """Place a two-bone chain so its endpoint lands on ``target``.

    ``upper`` and ``lower`` are the two bone lengths as the *target* skeleton has
    them; the endpoint is reached exactly wherever that is possible, and the
    chain is straightened towards the target when it is out of range. The bend
    plane is taken from ``previous_knee`` so the joint keeps bending the way it
    already did, which is what stops a knee from crossing or flipping.

    Returns ``(knee, ankle, reached)``. ``reached`` is False when the target had
    to be clamped, which the caller should report rather than hide.

    Pure geometry: no bpy, no rig, so it is testable on its own.
    """
    hip = tuple(float(v) for v in hip)
    target = tuple(float(v) for v in target)
    offset = vec_sub(target, hip)
    measured = vec_length(offset)
    reach = upper + lower
    shortest = abs(upper - lower)
    reached = True
    if measured > reach:
        # Fully extended; the endpoint cannot get any closer than this.
        distance, reached = reach, False
    elif measured < shortest:
        distance, reached = shortest, False
    else:
        distance = measured

    direction = (0.0, 0.0, -1.0) if measured < FIXED_DIRECTION_EPSILON else vec_scale(offset, 1.0 / measured)
    denominator = 2.0 * max(distance, FIXED_DIRECTION_EPSILON)
    along = (upper * upper - lower * lower + distance * distance) / denominator
    bend = math.sqrt(upper * upper - along * along) if upper * upper > along * along else 0.0
    # The bend plane comes from where the knee already is, projected off the
    # hip-to-target axis. A straight leg says nothing about the plane, so fall
    # back to any perpendicular rather than leaving the bend undefined.
    plane = None
    if previous_knee is not None:
        axis = vec_sub(previous_knee, hip)
        component = vec_sub(axis, vec_scale(direction, vec_dot(axis, direction)))
        if vec_length(component) > FIXED_DIRECTION_EPSILON:
            plane = vec_normalize(component)
    if plane is None:
        plane = any_perpendicular(direction)
    knee = vec_add(vec_add(hip, vec_scale(direction, along)), vec_scale(plane, bend))
    ankle = target if reached else vec_add(hip, vec_scale(direction, distance))
    return knee, ankle, reached
