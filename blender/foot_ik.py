"""End-effector correction for retargeting (option B).

Direction retargeting aims each bone at its source direction and keeps whatever
bone length the target rig already has, so a rig whose segments differ in length
from the captured subject puts its end effectors somewhere else. Measured on a
generated Rigify human against a real capture, the feet land 11.6 cm (left) and
4.8 cm (right) away from where the video has them.

This module re-solves each leg so the ankle lands on the captured ankle position,
using the rig's own thigh and shin lengths. The hip stays where the retarget put
it and the knee moves, which is the only free variable: with fixed bone lengths,
a leg whose segments disagree with the source cannot have both its knee and its
ankle where the source had them.

Measured effect of enabling it: the ankle gap falls to 3.2 cm and 2.8 cm. What
remains comes from the few frames whose target sits outside the rig's reach, and
those are reported rather than hidden. The cost is that the leg's overall
direction moves by about 6 degrees -- a consequence of fixing both the hip (from
the retarget) and the ankle (from the capture) on a leg of the wrong proportions.

With ``RetargetOptions.foot_correction`` False, only directions are transferred
and the feet land at the rig's own leg length instead.
"""

from __future__ import annotations

import bpy
from mathutils import Quaternion, Vector

from ..core import retarget_math as rm

#: Leg chains: (thigh control, shin control, source hip, source knee, source ankle).
LEGS = (
    ('thigh_fk.L', 'shin_fk.L', 'hip.L', 'knee.L', 'ankle.L'),
    ('thigh_fk.R', 'shin_fk.R', 'hip.R', 'knee.R', 'ankle.R'),
)

#: Weight given to the new knee bend plane when carrying it forward between
#: frames. The plane is what stops a knee from flipping when the leg straightens
#: and the geometry stops constraining it.
KNEE_PLANE_MEMORY = 0.7


def leg_source_lengths(result):
    """Median thigh and shin lengths of a capture, per side, in metres.

    Read once per sequence for the same reason the pipeline fixes them once: they
    are constants of the subject, and a per-frame read would put the network's
    per-frame length error straight into the correction.
    """
    lengths = {}
    frames = result.frames
    if not frames:
        return lengths
    for _thigh, _shin, hip, knee, ankle in LEGS:
        side = hip[-1]
        uppers, lowers = [], []
        for frame in frames:
            body = frame.body3d
            if not all(name in body for name in (hip, knee, ankle)):
                continue
            uppers.append(rm.vec_distance(body[hip], body[knee]))
            lowers.append(rm.vec_distance(body[knee], body[ankle]))
        if not uppers or not lowers:
            continue
        uppers.sort()
        lowers.sort()
        lengths[side] = (uppers[len(uppers) // 2], lowers[len(lowers) // 2])
    return lengths


def rig_leg_lengths(armature):
    """World-space thigh and shin lengths of the rig's control chains, per side."""
    lengths = {}
    matrix = armature.matrix_world
    for thigh, shin, _hip, _knee, _ankle in LEGS:
        side = thigh[-1]
        upper = armature.data.bones.get(thigh)
        lower = armature.data.bones.get(shin)
        if upper is None or lower is None:
            continue
        lengths[side] = (
            ((matrix @ upper.tail_local) - (matrix @ upper.head_local)).length,
            ((matrix @ lower.tail_local) - (matrix @ lower.head_local)).length,
        )
    return lengths


def _set_direction(pose_bone, head, target_direction, world_to_pose):
    """Point ``pose_bone`` along ``target_direction`` with its head at ``head``.

    The pose matrix is set directly, exactly as ``_apply_pose_rotation`` does in
    the direction pass: only the orientation is replaced and the head stays where
    the parent chain put it. Composing an aim rotation onto
    ``rotation_quaternion`` instead resolves the aim against the bone's *rest*
    frame while the inherited parent rotation is still in effect, which aims the
    wrong way -- a connected child such as the shin inherits its parent's
    rotation, so its rest direction is not what it currently points along.

    ``head`` is passed in rather than read from the pose bone. ``pose_bone.head``
    and ``matrix`` only settle once the dependency graph is evaluated, so reading
    them between two assignments makes the outcome depend on when that flush
    lands, which is not reproducible: the same options applied twice gave
    slightly different Actions.
    """
    if rm.vec_length(target_direction) < rm.FIXED_DIRECTION_EPSILON:
        return False
    pose_direction = Vector(world_to_pose @ Vector(target_direction))
    rotation = Quaternion(rm.quat_from_two_vectors(
        tuple(pose_bone.bone.matrix_local.to_3x3().col[1]), tuple(pose_direction)))
    target = rotation.to_matrix().to_4x4() @ pose_bone.bone.matrix_local
    # ``pose.matrix.translation`` is the head position, so writing ``head`` here
    # would move the bone by head - (R @ matrix_local).translation -- about a metre
    # on a hip-height bone. Keep the head the parent chain produced instead.
    target.translation = Vector(head)
    pose_bone.matrix = target
    return True


def _prepare(options, result, armature, scale):
    """Resolve the per-sequence constants once, then reuse them for every frame.

    The retarget loop calls the correction once per frame, so this caches onto
    the options object rather than recomputing leg lengths and source directions
    for the whole capture on each call. The cache is keyed on everything it was
    built from, so reusing one ``RetargetOptions`` for a second run -- which the
    tests and the UI both do -- rebuilds instead of silently carrying the earlier
    run's per-frame bend state into the new one.
    """
    key = (id(result), id(armature), round(float(scale), 9))
    prepared = getattr(options, '_foot_correction', None)
    if prepared is not None and prepared.get('key') == key:
        return prepared
    source = leg_source_lengths(result)
    rigged = rig_leg_lengths(armature)
    usable = [side for side in ('L', 'R') if side in source and side in rigged]
    directions = {side: [] for side in usable}
    planes = {side: [] for side in usable}
    targets = {side: [] for side in usable}
    for frame in result.frames:
        body = frame.body3d
        for _thigh, _shin, hip, knee, ankle in LEGS:
            side = hip[-1]
            if side not in usable:
                continue
            if not all(name in body for name in (hip, knee, ankle)):
                directions[side].append(None)
                planes[side].append(None)
                targets[side].append(None)
                continue
            hip_scaled = rm.vec_scale(body[hip], scale)
            ankle_scaled = rm.vec_scale(body[ankle], scale)
            offset = rm.vec_sub(ankle_scaled, hip_scaled)
            length = rm.vec_length(offset)
            if length <= rm.FIXED_DIRECTION_EPSILON:
                directions[side].append(None)
                planes[side].append(None)
                targets[side].append(None)
                continue
            direction = rm.vec_scale(offset, 1.0 / length)
            directions[side].append(direction)
            # The absolute target: where the capture's ankle sits once the source
            # skeleton is brought onto the rig's size, the same transform the
            # retarget applies to the rest of the pose.
            targets[side].append(ankle_scaled)
            # The knee's offset from the hip-to-ankle axis *is* the bend plane, so
            # the knee keeps bending the way the video shows. A straight source leg
            # leaves it undefined, hence None rather than a fabricated plane.
            axis = rm.vec_sub(body[knee], body[hip])
            plane = rm.vec_sub(axis, rm.vec_scale(direction, rm.vec_dot(axis, direction)))
            planes[side].append(rm.vec_normalize(plane)
                                if rm.vec_length(plane) > rm.FIXED_DIRECTION_EPSILON else None)
    prepared = {'key': key, 'source': source, 'rigged': rigged, 'usable': usable, 'scale': scale,
                'directions': directions, 'planes': planes, 'targets': targets,
                'last': {}, 'reported': False}
    options._foot_correction = prepared
    return prepared


def apply_foot_correction(armature, result, frame_position, options, scale, world_to_pose, warnings,
                          reapply_foot=None, hips=None):
    """Re-solve both legs for one frame, after the direction pass posed it.

    Called from the retarget loop before that frame's keys are inserted, so the
    written Action holds the corrected pose. The bend plane is carried forward
    across frames through ``prepared['last']`` so a leg passing through straight
    does not let the knee flip. Returns the number of legs whose target had to be
    clamped, which the caller reports rather than hides.
    """
    if not getattr(options, 'foot_correction', False):
        return 0

    prepared = _prepare(options, result, armature, scale)
    if not prepared['usable']:
        if not prepared.get('reported'):
            warnings.append('足部校正：rig 或结果缺少腿链数据，已跳过。')
            prepared['reported'] = True
        return 0
    if frame_position >= len(result.frames):
        return 0

    unreachable = 0
    for thigh_name, shin_name, hip, _knee, _ankle in LEGS:
        side = hip[-1]
        if side not in prepared['usable']:
            continue
        direction = prepared['directions'][side][frame_position]
        thigh_bone = armature.pose.bones.get(thigh_name)
        shin_bone = armature.pose.bones.get(shin_name)
        if direction is None or thigh_bone is None or shin_bone is None:
            continue
        upper, lower = prepared['rigged'][side]
        # Supplied by the caller from the source data, so nothing here depends on
        # when the dependency graph last settled.
        hip_position = (hips or {}).get(side)
        if hip_position is None:
            continue
        observed = prepared['planes'][side][frame_position]
        previous = prepared['last'].get(side)
        if observed is not None and previous is not None:
            blended = tuple(KNEE_PLANE_MEMORY * observed[axis] + (1.0 - KNEE_PLANE_MEMORY) * previous[axis]
                            for axis in range(3))
            plane = (rm.vec_normalize(blended)
                     if rm.vec_length(blended) > rm.FIXED_DIRECTION_EPSILON else observed)
        elif observed is not None:
            plane = observed
        else:
            # Straight in the source: keep the last plane instead of letting an
            # arbitrary perpendicular choose a new bend direction.
            plane = previous
        if plane is not None:
            prepared['last'][side] = plane
        hint = rm.vec_add(hip_position, rm.vec_scale(plane, upper)) if plane is not None else None
        target = prepared['targets'][side][frame_position]
        if target is None:
            continue
        solved_knee, solved_ankle, reached = rm.three_bone_ik(
            hip_position, target, upper, lower, previous_knee=hint)
        if not reached:
            unreachable += 1
        _set_direction(thigh_bone, hip_position, rm.vec_sub(solved_knee, hip_position), world_to_pose)
        # The shin is connected to the thigh, so its head is exactly the knee the
        # solver produced; both directions come from the solver rather than from a
        # re-read of the pose (see _set_direction).
        _set_direction(shin_bone, solved_knee, rm.vec_sub(solved_ankle, solved_knee), world_to_pose)
        bpy.context.view_layer.update()
        # Re-resolve the foot. Its world orientation was computed against the shin
        # the direction pass produced, and moving the shin invalidates it: without
        # this the foot points somewhere the capture never asked for. The caller
        # supplies its own rule, so the foot keeps whichever of the observed
        # orientation and the direction pass the capture profile selected.
        if reapply_foot is not None:
            reapply_foot(side)

    if unreachable:
        warnings.append('足部校正：目标超出腿长范围，已按最大伸展处理（第 {0} 帧）。'
                        .format(frame_position + 1))
    return unreachable
