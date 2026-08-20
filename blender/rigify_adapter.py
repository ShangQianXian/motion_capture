"""Rigify detection, control bone mapping and mocap retargeting.

Implements ``docs/DEVELOPMENT_GUIDE.md`` sections 4.4 and 10. Only generated
Rigify human rigs are supported in v0.1; meta-rigs are rejected with a clear
message and DEF deform bones are never written.

Design notes
------------

* Target world rotations are computed with the framework-free maths in
  ``core.retarget_math``; ``mathutils`` is used only to read rest matrices and to
  write pose matrices.
* Rotations are applied through ``pose_bone.matrix`` in hierarchy order with a
  ``view_layer.update()`` per depth level. That lets Blender resolve the MCH
  parent chain Rigify inserts above the FK controls instead of re-implementing
  Rigify's internal rig logic.
* Ordering is by armature depth, which is a valid topological order because depth
  strictly increases along parent links. Do **not** assume the spine controls are
  a simple upward chain: Rigify's torso rig is pivot-based, so on a 0.6.10 human
  rig the real parenting is::

      torso                                   (depth 1)
        MCH-spine.001 -> spine_fk.001         (depth 3)
          MCH-spine   -> spine_fk             (depth 5)   <- hips, a CHILD of .001
            tweak_spine -> ORG-spine -> MCH-thigh_parent.L -> thigh_fk.L (depth 9)
        MCH-spine.002 -> spine_fk.002         (depth 3)
          MCH-spine.003 -> spine_fk.003       (depth 5)
            MCH-ROT-neck -> neck              (depth 7) -> head (depth 9)

  The hip segment therefore hangs below the pivot and the legs hang off it.
* Only ``rotation_quaternion`` is keyframed, plus ``location`` on the root motion
  control, so no stray location keys leak onto connected bones.
* ``hips`` and ``chest`` are deliberately left at rest: their rest direction is
  +Y horizontal (widget-style controls), which makes rest-to-source aiming
  invalid. The ``spine_fk`` chain carries the torso motion instead.
"""

from __future__ import annotations

import bpy
from mathutils import Matrix, Quaternion, Vector

from ..core import errors, result_schema, skeleton
from ..core import retarget_math as rm

#: Rotation mode forced on every driven control bone (guide section 10.3).
ROTATION_MODE = "QUATERNION"

#: Minimum number of signature bones required to accept a rig without ``rig_id``.
SIGNATURE_THRESHOLD = 6

#: Identity quaternion in Blender's (w, x, y, z) order.
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)

#: Warning codes surfaced through the returned Action's custom properties.
CODE_CHAIN_SKIPPED = "CHAIN_SKIPPED"
CODE_SUSPECT_MIRROR = "SUSPECT_MIRROR"
CODE_SUSPECT_COORDS = "SUSPECT_COORDINATE_SYSTEM"

#: Guard against pathological bone hierarchies while measuring depth.
_MAX_DEPTH = 128


class RigifyDetection(object):
    """Outcome of :func:`detect_rigify_human`."""

    __slots__ = ("ok", "armature", "rig_id", "is_metarig", "found_bones", "missing_bones", "error")

    def __init__(self, armature=None) -> None:
        self.ok = False
        self.armature = armature
        self.rig_id = ""
        self.is_metarig = False
        self.found_bones = []
        self.missing_bones = []
        self.error = None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "RigifyDetection(ok={0}, rig_id={1!r})".format(self.ok, self.rig_id)


class ResolvedChain(object):
    """A chain spec bound to a concrete pose bone."""

    __slots__ = ("spec", "bone_name", "depth", "rest_direction", "rest_reference")

    def __init__(self, spec, bone_name: str, depth: int, rest_direction, rest_reference) -> None:
        self.spec = spec
        self.bone_name = bone_name
        self.depth = depth
        self.rest_direction = rest_direction
        self.rest_reference = rest_reference

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "ResolvedChain({0!r}, depth={1}, mode={2})".format(
            self.bone_name, self.depth, self.spec.mode
        )


class RigifyMapping(object):
    """Source skeleton to Rigify control bone mapping."""

    __slots__ = (
        "armature",
        "chains",
        "missing_required",
        "missing_optional",
        "pelvis_height",
        "warnings",
    )

    def __init__(self, armature) -> None:
        self.armature = armature
        self.chains = []
        self.missing_required = []
        self.missing_optional = []
        self.pelvis_height = 0.0
        self.warnings = []

    @property
    def bone_names(self) -> list:
        return [chain.bone_name for chain in self.chains]

    def body_chains(self) -> list:
        return [chain for chain in self.chains if chain.spec.group == "body"]

    def hand_chains(self) -> list:
        return [chain for chain in self.chains if chain.spec.group == "hand"]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "RigifyMapping(chains={0}, missing_required={1})".format(
            len(self.chains), len(self.missing_required)
        )


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


def detect_rigify_human(armature) -> RigifyDetection:
    """Detect whether ``armature`` is a Rigify generated human rig."""
    detection = RigifyDetection(armature)

    if armature is None:
        detection.error = errors.MocapError(errors.RIGIFY_NOT_FOUND, "未选择目标骨架。")
        return detection
    if getattr(armature, "type", None) != "ARMATURE":
        detection.error = errors.MocapError(
            errors.RIGIFY_NOT_FOUND,
            "目标对象 {0} 不是 Armature。".format(armature.name),
            details={"object": armature.name, "type": str(getattr(armature, "type", ""))},
        )
        return detection

    detection.rig_id = str(armature.data.get("rig_id") or "")

    bone_names = {bone.name for bone in armature.pose.bones}
    detection.found_bones = [name for name in skeleton.RIGIFY_SIGNATURE_BONES if name in bone_names]
    detection.missing_bones = [
        name for name in skeleton.RIGIFY_SIGNATURE_BONES if name not in bone_names
    ]

    has_rigify_types = False
    for pose_bone in armature.pose.bones:
        if str(getattr(pose_bone, "rigify_type", "") or ""):
            has_rigify_types = True
            break
    detection.is_metarig = has_rigify_types and not detection.rig_id
    if detection.is_metarig:
        detection.error = errors.MocapError(
            errors.RIGIFY_NOT_FOUND,
            "{0} 看起来是 Rigify metarig，不是生成后的 rig。".format(armature.name),
            suggestion="先在 metarig 上执行 Rigify 的 Generate Rig，然后选择生成出的 rig。",
            details={"object": armature.name},
        )
        return detection

    if detection.rig_id or len(detection.found_bones) >= SIGNATURE_THRESHOLD:
        detection.ok = True
        return detection

    detection.error = errors.MocapError(
        errors.RIGIFY_NOT_FOUND,
        "{0} 不像 Rigify Human 生成 rig（缺少 {1}）。".format(
            armature.name, "、".join(detection.missing_bones[:6])
        ),
        details={"object": armature.name, "missing_bones": detection.missing_bones},
    )
    return detection


# --------------------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------------------


def _bone_depth(bone) -> int:
    """Number of ancestors of ``bone`` in the armature hierarchy."""
    depth = 0
    parent = bone.parent
    while parent is not None and depth < _MAX_DEPTH:
        depth += 1
        parent = parent.parent
    return depth


def _rest_direction(bone) -> tuple:
    delta = bone.tail_local - bone.head_local
    return (delta.x, delta.y, delta.z)


def _rest_reference(armature, spec) -> tuple:
    """Rest-space right-to-left reference axis for aim-plus-reference chains."""
    if spec.ref == skeleton.REF_HIP_LINE:
        left = armature.data.bones.get("thigh_fk.L")
        right = armature.data.bones.get("thigh_fk.R")
    elif spec.ref == skeleton.REF_SHOULDER_LINE:
        left = armature.data.bones.get("upper_arm_fk.L")
        right = armature.data.bones.get("upper_arm_fk.R")
    else:
        return (1.0, 0.0, 0.0)
    if left is None or right is None:
        return (1.0, 0.0, 0.0)
    delta = left.head_local - right.head_local
    reference = (delta.x, delta.y, delta.z)
    return reference if rm.vec_length(reference) > 1e-6 else (1.0, 0.0, 0.0)


def rig_pelvis_height(armature) -> float:
    """Rest height of the rig's pelvis, used for root-motion scaling."""
    for name in skeleton.RIG_PELVIS_REFERENCE_BONES:
        bone = armature.data.bones.get(name)
        if bone is not None:
            return float(bone.head_local.z)
    heights = [bone.head_local.z for bone in armature.data.bones]
    return float(sum(heights) / len(heights)) if heights else 1.0


def build_rigify_mapping(armature, include_hands: bool = True) -> RigifyMapping:
    """Build source standard skeleton to Rigify control bone mapping."""
    detection = detect_rigify_human(armature)
    if not detection.ok:
        raise detection.error

    mapping = RigifyMapping(armature)
    mapping.pelvis_height = rig_pelvis_height(armature)
    bones = armature.data.bones

    for spec in skeleton.RETARGET_CHAINS:
        if spec.group == "hand" and not include_hands:
            continue
        bone = None
        for candidate in spec.candidates:
            bone = bones.get(candidate)
            if bone is not None:
                break
        if bone is None:
            if spec.required:
                mapping.missing_required.append(spec.target)
            else:
                mapping.missing_optional.append(spec.target)
            continue
        mapping.chains.append(
            ResolvedChain(
                spec,
                bone.name,
                _bone_depth(bone),
                _rest_direction(bone),
                _rest_reference(armature, spec),
            )
        )

    if mapping.missing_required:
        raise errors.MocapError(
            errors.RIGIFY_MAPPING_FAILED,
            "目标 rig 缺少必需的 Rigify 控制骨：{0}".format("、".join(mapping.missing_required[:8])),
            suggestion="确认该 rig 由标准 Rigify Human metarig 生成。",
            details={"object": armature.name, "missing_bones": mapping.missing_required},
        )
    if mapping.missing_optional:
        mapping.warnings.append(
            "以下可选控制骨不存在，已跳过：{0}".format("、".join(mapping.missing_optional[:8]))
        )
    return mapping


# --------------------------------------------------------------------------------------
# Retargeting
# --------------------------------------------------------------------------------------


class RetargetOptions(object):
    """Options for :func:`retarget_to_rigify`."""

    __slots__ = (
        "root_motion",
        "include_hands",
        "switch_limbs_to_fk",
        "flip_x",
        "scale_mode",
        "frame_step",
        "action_name",
        "progress",
    )

    def __init__(
        self,
        root_motion: str = "world",
        include_hands: bool = True,
        switch_limbs_to_fk: bool = True,
        flip_x: bool = False,
        scale_mode: str = "target_rig_height",
        frame_step: int = 1,
        action_name: str = "",
        progress=None,
    ) -> None:
        self.root_motion = root_motion
        self.include_hands = bool(include_hands)
        self.switch_limbs_to_fk = bool(switch_limbs_to_fk)
        self.flip_x = bool(flip_x)
        self.scale_mode = scale_mode
        self.frame_step = max(1, int(frame_step))
        self.action_name = action_name
        self.progress = progress


def _source_reference(frame, ref: str):
    """Source-space right-to-left reference axis, or ``None``."""
    if ref == skeleton.REF_HIP_LINE:
        left = frame.body3d.get("hip.L")
        right = frame.body3d.get("hip.R")
    elif ref == skeleton.REF_SHOULDER_LINE:
        left = frame.body3d.get("shoulder.L")
        right = frame.body3d.get("shoulder.R")
    else:
        return None
    if left is None or right is None:
        return None
    delta = rm.vec_sub(left, right)
    return delta if rm.vec_length(delta) > 1e-6 else None


def _target_rotation(chain, frame):
    """World-space rotation for ``chain`` at ``frame``, or ``None`` when unusable."""
    spec = chain.spec
    if spec.source is None:
        return None
    start_name, end_name = spec.source
    if end_name is None:
        return None
    start = frame.joint(start_name)
    end = frame.joint(end_name)
    if start is None or end is None:
        return None
    direction = rm.vec_sub(end, start)
    if rm.vec_length(direction) < 1e-6:
        return None

    if spec.mode == skeleton.MODE_AIM_REF:
        source_reference = _source_reference(frame, spec.ref)
        if source_reference is not None:
            return rm.aim_rotation_with_reference(
                chain.rest_direction, chain.rest_reference, direction, source_reference
            )
    return rm.aim_rotation(chain.rest_direction, direction)


def _ensure_action(armature, name: str):
    """Create and assign a fresh Action.

    Deliberately avoids the slotted-Action API: ``action.slots`` only exists in
    Blender 4.4+ and ``slots.new_for_id`` is unavailable in 4.5.0, while plain
    assignment plus ``keyframe_insert`` works identically on 4.0 through 4.5.
    """
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    return action


def _switch_to_fk(armature, frame: int) -> list:
    """Force the Rigify limbs into FK so written keys drive the deform bones."""
    switched = []
    for bone_name in skeleton.IK_FK_SWITCH_BONES:
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None or skeleton.IK_FK_PROPERTY not in pose_bone:
            continue
        pose_bone[skeleton.IK_FK_PROPERTY] = 1.0
        try:
            pose_bone.keyframe_insert('["{0}"]'.format(skeleton.IK_FK_PROPERTY), frame=frame)
        except (RuntimeError, TypeError):  # pragma: no cover - not animatable
            pass
        switched.append(bone_name)
    return switched


def _apply_world_rotation(pose_bone, armature, quaternion) -> None:
    """Set a bone's world rotation while preserving its evaluated head position."""
    world_quat = Quaternion(
        (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))
    )
    rest_world = armature.matrix_world @ pose_bone.bone.matrix_local
    target = world_quat.to_matrix().to_4x4() @ rest_world
    # Keep the position the parent chain already determined; only rotate.
    target.translation = pose_bone.matrix.translation
    pose_bone.matrix = target


def _apply_root(pose_bone, armature, frame, scale: float, origin, options) -> None:
    """Write the root motion translation onto the torso control."""
    pelvis = frame.body3d.get("pelvis")
    if pelvis is None:
        return
    scaled = rm.vec_scale(pelvis, scale)
    delta = rm.vec_sub(scaled, origin)
    if options.root_motion == "in_place":
        delta = (0.0, 0.0, delta[2])
    rest_world = armature.matrix_world @ pose_bone.bone.matrix_local
    pose_bone.matrix = Matrix.Translation(Vector(delta)) @ rest_world


def _root_scale(result, mapping, options) -> float:
    """Scale mapping source translation onto the target rig proportions."""
    if options.scale_mode != "target_rig_height":
        return 1.0
    heights = [
        frame.body3d["pelvis"][2] for frame in result.frames[:120] if "pelvis" in frame.body3d
    ]
    if not heights:
        return 1.0
    source_height = sorted(heights)[len(heights) // 2]
    if source_height <= 1e-3 or mapping.pelvis_height <= 1e-3:
        return 1.0
    return float(mapping.pelvis_height) / float(source_height)


def _root_origin(result, scale: float) -> tuple:
    """First-frame pelvis position, scaled; the reference for root motion."""
    pelvis = result.frames[0].body3d.get("pelvis")
    if pelvis is None:
        return (0.0, 0.0, 0.0)
    return rm.vec_scale(pelvis, scale)


def _depth_groups(chains) -> list:
    """Chains grouped by hierarchy depth, shallowest first."""
    groups = {}
    for chain in chains:
        groups.setdefault(chain.depth, []).append(chain)
    return [groups[depth] for depth in sorted(groups)]


def _activate(armature):
    """Make ``armature`` the active, selected object and enter pose mode."""
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = armature.mode
    try:
        armature.hide_set(False)
    except (AttributeError, RuntimeError):  # pragma: no cover - not in a view layer
        pass
    armature.select_set(True)
    view_layer.objects.active = armature
    if armature.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")
    return previous_active, previous_mode


def _restore(armature, previous_active, previous_mode) -> None:
    if previous_mode and armature.mode != previous_mode:
        try:
            bpy.ops.object.mode_set(mode=previous_mode)
        except RuntimeError:  # pragma: no cover - mode unavailable in this context
            pass
    if previous_active is not None:
        try:
            bpy.context.view_layer.objects.active = previous_active
        except (AttributeError, RuntimeError):  # pragma: no cover
            pass


def retarget_to_rigify(result, armature, options=None) -> bpy.types.Action:
    """Insert keyframes on Rigify control bones from a mocap result.

    Returns the created Action. Recoverable problems are collected on the
    Action's ``["mocap_warnings"]`` custom property so the UI can show them
    without failing the operation (guide section 15: warnings are not fatal).
    """
    options = options or RetargetOptions()
    warnings = []

    if not isinstance(result, result_schema.MocapResult):
        raise errors.MocapError(
            errors.RESULT_SCHEMA_INVALID, "retarget 需要一个已校验的 MocapResult。"
        )
    if not result.frames:
        raise errors.MocapError(errors.RESULT_SCHEMA_INVALID, "结果中没有任何帧。")

    if result_schema.looks_y_up(result):
        warnings.append(
            "[{0}] 结果看起来是 Y 轴向上的数据，但 v0.1 规定 Z 轴向上，动画可能躺倒。".format(
                CODE_SUSPECT_COORDS
            )
        )
    if result_schema.detect_mirror(result):
        if options.flip_x:
            warnings.append("已按 Flip X 选项镜像结果。")
        else:
            warnings.append(
                "[{0}] 结果疑似左右镜像（hip.L 的 X 小于 hip.R）。如动作左右颠倒请勾选 Flip X。".format(
                    CODE_SUSPECT_MIRROR
                )
            )
    if options.flip_x:
        result_schema.mirror_result_x(result)

    mapping = build_rigify_mapping(armature, include_hands=options.include_hands)
    warnings.extend(mapping.warnings)

    has_hands = result.has_hands
    if options.include_hands and not has_hands and mapping.hand_chains():
        warnings.append("结果中没有 hands3d 数据，已跳过手指链。")
    chains = mapping.chains if has_hands else mapping.body_chains()

    scale = _root_scale(result, mapping, options)
    origin = _root_origin(result, scale)
    depth_groups = _depth_groups(chains)

    previous_active, previous_mode = _activate(armature)
    action = _ensure_action(armature, options.action_name or result.action_name())
    scene = bpy.context.scene
    previous_frame = scene.frame_current

    try:
        for chain in chains:
            pose_bone = armature.pose.bones.get(chain.bone_name)
            if pose_bone is not None:
                pose_bone.rotation_mode = ROTATION_MODE

        if options.switch_limbs_to_fk:
            if not _switch_to_fk(armature, result.frames[0].frame):
                warnings.append("目标 rig 上没有 IK_FK 属性，跳过 FK 切换。")

        skipped = {}
        total = len(result.frames)
        written = 0

        for position, frame in enumerate(result.frames):
            if (
                options.frame_step > 1
                and position % options.frame_step
                and position not in (0, total - 1)
            ):
                continue
            scene.frame_set(frame.frame)

            for group in depth_groups:
                for chain in group:
                    pose_bone = armature.pose.bones.get(chain.bone_name)
                    if pose_bone is None:
                        continue
                    mode = chain.spec.mode
                    if mode == skeleton.MODE_ROOT:
                        _apply_root(pose_bone, armature, frame, scale, origin, options)
                        continue
                    if mode == skeleton.MODE_IDENTITY:
                        # Follow the parent exactly: rest orientation relative to it.
                        pose_bone.rotation_quaternion = IDENTITY_QUATERNION
                        continue
                    rotation = _target_rotation(chain, frame)
                    if rotation is None:
                        skipped[chain.bone_name] = skipped.get(chain.bone_name, 0) + 1
                        continue
                    _apply_world_rotation(pose_bone, armature, rotation)
                bpy.context.view_layer.update()

            for chain in chains:
                pose_bone = armature.pose.bones.get(chain.bone_name)
                if pose_bone is None:
                    continue
                pose_bone.keyframe_insert("rotation_quaternion", frame=frame.frame)
                if chain.spec.mode == skeleton.MODE_ROOT:
                    pose_bone.keyframe_insert("location", frame=frame.frame)
            written += 1

            if options.progress is not None and (position % 15 == 0 or position == total - 1):
                options.progress(position + 1, total)

        for bone_name, count in sorted(skipped.items()):
            warnings.append(
                "[{0}] {1} 有 {2} 帧缺少源数据，已保留上一姿态。".format(
                    CODE_CHAIN_SKIPPED, bone_name, count
                )
            )
    finally:
        try:
            scene.frame_set(previous_frame)
        except Exception:  # pragma: no cover - defensive
            pass
        _restore(armature, previous_active, previous_mode)

    action["mocap_warnings"] = warnings
    action["mocap_profile"] = result.profile
    action["mocap_source"] = result.source_path
    action["mocap_frame_start"] = result.frame_start
    action["mocap_frame_end"] = result.frame_end
    action["mocap_bones"] = [chain.bone_name for chain in chains]
    action["mocap_written_frames"] = written
    action["mocap_scale"] = scale
    return action
