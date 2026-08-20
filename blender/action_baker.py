"""Action baking and cleanup (guide section 10.4).

The primary retarget path writes final FK keyframes directly, so baking is a
consolidation step: it resolves any temporary constraints into plain keyframes,
removes the temporary data and hands the user an editable Action.
"""

from __future__ import annotations

import bpy

from ..core import errors
from . import temp_data

#: Default bake settings.
DEFAULT_STEP = 1


class BoneSelectionState(object):
    """Saves and restores bone selection plus bone-collection visibility.

    ``bpy.ops.nla.bake(only_selected=True)`` can only see bones that are both
    visible and selected, and Rigify hides several control collections by
    default, so they are temporarily revealed and restored afterwards.
    """

    def __init__(self, armature) -> None:
        self.armature = armature
        self.selection = {}
        self.hidden_bones = []
        self.hidden_collections = []

    def __enter__(self):
        data = self.armature.data
        for bone in data.bones:
            self.selection[bone.name] = bone.select
        for bone in data.bones:
            if bone.hide:
                self.hidden_bones.append(bone.name)
        collections = getattr(data, "collections", None)
        if collections is not None:
            for collection in collections:
                if not collection.is_visible:
                    self.hidden_collections.append(collection.name)
        return self

    def reveal(self, bone_names) -> None:
        """Make ``bone_names`` visible and selected, everything else unselected."""
        data = self.armature.data
        collections = getattr(data, "collections", None)
        if collections is not None:
            for collection in collections:
                collection.is_visible = True
        wanted = set(bone_names)
        for bone in data.bones:
            if bone.name in wanted:
                bone.hide = False
                bone.select = True
            else:
                bone.select = False

    def __exit__(self, *exc_info):
        data = self.armature.data
        for name, selected in self.selection.items():
            bone = data.bones.get(name)
            if bone is not None:
                bone.select = selected
        for name in self.hidden_bones:
            bone = data.bones.get(name)
            if bone is not None:
                bone.hide = True
        collections = getattr(data, "collections", None)
        if collections is not None:
            for name in self.hidden_collections:
                collection = collections.get(name)
                if collection is not None:
                    collection.is_visible = False
        return False


def action_bone_names(action) -> list:
    """Bone names recorded on the Action, falling back to its F-Curve paths."""
    recorded = action.get("mocap_bones") if action is not None else None
    if recorded:
        return [str(name) for name in recorded]
    names = []
    if action is None:
        return names
    for fcurve in action.fcurves:
        path = fcurve.data_path
        if not path.startswith('pose.bones["'):
            continue
        name = path.split('"')[1]
        if name not in names:
            names.append(name)
    return names


def bake_action(
    armature,
    frame_start: int,
    frame_end: int,
    step: int = DEFAULT_STEP,
    clean_curves: bool = False,
    only_mapped: bool = True,
) -> bpy.types.Action:
    """Bake visual transforms into an editable Blender Action.

    Clears the temporary constraints and objects the retarget stage may have
    created, then returns the Action still assigned to ``armature``.
    """
    if armature is None or getattr(armature, "type", None) != "ARMATURE":
        raise errors.MocapError(
            errors.RIGIFY_NOT_FOUND, "烘焙需要一个 Armature 对象。"
        )
    animation_data = armature.animation_data
    action = animation_data.action if animation_data is not None else None
    if action is None:
        raise errors.MocapError(
            errors.RIGIFY_MAPPING_FAILED,
            "目标 rig 上没有可烘焙的 Action，请先执行 Apply to Rigify。",
            details={"object": armature.name},
        )

    start = int(frame_start)
    end = int(max(frame_end, frame_start))
    bones = action_bone_names(action) if only_mapped else [b.name for b in armature.data.bones]

    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = armature.mode
    try:
        armature.hide_set(False)
    except (AttributeError, RuntimeError):  # pragma: no cover
        pass
    armature.select_set(True)
    view_layer.objects.active = armature
    if armature.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")

    try:
        with BoneSelectionState(armature) as state:
            state.reveal(bones)
            try:
                bpy.ops.nla.bake(
                    frame_start=start,
                    frame_end=end,
                    step=max(1, int(step)),
                    only_selected=bool(only_mapped),
                    visual_keying=True,
                    clear_constraints=True,
                    clear_parents=False,
                    use_current_action=True,
                    clean_curves=bool(clean_curves),
                    bake_types={"POSE"},
                )
            except RuntimeError as exc:
                raise errors.MocapError(
                    errors.RIGIFY_MAPPING_FAILED,
                    "烘焙失败：{0}".format(exc),
                    details={"object": armature.name, "frame_start": start, "frame_end": end},
                )
    finally:
        if previous_mode and armature.mode != previous_mode:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:  # pragma: no cover
                pass
        if previous_active is not None:
            try:
                view_layer.objects.active = previous_active
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass

    temp_data.cleanup_all()

    baked = armature.animation_data.action
    baked.use_fake_user = True
    baked["mocap_baked"] = True
    baked["mocap_bake_range"] = [start, end]
    return baked


def action_summary(action) -> dict:
    """Small report used by the operators and the log panel."""
    if action is None:
        return {"name": "", "fcurves": 0, "keyframes": 0, "bones": 0}
    keyframes = 0
    for fcurve in action.fcurves:
        keyframes += len(fcurve.keyframe_points)
    return {
        "name": action.name,
        "fcurves": len(action.fcurves),
        "keyframes": keyframes,
        "bones": len(action_bone_names(action)),
        "warnings": list(action.get("mocap_warnings") or []),
        "baked": bool(action.get("mocap_baked")),
    }
