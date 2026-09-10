"""Tracking and cleanup of temporary Blender data created during retargeting.

Guide sections 10.4 and 4.4 require every temporary object, armature and
constraint to be removed after baking. Everything this add-on creates carries the
:data:`TEMP_PREFIX` so a cleanup pass can find leftovers even after a crash or a
file reload.
"""

from __future__ import annotations

import bpy

#: Name prefix for every temporary datablock this add-on creates.
TEMP_PREFIX = "MOCAP_TMP_"

#: Custom property marking a constraint or object as add-on owned.
TEMP_MARKER = "mocap_temp"

#: In-session registry of created objects, keyed by name.
_created_objects = []

#: In-session registry of ``(armature_object_name, bone_name, constraint_name)``.
_created_constraints = []


def temp_name(suffix: str) -> str:
    """Build a prefixed name for a temporary datablock."""
    return "{0}{1}".format(TEMP_PREFIX, suffix)


def register_object(obj) -> None:
    """Track ``obj`` for later cleanup."""
    if obj is None:
        return
    obj[TEMP_MARKER] = True
    if obj.name not in _created_objects:
        _created_objects.append(obj.name)


def register_constraint(armature_object, bone_name: str, constraint) -> None:
    """Track a pose bone constraint for later removal."""
    if armature_object is None or constraint is None:
        return
    try:
        constraint[TEMP_MARKER] = True
    except TypeError:  # pragma: no cover - some constraint types reject id props
        pass
    entry = (armature_object.name, bone_name, constraint.name)
    if entry not in _created_constraints:
        _created_constraints.append(entry)


def create_empty(name: str, location=(0.0, 0.0, 0.0), collection=None):
    """Create a tracked empty object."""
    empty = bpy.data.objects.new(temp_name(name), None)
    empty.empty_display_size = 0.05
    empty.location = location
    target = collection or bpy.context.scene.collection
    target.objects.link(empty)
    register_object(empty)
    return empty


def remove_constraints(armature_object=None) -> int:
    """Remove tracked constraints; returns the number removed."""
    removed = 0
    for object_name, bone_name, constraint_name in list(_created_constraints):
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "ARMATURE":
            _created_constraints.remove((object_name, bone_name, constraint_name))
            continue
        if armature_object is not None and obj is not armature_object:
            continue
        pose_bone = obj.pose.bones.get(bone_name)
        if pose_bone is not None:
            constraint = pose_bone.constraints.get(constraint_name)
            if constraint is not None:
                pose_bone.constraints.remove(constraint)
                removed += 1
        _created_constraints.remove((object_name, bone_name, constraint_name))
    return removed


def _sweep_marked_constraints(armature_object=None) -> int:
    """Remove any prefixed/marked constraint, even from a previous session."""
    removed = 0
    for obj in bpy.data.objects:
        if armature_object is not None and obj is not armature_object:
            continue
        if obj.type != "ARMATURE" or obj.pose is None:
            continue
        for pose_bone in obj.pose.bones:
            for constraint in list(pose_bone.constraints):
                try:
                    marked = bool(constraint.get(TEMP_MARKER))
                except TypeError:
                    marked = False
                if constraint.name.startswith(TEMP_PREFIX) or marked:
                    pose_bone.constraints.remove(constraint)
                    removed += 1
    return removed


def cleanup_all(purge_orphans: bool = True) -> dict:
    """Remove every temporary object and constraint this add-on created.

    Safe to call repeatedly and safe to call from ``unregister()``.
    """
    report = {"objects": 0, "constraints": 0, "armatures": 0}
    report["constraints"] = remove_constraints()
    try:
        report["constraints"] += _sweep_marked_constraints()
    except Exception:  # pragma: no cover - defensive during teardown
        pass

    names = list(_created_objects)
    for obj in bpy.data.objects:
        if obj.name.startswith(TEMP_PREFIX) or obj.get(TEMP_MARKER):
            if obj.name not in names:
                names.append(obj.name)

    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        data = obj.data
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            report["objects"] += 1
        except (ReferenceError, RuntimeError):  # pragma: no cover
            continue
        if purge_orphans and data is not None and getattr(data, "users", 1) == 0:
            if isinstance(data, bpy.types.Armature):
                try:
                    bpy.data.armatures.remove(data)
                    report["armatures"] += 1
                except (ReferenceError, RuntimeError):  # pragma: no cover
                    pass
    del _created_objects[:]
    return report


def has_temp_data() -> bool:
    """True when tracked or prefixed temporary data still exists."""
    if _created_constraints:
        return True
    for obj in bpy.data.objects:
        if obj.name.startswith(TEMP_PREFIX) or obj.get(TEMP_MARKER):
            return True
    return False
