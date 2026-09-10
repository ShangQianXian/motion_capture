"""Blender background test: mock result -> Rigify Action -> bake.

Phase 4 acceptance (guide section, Phase 4):

* a standard Rigify human rig is detected,
* the mock result produces an Action,
* that Action contains keyframes,
* the baked animation is editable,
* temporary objects and constraints can be cleaned up.

The mock result is generated in-process by ``backend_worker.mock_source``, which
is standard-library only and therefore importable in Blender's bundled Python -
no worker subprocess and no models needed.

Run::

    blender --background --factory-startup --python tests/blender/test_mock_retarget.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness as harness  # noqa: E402

#: Bones that must receive rotation keyframes.
EXPECTED_BONES = (
    "torso",
    "spine_fk",
    "spine_fk.001",
    "spine_fk.002",
    "spine_fk.003",
    "neck",
    "head",
    "shoulder.L",
    "upper_arm_fk.L",
    "forearm_fk.L",
    "hand_fk.L",
    "shoulder.R",
    "upper_arm_fk.R",
    "forearm_fk.R",
    "hand_fk.R",
    "thigh_fk.L",
    "shin_fk.L",
    "foot_fk.L",
    "toe_fk.L",
    "thigh_fk.R",
    "shin_fk.R",
    "foot_fk.R",
    "toe_fk.R",
)


def write_mock_result(directory: str, frames: int = 30) -> str:
    """Generate and write a mock ``mocap_result.json``; returns its path."""
    from motion_capture.backend_worker import export_result, mock_source, postprocess

    generated = mock_source.generate_frames(
        frame_start=1, frame_end=frames, fps=30.0, include_hands=True
    )
    generated, warnings = postprocess.postprocess(
        generated, 30.0, {"smoothing_strength": 0.65, "foot_lock_strength": 0.7}
    )
    payload = export_result.build_result(
        generated,
        30.0,
        source_path="D:/media/mock_walk.mp4",
        source_type="video",
        profile="preview",
        warnings=postprocess.summarise_warnings(warnings),
    )
    return export_result.write_result(payload, directory)


def bone_curve_map(action) -> dict:
    """``{bone_name: {data_path_suffix: keyframe_count}}`` for an Action."""
    result = {}
    for fcurve in action.fcurves:
        path = fcurve.data_path
        if not path.startswith('pose.bones["'):
            continue
        name = path.split('"')[1]
        suffix = path.rsplit("].", 1)[-1] if "]." in path else path
        entry = result.setdefault(name, {})
        entry[suffix] = entry.get(suffix, 0) + len(fcurve.keyframe_points)
    return result


def main() -> None:
    print("Blender {0} / Python {1}".format(bpy.app.version_string, sys.version.split()[0]))

    harness.section("setup")
    try:
        module, used_operator = harness.enable_addon()
    except Exception as exc:  # noqa: BLE001
        harness.fatal("enabling the add-on raised", exc)
        return
    harness.check(module is not None, "add-on enabled")

    from motion_capture.blender import action_baker, rigify_adapter, temp_data
    from motion_capture.core import errors, result_schema, skeleton

    workdir = tempfile.mkdtemp(prefix="mocap_test_")
    result_path = write_mock_result(workdir, frames=30)
    harness.check(os.path.isfile(result_path), "mock result written")

    harness.section("result import")
    try:
        result = result_schema.load_mocap_result(result_path)
    except errors.MocapError as exc:
        harness.fatal("mock result failed validation: {0}".format(exc.message), exc)
        return
    harness.check_equal(len(result.frames), 30, "frame count")
    harness.check_equal(result.coordinate_system, "blender_world", "coordinate system")
    harness.check_equal(result.unit, "meter", "unit")
    harness.check_equal(result.skeleton_id, skeleton.SKELETON_ID, "skeleton id")
    harness.check(result.has_hands, "mock result carries hand data")
    harness.check(not result_schema.detect_mirror(result), "mock result is not mirrored")
    harness.check(not result_schema.looks_y_up(result), "mock result is Z-up")
    harness.check_equal(result.action_name(), "Mocap_mock_walk_preview", "action name pattern")

    harness.section("rigify rig")
    rig = harness.generate_rigify_human()
    if rig is None:
        harness.fatal("could not generate a Rigify human rig in this Blender build")
        return
    harness.check_equal(rig.type, "ARMATURE", "generated object is an armature")
    harness.check(bool(rig.data.get("rig_id")), "generated rig carries rig_id")

    detection = rigify_adapter.detect_rigify_human(rig)
    harness.check(detection.ok, "detect_rigify_human accepts the generated rig")
    harness.check_equal(len(detection.missing_bones), 0, "no signature bones missing")

    harness.section("metarig is rejected")
    metarig = bpy.data.objects.get("metarig")
    if metarig is not None:
        meta_detection = rigify_adapter.detect_rigify_human(metarig)
        harness.check(not meta_detection.ok, "metarig is rejected")
        if meta_detection.error is not None:
            harness.check_equal(
                meta_detection.error.code, errors.RIGIFY_NOT_FOUND, "metarig error code"
            )

    harness.section("non-armature is rejected")
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube_detection = rigify_adapter.detect_rigify_human(cube)
    harness.check(not cube_detection.ok, "mesh object is rejected")

    harness.section("mapping")
    try:
        mapping = rigify_adapter.build_rigify_mapping(rig, include_hands=True)
    except errors.MocapError as exc:
        harness.fatal("build_rigify_mapping failed: {0}".format(exc.message), exc)
        return
    harness.check_equal(len(mapping.missing_required), 0, "no required chain missing")
    harness.check(mapping.pelvis_height > 0.5, "pelvis height looks sane ({0:.3f})".format(mapping.pelvis_height))
    mapped = set(mapping.bone_names)
    missing = [name for name in EXPECTED_BONES if name not in mapped]
    harness.check(not missing, "all expected body bones mapped (missing {0})".format(missing))
    harness.check(len(mapping.hand_chains()) == 30, "30 finger chains mapped (got {0})".format(len(mapping.hand_chains())))

    depths = {chain.bone_name: chain.depth for chain in mapping.chains}
    harness.check(
        depths.get("forearm_fk.L", 0) > depths.get("upper_arm_fk.L", 99),
        "forearm depth is below upper arm (parents evaluate first)",
    )

    # The invariant the retarget ordering actually relies on: for every mapped
    # bone, any mapped ANCESTOR must have a strictly smaller depth, so grouping by
    # depth is a valid topological order.
    #
    # Note Rigify's spine is pivot-based, not a simple upward chain: spine_fk (the
    # hip segment) is a CHILD of spine_fk.001, and the legs hang off spine_fk. So
    # "spine_fk.003 deeper than spine_fk" is NOT true and must not be asserted.
    violations = []
    for chain in mapping.chains:
        bone = rig.data.bones.get(chain.bone_name)
        parent = bone.parent if bone is not None else None
        while parent is not None:
            if parent.name in depths and depths[parent.name] >= chain.depth:
                violations.append((chain.bone_name, parent.name))
            parent = parent.parent
    harness.check(
        not violations,
        "depth ordering is a valid topological order (violations {0})".format(violations),
    )

    harness.section("retarget")
    options = rigify_adapter.RetargetOptions(
        root_motion="world", include_hands=True, switch_limbs_to_fk=True
    )
    try:
        action = rigify_adapter.retarget_to_rigify(result, rig, options)
    except errors.MocapError as exc:
        harness.fatal("retarget_to_rigify failed: {0}".format(exc.message), exc)
        return
    except Exception as exc:  # noqa: BLE001
        harness.fatal("retarget_to_rigify raised", exc)
        return

    harness.check(action is not None, "action created")
    harness.check_equal(action.name, "Mocap_mock_walk_preview", "action name")
    harness.check(rig.animation_data.action is action, "action assigned to the rig")

    curves = bone_curve_map(action)
    harness.check(len(action.fcurves) > 0, "action has F-Curves ({0})".format(len(action.fcurves)))
    total_keys = sum(len(fc.keyframe_points) for fc in action.fcurves)
    harness.check(total_keys > 0, "action has keyframes ({0})".format(total_keys))

    unkeyed = [name for name in EXPECTED_BONES if name not in curves]
    harness.check(not unkeyed, "every expected bone has curves (missing {0})".format(unkeyed))
    for name in ("upper_arm_fk.L", "thigh_fk.R", "spine_fk"):
        entry = curves.get(name, {})
        harness.check_equal(
            entry.get("rotation_quaternion"), 120, "{0} has 4x30 rotation keys".format(name)
        )
    harness.check(
        curves.get("torso", {}).get("location") == 90,
        "torso has 3x30 location keys (got {0})".format(curves.get("torso", {}).get("location")),
    )
    harness.check(
        "location" not in curves.get("upper_arm_fk.L", {}),
        "no stray location keys on limb FK bones",
    )

    harness.section("IK/FK switch")
    for name in skeleton.IK_FK_SWITCH_BONES:
        pose_bone = rig.pose.bones.get(name)
        if pose_bone is None:
            continue
        harness.check_almost_equal(
            pose_bone[skeleton.IK_FK_PROPERTY], 1.0, "{0} switched to FK".format(name)
        )

    harness.section("pose actually moves and keeps handedness")
    bpy.context.scene.frame_set(result.frames[0].frame)
    bpy.context.view_layer.update()
    left = rig.pose.bones["upper_arm_fk.L"]
    right = rig.pose.bones["upper_arm_fk.R"]
    harness.check(
        (rig.matrix_world @ left.head).x > (rig.matrix_world @ right.head).x,
        "left arm stays on +X (no left/right flip)",
    )
    left_thigh = rig.pose.bones["thigh_fk.L"]
    right_thigh = rig.pose.bones["thigh_fk.R"]
    harness.check(
        (rig.matrix_world @ left_thigh.head).x > (rig.matrix_world @ right_thigh.head).x,
        "left leg stays on +X",
    )

    rotations = []
    for frame in (result.frames[0].frame, result.frames[len(result.frames) // 2].frame):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        rotations.append(rig.pose.bones["thigh_fk.L"].rotation_quaternion.copy())
    delta = sum(abs(a - b) for a, b in zip(rotations[0], rotations[1]))
    harness.check(delta > 1e-4, "thigh_fk.L actually animates (delta {0:.5f})".format(delta))

    harness.section("bake")
    protected = rig.pose.bones['upper_arm_fk.L'].constraints.new('LIMIT_ROTATION')
    protected.name = 'UserKeepConstraint'
    temporary = rig.pose.bones['upper_arm_fk.L'].constraints.new('LIMIT_ROTATION')
    temporary.name = temp_data.temp_name('bake_test')
    temp_data.register_constraint(rig, 'upper_arm_fk.L', temporary)
    other = bpy.data.objects.get('metarig')
    other.hide_set(False)
    other.select_set(True)
    other_action = other.animation_data.action if other.animation_data else None
    other_temp = other.pose.bones[0].constraints.new('LIMIT_ROTATION')
    other_temp.name = temp_data.temp_name('other_rig')
    other_temp_name = other_temp.name
    selected_before = {obj.name for obj in bpy.context.selected_objects}
    active_before = bpy.context.view_layer.objects.active
    mode_before = active_before.mode
    frame_before = bpy.context.scene.frame_current
    try:
        baked = action_baker.bake_action(rig, result.frame_start, result.frame_end)
    except errors.MocapError as exc:
        harness.fatal("bake_action failed: {0}".format(exc.message), exc)
        return
    except Exception as exc:  # noqa: BLE001
        harness.fatal("bake_action raised", exc)
        return

    summary = action_baker.action_summary(baked)
    harness.check(rig.pose.bones['upper_arm_fk.L'].constraints.get('UserKeepConstraint') is not None, 'baking preserves user constraints')
    harness.check(rig.pose.bones['upper_arm_fk.L'].constraints.get(temp_data.temp_name('bake_test')) is None, 'baking removes only target temporary constraints')
    harness.check(other.pose.bones[0].constraints.get(other_temp_name) is not None, 'baking preserves other rig temporary constraints')
    harness.check((other.animation_data.action if other.animation_data else None) is other_action, 'baking leaves other selected armature animation untouched')
    harness.check_equal({obj.name for obj in bpy.context.selected_objects}, selected_before, 'baking restores selection')
    harness.check(bpy.context.view_layer.objects.active is active_before, 'baking restores active object')
    harness.check_equal(active_before.mode, mode_before, 'baking restores mode')
    harness.check_equal(bpy.context.scene.frame_current, frame_before, 'baking restores frame')
    harness.check(summary["fcurves"] > 0, "baked action has curves ({0})".format(summary["fcurves"]))
    harness.check(summary["keyframes"] > 0, "baked action has keyframes ({0})".format(summary["keyframes"]))
    harness.check(bool(baked.get("mocap_baked")), "baked action is tagged")
    harness.check(baked.use_fake_user, "baked action has a fake user so it survives a save")
    harness.check(
        rig.animation_data.action is baked, "baked action is still assigned to the rig"
    )
    editable = [fc for fc in baked.fcurves if len(fc.keyframe_points) > 1]
    harness.check(bool(editable), "baked curves contain editable keyframe points")

    harness.section("cleanup")
    report = temp_data.cleanup_all()
    harness.check(isinstance(report, dict), "cleanup returns a report")
    leftovers = [o.name for o in bpy.data.objects if o.name.startswith(temp_data.TEMP_PREFIX)]
    harness.check(not leftovers, "no temporary objects left (found {0})".format(leftovers))
    harness.check(not temp_data.has_temp_data(), "has_temp_data reports clean")

    harness.section("in-place root motion")
    options_in_place = rigify_adapter.RetargetOptions(
        root_motion="in_place", include_hands=False, action_name="Mocap_inplace_test"
    )
    try:
        in_place = rigify_adapter.retarget_to_rigify(result, rig, options_in_place)
    except Exception as exc:  # noqa: BLE001
        harness.fatal("in-place retarget raised", exc)
        return
    torso_curves = [
        fc for fc in in_place.fcurves if fc.data_path == 'pose.bones["torso"].location'
    ]
    horizontal = [fc for fc in torso_curves if fc.array_index in (0, 1)]
    flat = all(
        all(abs(kp.co[1]) < 1e-6 for kp in fc.keyframe_points) for fc in horizontal
    )
    harness.check(flat, "in_place removes horizontal root translation")
    vertical = [fc for fc in torso_curves if fc.array_index == 2]
    moves = any(
        max(kp.co[1] for kp in fc.keyframe_points) - min(kp.co[1] for kp in fc.keyframe_points) > 1e-5
        for fc in vertical
    )
    harness.check(moves, "in_place keeps vertical bob")

    harness.section("repeat mirrored retarget")
    original_body = dict(result.frames[0].body3d)
    mirrored_options = rigify_adapter.RetargetOptions(flip_x=True, include_hands=False, action_name='MirrorRegression')
    def key_values(action):
        return {(fc.data_path, fc.array_index): tuple(round(float(kp.co[1]), 5) for kp in fc.keyframe_points)
                for fc in action.fcurves}
    first_mirror = key_values(rigify_adapter.retarget_to_rigify(result, rig, mirrored_options))
    second_mirror = key_values(rigify_adapter.retarget_to_rigify(result, rig, mirrored_options))
    harness.check(result.frames[0].body3d == original_body, 'mirrored retarget preserves cached source')
    harness.check(first_mirror == second_mirror, 'repeated mirrored application is stable')

    harness.section("teardown")
    try:
        harness.disable_addon(module, used_operator)
        harness.check(True, "add-on disabled cleanly")
    except Exception as exc:  # noqa: BLE001
        harness.check(False, "add-on disabled cleanly ({0})".format(exc))

    harness.finish()


if __name__ == "__main__":
    main()
