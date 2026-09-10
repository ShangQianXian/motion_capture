"""Check evaluated bone directions, object transforms and the calibration UI."""
import copy
import math
import os
import sys
import tempfile

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as harness


def main():
    harness.reset_scene()
    module, used_operator = harness.enable_addon()
    from motion_capture.backend_worker import export_result, mock_source
    from motion_capture.core import pose_calibration, result_schema
    from motion_capture.blender import rigify_adapter as adapter

    rig = harness.generate_rigify_human()
    if rig is None:
        harness.fatal('Rigify generation failed')
    payload = export_result.build_result(mock_source.generate_frames(frame_start=1,frame_end=3),24.)
    result = result_schema.MocapResult(payload)
    source = copy.deepcopy(result.frames[0].body3d)
    mapping = adapter.build_rigify_mapping(rig, include_hands=False)
    original_height = mapping.pelvis_height
    for degrees, size in ((0.,1.), (30.,1.7), (-25.,.6)):
        rig.location = (3.,-2.,1.)
        rig.rotation_euler = (math.radians(degrees),0.,.3)
        rig.scale = (size,)*3
        bpy.context.view_layer.update()
        adapter.retarget_to_rigify(result,rig,adapter.RetargetOptions(include_hands=False))
        errors = []
        for frame in result.frames:
            bpy.context.scene.frame_set(frame.frame)
            bpy.context.view_layer.update()
            for chain in mapping.body_chains():
                if not chain.spec.source or not chain.spec.source[1]:
                    continue
                start, end = (frame.joint(name) for name in chain.spec.source)
                if start is None or end is None:
                    continue
                pb = rig.pose.bones[chain.bone_name]
                actual = rig.matrix_world.to_3x3() @ (pb.tail-pb.head)
                expected = Vector(end)-Vector(start)
                errors.append(math.degrees(actual.angle(expected)))
        harness.check(max(errors) < .1, 'world bone directions match with X rotation {0} / scale {1}: {2:.4f} degrees'.format(degrees,size,max(errors)))
        bpy.context.scene.frame_set(1)
        rest_head = rig.data.bones['torso'].head_local
        harness.check((rig.pose.bones['torso'].head-rest_head).length < 1e-4,
                      'first-frame root does not apply object translation twice')
        harness.check(abs(adapter.rig_pelvis_height(rig)-original_height) < 1e-6,
                      'object transforms preserve local rig dimensions')

    rig.location = (0.,0.,0.)
    rig.rotation_euler = (0.,0.,0.)
    rig.scale = (1.,1.,1.)
    bpy.context.view_layer.update()
    with tempfile.TemporaryDirectory(prefix='mocap_calibration_') as directory:
        tilted = pose_calibration.calibrated_result(result, math.radians(20))
        path = export_result.write_result(tilted.data,directory)
        harness.check(bpy.ops.mocap.import_result(filepath=path) == {'FINISHED'}, 'import for calibration')
        props = bpy.context.scene.mocap_props
        props.target_armature = rig
        harness.check(bpy.ops.mocap.calibrate_pitch() == {'FINISHED'}, 'standing calibration operator')
        expected_pitch = pose_calibration.estimate_standing_pitch(tilted)
        harness.check(abs(props.pitch_correction-expected_pitch) < 1e-6, 'UI stores the estimated constant pitch')
        from motion_capture.addon import operators
        untouched = copy.deepcopy(operators.loaded_result().frames[0].body3d)
        for repeat in range(2):
            harness.check(bpy.ops.mocap.apply_to_rigify() == {'FINISHED'}, 'calibrated application {0}'.format(repeat+1))
        harness.check(operators.loaded_result().frames[0].body3d == untouched, 'repeated calibration leaves cached input intact')
    harness.check(result.frames[0].body3d == source, 'retarget preserves source coordinates')
    harness.disable_addon(module, used_operator)
    harness.finish()


if __name__ == '__main__':
    main()
