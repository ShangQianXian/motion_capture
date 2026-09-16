"""Actual Rigify control/deform axes, transforms and rollback for independent rotations."""
import copy
import math
import os
import sys
import bpy
from mathutils import Vector
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _harness as h


def main():
    h.reset_scene()
    module, operator=h.enable_addon()
    from motion_capture.backend_worker import export_result, mock_source
    from motion_capture.core import result_schema, retarget_math as rm, orientations as ori, camera_alignment as ca
    from motion_capture.blender import rigify_adapter as adapter
    rig=h.generate_rigify_human()
    frames=mock_source.generate_frames(frame_start=1,frame_end=4,fps=24)
    for i,f in enumerate(frames):
        f['orientations']={name:rm.quat_mul(rm.quat_from_axis_angle((0,0,1),.35*i),
                           rm.quat_from_axis_angle((1,0,0),.12*i)) for name in ori.NAMES}
        ori.apply_feet(f)
    # The pipeline's camera transform must also survive Rigify rest transforms,
    # parent inheritance and differently transformed armature objects.
    camera_frames = ca.transform(frames, dict(camera_view='left_front_45',
        quaternion_wxyz=rm.quat_from_axis_angle((0,0,1),-math.pi/4)))
    calibration = ca.resolve(camera_frames, 'left_front_45', True)
    h.check(calibration['initial_alignment']=='aligned','fixed camera initial facing aligns to -Y')
    restored = ca.transform(camera_frames,calibration)
    h.check(rm.vec_distance(frames[0]['body3d']['ankle.L'],restored[0]['body3d']['ankle.L'])<1e-8,
            'camera conversion restores character-space ankle before Rigify')
    frames = restored
    result=result_schema.MocapResult(export_result.build_result(frames,24))
    original=copy.deepcopy(result.data)
    for tilt,scale in ((0,1),(.4,1.7),(-.3,.6)):
        rig.rotation_euler=(tilt,0,.2)
        rig.scale=(scale,)*3
        bpy.context.view_layer.update()
        old_action=rig.animation_data.action if rig.animation_data else None
        action=adapter.retarget_to_rigify(result,rig,adapter.RetargetOptions(include_hands=False))
        h.check(action is not old_action,'new action preserves previous action')
        for f in result.frames:
            bpy.context.scene.frame_set(f.frame)
            bpy.context.view_layer.update()
            for name,bone in (('head','head'),('foot.L','foot_fk.L'),('foot.R','foot_fk.R')):
                expected=Vector(rm.quat_rotate_vector(f.orientations[name],(0,0,1))) if name=='head' else Vector(f.body3d['toe.'+name[-1]])-Vector(f.body3d['ankle.'+name[-1]])
                pb=rig.pose.bones[bone]
                actual=rig.matrix_world.to_3x3() @ (pb.tail-pb.head)
                h.check(math.degrees(actual.angle(expected))<.1,'control '+bone+' matches independent world orientation')
                # Compare the anatomical reference axis projected perpendicular to aim.
                ref=Vector(rm.quat_rotate_vector(f.orientations[name],(1,0,0)))
                actual_x=rig.matrix_world.to_3x3() @ pb.matrix.to_3x3().col[0]
                h.check(abs(actual_x.normalized().dot(ref))>.99,'control '+bone+' retains lateral axis')
            for side in ('L','R'):
                control=rig.pose.bones['foot_fk.'+side]
                deform=rig.pose.bones.get('DEF-foot.'+side)
                h.check(deform is not None and (control.tail-control.head).angle(deform.tail-deform.head)<.01,
                        'evaluated foot deformation follows FK '+side)
    h.check(result.data==original,'retarget leaves source result untouched')
    # Cancellation after partial application restores the previous action.
    original_action=rig.animation_data.action
    steps=adapter.retarget_steps(result,rig,adapter.RetargetOptions(include_hands=False))
    next(steps)
    steps.close()
    h.check(rig.animation_data.action is original_action,'cancellation rolls back action')
    h.disable_addon(module,operator)
    h.finish()


if __name__=='__main__':
    main()
