"""Apply all real capture frames to Rigify and save a reviewable .blend artifact."""
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / 'tests/blender'))
import _harness as harness


def main():
    harness.reset_scene()
    harness.enable_addon()
    from motion_capture.core import result_schema, retarget_math as rm, preview
    from motion_capture.blender import rigify_adapter as adapter
    folder = ROOT / '.cache/v03-orientations'
    result = result_schema.load_mocap_result(str(folder / 'capture_quality/mocap_result.json'))
    manifest = preview.load_manifest(result)
    assert manifest['processing_version'] == '0.3.1'
    rig = harness.generate_rigify_human()
    rig.show_in_front = True
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = len(result.frames)
    original = bpy.data.actions.new('Original preserved')
    rig.animation_data_create()
    rig.animation_data.action = original
    original.use_fake_user = True
    action = adapter.retarget_to_rigify(result, rig, adapter.RetargetOptions(
        include_hands=False, scene_fps=24, root_motion='in_place', action_name='v0.3.1 verified walk'))
    errors = {'head': [], 'foot.L': [], 'foot.R': []}
    for frame in result.frames:
        scene.frame_set(round(1 + frame.time*24))
        bpy.context.view_layer.update()
        for name, bone in (('head', 'head'), ('foot.L', 'foot_fk.L'), ('foot.R', 'foot_fk.R')):
            pb = rig.pose.bones[bone]
            expected = Vector(rm.quat_rotate_vector(frame.orientations[name], (0, 0, 1))) if name == 'head' else (
                Vector(frame.body3d['toe.'+name[-1]]) - Vector(frame.body3d['ankle.'+name[-1]]))
            actual = rig.matrix_world.to_3x3() @ (pb.tail-pb.head)
            errors[name].append(math.degrees(actual.angle(expected)))
    report = dict(frames=len(result.frames), action=action.name, previous_action_preserved=original.name in bpy.data.actions,
                  max_control_direction_error_degrees={name:max(values) for name,values in errors.items()})
    assert all(value < .1 for value in report['max_control_direction_error_degrees'].values()), report
    assert report['previous_action_preserved']
    scene.frame_set(5)
    scene.mocap_props.target_armature = rig
    scene.mocap_props.source_media = result.source_path
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'verified-animation.blend'))
    (folder / 'blender-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
