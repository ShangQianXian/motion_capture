"""All v0.3 motion families produce independent, correctly timed Rigify Actions."""
from pathlib import Path
import sys
import bpy
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import _harness as h
from tests.fixtures.motions import clip


def main():
    module, used = h.enable_addon()
    from motion_capture.core import motion_processing, result_schema
    from motion_capture.backend_worker import export_result
    from motion_capture.blender import rigify_adapter as adapter
    rig = h.generate_rigify_human()
    scene = bpy.context.scene
    original = bpy.data.actions.new('Original artist action')
    rig.animation_data_create()
    rig.animation_data.action = original
    rig.pose.bones['root'].keyframe_insert('location', frame=8)
    for mode in ('general', 'walk', 'run', 'attack', 'idle'):
        frames, _, _ = motion_processing.process(clip(mode, 24, 3),24,dict(motion_type=mode,coordinate_space='world'))
        result = result_schema.MocapResult(export_result.build_result(frames,24,'fixture.mp4','video','quality_feet'))
        scene.frame_set(8, subframe=.5)
        action = adapter.retarget_to_rigify(result,rig,adapter.RetargetOptions(scene_fps=30 / 1.001))
        h.check(action != original, mode + ' creates separate Action')
        h.check(action['mocap_frame_start'] == 1, mode + ' begins at frame 1')
        h.check(abs(action['mocap_frame_end'] - (1 + frames[-1]['time'] * 30 / 1.001)) < .001, mode + ' preserves duration')
        h.check(scene.frame_current == 8 and abs(scene.frame_subframe - .5) < 1e-5, mode + ' restores scene time')
        h.check(any(len(fc.keyframe_points) > 1 and max(k.co.y for k in fc.keyframe_points) - min(k.co.y for k in fc.keyframe_points) > .001
                    for fc in action.fcurves), mode + ' preserves animation including idle motion')
        rig.animation_data.action = original
    h.check(original.name in bpy.data.actions and original.use_fake_user, 'original action remains preserved')
    h.disable_addon(module, used)
    h.finish()


if __name__ == '__main__':
    main()
