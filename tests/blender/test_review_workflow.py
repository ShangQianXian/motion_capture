"""Headless state/transaction contracts; real drawing is covered by foreground QA."""
import copy
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib
import bpy

sys.path.insert(0, str(Path(__file__).parent))
import _harness as h


def write_png(path):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 2, 2, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress((b'\x00' + b'\x60\x80\xa0' * 2) * 2)) + chunk(b'IEND', b''))


def main():
    module, used = h.enable_addon()
    from motion_capture.addon import review, operators
    from motion_capture.core import preview, skeleton
    from motion_capture.backend_worker import mock_source, export_result
    from motion_capture.blender import rigify_adapter as adapter
    scene = bpy.context.scene
    props = scene.mocap_props
    rig = h.generate_rigify_human()
    h.check(rig is not None, 'Rigify available')
    props.target_armature = rig
    with tempfile.TemporaryDirectory(prefix='mocap_review_') as folder:
        root = Path(folder)
        source = root / '素材.png'
        write_png(source)
        frames = mock_source.generate_frames(frame_start=91, frame_end=96, fps=30)
        for index, frame in enumerate(frames):
            frame['time'] = 3 + index / 30
        payload = export_result.build_result(frames, 30, str(source), 'video', 'preview')
        path = export_result.write_result(payload, folder)
        objects, actions = len(bpy.data.objects), len(bpy.data.actions)
        review.load_result(scene, path, restore_settings=True)
        value = review.session(scene)
        h.check(len(bpy.data.objects) == objects and len(bpy.data.actions) == actions,
                'loading a review creates no scene objects or animation')
        h.check('预览' in review.apply_block_reason(scene), 'application requires a rendered review')
        h.check(value.state.manifest is None, 'legacy results remain loadable')
        h.check(not value.accept_frame({'request_id': -1, 'source_index': 999, 'path': str(source)}),
                'outdated media replies cannot change displayed frames')
        value.pending_id, value.pending_source, value.pending_sample = 7, 4, 2
        h.check(not value.accept_frame({'request_id': 6, 'source_index': 4, 'path': str(source)}),
                'request identity is enforced before loading images')
        h.check(value.accept_frame({'request_id': 7, 'source_index': 4, 'path': str(source)}),
                'matching reply advances the synchronized pair')
        h.check((value.displayed_source, value.displayed_sample) == (4, 2), 'source and pose indices commit together')
        h.check(value.image_bytes <= preview.IMAGE_MEMORY_LIMIT, 'image memory is bounded')

        props.smoothing_strength = .2
        h.check(value.state.stale and '重新生成' in review.apply_block_reason(scene), 'capture edits invalidate application')
        review.load_result(scene, path)
        value = review.session(scene)
        other = bpy.data.scenes.new('Other review scene')
        with bpy.context.temp_override(scene=other):
            h.check(operators.loaded_result() is None, 'results cannot leak across scenes')
            h.check(bool(review.apply_block_reason(other)), 'other scene cannot apply this result')

        props.pitch_correction = .1
        value.state.corrected(props.pitch_correction, props.flip_x)
        # Unit-level stand-in for the successful foreground draw callback.
        value.state.viewed = True
        h.check(not review.apply_block_reason(scene), 'reviewed matching result can be applied')
        props.flip_x = True
        h.check(not value.state.viewed, 'direction edits require displaying the updated pose')
        value.state.corrected(props.pitch_correction, props.flip_x)
        value.state.viewed = True
        old_body = copy.deepcopy(value.state.result.frames[0].body3d)
        original = bpy.data.actions.new('Artist original')
        rig.animation_data_create()
        rig.animation_data.action = original
        rig.pose.bones['torso'].keyframe_insert('location', frame=16)
        original_keys = [(fc.data_path, len(fc.keyframe_points)) for fc in original.fcurves]
        scene.render.fps = 24
        scene.render.fps_base = 1.001
        scene.frame_set(16, subframe=.25)
        h.check(bpy.ops.mocap.apply_to_rigify() == {'FINISHED'}, 'confirmed operator applies a new Action')
        applied = rig.animation_data.action
        h.check(applied != original and original.use_fake_user, 'previous Action remains saved')
        h.check([(fc.data_path, len(fc.keyframe_points)) for fc in original.fcurves] == original_keys,
                'original animation curves are unchanged')
        h.check(applied['mocap_frame_start'] == 1, 'new Action starts at frame 1')
        expected_end = 1 + (frames[-1]['time'] - frames[0]['time']) * 24 / scene.render.fps_base
        h.check(abs(applied['mocap_frame_end'] - expected_end) < .001, 'duration uses scene fps/base and source timestamps')
        key_times = [key.co.x for fc in applied.fcurves for key in fc.keyframe_points]
        h.check(any(abs(t - round(t)) > .01 for t in key_times), 'fractional keyframe timing is preserved')
        h.check(scene.frame_current == 16 and abs(scene.frame_subframe - .25) < 1e-6,
                'application restores scene time including subframe')
        h.check(scene.render.fps == 24 and abs(scene.render.fps_base - 1.001) < 1e-6, 'scene frame rate is unchanged')
        h.check(value.state.result.frames[0].body3d == old_body, 'preview and application preserve raw results')

        switch = rig.pose.bones[skeleton.IK_FK_SWITCH_BONES[0]]
        rig.animation_data.action = original
        switch[skeleton.IK_FK_PROPERTY] = 0.0
        rig.pose.bones['upper_arm_fk.L'].rotation_mode = 'XYZ'
        before_count = len(bpy.data.actions)
        before_matrix = rig.pose.bones['upper_arm_fk.L'].matrix_basis.copy()
        iterator = adapter.retarget_steps(value.state.result, rig, adapter.RetargetOptions(scene_fps=24))
        next(iterator)
        next(iterator)
        iterator.close()
        h.check(rig.animation_data.action == original and len(bpy.data.actions) == before_count,
                'cancel removes partial Action and restores the original')
        h.check(switch[skeleton.IK_FK_PROPERTY] == 0 and rig.pose.bones['upper_arm_fk.L'].rotation_mode == 'XYZ',
                'cancel restores FK settings and rotation mode')
        after_matrix = rig.pose.bones['upper_arm_fk.L'].matrix_basis
        h.check(max(abs(before_matrix[r][c] - after_matrix[r][c]) for r in range(4) for c in range(4)) < 1e-5,
                'cancel restores the previous bone transform')
        def fail(done, total):
            raise RuntimeError('injected retarget failure')
        try:
            adapter.retarget_to_rigify(value.state.result, rig, adapter.RetargetOptions(progress=fail))
        except RuntimeError:
            pass
        h.check(rig.animation_data.action == original and len(bpy.data.actions) == before_count,
                'failure rolls back partial application too')

        image_result = copy.deepcopy(value.state.result)
        image_result.data['source']['type'] = 'image'
        progress = []
        image_action = adapter.retarget_to_rigify(image_result, rig, adapter.RetargetOptions(
            scene_fps=24, progress=lambda done, total: progress.append((done, total))))
        h.check(progress == [(1, 1)], 'image input applies only the first reviewed pose, including legacy multi-frame data')
        h.check(all(abs(key.co.x - 1) < 1e-6 for curve in image_action.fcurves for key in curve.keyframe_points),
                'image pose is keyed only at frame 1')
        rig.animation_data.action = original
        from types import SimpleNamespace
        fake_modal = SimpleNamespace(_scene=scene, advance=lambda: {'RUNNING_MODAL'})
        h.check(operators.MOCAP_OT_apply_to_rigify.modal(fake_modal, bpy.context, SimpleNamespace(type='TIMER')) == {'RUNNING_MODAL'},
                'modal progress accepts Blender timer events without an Event.timer attribute')

        relocated = root / 'relocated.png'
        shutil.copy2(source, relocated)
        source.unlink()
        h.check('定位' in review.apply_block_reason(scene), 'missing source blocks confirmation')
        h.check(bpy.ops.mocap.relocate_source(filepath=str(relocated)) == {'FINISHED'}, 'matching source can be relocated')
        h.check(value.state.media_matches() and not value.state.viewed, 'relocation requires renewed visual review')
        review.cleanup()
        h.check(not review._sessions, 'all scene review sessions are released')
        h.check(not any(image.name.startswith('Mocap Preview') for image in bpy.data.images), 'preview images are released')
        h.check(not bpy.app.timers.is_registered(review._tick), 'preview timer is released')
    h.disable_addon(module, used)
    h.finish()


if __name__ == '__main__':
    main()
