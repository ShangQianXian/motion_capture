"""Known camera extrinsics, independent turns and compatible result contracts."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from core import camera_alignment as ca, retarget_math as rm, orientations as ori, preview, result_schema
from backend_worker import mock_source, export_result, pipeline


def rotate(frames, angle):
    return ca.transform(frames, dict(camera_view='left_front_45',
                         quaternion_wxyz=rm.quat_from_axis_angle((0, 0, 1), angle)))


class CameraAlignmentTests(unittest.TestCase):
    def frames(self):
        frames = mock_source.generate_frames(frame_start=1, frame_end=24, fps=24)
        for i, f in enumerate(frames):
            f['orientations'] = {n: rm.quat_from_axis_angle((1, 0, 0), .2) for n in ori.NAMES}
            f['hands3d'] = {'thumb.01.L': (.3, .2, 1.)}
            ori.apply_feet(f)
        return frames

    def assert_points(self, a, b, places=9):
        for fa, fb in zip(a, b):
            for key in ('body3d', 'hands3d'):
                for name, point in fa[key].items():
                    for x, y in zip(point, fb[key][name]):
                        self.assertAlmostEqual(x, y, places=places)

    def test_left_front_camera_sign_and_known_pose_round_trip(self):
        world = self.frames()
        camera = rotate(world, -math.pi / 4)
        c = ca.resolve(camera, 'left_front_45')
        restored = ca.transform(camera, c)
        self.assert_points(world, restored)
        # Forward points to the LEFT in the source image, not the right.
        self.assertLess(rm.quat_rotate_vector(rm.quat_from_axis_angle((0, 0, 1), -math.pi/4), (0,-1,0))[0], 0)
        for f, original in zip(restored, world):
            for n in ori.NAMES:
                self.assertGreater(abs(rm.quat_dot(f['orientations'][n], original['orientations'][n])), 1-1e-9)

    def test_constant_reference_preserves_later_turns_and_root_travel(self):
        world = self.frames()
        for i, f in enumerate(world):
            turned = rotate([f], 0. if i < 8 else math.pi / 2)[0]
            for key in ('body3d', 'hands3d'):
                turned[key] = {n: rm.vec_add(p, (0, -.02*i, 0)) for n,p in turned[key].items()}
            world[i] = turned
        camera = rotate(world, -math.pi/4 + .1)
        c = ca.resolve(camera, 'left_front_45', True)
        self.assertEqual(c['initial_alignment'], 'aligned')
        self.assertAlmostEqual(c['initial_heading_correction_degrees'], -math.degrees(.1), places=6)
        restored = ca.transform(camera, c)
        self.assert_points(world, restored)

    def test_bone_lengths_joint_angles_contacts_and_source_unchanged(self):
        frames = self.frames()
        before = copy.deepcopy(frames)
        output = ca.transform(frames, ca.resolve(frames, 'left_front_45'))
        self.assertEqual(before, frames)
        for old, new in zip(frames, output):
            self.assertEqual(old['contacts'], new['contacts'])
            for side in ('L', 'R'):
                a, b, c = ('hip.'+side, 'knee.'+side, 'ankle.'+side)
                def geometry(f):
                    x = rm.vec_sub(f['body3d'][a], f['body3d'][b])
                    y = rm.vec_sub(f['body3d'][c], f['body3d'][b])
                    return rm.vec_length(x), rm.vec_length(y), rm.vec_dot(rm.vec_normalize(x), rm.vec_normalize(y))
                for x, y in zip(geometry(old), geometry(new)):
                    self.assertAlmostEqual(x, y, places=9)

    def test_weak_or_conflicting_initial_heading_does_not_snap_or_borrow_later(self):
        frames = self.frames()
        self.assertEqual(ca.resolve(frames, 'left_front_45', True)['initial_alignment'], 'camera_prior_conflict')
        camera = rotate(frames, -math.pi/4)
        for f in camera[:7]:
            f['confidence'].update({'hip.L': 0., 'hip.R': 0.})
        c = ca.resolve(camera, 'left_front_45', True)
        self.assertEqual(c['initial_alignment'], 'insufficient_initial_observations')
        self.assertEqual(c['applied_yaw_degrees'], 45.)
        self.assertEqual(ca.resolve([], 'left_front_45', True)['reference_samples'], 0)

    def test_legacy_mirror_serialization_and_cache_identity(self):
        frames = self.frames()
        self.assertEqual(ca.transform(frames, ca.resolve(frames)), frames)
        c = ca.resolve(rotate(frames,-math.pi/4), 'left_front_45', True)
        result = result_schema.MocapResult(export_result.build_result(ca.transform(frames,c),24,capture_transform=c))
        mirrored = preview.corrected_result(result, 0, True)
        restored = preview.corrected_result(mirrored, 0, True)
        self.assert_points([f.to_dict() for f in result.frames], [f.to_dict() for f in restored.frames], places=6)
        state = preview.ReviewState(result, None, {}, 'example.mp4')
        self.assertTrue(state.matches({}, 'example.mp4'))
        self.assertFalse(state.matches(dict(camera_view='left_front_45'), 'example.mp4'))
        self.assertFalse(state.matches(dict(align_initial_facing=True), 'example.mp4'))
        # The 2D input scale changes the lifted pose, so a cached review must be
        # invalidated when it changes, and stay valid when it does not.
        self.assertTrue(state.matches(dict(input_normalisation='current'), 'example.mp4'))
        self.assertFalse(state.matches(dict(input_normalisation='canonical'), 'example.mp4'))
        self.assertEqual(restored.data['capture_transform'], c)

    def test_real_pipeline_exports_once_after_camera_space_fitting(self):
        # Replace only expensive inference and preflight, retaining production
        # postprocessing, finalization, serialization and preview sidecar writer.
        from ._util import scene_props, preferences
        from core import job_schema
        camera = rotate(self.frames(), -math.pi/4)
        for f in camera:
            f['hands3d'] = {}  # Hand schema not relevant to this integration.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.mp4'
            source.touch()
            job = job_schema.build_job(scene_props(source_media=str(source),
                camera_view='left_front_45', align_initial_facing=False),
                preferences(models_root=directory), blend_path=str(Path(directory)/'test.blend'))
            job['review_settings'] = dict(camera_view='left_front_45',align_initial_facing=False)
            seen = []
            def inference(job, profile, manifest, reporter, cancel, options):
                options['_preview_meta'] = dict(width=640,height=480,fps=24,effective_fps=24)
                options['_preview_rows'] = [dict(sample_frame=f['frame'],source_index=i,time=f['time'],body2d=[],status='detected') for i,f in enumerate(camera)]
                return copy.deepcopy(camera),24,[]
            def enrich(frames, *args):
                seen.append(copy.deepcopy(frames))
            reporter = SimpleNamespace(debug=lambda *a:None,warning=lambda *a,**k:None,progress=lambda *a,**k:None)
            report = SimpleNamespace(warnings=[],effective_profile='preview',missing_required=[])
            with patch.object(pipeline.model_manifest,'load_manifest',return_value={}), \
                 patch.object(pipeline.model_manifest,'check_profile_requirements',return_value=report), \
                 patch('backend_worker.environment.validate_environment',return_value={'ok':True}), \
                 patch.object(pipeline,'_run_mediapipe',side_effect=inference), \
                 patch('backend_worker.orientation_solver.enrich',side_effect=enrich):
                path = pipeline.run_job(job,reporter)
            self.assert_points(seen[0],camera)
            result = result_schema.load_mocap_result(path)
            raw = result_schema.load_mocap_result(str(Path(path).parent/'mocap_raw.json'))
            self.assert_points(rotate(camera,math.pi/4), [f.to_dict() for f in raw.frames], places=5)
            manifest = preview.load_manifest(result)
            self.assertEqual(manifest['processing_version'],ori.VERSION)
            self.assertEqual(manifest['diagnostics']['camera_alignment'],result.data['capture_transform'])
            self.assertEqual(raw.data['capture_transform'],result.data['capture_transform'])

    def test_invalid_camera_metadata_and_job_settings_are_rejected(self):
        from core import errors, job_schema
        from ._util import scene_props, preferences
        result=export_result.build_result(self.frames(),24)
        for invalid in (None, [], {}, {'applied_yaw_degrees':float('nan')}, {'applied_yaw_degrees':'45'}):
            with self.subTest(invalid=invalid), self.assertRaises(errors.MocapError):
                result_schema.validate_result(dict(result,capture_transform=invalid))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.mp4'
            path.touch()
            for props in (dict(camera_view='front'),dict(align_initial_facing='yes')):
                job=job_schema.build_job(scene_props(source_media=str(path)),preferences(models_root=directory),
                                         blend_path=str(Path(directory)/'test.blend'))
                job['options'].update(props)
                with self.assertRaises(errors.MocapError):
                    job_schema.validate_job(job)


if __name__ == '__main__':
    unittest.main()
