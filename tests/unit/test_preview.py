"""Review identity, real detector overlays and crop/rate synchronization contracts."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from core import errors, orientations, preview, result_schema
from backend_worker import export_result, mock_source, preview_export, postprocess


class PreviewTests(unittest.TestCase):
    def result(self):
        return result_schema.MocapResult(export_result.build_result(
            mock_source.generate_frames(frame_start=1, frame_end=3), 30.0))

    def test_scene_rate_preserves_seconds_and_fractional_keys(self):
        self.assertEqual(preview.action_frame(4.5, 4.0, 24), 13)
        self.assertAlmostEqual(preview.action_frame(4 + 1 / 30, 4, 24), 1.8)
        self.assertAlmostEqual(preview.action_frame(5, 4, 24000 / 1001), 1 + 24000 / 1001)
        self.assertEqual(preview.action_frame(9, 4, 24, image=True), 1)
        with self.assertRaises(errors.MocapError):
            preview.action_frame(float('nan'), 0, 30)

    def test_identity_invalidates_on_edit_and_relocation_preserves_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '素材.png'
            path.write_bytes(b'first')
            settings = {'profile': 'preview'}
            state = preview.ReviewState(self.result(), None, settings, str(path))
            state.corrected(0, False)
            state.viewed = True
            self.assertTrue(state.matches(settings, str(path)))
            self.assertTrue(state.media_matches())
            self.assertFalse(state.matches({'profile': 'quality'}, str(path)))
            state.corrected(.1, False)
            self.assertFalse(state.viewed)
            path.write_bytes(b'replaced image')
            self.assertFalse(state.media_matches())
            state.invalidate()
            self.assertFalse(state.matches(settings, str(path)))

    def test_corrections_are_shared_nonmutating_and_repeatable(self):
        original = self.result()
        before = copy.deepcopy(original.frames[0].body3d)
        a = preview.corrected_result(original, .2, True)
        b = preview.corrected_result(original, .2, True)
        self.assertEqual(a.frames[0].body3d, b.frames[0].body3d)
        self.assertEqual(before, original.frames[0].body3d)
        self.assertNotEqual(before, a.frames[0].body3d)

    def test_version_mismatch_has_its_own_reason(self):
        manifest = {'source': {}, 'processing_version': 'old-version'}
        state = preview.ReviewState(self.result(), manifest, {}, '')
        reason = state.mismatch_reason({}, '')
        self.assertIn('处理版本不一致', reason)
        self.assertNotIn('参数已改变', reason)
        self.assertFalse(state.matches({}, ''))
        manifest['processing_version'] = orientations.VERSION
        self.assertTrue(state.matches({}, ''))
        state.invalidate(reason)
        state.invalidate()
        self.assertEqual(state.stale_reason, reason)
        self.assertEqual(state.revision, 1)
        self.assertFalse(state.matches({}, ''))

    def test_native_2d_coordinates_and_confidence_are_preserved(self):
        self.assertEqual(preview_export.coco_points([(960, 240)], [.2], 1920, 480), [[.5, .5, .2]])
        landmarks = [SimpleNamespace(x=.2, y=.3, visibility=.4)]
        self.assertEqual(preview_export.mp_points(landmarks), [[.2, .3, .4]])
        self.assertEqual(preview_export.mp_points([SimpleNamespace(x=float('nan'), y=.2)]), [[0, 0, 0]])

    def test_sidecar_retains_missing_frames_and_rejects_mismatched_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'test.mp4'
            source.touch()
            payload = self.result().data
            result_path = export_result.write_result(payload, directory)
            rows = [dict(source_index=i * 2, time=i / 30, sample_frame=i + 1,
                         result_frame=None, body2d=[], hands2d=[], status='missing') for i in range(4)]
            job = dict(job_id='job', input={'path': str(source)}, options={})
            preview_export.write(job, payload, result_path, rows, {'fps': 60}, 'quality')
            result = result_schema.load_mocap_result(result_path)
            manifest = preview.load_manifest(result)
            self.assertEqual(len(manifest['frames']), 4)
            self.assertEqual(manifest['frames'][-1]['status'], 'missing')
            self.assertIsNone(manifest['frames'][-1]['result_frame'])
            self.assertEqual(preview.next_problem(manifest['frames'], 0, 1), 3)
            self.assertEqual(manifest['topology'], 'coco17')
            with open(result_path, 'a') as handle:
                handle.write(' ')
            with self.assertRaises(errors.MocapError):
                preview.load_manifest(result)
            Path(directory, preview.MANIFEST_FILENAME).unlink()
            self.assertIsNone(preview.load_manifest(result))

    def test_corrupt_sidecars_fail_instead_of_misaligned_overlay(self):
        data = dict(version='0.2', source={}, frames=[dict(source_index=0, time=0., body2d=[[1., 1., .9]])])
        preview.validate_manifest(data)
        for broken in ([dict(source_index=0, time=float('nan'))],
                       [dict(source_index=-1, time=0.)],
                       [dict(source_index=0, time=0., body2d=[[1, 2, 3]])],
                       [dict(source_index=0, time=0.), dict(source_index=0, time=1.)]):
            with self.subTest(broken=broken), self.assertRaises(errors.MocapError):
                preview.validate_manifest(dict(data, frames=broken))

    def test_postprocess_records_interpolated_joints(self):
        frames = mock_source.generate_frames(frame_start=1, frame_end=5)
        frames[2]['confidence']['elbow.L'] = .1
        output, _ = postprocess.postprocess(frames, 30, {'smoothing_strength': 0})
        self.assertIn('elbow.L', output[2]['interpolated_joints'])

    def test_v03_optional_stages_and_invalid_references(self):
        data = dict(version='0.2', source={}, frames=[dict(source_index=0, time=0., contact_states={'L':'air'})],
                    stages={'raw': {'file':'mocap_raw.json', 'sha256':'a' * 64}})
        preview.validate_manifest(data)
        for stage in ({'file':'../other.json','sha256':'a' * 64}, {'file':'mocap_raw.json','sha256':'broken'}):
            with self.assertRaises(errors.MocapError):
                preview.validate_manifest(dict(data, stages={'raw':stage}))
        with self.assertRaises(errors.MocapError):
            preview.validate_manifest(dict(data, diagnostics=[]))
        points = [[0, 0, .9]] * 23 + [[0, 0, .1]] * 110
        self.assertFalse(preview.is_problem(dict(body2d=points)))

    def test_consecutive_missing_rows_and_legacy_image(self):
        rows = [dict(source_index=i * 3, time=4 + i / 20, status='missing' if 1 <= i <= 3 else 'detected')
                for i in range(5)]
        preview.validate_manifest(dict(version='0.2', source={}, frames=rows))
        self.assertEqual(preview.next_problem(rows, 0, 1), 1)
        self.assertEqual(preview.next_problem(rows, 4, -1), 3)
        self.assertAlmostEqual(preview.action_frame(rows[-1]['time'], rows[0]['time'], 24), 5.8)
        result = self.result()
        result.data['source']['type'] = 'image'
        self.assertTrue(all(row['source_index'] == 0 for row in preview.legacy_rows(result, 30)))
        self.assertTrue(preview.settings_equal({'smoothing_strength': .65}, {'smoothing_strength': .649999976}))


if __name__ == '__main__':
    unittest.main()
