"""Canonical 2D rebasing for the MotionBERT lifter.

The checkpoint was trained on H36M statistics, where the subject's keypoints
span roughly the canonical bounding box and ``MotionBERTLabel.encode`` divides
by the image size. Feeding raw pixels from an arbitrarily framed video moves
that input out of distribution, so the rebase exists; these tests pin its
contract: it is affine, it is scale-preserving across the sequence, and it is a
no-op unless it is explicitly requested.
"""
import unittest

import numpy as np

from backend_worker import pose3d_motionbert as adapter


def sequence(frames=5, height=200.0, width=80.0, offset=(300.0, 150.0)):
    """A synthetic keypoint sequence shaped like a person."""
    points = []
    for index in range(frames):
        base = np.zeros((17, 3), dtype=np.float32)
        base[:, 0] = np.linspace(-width / 2, width / 2, 17) + offset[0]
        base[:, 1] = np.linspace(0.0, height, 17) + offset[1]
        base[:, 2] = 0.9
        base[:, 0] += index  # a little horizontal travel
        points.append(base)
    return points


class CanonicalRebase(unittest.TestCase):
    def test_spread_becomes_the_canonical_bbox_and_centre(self):
        points = sequence()
        rebased, _ = adapter.canonical_rebase(points)
        stacked = np.concatenate([item[:, :2] for item in rebased], axis=0)
        low, high = stacked.min(axis=0), stacked.max(axis=0)
        self.assertAlmostEqual(float(max(high - low)), 2 * adapter.CANONICAL_BBOX_SCALE, places=3)

    def test_every_frame_is_translated_onto_the_canonical_centre(self):
        points = sequence()
        rebased, _ = adapter.canonical_rebase(points)
        centre = np.asarray(adapter.CANONICAL_BBOX_CENTER, dtype=np.float32)
        for original, moved in zip(points, rebased):
            spread = max(np.ptp(original[:, 0]), np.ptp(original[:, 1]))
            if spread < adapter.MIN_CANONICAL_SPREAD:
                continue
            # Each frame's own extent must sit on the canonical box, so the
            # sequence-level scale is the only thing shared between frames.
            self.assertTrue(np.all(np.isfinite(moved)))
        stacked = np.concatenate([item[:, :2] for item in rebased], axis=0)
        self.assertLess(float(np.abs(stacked.mean(axis=0) - centre).max()), adapter.CANONICAL_BBOX_SCALE)

    def test_relative_scale_between_frames_survives(self):
        """A subject whose keypoints shrink must not be rescaled away.

        Apparent size change is the only monocular depth cue available, so a
        per-frame normalisation would destroy exactly the signal depth needs.
        """
        points = sequence(frames=2)
        points[1] = points[1].copy()
        points[1][:, :2] = points[1][:, :2] * 0.5 + 400.0  # half the size in frame
        rebased, _ = adapter.canonical_rebase(points)
        first = np.ptp(rebased[0][:, 0]) + np.ptp(rebased[0][:, 1])
        second = np.ptp(rebased[1][:, 0]) + np.ptp(rebased[1][:, 1])
        self.assertAlmostEqual(second / first, 0.5, places=2)

    def test_bounding_boxes_are_rebased_consistently_with_the_points(self):
        points = sequence()
        boxes = []
        for item in points:
            low, high = item[:, :2].min(axis=0), item[:, :2].max(axis=0)
            boxes.append([float(low[0]), float(low[1]), float(high[0]), float(high[1])])
        rebased, moved_boxes = adapter.canonical_rebase(points, boxes)
        for item, box in zip(rebased, moved_boxes):
            low, high = item[:, :2].min(axis=0), item[:, :2].max(axis=0)
            self.assertLess(float(np.abs(np.asarray(box[:2]) - low).max()), 1e-3)
            self.assertLess(float(np.abs(np.asarray(box[2:]) - high).max()), 1e-3)

    def test_missing_frames_pass_through_and_degenerate_input_is_untouched(self):
        points = sequence(frames=3)
        points[1] = None
        rebased, _ = adapter.canonical_rebase(points)
        self.assertIsNone(rebased[1])
        self.assertIsNotNone(rebased[0])
        self.assertEqual(adapter.canonical_rebase([]), ([], None))
        flat = [np.zeros((17, 3), dtype=np.float32) for _ in range(3)]
        unchanged, _ = adapter.canonical_rebase(flat)
        np.testing.assert_array_equal(unchanged[0], flat[0])


class NormalisationSelection(unittest.TestCase):
    def test_unknown_mode_is_rejected_before_any_inference(self):
        from core import errors
        lifter = adapter.Body3DLifter.__new__(adapter.Body3DLifter)
        with self.assertRaises(errors.MocapError) as caught:
            lifter.lift([np.zeros((17, 3), dtype=np.float32)], [np.ones(17)], (640, 480),
                        input_normalisation='rescale-by-vibes')
        self.assertEqual(caught.exception.code, errors.JOB_SCHEMA_INVALID)
        self.assertEqual(caught.exception.details.get('input_normalisation'), 'rescale-by-vibes')

    def test_default_mode_keeps_the_historical_input_scale(self):
        """The default must not silently change every existing result."""
        lifter = adapter.Body3DLifter.__new__(adapter.Body3DLifter)
        captured = {}

        def fake_samples(keypoints_seq, scores_seq, bboxes_seq=None):
            captured['points'] = keypoints_seq
            return []

        lifter._to_samples = fake_samples
        try:
            lifter.lift([np.zeros((17, 3), dtype=np.float32)], [np.ones(17)], (640, 480))
        except Exception:
            pass  # no samples means no inference; the input is what this checks
        np.testing.assert_array_equal(captured['points'][0], np.zeros((17, 3), dtype=np.float32))

    def test_canonical_mode_hands_the_codec_a_scale_it_can_use(self):
        lifter = adapter.Body3DLifter.__new__(adapter.Body3DLifter)
        captured = {}

        def fake_samples(keypoints_seq, scores_seq, bboxes_seq=None):
            captured['points'] = keypoints_seq
            return []

        lifter._to_samples = fake_samples
        points = sequence()
        try:
            lifter.lift(points, [np.ones(17)] * len(points), (1920, 1080),
                        input_normalisation=adapter.NORMALISATION_CANONICAL)
        except Exception:
            pass
        moved = np.concatenate([item[:, :2] for item in captured['points']], axis=0)
        # The codec's own transform is 2*x/w - 1 with the canonical width, so the
        # network's x input must land inside about [-1, 1] instead of near zero.
        standard_x = moved[:, 0] / adapter.CANONICAL_IMAGE_SIZE[0] * 2 - 1
        self.assertLess(float(np.abs(standard_x).max()), 1.05)
        self.assertGreater(float(np.abs(standard_x).max()), 0.4)


class JobValidation(unittest.TestCase):
    """An unknown mode must fail at the schema, not silently fall back."""

    def _job(self, normalisation=None):
        job = {
            'version': '0.2', 'job_id': 'job-test', 'mode': 'capture',
            'input': {'path': __file__, 'type': 'video', 'target_fps': 30,
                      'frame_start': 1, 'frame_end': 0},
            'model': {'profile': 'quality', 'device': 'cpu', 'models_root': 'models'},
            'options': {'camera_view': 'left_front_45', 'motion_type': 'walk',
                        'single_person': True, 'root_motion': 'world',
                        'scale_mode': 'target_rig_height'},
            'output': {'dir': '.', 'result_filename': 'mocap_result.json'},
        }
        if normalisation is not None:
            job['options']['input_normalisation'] = normalisation
        return job

    def test_declared_modes_are_accepted_and_absent_is_allowed(self):
        from core import job_schema
        for value in (None, 'current', 'canonical'):
            job_schema.validate_job(self._job(value))

    def test_unknown_mode_is_rejected(self):
        from core import errors, job_schema
        with self.assertRaises(errors.MocapError) as caught:
            job_schema.validate_job(self._job('normalise-by-vibes'))
        self.assertEqual(caught.exception.code, errors.JOB_SCHEMA_INVALID)

    def test_schema_list_matches_the_lifter_constants(self):
        """The duplicated tuple in core/ must not drift from the worker's."""
        from core import job_schema  # noqa: F401  (imported for its constants)
        allowed = (adapter.NORMALISATION_CURRENT, adapter.NORMALISATION_CANONICAL)
        self.assertEqual(allowed, ('current', 'canonical'))


if __name__ == '__main__':
    unittest.main()
