"""The reprojection diagnostic, checked against a synthetic known camera.

The diagnostic's job is to fail loudly when a capture does not reconcile with
its own 2D evidence, and to stay quiet when it does. Both directions are tested
here by constructing the 2D evidence from a known camera and then asking the
diagnostic to recover it.
"""
import math
import unittest

import numpy as np

from backend_worker import reprojection


def camera_axes(azimuth_degrees):
    angle = math.radians(azimuth_degrees)
    return ((math.cos(angle), math.sin(angle), 0.0),
            (-math.sin(angle), math.cos(angle), 0.0))


def synthetic_pose(offset=(0.0, 0.0, 0.0), step=0.0):
    """One plausible frame in camera space, metres, root-relative."""
    pelvis = np.array([0.0, 0.0, 0.9]) + np.asarray(offset, dtype=float)
    body = {
        'pelvis': pelvis,
        'hip.L': pelvis + np.array([0.10, 0.0, -0.02]),
        'hip.R': pelvis + np.array([-0.10, 0.0, -0.02]),
        'knee.L': pelvis + np.array([0.10, step, -0.45]),
        'knee.R': pelvis + np.array([-0.10, -step, -0.45]),
        'ankle.L': pelvis + np.array([0.10, step * 2, -0.85]),
        'ankle.R': pelvis + np.array([-0.10, -step * 2, -0.85]),
        'shoulder.L': pelvis + np.array([0.18, 0.0, 0.45]),
        'shoulder.R': pelvis + np.array([-0.18, 0.0, 0.45]),
        'elbow.L': pelvis + np.array([0.24, 0.0, 0.20]),
        'elbow.R': pelvis + np.array([-0.24, 0.0, 0.20]),
        'wrist.L': pelvis + np.array([0.28, 0.0, -0.02]),
        'wrist.R': pelvis + np.array([-0.28, 0.0, -0.02]),
        'head': pelvis + np.array([0.0, 0.0, 0.75]),
    }
    return {name: tuple(value) for name, value in body.items()}


def build_capture(azimuth=45.0, focal=1600.0, depth=4.0, frames=24,
                  image_size=(1280.0, 720.0), anchor=(640.0, 360.0)):
    """Return ``(frames, rows)`` for a synthetic capture seen by ``azimuth``."""
    lateral, depth_axis = camera_axes(azimuth)
    index_to_joint = {index: name for index, name in reprojection.COCO_TO_JOINT.items()}
    capture_frames, rows = [], []
    for number in range(1, frames + 1):
        body = synthetic_pose(step=0.06 * math.sin(number / 4.0))
        capture_frames.append({'frame': number, 'time': (number - 1) / 30.0, 'body3d': body})
        pelvis = np.asarray(body['pelvis'])
        points = [[0.0, 0.0, 0.0] for _ in range(133)]
        for index, name in index_to_joint.items():
            joint = np.asarray(body[name])
            delta_lateral = float(np.dot(joint - pelvis, lateral))
            delta_depth = float(np.dot(joint - pelvis, depth_axis))
            scale = focal / (depth + delta_depth)
            u = anchor[0] + scale * delta_lateral
            v = anchor[1] - scale * (joint[2] - pelvis[2])
            points[index] = [u / image_size[0], v / image_size[1], 0.9]
        rows.append({'result_frame': number, 'body2d': points,
                     'pelvis2d': [anchor[0] / image_size[0], anchor[1] / image_size[1]]})
    return capture_frames, rows


class ReprojectionDiagnostic(unittest.TestCase):
    def test_recovers_the_camera_the_capture_was_made_with(self):
        frames, rows = build_capture(azimuth=45.0)
        report = reprojection.diagnose(frames, rows, (1280.0, 720.0),
                                       {'camera_azimuth_degrees': 45.0})
        self.assertEqual(report['status'], 'ok')
        self.assertLess(abs(report['camera']['azimuth_degrees'] - 45.0), 8.0)
        self.assertLess(report['camera']['median_reprojection_px'], 2.0)
        self.assertFalse(report['camera_conflict'])
        self.assertEqual(reprojection.findings(report), [])

    def test_a_wrong_recorded_view_is_reported_as_a_conflict(self):
        frames, rows = build_capture(azimuth=45.0)
        report = reprojection.diagnose(frames, rows, (1280.0, 720.0),
                                       {'camera_azimuth_degrees': 135.0})
        self.assertTrue(report['camera_conflict'])
        codes = [warning['code'] for warning in reprojection.findings(report)]
        self.assertIn('CAMERA_AZIMUTH_CONFLICT', codes)
        # The fit must still land on the true camera rather than the recorded one.
        self.assertLess(abs(report['camera']['azimuth_degrees'] - 45.0), 12.0)

    def test_noise_raised_residual_is_reported_not_hidden(self):
        frames, rows = build_capture(azimuth=45.0)
        generator = np.random.default_rng(7)
        for row in rows:
            for point in row['body2d']:
                point[0] += generator.normal(0.0, 40.0) / 1280.0
                point[1] += generator.normal(0.0, 40.0) / 720.0
        report = reprojection.diagnose(frames, rows, (1280.0, 720.0),
                                       {'camera_azimuth_degrees': 45.0})
        self.assertGreater(report['reprojection']['median_px'], 25.0)
        codes = [warning['code'] for warning in reprojection.findings(report)]
        self.assertIn('REPROJECTION_HIGH', codes)

    def test_motion_along_the_view_axis_is_flagged_as_depth_dominated(self):
        frames, rows = build_capture(azimuth=45.0)
        _lateral, depth_axis = camera_axes(45.0)
        # Push one wrist straight into the screen: no in-plane evidence exists
        # for that displacement, so the diagnostic must say so.
        for step, frame in enumerate(frames):
            wrist = np.asarray(frame['body3d']['wrist.L'])
            frame['body3d']['wrist.L'] = tuple(wrist + np.asarray(depth_axis) * (0.02 * step))
        report = reprojection.diagnose(frames, rows, (1280.0, 720.0),
                                       {'camera_azimuth_degrees': 45.0})
        self.assertIn('wrist.L', report['depth_dominated_joints'])
        codes = [warning['code'] for warning in reprojection.findings(report)]
        self.assertIn('DEPTH_DOMINANT_MOTION', codes)

    def test_insufficient_observations_are_not_scored(self):
        report = reprojection.diagnose([], [], (1280.0, 720.0))
        self.assertEqual(report['status'], 'insufficient_observations')
        self.assertEqual(reprojection.findings(report), [])

    def test_low_confidence_and_missing_pixels_are_ignored(self):
        frames, rows = build_capture(azimuth=45.0)
        for row in rows:
            for index in (15, 16):
                row['body2d'][index][2] = 0.05
        report = reprojection.diagnose(frames, rows, (1280.0, 720.0),
                                       {'camera_azimuth_degrees': 45.0})
        self.assertNotIn('ankle.L', report['reprojection']['per_joint'])
        self.assertNotIn('ankle.R', report['reprojection']['per_joint'])
        self.assertIn('knee.L', report['reprojection']['per_joint'])


if __name__ == '__main__':
    unittest.main()
