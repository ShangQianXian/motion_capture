import copy
import math
import unittest

from core import errors, pose_calibration, result_schema
from backend_worker.pose3d_motionbert import h36m_to_standard, ground_and_stand


def tilted_result(angles=(16., 20., 24.)):
    frames = []
    for i, degrees in enumerate(angles):
        angle = math.radians(degrees)
        def rotate(point):
            x, y, z = point
            return (x, math.cos(angle)*y-math.sin(angle)*z,
                    math.sin(angle)*y+math.cos(angle)*z)
        frames.append(dict(frame=i+1, time=i/24,
                           body3d={name: rotate(point) for name, point in dict(
                               pelvis=(0,0,1), head=(0,0,2),
                               **{'ankle.L': (.1,0,0), 'ankle.R': (-.1,0,0),
                                  'wrist.L': (.4,0,1.2)}).items()},
                           hands3d={'index.01.L': rotate((.5,0,1.2))}))
    return result_schema.MocapResult(dict(frames=frames))


class PitchCalibrationTests(unittest.TestCase):
    def test_constant_pitch_removes_offset_and_preserves_motion(self):
        result = tilted_result()
        original = copy.deepcopy(result.data)
        angle = pose_calibration.estimate_standing_pitch(result)
        self.assertAlmostEqual(math.degrees(angle), -20.)
        corrected = pose_calibration.calibrated_result(result, angle)
        for frame, expected in zip(corrected.frames, (-4., 0., 4.)):
            up = [frame.joint('head')[i]-frame.joint('pelvis')[i] for i in range(3)]
            self.assertAlmostEqual(math.degrees(math.atan2(-up[1],up[2])), expected)
            self.assertAlmostEqual(math.dist(frame.joint('wrist.L'),frame.joint('index.01.L')), .1)
        self.assertEqual(result.data, original)
        self.assertEqual(corrected.data['frames'], [frame.to_dict() for frame in corrected.frames])
        self.assertEqual(corrected.data, pose_calibration.calibrated_result(result, angle).data)

    def test_ground_height_and_horizontal_root_travel_are_preserved(self):
        result = tilted_result()
        for frame in result.frames:
            for store in (frame.body3d, frame.hands3d):
                for name, point in store.items():
                    store[name] = (point[0]+frame.frame, point[1], point[2]+.2*frame.frame)
        fixed = pose_calibration.calibrated_result(result, math.radians(-20))
        for old, new in zip(result.frames,fixed.frames):
            self.assertEqual(old.joint('pelvis')[:2], new.joint('pelvis')[:2])
            self.assertAlmostEqual(old.joint('ankle.L')[2],new.joint('ankle.L')[2])

    def test_unreliable_or_nonstanding_input_cannot_estimate_pitch(self):
        for result in (tilted_result(()), tilted_result((70.,))):
            with self.assertRaises(errors.MocapError):
                pose_calibration.estimate_standing_pitch(result)
        result = tilted_result()
        for frame in result.frames:
            frame.confidence['head'] = .1
        with self.assertRaises(errors.MocapError):
            pose_calibration.estimate_standing_pitch(result)
        with self.assertRaises(errors.MocapError):
            pose_calibration.calibrated_result(result, float('nan'))


class MotionBERTFeetTests(unittest.TestCase):
    def test_toes_stay_below_ankles_before_and_after_grounding(self):
        points = [(0.,0.,0.) for _ in range(17)]
        for index in (2,5):
            points[index] = (0.,.5,0.)
        for index in (3,6):
            points[index] = (0.,1.,0.)
        body, _ = h36m_to_standard(points)
        self.assertAlmostEqual(body['toe.L'][2], -1.04)
        self.assertLess(math.dist(body['toe.L'],body['ankle.L']), .2)
        grounded = ground_and_stand(body)
        self.assertAlmostEqual(grounded['toe.L'][2], 0.)
        self.assertAlmostEqual(grounded['ankle.L'][2], .04)
        shifted, _ = h36m_to_standard([(x,y+2,z) for x,y,z in points])
        for name, point in ground_and_stand(shifted).items():
            for actual, expected in zip(point, grounded[name]):
                self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
