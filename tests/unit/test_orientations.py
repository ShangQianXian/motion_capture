import copy
import math
import unittest

from core import orientations as ori, retarget_math as rm, result_schema, pose_calibration, motion_processing as mp, preview
from backend_worker import export_result
from tests.fixtures.motions import clip


def error(a,b):
    return math.degrees(rm.quat_angle(rm.quat_mul(rm.quat_conjugate(a),b)))


class Orientations(unittest.TestCase):
    def frames(self):
        frames = clip('idle')[:20]
        for f in frames:
            f['orientations'] = {name: rm.IDENTITY_QUAT for name in ori.NAMES}
            f['orientation_quality'] = {name: dict(confidence=.9, estimated=False, source='test', length=.215) for name in ori.NAMES}
        return frames

    def test_roundtrip_optional_and_invalid(self):
        frames = self.frames()
        payload = export_result.build_result(frames,30)
        result_schema.validate_result(payload)
        result = result_schema.MocapResult(payload)
        self.assertEqual(result.frames[0].to_dict()['orientations'],payload['frames'][0]['orientations'])
        for q in ([0,0,0,0],[1,0,0,float('nan')],[1,0,0],['x',0,0,0], [True,0,0,0]):
            payload['frames'][0]['orientations']['head'] = q
            with self.assertRaises(Exception):
                result_schema.validate_result(payload)
        legacy = export_result.build_result(clip('idle')[:1],30)
        result_schema.validate_result(legacy)
        self.assertFalse(result_schema.MocapResult(legacy).frames[0].orientations)

    def test_mirror_and_calibration_transform_frames(self):
        frames = self.frames()
        q = rm.quat_from_axis_angle((1,2,3),.7)
        frames[0]['orientations']['head'] = q
        result = result_schema.MocapResult(export_result.build_result(frames,30))
        result_schema.mirror_result_x(result)
        expected = ori.mirror(q)
        self.assertLess(error(result.frames[0].orientations['head'],expected),.001)
        result_schema.mirror_result_x(result)
        self.assertLess(error(result.frames[0].orientations['head'],q),.001)
        calibrated = pose_calibration.calibrated_result(result,.3)
        self.assertLess(error(calibrated.frames[0].orientations['head'],rm.quat_mul(rm.quat_from_axis_angle((1,0,0),.3),q)),.001)

    def test_toe_noise_does_not_change_legs_with_or_without_lock(self):
        frames = self.frames()
        perturbed = copy.deepcopy(frames)
        for i,f in enumerate(perturbed):
            for side in ('L','R'):
                f['orientations']['foot.'+side] = rm.quat_from_axis_angle((1,0,1),(-1)**i)
            ori.apply_feet(f)
        for lock in (0,.7):
            a,_,_=mp.process(frames,30,dict(foot_lock_strength=lock))
            b,_,_=mp.process(perturbed,30,dict(foot_lock_strength=lock))
            origin_delta=rm.vec_sub(a[0]['body3d']['pelvis'],b[0]['body3d']['pelvis'])
            for fa,fb in zip(a,b):
                for name in ('pelvis','hip.L','knee.L','ankle.L','hip.R','knee.R','ankle.R'):
                    # A constant floor-origin offset is permitted, never motion.
                    self.assertLess(rm.vec_distance(rm.vec_sub(fa['body3d'][name],fb['body3d'][name]),origin_delta),1e-9,name)
            self.assertGreaterEqual(min(p[2] for f in b for n,p in f['body3d'].items() if n.startswith(('toe.','heel.'))),-1e-9)

    def test_short_occlusion_interpolates_and_long_resets(self):
        frames = self.frames()
        q=rm.quat_from_axis_angle((0,0,1),.7)
        for f in frames:
            f['orientations']['head']=q
        for i in (4,5,6):
            frames[i]['orientation_quality']['head']['confidence']=0
        for i in range(10,20):
            frames[i]['orientation_quality']['head']['confidence']=0
        ori.stabilize(frames)
        self.assertLess(error(frames[5]['orientations']['head'],q),.001)
        self.assertEqual(frames[5]['orientation_quality']['head']['source'],'interpolated')
        self.assertEqual(frames[-1]['orientation_quality']['head']['source'],'body_prior')
        gap=self.frames()[:2]
        gap[1]['time']=3
        gap[0]['orientations']['head']=q
        gap[1]['orientation_quality']['head']['confidence']=0
        ori.stabilize(gap)
        self.assertLess(error(gap[1]['orientations']['head'],ori.body_basis(gap[1]['body3d'])),.001)

    def test_antipodal_quaternions_do_not_flip(self):
        frames=self.frames()
        for i,f in enumerate(frames):
            f['orientations']['head']=((-1)**i,0,0,0)
        ori.stabilize(frames)
        self.assertTrue(all(error(f['orientations']['head'],rm.IDENTITY_QUAT)<.001 for f in frames))

    def test_degenerate_knee_keeps_bend_plane(self):
        b=self.frames()[0]['body3d']
        b.update({'hip.L':(0,0,1), 'knee.L':(0,0,.5), 'ankle.L':(0,0,.01)})
        lengths={('hip.L','knee.L'):.5,('knee.L','ankle.L'):.5}
        planes={'L':(0,1,0)}
        mp._leg_ik(b,'L',(0,0,.01),lengths,planes=planes)
        self.assertGreater(b['knee.L'][1],0)

    def test_continuous_rotation_preserves_motion_at_supported_rates(self):
        for fps in (15,24,30,60):
            frames=self.frames()
            for i,f in enumerate(frames):
                f['time']=i/fps
                f['orientations']['head']=rm.quat_from_axis_angle((0,0,1),i/fps)
            expected=[f['orientations']['head'] for f in frames]
            ori.stabilize(frames)
            self.assertTrue(all(error(f['orientations']['head'],q)<.001 for f,q in zip(frames,expected)))

    def test_old_processing_version_cannot_be_applied_as_fixed(self):
        result=result_schema.MocapResult(export_result.build_result(self.frames(),30))
        manifest={'source':{},'processing_version':'0.3'}
        state=preview.ReviewState(result,manifest,{},'fixture.mp4')
        self.assertFalse(state.matches({},'fixture.mp4'))
        manifest['processing_version']=ori.VERSION
        self.assertTrue(state.matches({},'fixture.mp4'))


if __name__=='__main__':
    unittest.main()
