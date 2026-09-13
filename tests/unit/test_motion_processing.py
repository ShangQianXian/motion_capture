import copy
import math
import unittest
from core import motion_processing as mp, retarget_math as rm
from backend_worker import postprocess
from tests.fixtures.motions import clip


class MotionFidelityTests(unittest.TestCase):
    def test_presets_retain_amplitude_timing_and_bone_lengths(self):
        for fps in (15, 24, 30, 60):
            for preset in mp.PRESETS:
                with self.subTest(fps=fps, preset=preset):
                    source = clip(preset, fps)
                    original = copy.deepcopy(source)
                    out, warnings, metrics = mp.process(source, fps, dict(motion_type=preset, coordinate_space='world'))
                    self.assertEqual(source, original)
                    self.assertEqual([f['time'] for f in source], [f['time'] for f in out])
                    for name, values in metrics['joints'].items():
                        if values['amplitude_ratio'] is not None:
                            self.assertGreaterEqual(values['amplitude_ratio'], .9, name)
                            self.assertLessEqual(abs(values['lag_frames']), 1, name)
                    for a, b in mp.LIMBS:
                        lengths = [rm.vec_distance(f['body3d'][a], f['body3d'][b]) for f in out]
                        self.assertLess(max(lengths) - min(lengths), 1e-5)
                    self.assertTrue(all(f['body3d']['hip.L'][0] > f['body3d']['hip.R'][0] for f in out))
                    for original_frame, frame in zip(source, out):
                        for side in ('L', 'R'):
                            def bend(pose):
                                hip, knee, ankle = (pose[n + '.' + side] for n in ('hip','knee','ankle'))
                                axis = rm.vec_normalize(rm.vec_sub(ankle, hip))
                                return rm.vec_sub(rm.vec_sub(knee, hip), rm.vec_scale(axis, rm.vec_dot(rm.vec_sub(knee, hip), axis)))
                            self.assertGreaterEqual(rm.vec_dot(bend(original_frame['body3d']), bend(frame['body3d'])), -1e-6)
                            self.assertGreaterEqual(frame['body3d']['toe.' + side][2], -.005)
                    self.assertEqual('support_events' in metrics, preset in ('walk', 'run'))

    def test_flight_is_not_locked_and_contact_events_are_close(self):
        for mode in ('walk', 'run'):
            fps = 60
            truth = clip(mode, fps)
            out, _, _ = mp.process(truth, fps, dict(motion_type=mode, coordinate_space='world'))
            for side in ('L', 'R'):
                expected = [i for i in range(1, len(truth)) if truth[i]['contacts']['foot.' + side] != truth[i - 1]['contacts']['foot.' + side]]
                actual = [i for i in range(1, len(out)) if out[i]['contacts']['foot.' + side] != out[i - 1]['contacts']['foot.' + side]]
                for event in expected:
                    self.assertLessEqual(min(abs(i - event) for i in actual) / fps, .1)
            for i, frame in enumerate(truth):
                if not any(frame['contacts'].values()):
                    self.assertFalse(any(out[i]['contacts'].values()))

    def test_long_gaps_do_not_interpolate_or_carry_constraints(self):
        truth = clip('walk')
        rows = [dict(sample_frame=f['frame'], time=f['time']) for f in truth]
        sparse = [f for f in truth if f['frame'] not in range(20, 25) and f['frame'] not in range(50, 70)]
        out, _, _ = mp.process(sparse, 30, dict(_preview_rows=rows, coordinate_space='world'))
        numbers = {f['frame'] for f in out}
        self.assertTrue(set(range(20, 25)) <= numbers)
        self.assertFalse(set(range(50, 70)) & numbers)
        self.assertTrue(all(row.get('unreliable') for row in rows[49:69]))
        attack, _, _ = mp.process(sparse, 30, dict(motion_type='attack', _preview_rows=copy.deepcopy(rows), coordinate_space='world'))
        self.assertFalse(set(range(20, 25)) & {f['frame'] for f in attack})

    def test_synthesized_toes_do_not_destroy_contact_confidence(self):
        source = clip('idle')
        for frame in source:
            for side in ('L', 'R'):
                frame['confidence']['toe.' + side] = .3
        states, _ = mp.foot_states(source, 'world', 'idle')
        self.assertTrue(all(s == 'contact' for s in states['L']))
        source[20]['confidence']['ankle.L'] = .1
        states, _ = mp.foot_states(source, 'world', 'idle')
        self.assertEqual(states['L'][20], 'unknown')

    def test_idle_denoises_without_freezing_breathing(self):
        truth = clip('idle', 60)
        noisy = copy.deepcopy(truth)
        for i, f in enumerate(noisy):
            p = f['body3d']['chest']
            f['body3d']['chest'] = (p[0] + .0015 * (-1) ** i, p[1], p[2])
        out, _, _ = mp.process(noisy, 60, dict(motion_type='idle', coordinate_space='world'))
        error = sum(f['body3d']['chest'][0] ** 2 for f in out)
        self.assertLess(error, sum(f['body3d']['chest'][0] ** 2 for f in noisy) * .5)
        span = lambda seq: max(f['body3d']['chest'][2] for f in seq) - min(f['body3d']['chest'][2] for f in seq)
        self.assertGreaterEqual(span(out), span(truth) * .9)

    def test_support_slip_reduced_against_legacy(self):
        source = clip('idle', 60)
        for i, f in enumerate(source):
            for side in ('L', 'R'):
                for joint in ('ankle.', 'toe.', 'heel.'):
                    p = f['body3d'][joint + side]
                    f['body3d'][joint + side] = (p[0], p[1] + .025 * math.sin(i / 25), p[2])
                f['confidence']['toe.' + side] = .3
        legacy, _ = postprocess.postprocess(copy.deepcopy(source), 60, {})
        out, _, _ = mp.process(source, 60, dict(motion_type='idle', coordinate_space='world'))
        slip = lambda seq: sum(rm.vec_distance(a['body3d']['ankle.L'], b['body3d']['ankle.L']) for a, b in zip(seq[10:-1], seq[11:]))
        self.assertLessEqual(slip(out), slip(legacy) * .5)

    def test_image_skips_temporal_processing_and_root_is_not_invented(self):
        source = clip('idle')[:1]
        out, _, metrics = mp.process(source, 30, dict(coordinate_space='world', motion_type='attack'))
        self.assertEqual(source, out)
        self.assertTrue(metrics['single_frame'])
        out, warnings, metrics = mp.process(clip('walk'), 30, {})
        self.assertFalse(metrics['root_trajectory_available'])
        self.assertTrue(warnings)
        self.assertTrue(all(f['body3d']['pelvis'][:2] == (0., 0.) for f in out))

    def test_single_detection_in_video_is_not_reported_as_an_image(self):
        frames = clip('idle')
        rows = [dict(sample_frame=f['frame'], time=f['time']) for f in frames]
        out, warnings, metrics = mp.process(frames[30:31],30,dict(_preview_rows=rows))
        self.assertFalse(metrics['single_frame'])
        self.assertTrue(metrics['sparse_capture'])
        self.assertEqual(warnings[0]['code'], 'SPARSE_CAPTURE')
        self.assertTrue(rows[0]['unreliable'])
