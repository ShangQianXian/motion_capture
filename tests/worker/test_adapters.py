"""Synthetic adapter tests; never download or load trained model weights."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from backend_worker.pose2d_mmpose import coco17_to_h36m17
from backend_worker.pose3d_motionbert import Body3DLifter, _extract_keypoints_3d
from core import errors


def prediction(array):
    return [SimpleNamespace(pred_instances=SimpleNamespace(keypoints=array))]


class AdapterShapes(unittest.TestCase):
    def test_head_matches_coco_eye_midpoint(self):
        points = np.arange(34, dtype=np.float32).reshape(17, 2)
        body, score = coco17_to_h36m17(points, np.ones(17))
        np.testing.assert_array_equal(body[10], (points[1] + points[2]) / 2)

    def test_person_and_temporal_axes(self):
        sequence = np.arange(5 * 17 * 3, dtype=np.float32).reshape(5, 17, 3)
        for output in (sequence, sequence[None]):
            np.testing.assert_array_equal(_extract_keypoints_3d(np, prediction(output)), sequence)
        self.assertIsNone(_extract_keypoints_3d(np, prediction(sequence) * 2))
        self.assertIsNone(_extract_keypoints_3d(np, prediction(np.zeros((1, 16, 3)))))
        self.assertIsNone(_extract_keypoints_3d(np, prediction(np.full((1, 17, 3), np.nan))))

    def lifter(self, inference, window=5):
        lifter = object.__new__(Body3DLifter)
        lifter._PoseDataSample = SimpleNamespace
        lifter._InstanceData = SimpleNamespace
        lifter.model = SimpleNamespace(cfg=SimpleNamespace(
            model=SimpleNamespace(backbone={"seq_len": window}),
            test_dataloader=SimpleNamespace(dataset={})))
        lifter._inference = inference
        return lifter

    def test_short_sequences_gaps_and_window_boundaries(self):
        seen = []
        def inference(model, samples, **kwargs):
            self.assertEqual(len(samples), 5)
            self.assertTrue(kwargs["with_track_id"])
            sample = samples[2][0]
            self.assertEqual(sample.track_id, 0)
            self.assertEqual(sample.pred_instances.bboxes.shape, (1, 4))
            seen.append(float(sample.pred_instances.keypoints[0, 0, 0]))
            return prediction(np.full((1, 17, 3), seen[-1], dtype=np.float32))
        for length in (1, 4, 5, 6, 11):
            with self.subTest(length=length):
                points = [np.ones((17, 2), dtype=np.float32) * i for i in range(length)]
                scores = [np.ones(17)] * length
                if length > 5:
                    points[2], scores[2] = None, None
                seen.clear()
                result = self.lifter(inference).lift(points, scores, (640, 480))
                expected = [i for i in range(length) if points[i] is not None]
                self.assertEqual(seen, expected)
                np.testing.assert_array_equal(result[:, 0, 0], expected)

    def test_bad_outputs_fail_instead_of_zero_fill(self):
        for result in ([], prediction(np.zeros((3, 17, 3))), prediction(np.full((1, 17, 3), np.inf))):
            with self.assertRaises(errors.MocapError):
                self.lifter(lambda *a, **kw: result).lift([np.ones((17, 2))], [np.ones(17)], (640, 480))

    def test_typeerror_is_not_retried(self):
        from unittest.mock import Mock
        inference = Mock(side_effect=TypeError("backend bug"))
        with self.assertRaises(errors.MocapError):
            self.lifter(inference).lift([np.ones((17, 2))], [np.ones(17)], (640, 480))
        inference.assert_called_once()

    def test_cancellation(self):
        token = SimpleNamespace(cancelled=lambda: True)
        with self.assertRaises(errors.MocapError) as caught:
            self.lifter(None).lift([np.ones((17, 2))], [np.ones(17)], (640, 480), cancel_token=token)
        self.assertEqual(caught.exception.code, errors.CANCELLED)

    def test_long_occlusion_cuts_receptive_field(self):
        seen = []
        def inference(model, temporal, **kwargs):
            values = [float(s[0].pred_instances.keypoints[0, 0, 0]) for s in temporal]
            seen.append(values)
            self.assertFalse(kwargs['norm_pose_2d'])
            return prediction(np.ones((1, 17, 3), dtype=np.float32))
        points = [np.zeros((17, 2)), None, None, None, np.ones((17, 2)) * 100]
        self.lifter(inference, 5).lift(points, [np.ones(17), None, None, None, np.ones(17)], (640, 480), sample_fps=10)
        self.assertEqual(seen, [[0.] * 5, [100.] * 5])

    def test_wholebody_initialization_does_not_fall_back(self):
        from backend_worker import pipeline, pose2d_mmpose
        from unittest.mock import Mock
        with patch.object(pose2d_mmpose, 'resolve_device', return_value='cpu'), patch.object(pose2d_mmpose, 'PersonDetector'), \
                patch.object(pose2d_mmpose, 'Body2DEstimator', side_effect=errors.MocapError(errors.MODEL_MISSING, 'missing')) as estimator:
            with self.assertRaises(errors.MocapError):
                pipeline._run_mmpose({}, 'quality_feet', {}, Mock(), None, {})
            estimator.assert_called_once()

    def test_crop_reads_neighboring_context_but_exports_selected_samples(self):
        from backend_worker import pipeline, pose2d_mmpose, pose3d_motionbert, media_decode
        from unittest.mock import Mock
        indexes = (0, 120, 121, 122, 123, 124, 245)
        decoded = [SimpleNamespace(index=i, source_index=28+i, timestamp=(28+i)/24, width=640, height=480,
                                   image=np.zeros((480,640,3),dtype=np.uint8)) for i in indexes]
        source = Mock(fps=24., native_fps=24., width=640, height=480, native_total=400, total_frames=246, media_type='video')
        source.__iter__ = Mock(return_value=iter(decoded))
        body2d = Mock()
        body2d.estimate.return_value = (np.arange(34).reshape(17,2), np.ones(17))
        options = {'motion_type':'attack'}
        with patch.object(pose2d_mmpose, 'resolve_device', return_value='cpu'), patch.object(pose2d_mmpose, 'PersonDetector'), \
                patch.object(pose2d_mmpose, 'Body2DEstimator', return_value=body2d), \
                patch.object(media_decode, 'open_media', return_value=source) as opened, \
                patch.object(pose3d_motionbert, 'Body3DLifter') as lifter:
            lifter.return_value.lift.return_value = np.tile(np.arange(51).reshape(17,3),(len(indexes),1,1))
            frames, fps, _, _ = pipeline._run_mmpose({'input':dict(path='fixture.mp4',frame_start=150,frame_end=152)},
                                                     'quality',{},Mock(),None,options)
            self.assertEqual(opened.call_args.args[3:5], (29,273))
            self.assertEqual([f['frame'] for f in frames], [150,151,152])
            self.assertEqual([r['source_index'] for r in options['_preview_rows']], [149,150,151])
            self.assertEqual(lifter.return_value.lift.call_args.kwargs['max_gap_seconds'], .1)
            self.assertEqual(lifter.return_value.lift.call_args.kwargs['sample_fps'],24)
            self.assertAlmostEqual(frames[0]['time'],149/24)

    def test_quality_plus_initialization_falls_back_to_quality(self):
        from unittest.mock import Mock
        from backend_worker import pipeline, media_decode, pose2d_mmpose
        source = Mock(fps=30., total_frames=0)
        source.__iter__ = Mock(return_value=iter([]))
        reporter = Mock()
        with patch.object(pose2d_mmpose, "resolve_device", return_value="cpu"), \
                patch.object(pose2d_mmpose, "PersonDetector"), \
                patch.object(pose2d_mmpose, "Body2DEstimator", side_effect=[
                    errors.MocapError(errors.CUDA_OOM, "out of memory"), Mock()]) as estimator, \
                patch.object(media_decode, "open_media", return_value=source):
            frames, fps, warnings, profile = pipeline._run_mmpose({'input': {'path': 'fixture.mp4'}}, "quality_plus", {}, reporter, None, {})
        self.assertEqual([call.args[1] for call in estimator.call_args_list], ["quality_plus", "quality"])
        self.assertEqual(profile, "quality")
        self.assertEqual(warnings[0]["code"], pipeline.CODE_PROFILE_FALLBACK)
        source.close.assert_called_once()

    def test_quality_plus_keeps_body_and_adds_wholebody_observations(self):
        from unittest.mock import Mock
        from backend_worker import pipeline, media_decode, pose2d_mmpose, pose3d_motionbert
        source=Mock(fps=24.,native_fps=24.,width=640,height=480,native_total=1,total_frames=1,media_type='image')
        source.__iter__=Mock(return_value=iter([SimpleNamespace(index=0,source_index=0,timestamp=0.,width=640,height=480,image=None)]))
        body,extra=Mock(),Mock()
        body.estimate.return_value=(np.full((17,2),50.),np.full(17,.9))
        extra.estimate.return_value=(np.full((133,2),100.),np.full(133,.8))
        options={}
        with patch.object(pose2d_mmpose,'resolve_device',return_value='cpu'), patch.object(pose2d_mmpose,'PersonDetector'), \
                patch.object(pose2d_mmpose,'Body2DEstimator',side_effect=[body,extra]), \
                patch.object(media_decode,'open_media',return_value=source), patch.object(pose3d_motionbert,'Body3DLifter') as lifter:
            lifter.return_value.lift.return_value=np.arange(51).reshape(1,17,3)
            pipeline._run_mmpose({'input':{'path':'fixture.png'}},'quality_plus',{},Mock(),None,options)
        points=options['_preview_rows'][0]['body2d']
        self.assertEqual(len(points),133)
        self.assertAlmostEqual(points[0][0],50/640)
        self.assertAlmostEqual(points[23][0],100/640)
        self.assertEqual(len(options['_preview_rows'][0]['feet2d']['L']),3)

    @unittest.skipUnless(importlib.util.find_spec("mmpose"), "requires Quality worker")
    def test_real_mmpose_api_contract_without_weights(self):
        from mmpose.apis import inference_pose_lifter_model
        from mmpose.structures import PoseDataSample
        from mmengine.structures import InstanceData
        from mmengine.config import Config
        captured = []
        class Model:
            cfg = Config(dict(default_scope="mmpose", model=dict(backbone=dict(seq_len=5)),
                              test_dataloader=dict(dataset=dict(pipeline=[]))))
            dataset_meta = dict(dataset_name="h36m", stats_info=dict(bbox_center=np.array([[320., 240.]]), bbox_scale=200.))
            def test_step(self, batch):
                captured.append(np.asarray(batch["keypoints"][0]))
                return prediction(np.ones((1, 5, 17, 3), dtype=np.float32))
        lifter = self.lifter(inference_pose_lifter_model)
        lifter.model = Model()
        lifter._PoseDataSample = PoseDataSample
        lifter._InstanceData = InstanceData
        result = lifter.lift([np.arange(34).reshape(17, 2)] * 2, [np.ones(17)] * 2, (640, 480))
        self.assertEqual(result.shape, (2, 17, 3))
        self.assertEqual(captured[0].shape, (5, 17, 2))
        result = lifter.lift([np.arange(34).reshape(17, 2), None, np.arange(34).reshape(17, 2)],
                             [np.ones(17), None, np.ones(17)], (640, 480))
        self.assertEqual(result.shape, (2, 17, 3))


if __name__ == "__main__":
    unittest.main()
