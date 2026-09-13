"""Exercise the real MotionBERT codec, packing and decoding without checkpoints."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from backend_worker import pose3d_motionbert as adapter


@unittest.skipUnless(importlib.util.find_spec('mmpose'), 'requires Quality worker')
class MotionBERTPipelineTests(unittest.TestCase):
    def test_real_243_frame_codec_packing_and_head_decode(self):
        import mmpose
        import torch
        from mmengine import Config
        from mmengine.structures import InstanceData
        from mmpose.apis import inference_pose_lifter_model
        from mmpose.models.heads import MotionRegressionHead
        from mmpose.structures import PoseDataSample

        config_path = Path(mmpose.__file__).parent / '.mim/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py'
        config = Config.fromfile(str(config_path))
        original_pipeline = list(config.test_dataloader.dataset.pipeline)
        # Run the real head and decoder with small features, avoiding the large backbone.
        head = MotionRegressionHead(in_channels=4, embedding_size=4,
                                    decoder=config.model.head.decoder).eval()
        captured = []

        class Model:
            cfg = config
            dataset_meta = dict(dataset_name='h36m', flip_indices=list(range(17)),
                                stats_info=dict(bbox_center=np.array([[320., 240.]]), bbox_scale=200.))

            def test_step(self, batch):
                tensor = batch['inputs'][0]
                sample = batch['data_samples'][0]
                captured.append(tuple(tensor.shape))
                self_shape = (243, 17, 3)
                assert tuple(tensor.shape) == self_shape
                # Codec alone normalizes original pixels; the generic API must
                # not resize each pose to an unrelated training bbox first.
                np.testing.assert_allclose(tensor[121, 0, :2].numpy(), point[0] / 640 * 2 - [1, 480 / 640], atol=.004)
                np.testing.assert_allclose(tensor[121, :, 2].numpy(), .42)
                assert sample.gt_instances.lifting_target.shape == self_shape
                assert sample.gt_instances.lifting_target_visible.shape[0] == 243
                assert sample.gt_instance_labels.lifting_target_weight.shape[0] == 243
                assert sample.metainfo['factor'].shape == (243, 1)
                predictions = head.predict(torch.zeros((1, 243, 17, 4)), batch['data_samples'])
                result = PoseDataSample()
                result.pred_instances = predictions[0]
                return [result]

        model = Model()
        with patch.object(adapter, 'require_lifter_api', return_value=(
                lambda *a, **kw: model, inference_pose_lifter_model, PoseDataSample, InstanceData)), \
                patch.object(adapter.model_manifest, 'require_artifact', return_value='unused'):
            lifter = adapter.Body3DLifter('unused', device='cpu', manifest={})
        point = np.arange(34, dtype=np.float32).reshape(17, 2)
        for points, scores in (([point], [np.ones(17) * .42]),
                               ([point, None, point + 1], [np.ones(17) * .42, None, np.ones(17) * .42])):
            result = lifter.lift(points, scores, (640, 480))
            self.assertEqual(result.shape, (sum(p is not None for p in points), 17, 3))
            self.assertTrue(np.isfinite(result).all())
        self.assertEqual(len(captured), 3)
        # Installation must not edit the original transforms or change window length.
        self.assertEqual(original_pipeline[0]['type'], 'GenerateTarget')
        self.assertEqual(config.model.backbone.seq_len, 243)


class MotionBERTTargetAlignmentTests(unittest.TestCase):
    def test_targets_are_writable_independent_frames_and_inputs_are_preserved(self):
        points = np.zeros((243, 17, 2), dtype=np.float32)
        target = np.arange(51, dtype=np.float32).reshape(1, 17, 3)
        visible = np.ones((1, 17, 1), dtype=np.float32)
        data = dict(keypoints=points, lifting_target=target, lifting_target_visible=visible)
        aligned = adapter._align_motionbert_targets(data)
        self.assertIs(aligned['keypoints'], points)
        np.testing.assert_array_equal(aligned['lifting_target'][121], target[0])
        aligned['lifting_target'][0] = -1
        np.testing.assert_array_equal(aligned['lifting_target'][1], target[0])
        self.assertEqual(data['lifting_target'].shape, (1, 17, 3))
        self.assertEqual(aligned['lifting_target_visible'].shape, (243, 17, 1))

    def test_already_aligned_targets_are_preserved_and_bad_lengths_rejected(self):
        for frames in (1, 243):
            data = dict(keypoints=np.zeros((frames, 17, 2)),
                        lifting_target=np.zeros((frames, 17, 3)))
            self.assertIs(adapter._align_motionbert_targets(data)['lifting_target'], data['lifting_target'])
        data['lifting_target'] = np.zeros((2, 17, 3))
        with self.assertRaises(ValueError):
            adapter._align_motionbert_targets(data)


if __name__ == '__main__':
    unittest.main()
