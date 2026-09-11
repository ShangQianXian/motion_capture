"""Decode real small videos through the same persistent JSONL service as Blender."""

import json
from pathlib import Path
import tempfile
import time
import unittest
import sys

import cv2
import numpy as np

from core import job_schema, preview, worker_client
from backend_worker import media_decode, media_preview


class MediaPreviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mocap_media_')
        self.root = Path(self.directory.name)
        self.video = self.root / '素材.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 60, (160, 240))
        if not writer.isOpened():
            self.skipTest('MJPG encoder unavailable')
        for i in range(16):
            writer.write(np.full((240, 160, 3), i * 12, dtype=np.uint8))
        writer.release()
        self.worker = None

    def tearDown(self):
        if self.worker:
            self.worker.close()
        self.directory.cleanup()

    def wait_event(self, name, request_id=None):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for event in self.worker.poll_events():
                if event.event in ('failed', 'media_error'):
                    self.fail(event.error.user_text())
                if event.event == name and (request_id is None or event.fields.get('request_id') == request_id):
                    return event.fields
            time.sleep(.01)
        self.fail('Timed out waiting for ' + name + '\n' + self.worker.stderr_tail())

    def test_unicode_image_and_bounded_cache(self):
        image = self.root / '竖屏图片.png'
        ok, encoded = cv2.imencode('.png', np.full((1600, 800, 3), 80, np.uint8))
        image.write_bytes(encoded.tobytes())
        decoder = media_preview.Decoder(str(image), 'auto', str(self.root / 'cache'))
        try:
            self.assertEqual((decoder.info['width'], decoder.info['height']), (800, 1600))
            path, index = decoder.frame(0)
            decoded = media_decode.read_image(cv2, path)
            self.assertEqual(decoded.shape[:2], (960, 480))
            total = media_preview.trim_cache(self.root / 'cache', path, limit=1)
            self.assertEqual(total, Path(path).stat().st_size)
        finally:
            decoder.close()

    def test_crop_decimation_source_frames_are_exact(self):
        source = media_decode.open_media(str(self.video), target_fps=30, frame_start=3, frame_end=6)
        frames = list(source)
        self.assertEqual([f.source_index for f in frames], [4, 6, 8, 10])
        self.assertEqual(source.fps, 30)
        for frame in frames:
            self.assertAlmostEqual(frame.timestamp, frame.source_index / 60, places=5)
            self.assertAlmostEqual(float(frame.image.mean()), frame.source_index * 12, delta=3)

    def test_landscape_rates_corrupt_media_and_disk_eviction(self):
        for fps in (24, 30, 60):
            path = self.root / ('landscape_%s.avi' % fps)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (320, 180))
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((180, 320, 3), i * 18, np.uint8))
            writer.release()
            decoder = media_preview.Decoder(str(path), 'video', str(self.root / 'cache'))
            try:
                self.assertEqual((decoder.info['width'], decoder.info['height'], decoder.info['fps']), (320, 180, fps))
                for i in (10, 2, 7):
                    cached, actual = decoder.frame(i)
                    self.assertEqual(actual, i)
                    self.assertAlmostEqual(float(media_decode.read_image(cv2, cached).mean()), i * 18, delta=3)
                remaining = media_preview.trim_cache(self.root / 'cache', cached, limit=Path(cached).stat().st_size)
                self.assertLessEqual(remaining, Path(cached).stat().st_size)
                self.assertEqual(len(list((self.root / 'cache').glob('*/*.jpg'))), 1)
            finally:
                decoder.close()
        broken = self.root / 'broken.png'
        broken.write_bytes(b'not an image')
        from core import errors
        with self.assertRaises(errors.MocapError):
            media_preview.Decoder(str(broken), 'image', str(self.root / 'cache'))

    def test_persistent_service_probe_latest_seek_and_close(self):
        job = dict(version='0.2', mode='media_preview', job_id='test',
                   input={'path': str(self.video)}, output={'dir': str(self.root / 'job')})
        path = job_schema.write_job(job)
        self.worker = worker_client.WorkerProcess(sys.executable, path, job['output']['dir'], duplex=True)
        info = self.wait_event('media_info')['info']
        self.assertEqual((info['width'], info['height'], info['fps']), (160, 240, 60))
        for i in range(1, 10):
            self.worker.send(dict(command='frame', request_id=i, source_index=i))
        fields = self.wait_event('media_frame', 9)
        self.assertEqual(fields['source_index'], 9)
        decoded = media_decode.read_image(cv2, fields['path'])
        self.assertAlmostEqual(float(decoded.mean()), 108, delta=3)
        self.worker.send(dict(command='frame', request_id=10, source_index=2))
        fields = self.wait_event('media_frame', 10)
        self.assertEqual(fields['source_index'], 2)
        self.worker.send(dict(command='probe', request_id=11))
        self.wait_event('media_info', 11)
        self.worker.send(dict(command='close', request_id=12))
        self.worker.wait(timeout=5)
        self.assertEqual(self.worker.returncode, 0)


if __name__ == '__main__':
    unittest.main()
