"""Downloader regression tests: integrity, HTTP resume, config graphs and selection."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from core import model_manifest
from core.model_download import Downloader, DownloadError, download_lock, safe_path, select_artifacts
from tools import download_models


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None, url='https://example.test/model.pth'):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Length': str(len(body)), **(headers or {})}
        self.url = url

    def geturl(self):
        return self.url


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / '.downloads').mkdir()
        self.url = 'https://example.test/model.pth'
        self.manifest = model_manifest.load_manifest(str(self.root))

    def downloader(self, responses=(), **kwargs):
        opener = Mock(side_effect=responses)
        return Downloader(self.root, opener=opener, sleep=lambda _: None, log=lambda _: None, **kwargs)

    def test_profile_selection_includes_configs_and_fallback(self):
        quality = select_artifacts(self.manifest, ['quality'])
        plus = select_artifacts(self.manifest, ['quality_plus'])
        self.assertEqual(len(quality), 6)
        self.assertEqual(len(plus), 8)
        self.assertEqual(quality[0]['id'], 'config_rtmdet_m_person')
        self.assertTrue({a['id'] for a in quality}.issubset({a['id'] for a in plus}))
        self.assertEqual(len(select_artifacts(self.manifest, all_models=True)), 21)
        feet = {a['id'] for a in select_artifacts(self.manifest, ['quality_feet'])}
        self.assertTrue({'rtmpose_m_wholebody', 'config_rtmpose_m_wholebody', 'motionbert_body3d', 'rtmdet_m_person'} <= feet)

    def test_explicit_model_always_includes_its_config(self):
        artifacts = select_artifacts(self.manifest, artifact_ids=['rtmpose_x_body'])
        self.assertEqual([a['id'] for a in artifacts], ['config_rtmpose_x_body', 'rtmpose_x_body'])

    def test_cpu_alternative_prefers_existing_full(self):
        first = select_artifacts(self.manifest, ['fallback_cpu'])
        self.assertEqual([a['id'] for a in first], ['mediapipe_pose_lite'])
        file = self.root / 'mediapipe/pose_landmarker_full.task'
        file.parent.mkdir()
        file.write_bytes(b'existing')
        self.assertEqual(select_artifacts(self.manifest, ['fallback_cpu'])[0]['id'], 'mediapipe_pose_full')

    def test_unknown_ids_duplicate_paths_and_cycles_fail(self):
        with self.assertRaises(DownloadError):
            select_artifacts(self.manifest, artifact_ids=['missing'])
        self.manifest.artifacts_by_id['rtmpose_m_wholebody']['config_id'] = 'rtmpose_m_wholebody'
        with self.assertRaises(DownloadError):
            select_artifacts(self.manifest, ['quality'])
        self.manifest.artifacts[0]['relative_path'] = self.manifest.artifacts[1]['relative_path']
        with self.assertRaises(DownloadError):
            select_artifacts(self.manifest, ['preview'])

    def test_unsafe_paths_are_rejected(self):
        for path in ('../escape', 'C:/escape', 'C:escape', '//server/file', '/tmp/escape',
                     'model:stream', 'safe/../../escape', 'safe\\..\\escape', 'trailing. ', 'CON.txt', ''):
            with self.subTest(path=path), self.assertRaises(DownloadError):
                safe_path(self.root, path)

    def test_second_downloader_cannot_share_root_lock(self):
        with download_lock(self.root):
            with self.assertRaises(DownloadError):
                with download_lock(self.root):
                    self.fail('second lock was acquired')
        with download_lock(self.root):
            pass  # Lock is released after the first operation finishes.

    def test_download_then_verify_skips_network_and_detects_tampering(self):
        data = b'checkpoint bytes'
        downloader = self.downloader([Response(data)])
        target = downloader.fetch(self.url, 'body/model.pth', expected=hashlib.sha256(data).hexdigest())
        self.assertEqual(target.read_bytes(), data)
        downloader.fetch(self.url, 'body/model.pth')
        downloader.opener.assert_called_once()
        target.write_bytes(b'broken')
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'body/model.pth')

    def test_interrupted_transfer_resumes_with_if_range(self):
        downloader = self.downloader([
            Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'}),
            Response(b'def', 206, {'Content-Range': 'bytes 3-5/6', 'ETag': '"v1"'}),
        ])
        target = downloader.fetch(self.url, 'model.pth')
        self.assertEqual(target.read_bytes(), b'abcdef')
        request = downloader.opener.call_args_list[1].args[0]
        self.assertEqual(request.get_header('Range'), 'bytes=3-')
        self.assertEqual(request.get_header('If-range'), '"v1"')

    def test_resume_works_across_process_restarts(self):
        first = self.downloader([Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'})], retries=0)
        with self.assertRaises(DownloadError):
            first.fetch(self.url, 'model.pth')
        second = self.downloader([Response(b'def', 206, {'Content-Range': 'bytes 3-5/6'})])
        self.assertEqual(second.fetch(self.url, 'model.pth').read_bytes(), b'abcdef')

    def test_ignored_range_restarts_instead_of_appending(self):
        downloader = self.downloader([
            Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'}), Response(b'NEWFILE')])
        self.assertEqual(downloader.fetch(self.url, 'model.pth').read_bytes(), b'NEWFILE')

    def test_without_validator_restart_does_not_mix_versions(self):
        downloader = self.downloader([Response(b'abc', headers={'Content-Length': '6'}), Response(b'new')])
        self.assertEqual(downloader.fetch(self.url, 'model.pth').read_bytes(), b'new')
        self.assertIsNone(downloader.opener.call_args_list[1].args[0].get_header('Range'))

    def test_changed_etag_restarts_without_corruption(self):
        downloader = self.downloader([
            Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'}),
            Response(b'def', 206, {'Content-Range': 'bytes 3-5/6', 'ETag': '"v2"'}), Response(b'new')])
        self.assertEqual(downloader.fetch(self.url, 'model.pth').read_bytes(), b'new')

    def test_invalid_content_range_is_rejected(self):
        downloader = self.downloader([
            Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'}),
            Response(b'def', 206, {'Content-Range': 'bytes 2-4/6'})])
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth')
        self.assertFalse((self.root / 'model.pth').exists())

    def test_416_discards_stale_partial_and_retries(self):
        downloader = self.downloader([
            Response(b'abc', headers={'Content-Length': '6', 'ETag': '"v1"'}),
            HTTPError(self.url, 416, 'range', {}, None), Response(b'new')])
        self.assertEqual(downloader.fetch(self.url, 'model.pth').read_bytes(), b'new')

    def test_bad_hash_and_http_failure_preserve_existing_file(self):
        target = self.root / 'model.pth'
        target.write_bytes(b'user file')
        downloader = self.downloader([Response(b'wrong')], force=True)
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth', expected='0' * 64)
        self.assertEqual(target.read_bytes(), b'user file')
        downloader = self.downloader([HTTPError(self.url, 404, 'missing', {}, None)], force=True)
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth')
        self.assertEqual(target.read_bytes(), b'user file')
        downloader.opener.assert_called_once()

    def test_html_response_is_not_installed_as_weights(self):
        downloader = self.downloader([Response(b'<!DOCTYPE html><title>Error</title>')])
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth')
        self.assertFalse((self.root / 'model.pth').exists())

    def test_existing_unverified_file_needs_explicit_force(self):
        (self.root / 'model.pth').write_bytes(b'old')
        downloader = self.downloader()
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth')
        downloader.opener.assert_not_called()

    def test_verify_only_missing_file_does_not_download(self):
        downloader = self.downloader(verify_only=True)
        with self.assertRaises(DownloadError):
            downloader.fetch(self.url, 'model.pth')
        downloader.opener.assert_not_called()

    def config_artifact(self):
        return {'id': 'config', 'kind': 'config', 'display_name': 'Config',
                'relative_path': 'openmmlab/configs/model.py',
                'download_url': 'https://example.test/configs/body/model.py'}

    def test_config_dependencies_keep_tree_and_are_not_executed(self):
        sentinel = self.root / 'must-not-exist'
        source = "_base_ = ['../_base_/runtime.py', 'mmdet::base.py']\nfrom pathlib import Path\nPath({0!r}).touch()\n".format(str(sentinel))
        downloader = self.downloader([Response(source.encode()), Response(b'answer = 42\n')])
        target = downloader.config(self.config_artifact())
        self.assertFalse(sentinel.exists())
        wrapper_base = downloader.config_bases(target.read_text())[0]
        original = (target.parent / wrapper_base).resolve()
        self.assertTrue(original.is_file())
        self.assertTrue((original.parent / '../_base_/runtime.py').resolve().is_file())
        self.assertEqual(downloader.package_bases, {'mmdet'})
        verifier = self.downloader(verify_only=True)
        verifier.config(self.config_artifact())
        verifier.opener.assert_not_called()

    def test_missing_base_is_repaired_even_when_wrapper_exists(self):
        downloader = self.downloader([Response(b"_base_ = '../base.py'"), Response(b'a = 1')])
        target = downloader.config(self.config_artifact())
        source = (target.parent / downloader.config_bases(target.read_text())[0]).resolve()
        (source.parent / '../base.py').resolve().unlink()
        downloader.opener.side_effect = [Response(b'a = 1')]
        downloader.config(self.config_artifact())
        self.assertEqual(downloader.opener.call_count, 3)

    def test_config_cycle_and_invalid_syntax_fail_before_entry_written(self):
        for source in (b"_base_ = 'model.py'", b'<html>error</html>', b'_base_ = some_function()'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / '.downloads').mkdir()
                downloader = Downloader(root, opener=Mock(return_value=Response(source)), log=lambda _: None)
                with self.assertRaises(DownloadError):
                    downloader.config(self.config_artifact())
                self.assertFalse((root / 'openmmlab/configs/model.py').exists())

    def test_list_only_uses_user_manifest_without_network_or_writes(self):
        destination = self.root / 'new models'
        source = self.root / 'custom.json'
        source.write_text(json.dumps({'artifacts': [{'id': 'custom', 'display_name': 'Custom',
                          'relative_path': 'custom.pth', 'download_url': self.url}]}))
        output = io.StringIO()
        with redirect_stdout(output), patch.object(Downloader, 'artifact') as fetch:
            result = download_models.main(['--models-root', str(destination), '--manifest', str(source),
                                           '--artifact', 'custom', '--list'])
        self.assertEqual(result, 0)
        self.assertFalse(destination.exists())
        fetch.assert_not_called()
        self.assertIn('custom.pth', output.getvalue())

    def test_cli_failure_is_nonzero_and_reported(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(Downloader, 'artifact', side_effect=DownloadError('404')):
            result = download_models.main(['--models-root', str(self.root), '--profile', 'preview'])
        self.assertEqual(result, 1)
        report = json.loads((self.root / '.downloads/last_report.json').read_text())
        self.assertFalse(report['ok'])
        self.assertEqual(report['artifacts'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
