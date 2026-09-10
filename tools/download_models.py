"""Download the model files listed in the project's manifest (Python 3.10+)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import model_manifest
from core.errors import MocapError
from core.model_download import DownloadError, Downloader, atomic_json, download_lock, safe_path, select_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', default=str(ROOT / 'models'))
    parser.add_argument('--manifest', help='Optional manifest file; otherwise Models Root/manifest.json wins.')
    parser.add_argument('--profile', action='append', choices=model_manifest.ALL_PROFILES,
                        help='Repeat to combine profiles; default: preview when no selection is supplied.')
    parser.add_argument('--artifact', action='append', default=[], help='Artifact ID; repeat to add models.')
    parser.add_argument('--all', action='store_true', help='Download every artifact, including optional models.')
    parser.add_argument('--include-optional', action='store_true')
    parser.add_argument('--configs-only', action='store_true', help='Prepare configs without downloading weights.')
    parser.add_argument('--list', '--dry-run', dest='list_only', action='store_true', help='Show plan without network or writes.')
    parser.add_argument('--verify-only', action='store_true', help='Verify files and receipts without network requests.')
    parser.add_argument('--force', action='store_true', help='Redownload files; replace existing files only after verification.')
    parser.add_argument('--timeout', type=float, default=30, help='Network timeout in seconds (default: 30).')
    parser.add_argument('--retries', type=int, default=3, help='Retries per file (default: 3).')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0:
        parser.error('--timeout must be positive and --retries must be nonnegative')
    if args.all and (args.profile or args.artifact):
        parser.error('--all cannot be combined with --profile or --artifact')
    if args.force and args.verify_only:
        parser.error('--force cannot be combined with --verify-only')
    root = Path(args.models_root).expanduser().resolve()
    profiles = args.profile or ([] if args.artifact or args.all else ['preview'])
    try:
        if args.manifest:
            source = Path(args.manifest).expanduser().resolve()
            data = json.loads(source.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or not isinstance(data.get('artifacts'), list):
                raise DownloadError('Manifest must contain an artifacts list.')
            manifest = model_manifest.ModelManifest(data, str(source), str(root))
        else:
            manifest = model_manifest.load_manifest(str(root))
        artifacts = select_artifacts(manifest, profiles, args.artifact, args.include_optional, args.all)
        if args.configs_only:
            artifacts = [a for a in artifacts if a.get('kind') == 'config']
        print('Manifest: ' + manifest.source_path)
        print('Models Root: ' + str(root))
        print('Selected: {0} artifacts'.format(len(artifacts)))
        for artifact in artifacts:
            target = safe_path(root, artifact['relative_path'])
            print('  {0} -> {1}{2}'.format(artifact['id'], artifact['relative_path'],
                                         ' [exists; will verify]' if target.exists() else ''))
            if args.list_only:
                print('    ' + artifact['download_url'])
        if args.list_only or not artifacts:
            return 0
        print('Sources and model licenses: docs/LICENSES.md. Downloads occur only when this tool is run.')
        print('SHA256 receipts detect later file changes; expected hashes are checked when supplied in the manifest.')
        results = []
        with download_lock(root):
            downloader = Downloader(root, timeout=args.timeout, retries=args.retries,
                                    force=args.force, verify_only=args.verify_only)
            failed_ids = set()
            for index, artifact in enumerate(artifacts, 1):
                print('[{0}/{1}] {2}'.format(index, len(artifacts), artifact['display_name']), flush=True)
                try:
                    if artifact.get('config_id') in failed_ids:
                        raise DownloadError('Its required config failed; fix the config and rerun.')
                    target = downloader.artifact(artifact)
                    results.append({'id': artifact['id'], 'status': 'ok', 'path': str(target)})
                except (DownloadError, OSError, ValueError) as exc:
                    failed_ids.add(artifact['id'])
                    results.append({'id': artifact['id'], 'status': 'failed', 'error': str(exc)})
                    print('[FAILED] ' + str(exc), file=sys.stderr)
            report = {'manifest': manifest.source_path, 'models_root': str(root), 'artifacts': results,
                      'required_runtime_packages': sorted(downloader.package_bases), 'ok': not failed_ids}
            report_path = safe_path(root, '.downloads/last_report.json')
            atomic_json(report_path, report)
            if downloader.package_bases:
                print('Package-scoped config bases require the installed Quality environment: ' +
                      ', '.join(sorted(downloader.package_bases)))
            print('Finished: {0} OK, {1} failed. Report: {2}'.format(len(results) - len(failed_ids), len(failed_ids), report_path))
            return 1 if failed_ids else 0
    except KeyboardInterrupt:
        print('\nCancelled. Partial downloads are retained for the next run.', file=sys.stderr)
        return 130
    except (DownloadError, MocapError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print('Download setup failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', line_buffering=True)
    sys.exit(main())
