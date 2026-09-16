"""Replay the reprojection diagnostic over a cached capture, with no models.

    .venv/Scripts/python.exe tools/validate_reprojection.py .cache/v03-camera

Reports the fitted camera, the per-joint reprojection residual and how much of
each joint's travel lies along the unobservable camera axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default=str(ROOT / '.cache/v03-camera'))
    parser.add_argument('--result', default='mocap_result.json')
    parser.add_argument('--no-sweep', action='store_true',
                        help='fit only near the recorded azimuth')
    args = parser.parse_args()

    from backend_worker import reprojection
    folder = Path(args.folder)
    with open(folder / args.result, encoding='utf-8') as handle:
        result = json.load(handle)
    with open(folder / 'preview_manifest.json', encoding='utf-8') as handle:
        manifest = json.load(handle)
    image_size = (manifest['source']['width'], manifest['source']['height'])
    recorded = (result.get('capture_transform') or {}).get('applied_yaw_degrees')

    diagnostic = reprojection.diagnose(
        result['frames'], manifest['frames'], image_size,
        {'camera_azimuth_degrees': recorded, 'sweep_azimuth': not args.no_sweep})
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    warnings = reprojection.findings(diagnostic)
    print('\nfindings:')
    for warning in warnings:
        print('  [{0}] {1}'.format(warning['code'], warning['message']))
    if not warnings:
        print('  (none)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
