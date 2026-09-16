"""Measure how well a captured 3D result reconciles with its own 2D evidence.

Diagnostic only: reads a cached capture (``mocap_result.json`` plus
``preview_manifest.json``) and reports, per joint,
  * the MotionBERT input magnitude implied by the bounding-box size,
  * the reprojection error of the lifted 3D pose under the calibrated camera,
  * how much of each limb's 3D excursion is depth (unobservable from one view).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

COCO_TO_BODY = {
    11: 'hip.L', 12: 'hip.R', 13: 'knee.L', 14: 'knee.R',
    15: 'ankle.L', 16: 'ankle.R', 5: 'shoulder.L', 6: 'shoulder.R',
    7: 'elbow.L', 8: 'elbow.R', 9: 'wrist.L', 10: 'wrist.R',
    0: 'head',
}
FOCAL_RATIO = 1.2


def quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load(folder):
    with open(os.path.join(folder, 'mocap_result.json'), encoding='utf-8') as handle:
        result = json.load(handle)
    with open(os.path.join(folder, 'preview_manifest.json'), encoding='utf-8') as handle:
        manifest = json.load(handle)
    return result, manifest


def main(folder):
    result, manifest = load(folder)
    width = manifest['source']['width']
    height = manifest['source']['height']
    focal = FOCAL_RATIO * max(width, height)
    q = result['capture_transform']['quaternion_wxyz']
    rotation = quat_matrix(np.asarray(q, dtype=float))
    frames = {f['frame']: f for f in result['frames']}
    rows = manifest['frames']

    print('source {0}x{1} fps={2} focal={3:.1f}px yaw={4:.2f}deg'.format(
        width, height, manifest['source'].get('fps'), focal,
        result['capture_transform']['applied_yaw_degrees']))

    # ---- 1. how much of the frame the subject occupies (MotionBERT input scale)
    ratios, heights = [], []
    for row in rows:
        points = row.get('body2d') or []
        if len(points) < 23:
            continue
        xy = np.asarray([[p[0] * width, p[1] * height] for p in points[:23]])
        top, bottom, left, right = xy[:, 1].min(), xy[:, 1].max(), xy[:, 0].min(), xy[:, 0].max()
        heights.append(bottom - top)
        ratios.append(max(bottom - top, right - left) / max(width, height))
    print('\n[1] subject scale')
    print('    bbox long side / max(image side): mean={0:.3f} median={1:.3f}'.format(
        float(np.mean(ratios)), float(np.median(ratios))))
    print('    person height: median={0:.1f}px = {1:.1f}% of frame height'.format(
        float(np.median(heights)), 100 * float(np.median(heights)) / height))
    print('    MotionBERT sees |x|,|y| ~ {0:.3f} after /w*2-1 (training scale ~1.0)'.format(
        float(np.median(ratios))))

    # ---- 2. reprojection error of the lifted 3D pose under the known camera
    errors = {}
    image_span = {}
    depth_span = {}
    for row in rows:
        frame = frames.get(row.get('result_frame'))
        points = row.get('body2d') or []
        if frame is None or len(points) < 23:
            continue
        body = frame['body3d']
        for index, name in COCO_TO_BODY.items():
            if name not in body:
                continue
            camera = rotation @ np.asarray(body[name], dtype=float)
            if camera[1] <= 0.05:
                continue
            u = focal * camera[0] / camera[1] + width / 2
            v = focal * camera[2] / camera[1] + height / 2
            px, py = points[index][0] * width, points[index][1] * height
            errors.setdefault(name, []).append(math.hypot(u - px, v - py))
    print('\n[2] reprojection error against detected 2D (pixels)')
    print('    joint            median    mean     n')
    for name in sorted(errors, key=lambda n: -float(np.median(errors[n]))):
        values = np.asarray(errors[name], dtype=float)
        print('    {0:<14} {1:7.1f} {2:7.1f} {3:5d}'.format(
            name, float(np.median(values)), float(values.mean()), len(values)))

    # ---- 3. limb excursion: how much lives in depth (invisible in one view)
    print('\n[3] per-limb excursion in camera space (metres)')
    print('    joint       image-plane   depth     depth share')
    for name in ('ankle.L', 'ankle.R', 'knee.L', 'knee.R', 'wrist.L', 'wrist.R'):
        track = np.asarray([frames[r['result_frame']]['body3d'][name] for r in rows
                            if r.get('result_frame') in frames
                            and name in frames[r['result_frame']]['body3d']], dtype=float)
        if len(track) < 3:
            continue
        camera = (rotation @ track.T).T
        lateral = camera[:, 0].max() - camera[:, 0].min()
        vertical = camera[:, 2].max() - camera[:, 2].min()
        depth = camera[:, 1].max() - camera[:, 1].min()
        plane = math.hypot(lateral, vertical)
        share = depth / max(1e-6, math.hypot(plane, depth))
        print('    {0:<11} {1:9.3f} {2:9.3f} {3:11.0%}'.format(name, plane, depth, share))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.cache/v03-camera')
