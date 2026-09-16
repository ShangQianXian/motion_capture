"""Measure MotionBERT lift accuracy against known 3D, with a synthetic camera.

The benchmark owns the whole forward problem: it takes an analytic 3D clip,
projects it through a known camera, adds 2D noise, and only then calls the real
lifter. That makes every input-distribution choice (image size, subject size in
frame, bounding box) an explicit parameter, which is exactly the dimension the
input-normalisation fix changes.

Ground truth never enters the lift; it is used only to score the result.

Usage::

    .venv/Scripts/python.exe tools/benchmark_lifter.py --windows 2

Requires the OpenMMLab stack and the downloaded MotionBERT weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import skeleton  # noqa: E402
from tests.fixtures import motions  # noqa: E402

#: H36M-17 index -> standard skeleton joint, matching skeleton.H36M_17_MAP.
H36M_ORDER = tuple(skeleton.H36M_17_MAP[index] for index in range(17))

#: H36M-17 head (index 10) is the midpoint between the eyes. The standard
#: skeleton has no eye joint, so the head is excluded from every score rather
#: than scored against an unmatched reference.
SCORED_JOINTS = tuple(index for index in range(17) if index != 10)

#: Bones scored for length consistency, by H36M index.
SCORED_BONES = (
    (1, 4), (2, 5), (3, 6),          # hip/knee/ankle pairs
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (11, 14), (11, 12), (12, 13), (14, 15), (15, 16),
    (0, 8), (8, 11), (8, 14),
)

#: H36M statistics shipped with MMPose (``_base_/datasets/h36m.py``).
H36M_BBOX_CENTER = (528.0, 427.0)
H36M_BBOX_SCALE = 400.0
H36M_IMAGE_SIZE = (1000.0, 1000.0)


def h36m_points(body: dict) -> np.ndarray:
    """Stack a standard-skeleton frame into the H36M-17 order."""
    return np.asarray([body[name] for name in H36M_ORDER], dtype=float)


def camera_axes(azimuth_degrees: float):
    """Camera basis consistent with the addon's calibration.

    A camera at ``azimuth`` sees screen right along ``(cos, sin, 0)``; its
    viewing direction (into the screen) is ``(-sin, cos, 0)``.
    """
    angle = math.radians(azimuth_degrees)
    right = np.array([math.cos(angle), math.sin(angle), 0.0])
    depth = np.array([-math.sin(angle), math.cos(angle), 0.0])
    return right, depth


def build_sequence(kind: str, frames: int, fps: float, image_size, span_pixels: float,
                   azimuth: float, noise: float, seed: int):
    """Return ``(keypoints_2d, keypoints_3d, bboxes, scale)`` for one clip."""
    clip = motions.clip(kind, fps=fps, seconds=frames / fps)
    truth = np.stack([h36m_points(frame['body3d']) for frame in clip])
    right, _ = camera_axes(azimuth)
    world_span = float(max(np.ptp(truth.reshape(-1, 3), axis=0)))
    scale = span_pixels / max(world_span, 1e-6)
    center = np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    pixels = np.stack([truth @ right * scale + center[0],
                       -(truth[..., 2] * scale) + center[1]], axis=-1)
    generator = np.random.default_rng(seed)
    if noise > 0:
        pixels = pixels + generator.normal(0.0, noise, pixels.shape)
    bboxes = []
    for points in pixels:
        low, high = points.min(axis=0), points.max(axis=0)
        bboxes.append([float(low[0]), float(low[1]), float(high[0]), float(high[1])])
    return pixels.astype(np.float32), truth, bboxes, scale


def similarity_align(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Scale/rotation/translation alignment, so only pose shape is scored."""
    a = predicted - predicted.mean(axis=0)
    b = truth - truth.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    denominator = float(np.sum(np.linalg.norm(a, axis=1) ** 2))
    scale = float(np.sum((a @ rotation * b).sum())) / max(denominator, 1e-9)
    return (a @ rotation) * scale


def score(predicted: np.ndarray, truth: np.ndarray, azimuth: float) -> dict:
    """Aligned MPJPE, bone-length error and camera-depth error for one frame."""
    aligned_truth = truth - truth.mean(axis=0)
    aligned = similarity_align(predicted, aligned_truth)
    difference = np.linalg.norm(aligned[list(SCORED_JOINTS)] - aligned_truth[list(SCORED_JOINTS)], axis=1)
    worst_bone = 0.0
    for a, b in SCORED_BONES:
        reference = float(np.linalg.norm(truth[a] - truth[b]))
        measured = float(np.linalg.norm(predicted[a] - predicted[b]))
        if reference > 1e-3:
            worst_bone = max(worst_bone, abs(measured - reference) / reference)
    _, depth_axis = camera_axes(azimuth)
    truth_depth = (truth @ depth_axis)
    predicted_depth = (predicted @ depth_axis)
    truth_depth = truth_depth - truth_depth.mean()
    predicted_depth = predicted_depth - predicted_depth.mean()
    # The depth sign is unrecoverable from one view, so compare magnitudes.
    if float(np.dot(predicted_depth, truth_depth)) < 0:
        predicted_depth = -predicted_depth
    return {
        'mpjpe_aligned_m': float(np.mean(difference)),
        'bone_length_error': worst_bone,
        'depth_mae_m': float(np.mean(np.abs(predicted_depth - truth_depth))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', default='walk', choices=('walk', 'run', 'attack', 'idle'))
    parser.add_argument('--windows', type=int, default=2, help='number of 243-frame windows')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--noise', type=float, default=3.0, help='2D gaussian noise, pixels')
    parser.add_argument('--azimuth', type=float, default=45.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--models-root', default=str(ROOT / 'models'))
    parser.add_argument('--output', default=str(ROOT / '.cache' / 'lifter-benchmark.json'))
    parser.add_argument('--variants', default='current,canonical')
    parser.add_argument('--scenes', default='all',
                        help="'all' or a comma list of: realistic,large,small")
    args = parser.parse_args()

    catalog = {
        # label: (image size, subject bounding-box span in pixels)
        'realistic': ('subject 57% of frame (this capture)', (1344.0, 768.0), 573.0),
        'hd': ('subject 46% of a 1080p frame', (1920.0, 1080.0), 500.0),
        'hd_small': ('subject 23% of a 1080p frame', (1920.0, 1080.0), 250.0),
        'large': ('subject fills frame (H36M-like)', H36M_IMAGE_SIZE, 900.0),
        'small': ('subject 25% of frame (small in shot)', (1920.0, 1080.0), 270.0),
    }
    names = list(catalog) if args.scenes == 'all' else args.scenes.split(',')
    scenes = [(catalog[name][0], catalog[name][1], catalog[name][2]) for name in names]

    frames = 243 * max(1, args.windows)
    report = {'cases': [], 'input': vars(args), 'scenes': [scene[0] for scene in scenes]}

    from backend_worker.pose3d_motionbert import Body3DLifter
    lifter = Body3DLifter(args.models_root, args.device, None, None)

    for label, image_size, span in scenes:
        pixels, truth, bboxes, projection_scale = build_sequence(
            args.kind, frames, args.fps, image_size, span, args.azimuth, args.noise, 20260915)
        scores = [np.ones(17, dtype=np.float32)] * len(pixels)
        for variant in args.variants.split(','):
            started = time.time()
            lifted = lifter.lift(list(pixels), scores, image_size, bboxes_seq=list(bboxes),
                                 sample_fps=args.fps, input_normalisation=variant)
            elapsed = time.time() - started
            # Fitted exactly like production does: one median pelvis-to-head
            # scalar for the whole clip.
            distance = float(np.median([np.linalg.norm(lifted[i][0] - lifted[i][10])
                                        for i in range(len(lifted))]))
            metric = 0.78 / distance if distance > 1e-6 else 1.0
            lifted = np.asarray(lifted, dtype=float) * metric
            margin = 121
            window = slice(margin, len(lifted) - margin)
            scores_list = [score(lifted[i], truth[i], args.azimuth) for i in range(*window.indices(len(lifted)))]
            entry = {
                'scene': label,
                'image_size': list(image_size),
                'subject_pixels': span,
                'variant': variant,
                'normalised_input_span': round(2.0 * span / image_size[0], 3),
                'mpjpe_aligned_m': round(float(np.mean([s['mpjpe_aligned_m'] for s in scores_list])), 4),
                'bone_length_error': round(float(np.max([s['bone_length_error'] for s in scores_list])), 4),
                'depth_mae_m': round(float(np.mean([s['depth_mae_m'] for s in scores_list])), 4),
                'seconds': round(elapsed, 1),
            }
            report['cases'].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nwritten', output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
