"""Reprojection diagnostic: how well does the 3D result reconcile with the 2D evidence?

The capture pipeline had no stage that ever compared its own output against the
detections it was built from. This module adds that comparison, three things:

* fits the camera the capture was actually shot with (azimuth is swept, so a
  wrong ``camera_view`` setting shows up as a large residual instead of hiding),
* reports per-joint reprojection error, which is the accuracy ceiling for the
  whole pipeline: nothing downstream can be more accurate than the 3D pose's
  agreement with the pixels,
* reports how much of each limb's excursion lies along the camera axis. Depth
  along that axis is unobservable from one view, so a large share there means
  the visible lateral placement of that joint came from the model's prior, not
  from the video. Surfacing it is the honest alternative to inventing motion.

Standard library plus numpy/scipy, which the worker already requires.
"""

from __future__ import annotations

import math

#: COCO-17 index -> standard skeleton joint, matching the H36M-17 derivation in
#: ``pose2d_mmpose.coco17_to_h36m17``.
COCO_TO_JOINT = {
    0: 'head', 12: 'hip.R', 14: 'knee.R', 16: 'ankle.R',
    11: 'hip.L', 13: 'knee.L', 15: 'ankle.L',
    5: 'shoulder.L', 6: 'shoulder.R',
    7: 'elbow.L', 8: 'elbow.R', 9: 'wrist.L', 10: 'wrist.R',
}

#: ``(COCO index, 3D joint)`` pairs used to anchor the camera fit.
#: The pelvis is derived from the two hips, so it is added separately.
PELVIS_HIPS = ((11, 12),)

#: Azimuths probed when the recorded camera view is not trustworthy.
AZIMUTH_SWEEP = tuple(range(0, 180, 5))

#: Residual above which the fitted camera disagrees with the recorded one.
CAMERA_CONFLICT_PX = 12.0

#: Fraction of a limb's excursion along the camera axis above which its
#: in-plane placement is mostly prior rather than observation.
DEPTH_DOMINANT_SHARE = 0.8

#: Minimum excursion, in metres, before depth dominance is worth reporting.
DEPTH_DOMINANT_TRAVEL = 0.10


def camera_basis(azimuth_degrees: float):
    """Screen-right and into-screen axes, matching the addon's calibration.

    A camera at ``azimuth`` looking at the origin sees screen right along
    ``(cos, sin, 0)``; the direction into the screen is ``(-sin, cos, 0)``.
    A character facing ``-Y`` with the camera at ``+X, -Y`` gives azimuth 45,
    which is what ``camera_view='left_front_45'`` records.
    """
    angle = math.radians(azimuth_degrees)
    return ((math.cos(angle), math.sin(angle), 0.0),
            (-math.sin(angle), math.cos(angle), 0.0))


def _dot(vector, axis) -> float:
    return vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2]


def _observations(frames, rows, image_size, minimum_confidence=0.25):
    """Per-frame matched pairs of 3D joints and detected pixels.

    Returns ``(entries, width, height)`` where each entry is
    ``(frame, {joint: (pixel_x, pixel_y, confidence)})``. Frames without a 2D
    row or without a usable pelvis anchor are skipped, never guessed.
    """
    width, height = float(image_size[0]), float(image_size[1])
    by_frame = {frame['frame']: frame for frame in frames}
    entries = []
    for row in rows:
        frame = by_frame.get(row.get('result_frame'))
        points = row.get('body2d') or []
        pelvis = row.get('pelvis2d')
        if frame is None or len(points) < 23 or not pelvis:
            continue
        body = frame.get('body3d') or {}
        observed = {}
        for index, name in COCO_TO_JOINT.items():
            if name not in body:
                continue
            x, y, confidence = (float(value) for value in points[index][:3])
            if not (math.isfinite(x) and math.isfinite(y)) or confidence < minimum_confidence:
                continue
            observed[name] = (x * width, y * height, confidence)
        if len(observed) < 4 or 'pelvis' not in body:
            continue
        entries.append((frame, observed, (float(pelvis[0]) * width, float(pelvis[1]) * height)))
    return entries, width, height


def _depth_axis(lateral):
    # The lateral axis is (cos, sin, 0); the depth axis is that rotated by 90
    # degrees, i.e. (-sin, cos, 0).
    return (-lateral[1], lateral[0], 0.0)


def _predict_uv(joint, pelvis, anchor, lateral, focal, depth_offset):
    """Project one joint to pixels, anchored on the observed pelvis.

    A joint one metre above the pelvis stays a metre above it on screen; only
    the horizontal component is divided by depth, and that division is the sole
    reason depth is observable from a single view.
    """
    delta_lateral = _dot(joint, lateral) - _dot(pelvis, lateral)
    delta_depth = _dot(joint, _depth_axis(lateral)) - _dot(pelvis, _depth_axis(lateral))
    depth = depth_offset + delta_depth
    if depth <= 0.05:
        return None
    scale = focal / depth
    return (anchor[0] + scale * delta_lateral, anchor[1] - scale * (joint[2] - pelvis[2]))


def _errors(entries, lateral, focal, depth_offset):
    """Per-joint pixel residuals for one camera candidate."""
    collected = []
    for frame, observed, anchor in entries:
        pelvis = frame['body3d']['pelvis']
        for name, (x, y, _confidence) in observed.items():
            projected = _predict_uv(frame['body3d'][name], pelvis, anchor,
                                    lateral, focal, depth_offset)
            if projected is None:
                continue
            collected.append((name, math.hypot(projected[0] - x, projected[1] - y)))
    return collected


def _median_residual(entries, lateral, focal, depth_offset):
    values = sorted(error for _name, error in _errors(entries, lateral, focal, depth_offset))
    if not values:
        return float('inf'), 0
    return values[len(values) // 2], len(values)


def fit_camera(entries, image_size, azimuth_hint=None, sweep=True):
    """Fit ``(azimuth, focal, depth)`` by minimising median reprojection error.

    Azimuth is searched because a wrong ``camera_view`` setting must not be able
    to hide behind a good-looking residual. The cost is nearly flat in focal
    length, so each candidate starts from a coarse grid instead of a single
    guess; the depth offset is swept too, for the same reason.
    """
    from scipy.optimize import minimize_scalar
    longest_side = float(max(image_size))
    candidates = set()
    if azimuth_hint is not None:
        candidates.update((float(azimuth_hint) - 2.0, float(azimuth_hint),
                           float(azimuth_hint) + 2.0))
    if sweep or azimuth_hint is None:
        candidates.update(float(value) for value in AZIMUTH_SWEEP)
    best = None
    for azimuth in sorted(candidates):
        lateral, _ = camera_basis(azimuth)

        def median_for(focal, offset):
            return _median_residual(entries, lateral, focal, offset)[0]

        coarse = None
        for ratio in (0.4, 0.8, 1.2, 1.6, 2.4, 3.5):
            for offset in (1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 14.0):
                value = median_for(ratio * longest_side, offset)
                if coarse is None or value < coarse[0]:
                    coarse = (value, ratio * longest_side, offset)
        focal = float(minimize_scalar(
            lambda value: median_for(value, coarse[2]),
            bounds=(max(20.0, coarse[1] * 0.5), coarse[1] * 2.0), method='bounded').x)
        offset = float(minimize_scalar(
            lambda value: median_for(focal, value),
            bounds=(max(0.3, coarse[2] * 0.4), max(1.0, coarse[2] * 2.5)),
            method='bounded').x)
        median, count = _median_residual(entries, lateral, focal, offset)
        if best is None or median < best['median_px']:
            best = dict(azimuth=azimuth, focal=focal, depth_offset=offset,
                        median_px=median, samples=count, lateral=lateral)
    return best


def diagnose(frames, rows, image_size, options=None):
    """Full reprojection report for one capture. Never raises on good input."""
    options = options or {}
    entries, width, height = _observations(frames, rows, image_size)
    if len(entries) < 3:
        return {'status': 'insufficient_observations', 'frames': len(entries)}
    hint = options.get('camera_azimuth_degrees')
    fit = fit_camera(entries, image_size, azimuth_hint=hint, sweep=options.get('sweep_azimuth', True))
    lateral = fit['lateral']
    depth_axis = _depth_axis(lateral)
    focal, offset = fit['focal'], fit['depth_offset']

    per_joint = {}
    per_frame = []
    for frame, observed, anchor in entries:
        pelvis = frame['body3d']['pelvis']
        worst = 0.0
        for name, (x, y, _confidence) in observed.items():
            projected = _predict_uv(frame['body3d'][name], pelvis, anchor, lateral, focal, offset)
            if projected is None:
                continue
            error = math.hypot(projected[0] - x, projected[1] - y)
            per_joint.setdefault(name, []).append(error)
            worst = max(worst, error)
        per_frame.append({'frame': frame['frame'], 'max_error_px': round(worst, 2)})

    summary = {}
    for name, values in per_joint.items():
        values.sort()
        summary[name] = {
            'median_px': round(values[len(values) // 2], 2),
            'p90_px': round(values[min(len(values) - 1, int(len(values) * 0.9))], 2),
            'samples': len(values),
        }

    travel = {}
    body_names = sorted({name for frame, observed, _ in entries for name in observed})
    for name in body_names:
        track = [frame['body3d'][name] for frame, _observed, _anchor in entries if name in frame['body3d']]
        if len(track) < 3:
            continue
        along = [_dot(point, depth_axis) for point in track]
        plane = [_dot(point, lateral) for point in track]
        heights = [point[2] for point in track]
        depth_travel = max(along) - min(along)
        plane_travel = math.hypot(max(plane) - min(plane), max(heights) - min(heights))
        total = math.hypot(plane_travel, depth_travel)
        travel[name] = {
            'depth_m': round(depth_travel, 4),
            'image_plane_m': round(plane_travel, 4),
            'depth_share': round(depth_travel / total, 3) if total > 1e-6 else 0.0,
        }

    dominant = sorted((name for name, value in travel.items()
                       if value['depth_share'] >= DEPTH_DOMINANT_SHARE
                       and value['depth_m'] >= DEPTH_DOMINANT_TRAVEL))
    median_all = sorted(error for values in per_joint.values() for error in values)
    return {
        'status': 'ok',
        'frames': len(entries),
        'image_size': [width, height],
        'camera': {
            'azimuth_degrees': round(fit['azimuth'], 2),
            'recorded_azimuth_degrees': (round(float(hint), 2) if hint is not None else None),
            'focal_px': round(focal, 1),
            'depth_offset_m': round(offset, 3),
            'median_reprojection_px': round(fit['median_px'], 2),
            'samples': fit['samples'],
        },
        'reprojection': {
            'median_px': round(median_all[len(median_all) // 2], 2),
            'p90_px': round(median_all[min(len(median_all) - 1, int(len(median_all) * 0.9))], 2),
            'per_joint': summary,
        },
        'depth_dominance': travel,
        'depth_dominated_joints': dominant,
        'camera_conflict': bool(hint is not None and abs(fit['azimuth'] - float(hint)) > 15.0),
    }


def findings(diagnostic):
    """Warning payloads for a diagnostic, so the pipeline can surface them.

    Only real, measured conditions are reported; a healthy capture returns none.
    A reference implementation of the same shape as the 3D result would agree
    with the pixels to within roughly the detector's own jitter (a few pixels),
    so the thresholds below describe disagreement that cannot be blamed on noise.
    """
    warnings = []
    if diagnostic.get('status') != 'ok':
        return warnings
    camera = diagnostic['camera']
    if diagnostic.get('camera_conflict'):
        warnings.append({
            'code': 'CAMERA_AZIMUTH_CONFLICT',
            'message': '素材相机视角与结果记录不一致（拟合 {0:.0f}°，记录 {1:.0f}°），'
                       '请核对“素材相机视角”后重新生成。'.format(
                           camera['azimuth_degrees'], camera['recorded_azimuth_degrees']),
        })
    median = diagnostic['reprojection']['median_px']
    if median > 25.0:
        warnings.append({
            'code': 'REPROJECTION_HIGH',
            'message': '三维结果与二维观测中位差 {0:.1f} 像素，明显高于检测抖动，'
                       '该结果不能代表素材动作。'.format(median),
        })
    dominant = diagnostic.get('depth_dominated_joints') or []
    if dominant:
        warnings.append({
            'code': 'DEPTH_DOMINANT_MOTION',
            'message': '{0} 的位移主要在相机光轴方向（单目不可观测），'
                       '其横向位置来自模型先验，请结合正面／背面核对。'.format('、'.join(dominant)),
        })
    return warnings
