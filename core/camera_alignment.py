"""Constant camera-to-character calibration; no per-frame gait or yaw fitting.

Camera-standard coordinates are image right, optical depth, up. A camera at
the character's left (+X), front (-Y), at 45 degrees has world right vector
(sqrt(.5), sqrt(.5), 0), so its inverse extrinsic is a POSITIVE Z rotation.
The optional initial pelvis reference corrects only a small model yaw bias.
It never removes subsequent turns or estimates limb depth from a gait prior.
"""
from __future__ import annotations

import copy
import math
import statistics

from . import retarget_math as rm

VIEWS = ('unspecified', 'left_front_45')
REFERENCE_SECONDS = .25
MAX_HEADING_CORRECTION = math.radians(20)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def resolve(frames, view='unspecified', align_initial_facing=False):
    """Resolve ONE transform from the initial reliable pelvis observations.

Use hips rather than shoulders: independent torso twist is an actual motion.
Do not search arbitrarily far ahead through occlusion or a later turn.
"""
    if view not in VIEWS:
        raise ValueError('Unknown camera_view: ' + str(view))
    yaw = math.pi / 4 if view == 'left_front_45' else 0.
    correction = 0.
    status = 'disabled'
    headings = []
    if view != 'unspecified' and align_initial_facing:
        status = 'insufficient_initial_observations'
        start = frames[0]['time'] if frames else 0.
        for frame in frames:
            if frame['time'] - start > REFERENCE_SECONDS + 1e-6:
                break
            body, confidence = frame['body3d'], frame.get('confidence', {})
            if any(n not in body or confidence.get(n, confidence.get('body_mean', 0)) < .4
                   for n in ('hip.L', 'hip.R')):
                continue
            lateral = rm.vec_sub(body['hip.L'], body['hip.R'])
            horizontal = math.hypot(lateral[0], lateral[1])
            # A near-vertical pelvis axis cannot give a reliable yaw reference.
            if horizontal < .05 or horizontal < .7 * rm.vec_length(lateral):
                continue
            headings.append(_wrap(math.atan2(lateral[1], lateral[0]) + yaw))
        if headings:
            center = math.atan2(sum(math.sin(a) for a in headings), sum(math.cos(a) for a in headings))
            residual = _wrap(center + statistics.median(_wrap(a - center) for a in headings))
            spread = max(abs(_wrap(a - residual)) for a in headings)
            if spread > math.radians(10):
                status = 'initial_heading_unstable'
            elif abs(residual) > MAX_HEADING_CORRECTION:
                status = 'camera_prior_conflict'
            else:
                correction, status = -residual, 'aligned'
    return dict(camera_view=view, target_forward='-Y' if view != 'unspecified' else None,
                camera_to_world_yaw_degrees=math.degrees(yaw),
                initial_heading_correction_degrees=math.degrees(correction),
                applied_yaw_degrees=math.degrees(yaw + correction),
                quaternion_wxyz=list(rm.quat_from_axis_angle((0, 0, 1), yaw + correction)),
                initial_alignment=status, reference_seconds=REFERENCE_SECONDS,
                reference_samples=len(headings), constant_over_sequence=True,
                depth_refined=False, camera_elevation='unspecified')


def transform(frames, calibration):
    """Rotate positions, root travel, hands and world orientations together.

About the fixed world origin (not each frame's pelvis), so world trajectories
and contacts undergo the same rigid change of coordinates as the limbs.
"""
    output = copy.deepcopy(frames)
    if calibration['camera_view'] == 'unspecified':
        return output
    q = calibration['quaternion_wxyz']
    for frame in output:
        for key in ('body3d', 'hands3d'):
            frame[key] = {name: rm.quat_rotate_vector(q, point)
                          for name, point in frame.get(key, {}).items()}
        if frame.get('orientations'):
            frame['orientations'] = {name: rm.quat_normalize(rm.quat_mul(q, rotation))
                                     for name, rotation in frame['orientations'].items()}
    return output


def finalize(raw_frames, processed_frames, options, diagnostics):
    """Called after camera-space image fitting and temporal processing, once."""
    calibration = resolve(raw_frames, options.get('camera_view', 'unspecified'),
                          options.get('align_initial_facing', False))
    diagnostics['camera_alignment'] = calibration
    diagnostics['motion_metric_axes'] = 'camera_standard_before_alignment'
    return transform(raw_frames, calibration), transform(processed_frames, calibration), calibration
