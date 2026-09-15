"""Observed head and foot frames. Numeric packages are imported only on use."""
from __future__ import annotations

import math
from statistics import median
from ._core import orientations as ori, retarget_math as rm, errors

# COCO WholeBody face-0 starts at 23; nose, chin, outer eyes, mouth corners.
FACE_IDS = (30, 8, 36, 45, 48, 54)
# Generic rigid face reference, metres, in the documented anatomical frame.
FACE_MODEL = ((0, -.09, .02), (0, -.01, -.11), (-.045, -.035, .05),
              (.045, -.035, .05), (-.035, -.045, -.045), (.035, -.045, -.045))
CAMERA_TO_STANDARD = ((1., 0., 0.), (0., 0., 1.), (0., -1., 0.))


def solve_head(points, image_size):
    import cv2
    import numpy as np
    if len(points) < 91:
        return None, {'source': 'face_pnp', 'confidence': 0., 'estimated': True}
    observed = np.asarray([points[23+i] for i in FACE_IDS], dtype=float)
    confidence = float(observed[:, 2].min())
    if confidence < .4:
        return None, {'source': 'face_pnp', 'confidence': confidence, 'estimated': True}
    width, height = image_size
    pixels = observed[:, :2] * (width, height)
    if not np.isfinite(pixels).all() or np.linalg.norm(pixels[2]-pixels[3]) < 4:
        return None, dict(source='face_pnp', confidence=0., estimated=True)
    focal = max(width, height)
    camera = np.array(((focal, 0, width/2), (0, focal, height/2), (0, 0, 1.)), dtype=float)
    model = np.asarray(FACE_MODEL, dtype=float)
    # SQPnP accepts the six noncoplanar reference points without a pose guess.
    info = dict(source='face_pnp', confidence=0., estimated=True, camera='estimated_center_max_focal')
    try:
        ok, rvec, tvec = cv2.solvePnP(model, pixels, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None, info
    if not ok:
        return None, info
    try:
        rvec, tvec = cv2.solvePnPRefineLM(model, pixels, camera, None, rvec, tvec)
    except cv2.error:
        return None, info
    rotation = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(model, rvec, tvec, camera, None)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))
    span = max(float(np.linalg.norm(pixels[2]-pixels[3])), 1.)
    info.update(reprojection_px=error, relative_error=error/span)
    if error/span > .18 or np.min((rotation @ model.T + tvec)[2]) <= 0:
        return None, info
    q = rm.quat_from_matrix3(np.asarray(CAMERA_TO_STANDARD) @ rotation)
    info.update(confidence=confidence, estimated=False)
    return q, info


def image_scale(body, points, image_size):
    """Robust image-plane pixels/metre; not shoulder-width-only (side-view singular)."""
    pairs = ((5, 6, 'shoulder.L', 'shoulder.R'), (11, 12, 'hip.L', 'hip.R'),
             (5, 11, 'shoulder.L', 'hip.L'), (6, 12, 'shoulder.R', 'hip.R'))
    values = []
    for a, b, ja, jb in pairs:
        if min(points[a][2], points[b][2]) < .5:
            continue
        delta = rm.vec_sub(body[ja], body[jb])
        distance = math.hypot(delta[0], delta[2])
        pixels = math.hypot((points[a][0]-points[b][0])*image_size[0],
                            (points[a][1]-points[b][1])*image_size[1])
        if distance > .08 and pixels > 8:
            values.append(pixels/distance)
    return median(values) if values else None


def solve_foot(points, side, image_size, scale, length, prior):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    ids = (15, 17, 18, 19) if side == 'L' else (16, 20, 21, 22)
    info = dict(source='wholebody_foot_fit', confidence=0., estimated=True, length=length)
    if len(points) < 23 or scale is None:
        return None, info
    obs = np.asarray([points[i] for i in ids], dtype=float)
    if min(obs[:, 2]) < .4:
        return None, info
    pixels = obs[:, :2] * image_size
    pixels -= pixels[0].copy()
    model = np.asarray(ori.foot_template(side, length))
    weight = np.sqrt(obs[:, 2:3])
    prior_matrix = np.asarray(rm.matrix3_from_quat(prior))
    def residual(vector):
        rotation = Rotation.from_rotvec(vector).as_matrix()
        rotated = model @ rotation.T
        reprojection = (rotated[:, (0, 2)] * (scale, -scale) - pixels) * weight
        # A weak angular prior selects the continuous depth branch, without
        # overriding image-plane evidence or forcing forward to a world sign.
        continuity = (rotation-prior_matrix).ravel() * .35
        return np.concatenate((reprojection.ravel(), continuity))
    seeds = [prior_matrix]
    # Include a pitched seed: the ankle-to-sole offset resolves many mirror fits.
    seeds.extend(prior_matrix @ Rotation.from_euler('x', a).as_matrix() for a in (-.7, .7))
    solutions = [least_squares(residual, Rotation.from_matrix(seed).as_rotvec(),
                               max_nfev=45, loss='soft_l1', f_scale=2.) for seed in seeds]
    solution = min(solutions, key=lambda s: float(np.sum(residual(s.x)**2)))
    rotation = Rotation.from_rotvec(solution.x).as_matrix()
    rotated = model @ rotation.T
    error = float(np.sqrt(np.mean(np.sum((rotated[:, (0,2)]*(scale,-scale)-pixels)**2, axis=1))))
    span = max(float(np.linalg.norm((pixels[1]+pixels[2])/2-pixels[3])), 8.)
    info.update(reprojection_px=error, relative_error=error/span)
    if error/span > .3:
        return None, info
    info.update(confidence=float(obs[:, 2].min()), estimated=False)
    return rm.quat_from_matrix3(rotation), info


def enrich(frames, rows, image_size, profile, cancel_token=None):
    """Add independent rotations before motion processing; retain raw evidence."""
    by_frame = {row['sample_frame']: row for row in rows}
    quality = profile in ('quality', 'quality_plus', 'quality_feet')
    scales = [image_scale(f['body3d'], by_frame[f['frame']]['body2d'], image_size)
              for f in frames] if quality else []
    scale = median([s for s in scales if s is not None]) if any(s is not None for s in scales) else None
    # Fix physical proportions once per sequence, not once per noisy frame.
    legs = [rm.vec_distance(f['body3d']['hip.L'], f['body3d']['knee.L'])
            + rm.vec_distance(f['body3d']['knee.L'], f['body3d']['ankle.L']) for f in frames]
    length = rm.clamp(median(legs)*.27, .14, .3) if legs else .215
    previous, seen, previous_time = {}, {}, None
    for f in frames:
        if cancel_token is not None and cancel_token.cancelled():
            raise errors.MocapError(errors.CANCELLED, '头部与脚部姿态求解已取消。')
        body = f['body3d']
        row = by_frame[f['frame']]
        points = row['body2d']
        prior = ori.body_basis(body)
        if previous_time is not None and f['time']-previous_time > .2:
            previous.clear()
            seen.clear()
        previous_time = f['time']
        rotations, infos = {}, {}
        if quality:
            head, info = solve_head(points, image_size)
        else:
            # MediaPipe world face geometry: ears define lateral, ear centre to
            # nose defines forward; pose landmarks remain approximate depth.
            head, info = None, dict(source='mediapipe_face', confidence=0., estimated=True)
            face = f.pop('_face_world', {})
            if len(points) > 8 and all(k in face for k in ('nose', 'left_ear', 'right_ear')):
                lateral = rm.vec_sub(face['left_ear'], face['right_ear'])
                forward = rm.vec_sub(face['nose'], rm.vec_midpoint(face['left_ear'], face['right_ear']))
                if min(points[i][2] for i in (0,7,8)) >= .4 and rm.vec_length(forward) > .01:
                    head = rm.quat_from_matrix3(rm.basis_from_axes(lateral, 0, rm.vec_neg(forward), 1))
                    info.update(confidence=min(.69, *(points[i][2] for i in (0,7,8))), estimated=True)
        rotations['head'], infos['head'] = head or prior, info
        if quality:
            # H36M output 9 is a training-layout proxy, not an anatomical neck.
            # Locate head centre behind/below the eye proxy, then neck above shoulders.
            body['head'] = rm.vec_sub(body['head'], rm.quat_rotate_vector(head or prior, (0, -.02, .035)))
            shoulders = rm.vec_midpoint(body['shoulder.L'], body['shoulder.R'])
            body['neck'] = rm.vec_lerp(shoulders, body['head'], .25)
        for side in ('L', 'R'):
            name = 'foot.'+side
            if name in seen and f['time']-seen[name] > .2:
                previous.pop(name, None)
            if quality:
                foot, info = solve_foot(points, side, image_size, scale, length, previous.get(name, prior))
            else:
                ankle, toe, heel = (body.get(j+'.'+side) for j in ('ankle','toe','heel'))
                score = min(f['confidence'].get(j+'.'+side, 0) for j in ('ankle','toe','heel'))
                foot, info = None, dict(source='mediapipe_world', confidence=min(.69, score), estimated=True, length=length)
                if ankle and toe and heel and score >= .4:
                    forward = rm.vec_sub(toe, heel)
                    up = rm.vec_sub(ankle, rm.vec_midpoint(toe, heel))
                    foot = ori.foot_basis(forward, up)
            rotations[name], infos[name] = foot or previous.get(name, prior), info
            if foot is not None:
                previous[name] = foot
                seen[name] = f['time']
        f['orientations'], f['orientation_quality'] = rotations, infos
        row['orientation_quality'] = infos.copy()
        ori.apply_feet(f)
