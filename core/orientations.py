"""Anatomical orientations: +X left, -Y forward, +Z up; (w,x,y,z).

These are world-space anatomical frames, not bone-local rotations. No bpy or
numeric runtime dependencies; shared by capture, review and retargeting.
"""
from __future__ import annotations

from . import retarget_math as rm

NAMES = ('head', 'foot.L', 'foot.R')
VERSION = '0.3.1'


def body_basis(body):
    left = rm.vec_sub(body['shoulder.L'], body['shoulder.R'])
    up = rm.vec_sub(body['chest'], body['pelvis'])
    return rm.quat_from_matrix3(rm.basis_from_axes(up, 2, left, 0))


def foot_basis(forward, up=(0, 0, 1)):
    return rm.quat_from_matrix3(rm.basis_from_axes(rm.vec_neg(forward), 1, up, 2))


def mirror(q):
    # S R S, S=diag(-1,1,1): reflection of both world and anatomical basis.
    return (q[0], q[1], -q[2], -q[3])


def foot_template(side, length=.215):
    sign = 1 if side == 'L' else -1
    scale = length / .215
    return tuple(rm.vec_scale(p, scale) for p in (
        (0, 0, 0), (-sign * .025, -.16, -.045),
        (sign * .04, -.14, -.045), (0, .055, -.045)))


def apply_feet(frame):
    """Reconstruct toes/heels about the final ankle, without moving leg joints."""
    for side in ('L', 'R'):
        name = 'foot.' + side
        q = frame.get('orientations', {}).get(name)
        if q is None:
            continue
        info = frame.get('orientation_quality', {}).get(name, {})
        template = foot_template(side, info.get('length', .215))
        ankle = frame['body3d']['ankle.' + side]
        for joint, point in (('toe.', rm.vec_midpoint(template[1], template[2])), ('heel.', template[3])):
            frame['body3d'][joint + side] = rm.vec_add(ankle, rm.quat_rotate_vector(q, point))
            frame.setdefault('confidence', {})[joint + side] = min(.69, info.get('confidence', .3))


def stabilize(frames, max_gap=.2, strength=.65):
    """Bounded gaps, shortest-arc interpolation and centered angular denoising.

    Long gaps never borrow a future observation; after a short hold the pose
    returns gradually to the torso prior. Missing whole-body spans reset state.
    """
    if not frames or not any(f.get('orientations') for f in frames):
        return
    for name in NAMES:
        if not any(name in f.get('orientations', {}) for f in frames):
            continue
        valid = [i for i, f in enumerate(frames) if name in f.get('orientations', {})
                 and f.get('orientation_quality', {}).get(name, {}).get('confidence', 0) >= .4]
        left = None
        for i, f in enumerate(frames):
            info = f.setdefault('orientation_quality', {}).setdefault(name, {})
            if i and f['time'] - frames[i-1]['time'] > max_gap:
                left = None
            if i in valid:
                left = i
                continue
            right = next((j for j in valid if j > i), None)
            prior = body_basis(f['body3d'])
            if left is not None and right is not None and frames[right]['time'] - frames[left]['time'] <= max_gap:
                t = (f['time'] - frames[left]['time']) / (frames[right]['time'] - frames[left]['time'])
                q = rm.slerp(frames[left]['orientations'][name], frames[right]['orientations'][name], t)
                source = 'interpolated'
            elif left is not None:
                age = f['time'] - frames[left]['time']
                q = rm.slerp(frames[left]['orientations'][name], prior, rm.clamp((age-max_gap)/.3, 0, 1))
                source = 'held' if age <= max_gap else 'body_prior'
            else:
                q, source = prior, 'body_prior'
            f.setdefault('orientations', {})[name] = q
            info.update(confidence=.3, source=source, estimated=True)
        values = [f['orientations'][name] for f in frames]
        for i in range(1, len(frames)-1):
            if all(j in valid for j in (i-1, i, i+1)) and frames[i+1]['time']-frames[i-1]['time'] <= max_gap:
                # Centered filtering adds no phase delay. High-confidence sharp
                # motion is preserved; weak detections get more angular denoising.
                angle = rm.quat_angle(rm.quat_mul(rm.quat_conjugate(values[i-1]), values[i+1]))
                confidence = frames[i]['orientation_quality'][name].get('confidence', 0)
                if angle < .35 or (confidence < .7 and angle < 1.2):
                    midpoint = rm.slerp(values[i-1], values[i+1], .5)
                    frames[i]['orientations'][name] = rm.slerp(values[i], midpoint, strength * (.85 if confidence < .7 else .5))
        previous = None
        for f in frames:
            q = rm.quat_normalize(f['orientations'][name])
            if previous is not None and rm.quat_dot(previous, q) < 0:
                q = tuple(-v for v in q)
            f['orientations'][name] = q
            previous = q
    for f in frames:
        apply_feet(f)
