"""Analytic, labelled motion truth for processing tests and Blender rendering.

These fixtures are test inputs only; capture code never imports motion templates.
"""
import math
from backend_worker.mock_source import rest_pose, _rotate_about
from core import retarget_math as rm


def leg(body, side, target):
    hip = body['hip.' + side]
    upper, lower = .44, .45
    vector = rm.vec_sub(target, hip)
    distance = rm.vec_length(vector)
    assert abs(upper - lower) < distance < upper + lower
    axis = rm.vec_scale(vector, 1 / distance)
    forward = (0, -1, 0)
    plane = rm.vec_normalize(rm.vec_sub(forward, rm.vec_scale(axis, rm.vec_dot(forward, axis))))
    along = (upper ** 2 - lower ** 2 + distance ** 2) / (2 * distance)
    height = math.sqrt(upper ** 2 - along ** 2)
    body['knee.' + side] = rm.vec_add(hip, rm.vec_add(rm.vec_scale(axis, along), rm.vec_scale(plane, height)))
    body['ankle.' + side] = target
    body['toe.' + side] = rm.vec_add(target, (0, -.14, -.07))
    body['heel.' + side] = rm.vec_add(target, (0, .05, -.07))


def clip(kind, fps=30, seconds=4):
    frames = []
    for i in range(round(fps * seconds)):
        time = i / fps
        mode = ('walk' if time < 1.4 else 'attack' if time < 2.8 else 'idle') if kind == 'general' else kind
        period = .8 if mode == 'run' else 1.2
        phase = (time / period) % 1
        body = {name: (p[0], p[1], p[2] - .12) for name, p in rest_pose().items()}
        bob = (.035 if mode == 'run' else .015) * math.sin(2 * math.pi * phase) ** 2 if mode in ('walk', 'run') else 0
        body = {name: (p[0], p[1], p[2] + bob) for name, p in body.items()}
        for side, sign in (('L', 1), ('R', -1)):
            for name in ('elbow.' + side, 'wrist.' + side):
                body[name] = _rotate_about(body[name], body['shoulder.' + side], (0, 1, 0), sign * .9)
        contacts = {}
        for side, offset in (('L', 0), ('R', .5)):
            foot_phase = (phase + offset) % 1
            stance = .28 if mode == 'run' else .55
            contact = mode in ('idle', 'attack') or foot_phase <= stance
            swing = 0 if contact else (foot_phase - stance) / (1 - stance)
            lift = (.23 if mode == 'run' else .14) * math.sin(math.pi * swing) ** 2
            target = (body['hip.' + side][0], .16 * math.sin(2 * math.pi * swing), .08 + lift)
            if mode == 'attack' and side == 'R':
                kick = max(0, 1 - abs(time % 1.3 - .58) / .18)
                target = (target[0], -.3 * kick, .08 + .32 * kick)
                contact = kick == 0
            leg(body, side, target)
            contacts['foot.' + side] = contact
            angle = (.55 if mode == 'run' else .3) * math.sin(2 * math.pi * (phase + offset))
            if mode == 'attack':
                t = time % 1.3
                # Wind-up, a 100 ms strike, stop and slow recovery.
                angle = -.25 * min(1, t / .4) if t < .4 else -.25 + 1.65 * min(1, (t - .4) / .1) if t < .5 else 1.4 if t < .7 else 1.4 * max(0, 1 - (t - .7) / .6)
                if side == 'L':
                    angle *= .35
            if mode == 'idle':
                angle = .015 * math.sin(2 * math.pi * time / 2)
            pivot = body['shoulder.' + side]
            for name in ('elbow.' + side, 'wrist.' + side):
                body[name] = _rotate_about(body[name], pivot, (1, 0, 0), angle)
        if mode == 'attack':
            turn = .18 * math.sin(2 * math.pi * time / 1.3)
            for name, p in list(body.items()):
                if name in ('spine', 'chest', 'neck', 'head') or name.startswith(('shoulder.', 'elbow.', 'wrist.')):
                    body[name] = _rotate_about(p, body['pelvis'], (0, 0, 1), turn)
        breathing = .008 * math.sin(math.pi * time)
        if mode == 'idle':
            for name, p in list(body.items()):
                if name in ('spine', 'chest', 'neck', 'head') or name.startswith(('shoulder.', 'elbow.', 'wrist.')):
                    body[name] = (p[0], p[1], p[2] + breathing)
        frames.append(dict(frame=i + 1, time=time, body3d=body, hands3d={},
                           confidence={name: .95 for name in body}, contacts=contacts))
    return frames
