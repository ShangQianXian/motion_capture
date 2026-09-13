"""Offline, evidence-driven motion processing. No motion templates or bpy dependencies."""
from __future__ import annotations

import copy
import math
from statistics import median
from . import retarget_math as rm, skeleton, smoothing

VERSION = '0.3'
PRESETS = {'general': .10, 'walk': .15, 'run': .10, 'attack': .067, 'idle': .20}
LABELS = {'general': '通用／混合', 'walk': '走路', 'run': '跑步', 'attack': '攻击', 'idle': '待机'}
LIMBS = tuple((a + '.' + side, b + '.' + side) for side in ('L', 'R')
              for a, b in (('hip', 'knee'), ('knee', 'ankle'), ('shoulder', 'elbow'), ('elbow', 'wrist')))


def confidence(frame, name):
    return float(frame.get('confidence', {}).get(name, frame.get('confidence', {}).get('body_mean', 1)))


def fill_timeline(frames, rows, max_gap):
    """Insert only short bounded gaps; long missing intervals remain absent in 3D."""
    lookup = {f['frame']: copy.deepcopy(f) for f in frames}
    valid = sorted(frames, key=lambda f: f['time'])
    cursor = 0
    for row in rows:
        number, timestamp = row['sample_frame'], row['time']
        if number in lookup:
            continue
        while cursor < len(valid) and valid[cursor]['time'] < timestamp:
            cursor += 1
        if cursor == 0 or cursor == len(valid):
            row['unreliable'] = True
            continue
        a, b = valid[cursor - 1], valid[cursor]
        # Duration of the missing samples, excluding the endpoint interval.
        dt = (b['time'] - a['time']) / max(1, b['frame'] - a['frame'])
        if b['time'] - a['time'] - dt > max_gap + 1e-7:
            row['unreliable'] = True
            continue
        fraction = (timestamp - a['time']) / (b['time'] - a['time'])
        frame = dict(frame=number, time=timestamp, confidence={}, contacts={}, interpolated_joints=[])
        for key in ('body3d', 'hands3d'):
            frame[key] = {name: rm.vec_lerp(point, b[key][name], fraction)
                          for name, point in a.get(key, {}).items() if name in b.get(key, {})}
            frame['interpolated_joints'].extend(frame[key])
            frame['confidence'].update({name: min(confidence(a, name), confidence(b, name), .39) for name in frame[key]})
        if all(name in frame['body3d'] for name in skeleton.BODY_JOINTS):
            lookup[number] = frame
    return sorted(lookup.values(), key=lambda frame: frame['time'])


def _solve(matrix, values):
    rows = [list(a) + [b] for a, b in zip(matrix, values)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        if abs(rows[i][i]) < 1e-12:
            return None
        scale = rows[i][i]
        rows[i] = [x / scale for x in rows[i]]
        for j in range(3):
            if j != i:
                factor = rows[j][i]
                rows[j] = [a - factor * b for a, b in zip(rows[j], rows[i])]
    return [row[-1] for row in rows]


def symmetric_track(track, times, window, strength, reliable=None):
    """Centered weighted quadratic fit; keep endpoints, sharp impulses and gaps."""
    if len(track) < 5 or strength <= 0:
        return list(track)
    typical = median(b - a for a, b in zip(times, times[1:]) if b > a)
    radius = max(2, round(window / typical / 2))
    output = list(track)
    for i in range(radius, len(track) - radius):
        indexes = list(range(i - radius, i + radius + 1))
        if any(track[j] is None or (reliable is not None and not reliable[j]) for j in indexes):
            continue
        if any(times[j + 1] - times[j] > typical * 1.6 for j in indexes[:-1]):
            continue
        before = rm.vec_sub(track[i], track[i - 1])
        after = rm.vec_sub(track[i + 1], track[i])
        speed = max(rm.vec_length(before), rm.vec_length(after)) / typical
        abrupt = rm.vec_length(rm.vec_sub(after, before)) / typical
        if speed > 2.5 or abrupt > 1.0:
            continue
        xs = [(times[j] - times[i]) / typical for j in indexes]
        sums = [sum(x ** power for x in xs) for power in range(5)]
        matrix = [[sums[a + b] for b in range(3)] for a in range(3)]
        fitted = []
        for axis in range(3):
            solution = _solve(matrix, [sum(x ** power * track[j][axis] for x, j in zip(xs, indexes)) for power in range(3)])
            fitted.append(solution[0] if solution else track[i][axis])
        output[i] = rm.vec_lerp(track[i], fitted, min(1., strength))
    return output


def _joint_gaps(frames, max_gap):
    names = set().union(*(frame['body3d'] for frame in frames))
    for name in names:
        valid = [i for i, frame in enumerate(frames) if confidence(frame, name) >= skeleton.CONFIDENCE_LOW
                 and name in frame['body3d']]
        for i, frame in enumerate(frames):
            if not valid or i < valid[0] or i > valid[-1]:
                frame.setdefault('unreliable_joints', []).append(name)
        for a, b in zip(valid, valid[1:]):
            if b == a + 1:
                continue
            dt = (frames[b]['time'] - frames[a]['time']) / (b - a)
            bounded = frames[b]['time'] - frames[a]['time'] - dt <= max_gap + 1e-7
            for i in range(a + 1, b):
                if bounded:
                    fraction = (frames[i]['time'] - frames[a]['time']) / (frames[b]['time'] - frames[a]['time'])
                    frames[i]['body3d'][name] = rm.vec_lerp(frames[a]['body3d'][name], frames[b]['body3d'][name], fraction)
                    if name not in frames[i].setdefault('interpolated_joints', []):
                        frames[i]['interpolated_joints'].append(name)
                else:
                    frames[i].setdefault('unreliable_joints', []).append(name)


def constrain_lengths(frames):
    lengths = {}
    for a, b in LIMBS:
        values = [rm.vec_distance(f['body3d'][a], f['body3d'][b]) for f in frames
                  if a in f['body3d'] and b in f['body3d'] and min(confidence(f, a), confidence(f, b)) >= .4]
        if values:
            lengths[a, b] = median(values)
    for frame in frames:
        body = frame['body3d']
        # Fit the pelvis to reachable legs before solving the knees, rather than
        # shortening ankle excursions when a noisy leg is slightly overextended.
        drop = 0.
        for side in ('L', 'R'):
            hip, knee, ankle = ('hip.' + side, 'knee.' + side, 'ankle.' + side)
            if min(confidence(frame, name) for name in (hip, knee, ankle)) < .4 or set(frame.get('unreliable_joints', [])).intersection((hip, knee, ankle)):
                continue
            if (hip, knee) not in lengths or (knee, ankle) not in lengths:
                continue
            vector = rm.vec_sub(body[hip], body[ankle])
            reach = lengths[hip, knee] + lengths[knee, ankle] - 1e-5
            horizontal = vector[0] ** 2 + vector[1] ** 2
            if vector[2] > 0 and horizontal < reach ** 2:
                drop = min(drop, math.sqrt(reach ** 2 - horizontal) - vector[2])
        drop = max(-.1, drop)
        if drop:
            for name, point in list(body.items()):
                if not name.startswith(('ankle.', 'toe.', 'heel.')):
                    body[name] = (point[0], point[1], point[2] + drop)
            frame['hands3d'] = {name: (p[0], p[1], p[2] + drop) for name, p in frame.get('hands3d', {}).items()}
        # Solve whole limbs about their observed endpoints.
        for side in ('L', 'R'):
            if not set(frame.get('unreliable_joints', [])).intersection(('hip.' + side, 'knee.' + side, 'ankle.' + side)):
                _leg_ik(body, side, body['ankle.' + side], lengths)
            if not set(frame.get('unreliable_joints', [])).intersection(('shoulder.' + side, 'elbow.' + side, 'wrist.' + side)):
                old_wrist = body['wrist.' + side]
                _leg_ik(body, side, old_wrist, lengths, arm=True)
                delta = rm.vec_sub(body['wrist.' + side], old_wrist)
                for name, point in frame.get('hands3d', {}).items():
                    if name.endswith('.' + side):
                        frame['hands3d'][name] = rm.vec_add(point, delta)
    return lengths


def foot_states(frames, coordinate_space='root_relative', preset='general', rows=()):
    """Tri-state contact evidence. Root-relative foot movement is not world slip."""
    row_map = {row['sample_frame']: row for row in rows}
    states = {side: [] for side in ('L', 'R')}
    grounds = {}
    typical = median([b['time'] - a['time'] for a, b in zip(frames, frames[1:])] or [1 / 30])
    for side in states:
        ankle, toe = 'ankle.' + side, 'toe.' + side
        # The ground here is the ankle support height (toes may be synthesized).
        ground = smoothing.percentile([f['body3d'][ankle][2] for f in frames if ankle in f['body3d'] and confidence(f, ankle) >= .5], .15)
        grounds[side] = ground
        stable, pending_start = False, None
        for i, frame in enumerate(frames):
            point = frame['body3d'].get(ankle)
            unreliable = set(frame.get('unreliable_joints', []))
            if point is None or confidence(frame, ankle) < .5 or unreliable.intersection((ankle, 'knee.' + side, 'hip.' + side)):
                states[side].append('unknown')
                stable, pending_start = False, None
                continue
            if i and frame['time'] - frames[i - 1]['time'] > typical * 1.6:
                stable, pending_start = False, None
            dt = frame['time'] - frames[i - 1]['time'] if i else typical
            previous = frames[i - 1]['body3d'].get(ankle, point) if i else point
            velocity = rm.vec_scale(rm.vec_sub(point, previous), 1 / max(dt, 1e-6))
            height_ok = point[2] - ground <= (.055 if stable else .035)
            motion_ok = abs(velocity[2]) < (.65 if stable else .45)
            if coordinate_space == 'world':
                motion_ok = motion_ok and math.hypot(velocity[0], velocity[1]) < (.5 if stable else .35)
            foot = row_map.get(frame['frame'], {}).get('feet2d', {}).get(side, [])
            if foot and max(p[2] for p in foot) < .4:
                states[side].append('unknown')
                stable, pending_start = False, None
                continue
            # Visible toe/heel motion relative to the pelvis can veto a putative
            # stance. This is image evidence, not a measured 3D foot position.
            row = row_map.get(frame['frame'], {})
            prior_row = row_map.get(frames[i - 1]['frame'], {}) if i else row
            prior_foot = prior_row.get('feet2d', {}).get(side, [])
            points, prior_points = row.get('body2d', []), prior_row.get('body2d', [])
            pelvis_indices = (23, 24) if len(points) == 33 else (11, 12)
            if foot and prior_foot and len(points) > max(pelvis_indices) and len(prior_points) > max(pelvis_indices):
                if min(points[j][2] for j in pelvis_indices) >= .5 and min(prior_points[j][2] for j in pelvis_indices) >= .5:
                    relative = sum(p[1] for p in foot) / len(foot) - sum(points[j][1] for j in pelvis_indices) / 2
                    prior_relative = sum(p[1] for p in prior_foot) / len(prior_foot) - sum(prior_points[j][1] for j in pelvis_indices) / 2
                    motion_ok = motion_ok and abs(relative - prior_relative) / max(dt, 1e-6) < .45
            if height_ok and motion_ok:
                if pending_start is None:
                    pending_start = i
                duration = frame['time'] - frames[pending_start]['time'] + typical
                required = .12 if preset == 'idle' else .045
                if duration >= required:
                    stable = True
                    for j in range(pending_start, i):
                        states[side][j] = 'contact'
                states[side].append('contact' if stable else 'unknown')
            else:
                states[side].append('air' if point[2] - ground > .055 or abs(velocity[2]) > .65 else 'unknown')
                stable, pending_start = False, None
    return states, grounds


def _leg_ik(body, side, target, lengths, arm=False):
    hip, knee, ankle = tuple(name + '.' + side for name in (('shoulder', 'elbow', 'wrist') if arm else ('hip', 'knee', 'ankle')))
    if any(key not in body for key in (hip, knee, ankle)):
        return
    l1, l2 = lengths.get((hip, knee)), lengths.get((knee, ankle))
    if not l1 or not l2:
        return
    direction = rm.vec_normalize(rm.vec_sub(target, body[hip]))
    distance = max(abs(l1 - l2) + 1e-6, min(l1 + l2 - 1e-6, rm.vec_distance(body[hip], target)))
    target = rm.vec_add(body[hip], rm.vec_scale(direction, distance))
    axis = rm.vec_sub(body[knee], body[hip])
    plane = rm.vec_sub(axis, rm.vec_scale(direction, rm.vec_dot(axis, direction)))
    if rm.vec_length(plane) < 1e-6:
        plane = rm.vec_cross(direction, (1, 0, 0))
    along = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
    bend = math.sqrt(max(0, l1 * l1 - along * along))
    body[knee] = rm.vec_add(rm.vec_add(body[hip], rm.vec_scale(direction, along)), rm.vec_scale(rm.vec_normalize(plane), bend))
    delta = rm.vec_sub(target, body[ankle])
    body[ankle] = target
    for prefix in (() if arm else ('toe.', 'heel.')):
        if prefix + side in body:
            body[prefix + side] = rm.vec_add(body[prefix + side], delta)


def process(frames, fps, options=None):
    options = options or {}
    preset = options.get('motion_type', 'general')
    if preset not in PRESETS:
        raise ValueError('Unknown motion_type: ' + str(preset))
    rows = options.get('_preview_rows', [])
    raw = copy.deepcopy(frames)
    if len(frames) <= 1:
        # Static placement is independent of the temporal preset.
        if raw and options.get('coordinate_space', 'root_relative') != 'world':
            height = min(p[2] for n, p in raw[0]['body3d'].items() if n.startswith(('ankle.', 'toe.', 'heel.')))
            for key in ('body3d', 'hands3d'):
                raw[0][key] = {n: (p[0], p[1], p[2] - height) for n, p in raw[0].get(key, {}).items()}
        sparse_video = len(rows) > 1
        for row in rows:
            if not raw or row['sample_frame'] != raw[0]['frame']:
                row['unreliable'] = True
        warnings = [dict(code='SPARSE_CAPTURE', message='视频仅恢复到一个有效姿态，不能代表完整动作，请核对漏检区间。')] if sparse_video else []
        return raw, warnings, {'processing_version': VERSION, 'motion_type': preset,
                              'single_frame': not sparse_video, 'sparse_capture': sparse_video}
    gap = .1 if preset == 'attack' else .2
    output = fill_timeline(frames, rows, gap)
    _joint_gaps(output, gap)
    times = [f['time'] for f in output]
    space = options.get('coordinate_space', 'root_relative')
    warnings = []
    if space != 'world':
        warnings.append({'code': 'ROOT_TRAJECTORY_UNAVAILABLE', 'message': '当前三维为根相对坐标；未恢复真实水平轨迹，腾空高度存在单目歧义。'})
        # One translation for the whole sequence; never ground each frame separately.
        height = smoothing.percentile([p[2] for f in output for name, p in f['body3d'].items()
                                       if name.startswith(('ankle.', 'toe.', 'heel.'))], .05)
        for frame in output:
            for key in ('body3d', 'hands3d'):
                frame[key] = {name: (p[0], p[1], p[2] - height) for name, p in frame.get(key, {}).items()}
    states, grounds = foot_states(output, space, preset, rows)
    before_smooth = copy.deepcopy(output)
    for key in ('body3d', 'hands3d'):
        names = set().union(*(f.get(key, {}) for f in output))
        for name in names:
            reliable = [confidence(f, name) >= .4 and name not in f.get('unreliable_joints', []) for f in output]
            if name.startswith(('ankle.', 'knee.', 'toe.', 'heel.')):
                side = name[-1]
                for i in range(1, len(output)):
                    if states[side][i] != states[side][i - 1]:
                        reliable[i] = reliable[i - 1] = False
            values = symmetric_track([f.get(key, {}).get(name) for f in output], times, PRESETS[preset],
                                     options.get('smoothing_strength', .65), reliable)
            for frame, point in zip(output, values):
                if point is not None:
                    frame[key][name] = point
    lengths = constrain_lengths(output)
    strength = max(0., min(1., float(options.get('foot_lock_strength', .7))))
    anchors = {}
    previous_time = None
    for i, frame in enumerate(output):
        if previous_time is not None and frame['time'] - previous_time > 1.6 / fps:
            anchors.clear()
        previous_time = frame['time']
        frame['contact_states'] = {side: states[side][i] for side in states}
        frame['contacts'] = {'foot.' + side: states[side][i] == 'contact' for side in states}
        # A bounded shared pelvis translation keeps both legs reachable. No
        # correction is generated during flight or uncertain contact.
        support = [side for side in states if states[side][i] == 'contact']
        if support and strength:
            offsets = [grounds[side] - frame['body3d']['ankle.' + side][2] for side in support]
            dz = max(-.035, min(.035, median(offsets))) * strength
            for key in ('body3d', 'hands3d'):
                frame[key] = {n: (p[0], p[1], p[2] + dz) for n, p in frame.get(key, {}).items()}
        for side in states:
            if states[side][i] != 'contact':
                anchors.pop(side, None)
                continue
            ankle = frame['body3d']['ankle.' + side]
            anchor = anchors.setdefault(side, ankle)
            # Root-relative backward motion during support is intentional locomotion.
            target = (anchor[0], anchor[1], grounds[side]) if space == 'world' else (ankle[0], ankle[1], grounds[side])
            if strength:
                _leg_ik(frame['body3d'], side, rm.vec_lerp(ankle, target, strength), lengths)
    diagnostics = compare(before_smooth, output, fps)
    diagnostics.update(processing_version=VERSION, motion_type=preset, coordinate_space=space,
                       root_trajectory_available=space == 'world',
                       contact_counts={side: {state: states[side].count(state) for state in ('contact', 'air', 'unknown')} for side in states})
    diagnostics['contact_intervals'] = {}
    timeline = rows or [{'sample_frame': f['frame'], 'time': f['time']} for f in output]
    state_map = {f['frame']: f['contact_states'] for f in output}
    for side in states:
        intervals = []
        for i, row in enumerate(timeline):
            state = state_map.get(row['sample_frame'], {}).get(side, 'unknown')
            end = timeline[i + 1]['time'] if i + 1 < len(timeline) else row['time'] + 1 / fps
            if intervals and intervals[-1]['state'] == state:
                intervals[-1]['end_time'] = end
            else:
                intervals.append(dict(state=state, start_time=row['time'], end_time=end))
        diagnostics['contact_intervals'][side] = intervals
    if any(v['amplitude_ratio'] is not None and (v['amplitude_ratio'] < .9 or abs(v['lag_frames']) > 1) for v in diagnostics['joints'].values()):
        warnings.append(dict(code='PROCESSING_FIDELITY_REVIEW', message='部分关节处理前后幅度或时序变化较大，请切换原始／处理后逐帧核对。'))
    if preset in ('walk', 'run'):
        diagnostics['support_events'] = {side: sum(state == 'contact' and (i == 0 or states[side][i - 1] != 'contact')
                                                  for i, state in enumerate(states[side])) for side in states}
    frame_map = {f['frame']: f for f in output}
    for row in rows:
        frame = frame_map.get(row['sample_frame'])
        if frame:
            row['contact_states'] = frame['contact_states']
            row['interpolated_joints'] = frame.get('interpolated_joints', [])
            row['unreliable_joints'] = frame.get('unreliable_joints', [])
    return output, warnings, diagnostics


def compare(raw, processed, fps):
    """Amplitude and lag are processing diagnostics, never recognition accuracy."""
    original = {frame['frame']: frame for frame in raw}
    pairs = [(original[f['frame']], f) for f in processed if f['frame'] in original]
    joints = {}
    for name in ('wrist.L', 'wrist.R', 'ankle.L', 'ankle.R', 'pelvis'):
        tracks = []
        for side in (0, 1):
            tracks.append([pair[side]['body3d'][name] for pair in pairs if name in pair[0]['body3d'] and name in pair[1]['body3d']])
        if len(tracks[0]) < 3:
            continue
        spans = [[max(p[a] for p in track) - min(p[a] for p in track) for a in range(3)] for track in tracks]
        axis = max(range(3), key=lambda a: spans[0][a] if max(spans[0]) > 1e-8 else spans[1][a])
        amplitude = spans[0][axis]
        lag = None
        if amplitude > .005:
            def cost(shift):
                values = [(tracks[0][i][axis], tracks[1][i + shift][axis]) for i in range(len(tracks[0])) if 0 <= i + shift < len(tracks[1])]
                a_mean, b_mean = (sum(p[k] for p in values) / len(values) for k in (0, 1))
                return sum(((a - a_mean) - (b - b_mean)) ** 2 for a, b in values) / len(values)
            lag = min(range(-min(3, len(pairs) // 3), min(3, len(pairs) // 3) + 1), key=lambda n: (cost(n), abs(n)))
        joints[name] = {'raw_amplitude': amplitude, 'processed_amplitude': spans[1][axis],
                        'amplitude_ratio': spans[1][axis] / amplitude if amplitude > .005 else None,
                        'lag_frames': lag, 'lag_seconds': lag / fps if lag is not None else None}
    return {'joints': joints}
