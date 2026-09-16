"""Compare a Rigify rig's proportions against a capture's, per segment.

Retargeting only aims bones (``_target_rotation``) and scales the root by the
pelvis height (``_root_scale``); every bone's *length* comes from the target
rig. So a directionally perfect retarget still lands an end effector somewhere
else when the rig's proportions differ, and nothing in the existing validation
catches that -- ``validate_orientation_blender.py`` only asserts direction
errors under 0.1 degrees.

This script reports both the raw length ratios and, in ``--positions`` mode,
the world-space displacement each direction produces. The displacement is the
part that actually shows up on screen: a bone that is 10% too long moves its
tail 10% of its own length away from where the capture put it.

Run inside Blender::

    blender --background --factory-startup --python tools/validate_rig_proportions.py -- \
        --folder .cache/v03-camera --positions
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / 'tests/blender'))

import bpy  # noqa: E402
import _harness as harness  # noqa: E402


#: Segments compared, as (label, rig bone chain, source joint pair).
SEGMENTS = (
    ('thigh.L', ('thigh_fk.L',), ('hip.L', 'knee.L')),
    ('shin.L', ('shin_fk.L',), ('knee.L', 'ankle.L')),
    ('thigh.R', ('thigh_fk.R',), ('hip.R', 'knee.R')),
    ('shin.R', ('shin_fk.R',), ('knee.R', 'ankle.R')),
    ('upper_arm.L', ('upper_arm_fk.L',), ('shoulder.L', 'elbow.L')),
    ('forearm.L', ('forearm_fk.L',), ('elbow.L', 'wrist.L')),
    ('spine', ('spine_fk', 'spine_fk.001', 'spine_fk.002'),
     ('pelvis', 'chest')),
    ('shoulder_width', ('shoulder.L', 'shoulder.R'), ('shoulder.L', 'shoulder.R')),
)

#: Chains whose accumulated position error is reported per frame.
POSITION_CHAINS = (
    ('thigh_fk.L', ('hip.L', 'knee.L')),
    ('shin_fk.L', ('knee.L', 'ankle.L')),
    ('thigh_fk.R', ('hip.R', 'knee.R')),
    ('shin_fk.R', ('knee.R', 'ankle.R')),
)

#: Ratios beyond this are large enough to move a foot by centimetres.
TOLERANCE = 0.05


def rig_stature(armature):
    """Approximate rig stature: head-top control height above the ground plane.

    Rigify's ``head`` bone tail is the top of the skull, which is the same
    landmark the source skeleton's ``head`` joint approximates, so the two are
    comparable without a second tolerance decision.
    """
    matrix = armature.matrix_world
    head = armature.data.bones.get('head')
    if head is None:
        return None
    top = matrix @ head.tail_local
    lowest = min((matrix @ bone.head_local).z for bone in armature.data.bones)
    return float(top.z - lowest)


def rig_lengths(armature):
    """Rest-pose lengths of the rig's control chains, in world metres."""
    matrix = armature.matrix_world
    lengths = {}
    for label, bones, _source in SEGMENTS:
        total = 0.0
        for name in bones:
            bone = armature.data.bones.get(name)
            if bone is None:
                total = None
                break
            head = matrix @ bone.head_local
            tail = matrix @ bone.tail_local
            total += (tail - head).length
        if total:
            lengths[label] = float(total)
    return lengths


def source_lengths(folder, result_name):
    """Median segment lengths of a capture, in metres."""
    with open(Path(folder) / result_name, encoding='utf-8') as handle:
        result = json.load(handle)
    frames = result['frames']
    lengths = {}
    for label, _bones, source in SEGMENTS:
        values = []
        for frame in frames:
            body = frame.get('body3d') or {}
            a, b = body.get(source[0]), body.get(source[1])
            if a is None or b is None:
                continue
            values.append(math.dist(a, b))
        if values:
            values.sort()
            lengths[label] = values[len(values) // 2]
    return lengths


def measure_positions(rig, result, scene_fps, root_scale, options):
    """World-space gap between the retargeted rig and the capture, per frame.

    The rig is driven by the real retargeting code and read back through the
    dependency graph, so what is measured is the pose the animator would see,
    not a re-derivation of it.
    """
    from motion_capture.blender import rigify_adapter as adapter
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.fps = int(round(scene_fps))
    scene.frame_start = 1
    scene.frame_end = max(1, len(result.frames))
    adapter.retarget_to_rigify(result, rig, options)

    samples = []
    for frame in result.frames:
        number = round(1 + frame.time * scene_fps)
        scene.frame_set(number)
        bpy.context.view_layer.update()
        row = {'frame': frame.frame, 'time': round(frame.time, 4)}
        for bone_name, (start_joint, end_joint) in POSITION_CHAINS:
            pose_bone = rig.pose.bones.get(bone_name)
            if pose_bone is None:
                continue
            head = rig.matrix_world @ pose_bone.head
            tail = rig.matrix_world @ pose_bone.tail
            source_head = Vector(frame.body3d[start_joint]) * root_scale
            source_tail = Vector(frame.body3d[end_joint]) * root_scale
            # Align both chains on the same joint head: the question is how far
            # the rig's own bone length moves the tail away from the capture.
            row[bone_name] = {
                'error_m': round((tail - head).length - (source_tail - source_head).length, 4),
                'tail_gap_m': round(((head + (source_tail - source_head)) - tail).length, 4),
            }
        samples.append(row)

    summary = {}
    for bone_name, (start_joint, end_joint) in POSITION_CHAINS:
        errors = [abs(sample[bone_name]['error_m']) for sample in samples if bone_name in sample]
        gaps = [sample[bone_name]['tail_gap_m'] for sample in samples if bone_name in sample]
        if not errors:
            continue
        errors.sort()
        gaps.sort()
        summary[bone_name] = {
            'source_joint': '{0}->{1}'.format(start_joint, end_joint),
            'median_length_error_m': round(errors[len(errors) // 2], 4),
            'median_tail_gap_m': round(gaps[len(gaps) // 2], 4),
            'max_tail_gap_m': round(gaps[-1], 4),
        }
    return {'samples': samples, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=str(ROOT / '.cache/v03-camera'))
    parser.add_argument('--result', default='mocap_result.json')
    parser.add_argument('--output', default='')
    parser.add_argument('--positions', action='store_true',
                        help='also retarget onto the rig and measure world-space gaps')
    parser.add_argument('--fps', type=float, default=24.0)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])

    harness.reset_scene()
    harness.enable_addon()
    from motion_capture.core import result_schema
    from motion_capture.blender import rigify_adapter as adapter

    rig = harness.generate_rigify_human()
    result = result_schema.load_mocap_result(str(Path(args.folder) / args.result))
    capture = source_lengths(args.folder, args.result)
    rigged = rig_lengths(rig)

    # The retarget scales the root so the rig matches the capture's pelvis
    # height; a rig that is taller overall would otherwise show a uniform bias
    # rather than a proportional one.
    rig_pelvis = adapter.rig_pelvis_height(rig)
    heights = sorted(frame.body3d['pelvis'][2] for frame in result.frames
                     if 'pelvis' in frame.body3d)
    source_pelvis = heights[len(heights) // 2] if heights else 0.0
    root_scale = rig_pelvis / source_pelvis if source_pelvis > 1e-3 else 1.0

    rows = []
    for label, _bones, _source in SEGMENTS:
        if label not in capture or label not in rigged:
            continue
        expected = capture[label] * root_scale
        ratio = rigged[label] / expected if expected > 1e-6 else float('nan')
        rows.append({'segment': label, 'source_m': round(capture[label], 4),
                     'rig_m': round(rigged[label], 4), 'expected_m': round(expected, 4),
                     'ratio': round(ratio, 4), 'ok': abs(ratio - 1.0) <= TOLERANCE})

    report = {
        'object': rig.name,
        'root_scale': round(root_scale, 4),
        'rig_pelvis_height_m': round(rig_pelvis, 4),
        'source_pelvis_height_m': round(source_pelvis, 4),
        'segments': rows,
        'mismatched': [row['segment'] for row in rows if not row['ok']],
    }
    stature = rig_stature(rig)
    if stature:
        report['rig_stature_m'] = round(stature, 4)
        # How much of the rig's height is legs decides whether a root scale
        # driven by the pelvis can ever make the feet land on the source.
        report['rig_leg_share_of_stature'] = round(
            (rigged.get('thigh.L', 0.0) + rigged.get('shin.L', 0.0)) / stature, 4)
    if args.positions:
        options = adapter.RetargetOptions(include_hands=False, scene_fps=args.fps,
                                          root_motion='in_place', action_name='proportion check')
        measured = measure_positions(rig, result, args.fps, root_scale, options)
        report['positions'] = measured['summary']
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['mismatched']:
        print('\n比例不匹配：这些骨段在重定向后末端会偏离素材位置（方向正确也一样）。')
    target = Path(args.output) if args.output else Path(args.folder) / 'rig-proportions.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', target)


if __name__ == '__main__':
    main()
