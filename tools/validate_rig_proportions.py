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

Measure your own scene (the file is opened, never saved)::

    blender --background "C:\\path\\to\\your.blend" \
        --python tools/validate_rig_proportions.py -- \
        --armature "rig" --folder "C:\\path\\to\\capture" --positions

Leave ``--armature`` out to take the first Rigify human found in the scene, and
leave ``--folder`` pointing at any folder that has no capture to report the
rig's own proportions only. ``--positions`` assigns a new Action to the rig, so
it refuses to run when the rig already carries animation unless you pass
``--allow-animated``.
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


def find_rigify_armature(adapter, name=""):
    """Return ``(armature, note)`` for the rig to measure.

    ``name`` selects a specific object; an empty name takes the first Rigify
    human in the scene. Returning ``None`` lets the caller fall back to a freshly
    generated rig rather than failing on an empty file.
    """
    if name:
        armature = bpy.data.objects.get(name)
        if armature is None:
            raise SystemExit(
                "找不到骨架对象 {0!r}。用 --armature 指定场景中的对象名；"
                "不带 --armature 时会自动使用场景里的 Rigify 骨架。".format(name))
        if getattr(armature, 'type', None) != 'ARMATURE':
            raise SystemExit("{0!r} 不是骨架对象。".format(name))
        detection = adapter.detect_rigify_human(armature)
        if not detection.ok:
            raise SystemExit("{0!r} 不是可用的 Rigify Human 骨架：{1}\n缺少的骨：{2}".format(
                name, detection.error.message if detection.error else "未识别",
                ", ".join(detection.missing_bones) or "无"))
        return armature, "场景对象 {0!r}".format(name)
    for armature in bpy.data.objects:
        if getattr(armature, 'type', None) != 'ARMATURE':
            continue
        if adapter.detect_rigify_human(armature).ok:
            return armature, "场景中自动找到的 Rigify 骨架 {0!r}".format(armature.name)
    return None, ""


def referenced_by_animation(armature):
    """Animation on the rig that retargeting would replace.

    Retargeting assigns a new Action, so an existing Action, NLA track or
    shape-key animation would be displaced. Rigify's own drivers are *not*
    reported: they are part of the rig, retargeting leaves them alone, and every
    generated Rigify human has them, so flagging them would block the common
    case for no reason.
    """
    found = []
    animation = getattr(armature, 'animation_data', None)
    if animation is not None:
        if animation.action is not None:
            found.append("Action {0!r}".format(animation.action.name))
        if getattr(animation, 'nla_tracks', None):
            found.append("{0} 条 NLA 轨道".format(len(animation.nla_tracks)))
    shape_keys = getattr(getattr(armature, 'data', None), 'shape_keys', None)
    if shape_keys is not None and shape_keys.animation_data is not None:
        if shape_keys.animation_data.action is not None:
            found.append("形态键 Action")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default=str(ROOT / '.cache/v03-camera'),
                        help='cached capture folder holding mocap_result.json')
    parser.add_argument('--result', default='mocap_result.json')
    parser.add_argument('--armature', default='',
                        help='name of the armature object to measure; empty takes the '
                             'first Rigify human found in the current scene')
    parser.add_argument('--output', default='')
    parser.add_argument('--positions', action='store_true',
                        help='also retarget onto the rig and measure world-space gaps')
    parser.add_argument('--allow-animated', action='store_true',
                        help='let --positions run on a rig that already has animation; '
                             'it will assign a new Action and is not undoable in background mode')
    parser.add_argument('--fps', type=float, default=24.0)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])

    harness.enable_addon()
    from motion_capture.core import result_schema
    from motion_capture.blender import rigify_adapter as adapter

    if args.armature:
        # The user pointed at a scene object: leave their file exactly as it is.
        rig, note = find_rigify_armature(adapter, args.armature)
    else:
        rig, note = find_rigify_armature(adapter)
        if rig is None:
            harness.reset_scene()
            harness.enable_addon()
            rig = harness.generate_rigify_human()
            note = "当前场景没有 Rigify 骨架，改为生成一个默认骨架"
    print("测量对象：{0}\n".format(note))

    result_path = Path(args.folder) / args.result
    if result_path.is_file():
        result = result_schema.load_mocap_result(str(result_path))
        capture = source_lengths(args.folder, args.result)
    else:
        # Without a capture the rig's own proportions are still worth reporting:
        # comparing two rigs, or checking one against known anatomy, needs no
        # motion data at all.
        result, capture = None, {}
        print("未找到 {0}，只报告骨架自身比例（无素材对照）。\n".format(result_path))

    rigged = rig_lengths(rig)

    # The retarget scales the root so the rig matches the capture's pelvis
    # height; a rig that is taller overall would otherwise show a uniform bias
    # rather than a proportional one.
    rig_pelvis = adapter.rig_pelvis_height(rig)
    heights = sorted(frame.body3d['pelvis'][2] for frame in result.frames
                     if 'pelvis' in frame.body3d) if result else []
    source_pelvis = heights[len(heights) // 2] if heights else 0.0
    root_scale = rig_pelvis / source_pelvis if source_pelvis > 1e-3 else 1.0

    rows = []
    for label, _bones, _source in SEGMENTS:
        expected = capture.get(label, 0.0) * root_scale
        ratio = rigged[label] / expected if label in rigged and expected > 1e-6 else None
        rows.append({'segment': label,
                     'source_m': round(capture[label], 4) if label in capture else None,
                     'rig_m': round(rigged[label], 4) if label in rigged else None,
                     'expected_m': round(expected, 4) if expected > 1e-6 else None,
                     'ratio': round(ratio, 4) if ratio is not None else None,
                     'ok': (abs(ratio - 1.0) <= TOLERANCE) if ratio is not None else None})

    report = {
        'object': rig.name,
        'source': note,
        'capture_result': str(result_path) if result is not None else None,
        'root_scale': round(root_scale, 4),
        'rig_pelvis_height_m': round(rig_pelvis, 4),
        'source_pelvis_height_m': round(source_pelvis, 4) if source_pelvis else None,
        'segments': rows,
        'mismatched': [row['segment'] for row in rows if row['ok'] is False],
    }
    stature = rig_stature(rig)
    if stature:
        report['rig_stature_m'] = round(stature, 4)
        # How much of the rig's height is legs decides whether a root scale
        # driven by the pelvis can ever make the feet land on the source.
        report['rig_leg_share_of_stature'] = round(
            (rigged.get('thigh.L', 0.0) + rigged.get('shin.L', 0.0)) / stature, 4)
    if args.positions and result is not None:
        existing = referenced_by_animation(rig)
        if existing and not args.allow_animated:
            raise SystemExit(
                "停止：{0!r} 上已有 {1}。--positions 会重定向并给它换上一个新 Action，"
                "后台模式下无法撤销。\n"
                "请改用 --armature 指向一个没有动画的骨架，或在不带 --positions 的情况下运行"
                "（只测比例，不改动文件），\n"
                "确认过风险后加 --allow-animated 强制执行。".format(rig.name, "、".join(existing)))
        options = adapter.RetargetOptions(include_hands=False, scene_fps=args.fps,
                                          root_motion='in_place', action_name='proportion check')
        measured = measure_positions(rig, result, args.fps, root_scale, options)
        report['positions'] = measured['summary']
    elif args.positions:
        print("跳过 --positions：需要一份捕捉结果才能重定向。\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['mismatched']:
        print('\n比例不匹配：这些骨段在重定向后末端会偏离素材位置（方向正确也一样）。')
    target = Path(args.output) if args.output else Path(args.folder) / 'rig-proportions.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('written', target)


if __name__ == '__main__':
    main()
