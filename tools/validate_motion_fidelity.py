"""Reproducible processing acceptance; no trained models or Blender required.

Optional --encode adds silent MP4s from the Blender fixture PNGs (requires cv2).
This report tests processing fidelity, not monocular reconstruction accuracy.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.fixtures.motions import clip
from core import motion_processing as mp, retarget_math as rm
from backend_worker.postprocess import postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    report = {'scope': 'processing of known 3D; not detector/reconstruction accuracy', 'cases': []}
    for fps in (15, 24, 30, 60):
        for preset in mp.PRESETS:
            truth = clip(preset, fps)
            out, _, metrics = mp.process(truth, fps, dict(motion_type=preset, coordinate_space='world'))
            ratios = [item['amplitude_ratio'] for item in metrics['joints'].values() if item['amplitude_ratio'] is not None]
            lag = max(abs(item['lag_frames']) for item in metrics['joints'].values() if item['lag_frames'] is not None)
            bone_error = max(max(rm.vec_distance(f['body3d'][a], f['body3d'][b]) for f in out) -
                             min(rm.vec_distance(f['body3d'][a], f['body3d'][b]) for f in out) for a, b in mp.LIMBS)
            contact_error = 0
            if preset in ('walk', 'run'):
                for side in ('L', 'R'):
                    events = lambda frames: [f['time'] for a, f in zip(frames, frames[1:]) if a['contacts']['foot.' + side] != f['contacts']['foot.' + side]]
                    measured = events(out)
                    contact_error = max([contact_error] + [min(abs(t - p) for p in measured) for t in events(truth)])
            report['cases'].append(dict(preset=preset, fps=fps, amplitude_min=min(ratios), lag_frames=lag,
                                        bone_length_range_m=bone_error, contact_event_error_s=contact_error))
    truth = clip('idle', 60)
    for i, frame in enumerate(truth):
        for side in ('L', 'R'):
            for name in ('ankle.', 'toe.', 'heel.'):
                p = frame['body3d'][name + side]
                frame['body3d'][name + side] = (p[0], p[1] + .025 * math.sin(i / 25), p[2])
            frame['confidence']['toe.' + side] = .3
    legacy, _ = postprocess(copy.deepcopy(truth), 60, {})
    final, _, _ = mp.process(truth, 60, dict(motion_type='idle', coordinate_space='world'))
    slip = lambda seq: sum(rm.vec_distance(a['body3d']['ankle.L'], b['body3d']['ankle.L']) for a, b in zip(seq[10:-1], seq[11:]))
    report['support_slip'] = dict(v02_m=slip(legacy), v03_m=slip(final), reduction=1 - slip(final) / slip(legacy))
    report['passed'] = all(c['amplitude_min'] >= .9 and c['lag_frames'] <= 1 and c['bone_length_range_m'] < 1e-5
                           and c['contact_event_error_s'] <= .10001 for c in report['cases']) and report['support_slip']['reduction'] >= .5
    output = ROOT / '.cache/v03-truth'
    output.mkdir(exist_ok=True)
    (output / 'processing-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.encode:
        import cv2
        for directory in sorted(output.glob('*/*')):
            files = sorted(directory.glob('frame_*.png'))
            if not files:
                continue
            frame = cv2.imread(str(files[0]))
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(directory / 'input.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
            assert writer.isOpened()
            try:
                for file in files:
                    writer.write(cv2.imread(str(file)))
            finally:
                writer.release()
            print('ENCODED', directory, len(files), flush=True)
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'support_slip': report['support_slip']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
