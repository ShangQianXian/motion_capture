"""Full-source regression and before/after evidence for the v0.3 orientation fix.

Run with the quality worker Python. --capture performs fresh model inference;
without it replay the existing WholeBody observations for quick solver diagnosis.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import orientations as ori, motion_processing as mp, retarget_math as rm
from backend_worker import orientation_solver, export_result


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def metrics(frames):
    result={}
    for name in ori.NAMES:
        qs=[f['orientations'][name] for f in frames]
        changes=[math.degrees(rm.quat_angle(rm.quat_mul(rm.quat_conjugate(a),b))) for a,b in zip(qs,qs[1:])]
        result[name]={'max_step_degrees':max(changes,default=0), 'steps_over_45_degrees':sum(a>45 for a in changes)}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',action='store_true')
    parser.add_argument('--reprocess',action='store_true',help='Reprocess saved raw orientations without loading models')
    parser.add_argument('--profile',default='quality',choices=('quality','preview','quality_feet','quality_plus'))
    parser.add_argument('--source',default='C:/Users/FanZhen/Desktop/models/角色/角色行走.mp4')
    args=parser.parse_args()
    old=ROOT/'.cache/v03-real/quality_feet_video'
    output=ROOT/'.cache/v03-orientations'/('capture_'+args.profile if args.capture or args.reprocess else 'replay')
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    if args.reprocess:
        from backend_worker import preview_export
        job=read(output/'job.json')
        manifest=read(output/'preview_manifest.json')
        rows=manifest['frames']
        raw=read(output/'mocap_raw.json')['frames']
        by_frame={r['sample_frame']:r for r in rows}
        for f in raw:
            row=by_frame[f['frame']]
            f['orientation_quality']=copy.deepcopy(row.get('raw_orientation_quality') or row['orientation_quality'])
            if args.profile=='preview':
                for info in f['orientation_quality'].values():
                    info['confidence']=min(.69,info['confidence'])
        frames,warnings,diagnostics=mp.process(raw,manifest['source']['effective_fps'],dict(job['options'],_preview_rows=rows))
        result=export_result.build_result(frames,manifest['source']['effective_fps'],source_path=job['input']['path'],
                    source_type=job['input']['type'],profile=args.profile,warnings=warnings)
        path=export_result.write_result(result,str(output))
        for serialized,frame in zip(result['frames'],frames):
            serialized['interpolated_joints']=frame.get('interpolated_joints',[])
        preview_export.write(job,result,path,rows,manifest['source'],args.profile,raw_frames=raw,diagnostics=diagnostics)
        manifest=read(output/'preview_manifest.json')
    elif args.capture:
        job=read(old/'job.json')
        job['input']['path']=args.source
        job['model'].update(profile=args.profile,device='cpu' if args.profile=='preview' else 'cuda:0')
        job['options'].update(processing_version=ori.VERSION,include_hands=False)
        job['review_settings'].update(capture_profile=args.profile,include_hands=False)
        job['output']['dir']=str(output)
        (output/'job.json').write_text(json.dumps(job,ensure_ascii=False),encoding='utf-8')
        worker=ROOT/('.venv-preview' if args.profile=='preview' else '.venv')/'Scripts/python.exe'
        with (output/'events.jsonl').open('w',encoding='utf-8') as stdout, (output/'worker.log').open('w',encoding='utf-8') as stderr:
            subprocess.run([str(worker),'-m','backend_worker.cli','--job',str(output/'job.json')],cwd=ROOT,
                           stdout=stdout,stderr=stderr,check=True,timeout=900)
        frames=read(output/'mocap_result.json')['frames']
        manifest=read(output/'preview_manifest.json')
        rows=manifest['frames']
    else:
        raw=read(old/'mocap_raw.json')
        frames=copy.deepcopy(raw['frames'])
        manifest=read(old/'preview_manifest.json')
        rows=manifest['frames']
        size=(manifest['source']['width'],manifest['source']['height'])
        orientation_solver.enrich(frames,rows,size,'quality')
        export_result.write_result(export_result.build_result(frames,24),str(output),'mocap_raw.json')
        frames,_,diagnostics=mp.process(frames,24,dict(motion_type='walk',_preview_rows=rows))
        export_result.write_result(export_result.build_result(frames,24),str(output))
        manifest['diagnostics']=diagnostics
        (output/'preview_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    report=dict(seconds=round(time.monotonic()-start,2),frames=len(frames),rotation=metrics(frames),
                coverage=manifest['diagnostics'].get('orientation_coverage',{}),
                note='Visible 2D agreement and rotation continuity are not ground-truth 3D accuracy.')
    for name in ori.NAMES:
        observations=[r.get('orientation_quality',{}).get(name,{}) for r in rows]
        errors=sorted(o['relative_error'] for o in observations if 'relative_error' in o and o.get('confidence',0)>=.4)
        report['rotation'][name]['median_relative_reprojection']=errors[len(errors)//2] if errors else None
        report['rotation'][name]['sources']={s:sum(o.get('source')==s for o in observations) for s in set(o.get('source') for o in observations)}
    report['focus_frames']=[dict(time=f['time'],body3d={k:f['body3d'][k] for k in ('neck','head','knee.L','knee.R')},orientations=f['orientations'])
                            for i in (4,19) for f in frames[i:i+1]]
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
