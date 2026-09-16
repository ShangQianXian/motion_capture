"""Replay all saved real observations through production processing and calibration.

No model inference is repeated: camera extrinsics are applied after inference.
Metrics measure coordinate alignment and rigid-motion invariants, not 3D truth.
"""
import copy
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import camera_alignment as ca, orientations as ori, retarget_math as rm
from backend_worker import postprocess, export_result, preview_export


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def yaw(frame, joint='hip'):
    d=rm.vec_sub(frame['body3d'][joint+'.L'],frame['body3d'][joint+'.R'])
    return math.degrees(math.atan2(d[1],d[0]))


def ranges(frames,side):
    tracks=[[f['body3d']['ankle.'+side][i]-f['body3d']['hip.'+side][i] for f in frames] for i in (0,1)]
    return dict(lateral_x_m=max(tracks[0])-min(tracks[0]),fore_aft_y_m=max(tracks[1])-min(tracks[1]))


def main():
    source=ROOT/'.cache/v03-orientations/capture_quality'
    output=ROOT/'.cache/v03-camera'
    output.mkdir(parents=True,exist_ok=True)
    job,manifest=read(source/'job.json'),read(source/'preview_manifest.json')
    raw=read(source/'mocap_raw.json')['frames']
    rows=manifest['frames']
    by_frame={r['sample_frame']:r for r in rows}
    for frame in raw:
        frame['orientation_quality']=copy.deepcopy(by_frame[frame['frame']]['raw_orientation_quality'])
    job['options'].update(processing_version=ori.VERSION,camera_view='left_front_45',align_initial_facing=True)
    job.setdefault('review_settings',{}).update(camera_view='left_front_45',align_initial_facing=True)
    job['output']['dir']=str(output)
    job['job_id']='v032-fixed-camera-validation'
    options=dict(job['options'],_preview_rows=rows,coordinate_space='root_relative')
    fps=manifest['source']['effective_fps']
    processed,warnings=postprocess.postprocess(raw,fps,options)
    before=copy.deepcopy(processed)
    raw,processed,calibration=ca.finalize(raw,processed,options,options['_diagnostics'])
    result=export_result.build_result(processed,fps,source_path=job['input']['path'],source_type='video',
             profile='quality',warnings=warnings,capture_transform=calibration)
    path=export_result.write_result(result,str(output))
    preview_export.write(job,result,path,rows,manifest['source'],'quality',raw_frames=raw,diagnostics=options['_diagnostics'])
    (output/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
    inverse=dict(camera_view='left_front_45',quaternion_wxyz=rm.quat_conjugate(calibration['quaternion_wxyz']))
    roundtrip=ca.transform(processed,inverse)
    max_error=max(rm.vec_distance(a['body3d'][n],b['body3d'][n]) for a,b in zip(before,roundtrip) for n in a['body3d'])
    old=read(source/'mocap_result.json')['frames']
    report=dict(frames=len(processed),fps=fps,validation_source='cached real 192-frame inference; production postprocess + camera alignment',
        calibration=calibration,source_view_roundtrip_error_m=max_error,
        initial_hip_yaw_before=statistics.median(yaw(f) for f in before[:7]),
        initial_hip_yaw_after=statistics.median(yaw(f) for f in processed[:7]),
        hip_yaw_range_after=[min(yaw(f) for f in processed),max(yaw(f) for f in processed)],
        ankle_relative_to_hip={s:dict(v031_camera_axes=ranges(old,s),v032_character_axes=ranges(processed,s)) for s in ('L','R')},
        frames_checked={str(f['time']):dict(frame=f['frame'],hip_yaw=yaw(f),shoulder_yaw=yaw(f,'shoulder'))
                        for f in processed if f['frame'] in (5,20)},
        note='Axis ranges are not 3D accuracy. Residual sideways motion is retained; this change does not infer new limb depth.')
    assert len(processed)==192 and max_error<1e-10
    assert abs(report['initial_hip_yaw_after'])<.2,report
    for side in ('L','R'):
        span=report['ankle_relative_to_hip'][side]['v032_character_axes']
        assert span['fore_aft_y_m']>2*span['lateral_x_m'],span
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
