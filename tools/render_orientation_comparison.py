"""Render same-video old/new evidence and image-plane direction metrics."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import retarget_math as rm, preview


def read(path):
    return json.loads(path.read_text('utf-8'))


def angle(a,b):
    if math.hypot(*a)<1e-5 or math.hypot(*b)<1e-5:
        return None
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))/(math.hypot(*a)*math.hypot(*b))))))


def plane(p):
    return p[0],-p[2]


def main():
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=ROOT/'.cache/v03-orientations'
    before=read(root/'baseline/mocap_result.json')['frames']
    after=read(root/'capture_quality/mocap_result.json')['frames']
    manifest=read(root/'capture_quality/preview_manifest.json')
    old_manifest=read(root/'baseline/preview_manifest.json')
    rows=manifest['frames']
    # The new Quality path and old quality_feet path must have identical observations.
    maximum=max(abs(a-b) for ra,rb in zip(rows,old_manifest['frames'])
                for pa,pb in zip(ra['body2d'],rb['body2d']) for a,b in zip(pa,pb))
    width,height=manifest['source']['width'],manifest['source']['height']
    metrics={'frames':len(after),'max_2d_difference':maximum,'direction_error_degrees':{},
             'note':'Image-plane direction relative to detected landmarks; not 3D ground truth.'}
    for name in ('foot.L','foot.R','head'):
        errors={'before':[],'after':[]}
        for old,new,row in zip(before,after,rows):
            if name=='head':
                points=row['body2d']
                if min(points[i][2] for i in (59,68))<.4:
                    continue
                observed=((points[68][0]-points[59][0])*width,(points[68][1]-points[59][1])*height)
                up=rm.vec_sub(old['body3d']['head'],old['body3d']['neck'])
                lateral=rm.vec_sub(old['body3d']['shoulder.L'],old['body3d']['shoulder.R'])
                olddir=rm.mat_column(rm.basis_from_axes(up,2,lateral,0),0)
                newdir=rm.quat_rotate_vector(new['orientations']['head'],(1,0,0))
            else:
                side=name[-1]
                feet=row['feet2d'][side]
                ankle=row['body2d'][15 if side == 'L' else 16]
                if min(p[2] for p in feet)<.4 or ankle[2]<.4:
                    continue
                # Identical endpoints on both versions and in the detector;
                # this is also the direction actually applied to foot_fk.
                observed=(((feet[0][0]+feet[1][0])/2-ankle[0])*width,
                          ((feet[0][1]+feet[1][1])/2-ankle[1])*height)
                olddir=rm.vec_sub(old['body3d']['toe.'+side],old['body3d']['ankle.'+side])
                newdir=rm.vec_sub(new['body3d']['toe.'+side],new['body3d']['ankle.'+side])
            for key,direction in (('before',olddir),('after',newdir)):
                value=angle(plane(direction),observed)
                if value is not None:
                    errors[key].append(value)
        metrics['direction_error_degrees'][name]={key:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for key,v in errors.items()}
    for joint in ('knee.L','knee.R','ankle.L','ankle.R'):
        # Relative to pelvis removes the intentional grounding convention change.
        differences=[rm.vec_distance(rm.vec_sub(a['body3d'][joint],a['body3d']['pelvis']),
                                     rm.vec_sub(b['body3d'][joint],b['body3d']['pelvis'])) for a,b in zip(before,after)]
        metrics[joint+'_relative_change_m']=dict(median=float(np.median(differences)),max=max(differences))
    metrics['lowest_foot_z_m']=min(p[2] for f in after for n,p in f['body3d'].items() if n.startswith(('heel.','toe.')))
    (root/'comparison.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    colors={'L':'#16aabc','R':'#ed8544'}
    figure,axes=plt.subplots(2,3,figsize=(13,10),layout='constrained')
    video=cv2.VideoCapture(manifest['source']['path'])
    for line,index in enumerate((4,19)):
        video.set(cv2.CAP_PROP_POS_FRAMES,rows[index]['source_index'])
        ok,image=video.read()
        if not ok:
            raise RuntimeError('Cannot decode comparison frame')
        axes[line,0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        axes[line,0].set_title('Source / %.3f s'%after[index]['time'])
        axes[line,0].axis('off')
        for col,frame in ((1,before[index]),(2,after[index])):
            ax=axes[line,col]
            body=frame['body3d']
            for a,b in preview.BODY_EDGES:
                if a in body and b in body:
                    ax.plot([body[a][0],body[b][0]],[body[a][2],body[b][2]],'-o',color=colors.get(b[-1],'#414d65'),lw=2.5,ms=4)
            if col==2:
                for name,q in frame['orientations'].items():
                    anchor=body['head' if name=='head' else 'ankle.'+name[-1]]
                    for vector in ((0,-1,0),(0,0,1)):
                        end=rm.vec_add(anchor,rm.vec_scale(rm.quat_rotate_vector(q,vector),.15 if name=='head' else .10))
                        ax.annotate('',xy=(end[0],end[2]),xytext=(anchor[0],anchor[2]),arrowprops=dict(arrowstyle='->',color=colors.get(name[-1],'#414d65')))
                    if name!='head':
                        toe,heel=body['toe.'+name[-1]],body['heel.'+name[-1]]
                        ax.plot([toe[0],heel[0]],[toe[2],heel[2]],color=colors[name[-1]],lw=3)
            ax.set_aspect('equal');ax.set_xlim(-.55,.55);ax.set_ylim(-.16,1.85)
            ax.axhline(0,color='#cccccc',lw=1);ax.grid(alpha=.15)
            ax.set_title('Before / 0.3.0' if col==1 else 'After / 0.3.1 + orientation axes')
    video.release()
    figure.savefig(root/'comparison.png',dpi=150)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
