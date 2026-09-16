"""Source, old rear view, new rear view and new side view, all 192 frames."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import preview


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    import cv2
    import numpy as np
    folder=ROOT/'.cache/v03-camera'
    old=read(ROOT/'.cache/v03-orientations/capture_quality/mocap_result.json')['frames']
    new=read(folder/'mocap_result.json')['frames']
    manifest=read(folder/'preview_manifest.json')
    source=cv2.VideoCapture(manifest['source']['path'])
    video=cv2.VideoWriter(str(folder/'camera-comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(1280,800))
    if not source.isOpened() or not video.isOpened():
        raise RuntimeError('Cannot open source or output video')
    def text(image,title,x,y,color=(48,44,35)):
        cv2.putText(image,title,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.62,color,1,cv2.LINE_AA)
    def skeleton(image,frame,x,y,yaw,title):
        text(image,title,x+20,y+29)
        body=frame['body3d']
        root=body['pelvis']
        def project(p):
            px,py=p[0]-root[0],p[1]-root[1]
            horizontal=math.cos(yaw)*px+math.sin(yaw)*py
            return (int(x+320+horizontal*205),int(y+370-p[2]*205))
        cv2.line(image,(x+50,y+370),(x+590,y+370),(209,208,207),1)
        for a,b in preview.BODY_EDGES:
            if a in body and b in body:
                color=(189,163,15) if b.endswith('.L') else (64,135,237) if b.endswith('.R') else (92,72,53)
                cv2.line(image,project(body[a]),project(body[b]),color,3,cv2.LINE_AA)
                cv2.circle(image,project(body[b]),4,color,-1,cv2.LINE_AA)
        if abs(yaw-math.pi/2)<.01:
            cv2.arrowedLine(image,(x+560,y+345),(x+480,y+345),(189,163,15),2,cv2.LINE_AA,tipLength=.15)
            text(image,'-Y forward',x+465,y+330)
        text(image,'L: cyan   R: orange',x+20,y+390)
    snapshots=[]
    for index,frame in enumerate(new):
        source.set(cv2.CAP_PROP_POS_FRAMES,manifest['frames'][index]['source_index'])
        ok,reference=source.read()
        if not ok:
            raise RuntimeError('Missing reference frame '+str(index))
        canvas=np.full((800,1280,3),249,dtype=np.uint8)
        factor=min(620/reference.shape[1],340/reference.shape[0])
        ref=cv2.resize(reference,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        xx=(640-ref.shape[1])//2; yy=45+(340-ref.shape[0])//2
        canvas[yy:yy+ref.shape[0],xx:xx+ref.shape[1]]=ref
        text(canvas,'Source - fixed left front 45 deg',20,29)
        text(canvas,'Frame %d / 192   %.3f s'%(frame['frame'],frame['time']),20,395)
        skeleton(canvas,old[index],640,0,math.pi,'v0.3.1 - rear (camera axes)')
        skeleton(canvas,frame,0,400,math.pi,'v0.3.2 - rear (character axes)')
        skeleton(canvas,frame,640,400,math.pi/2,'v0.3.2 - side (character axes)')
        cv2.line(canvas,(640,0),(640,800),(225,225,225),1)
        cv2.line(canvas,(0,400),(1280,400),(225,225,225),1)
        video.write(canvas)
        if index in (4,19):
            target=folder/('comparison-frame-%d.png'%frame['frame'])
            cv2.imwrite(str(target),canvas)
            snapshots.append(str(target))
    source.release(); video.release()
    decoded=cv2.VideoCapture(str(folder/'camera-comparison.mp4'))
    count=int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)); decoded.release()
    assert count==192,count
    print(json.dumps(dict(video=str(folder/'camera-comparison.mp4'),frames=count,snapshots=snapshots)))


if __name__=='__main__':
    main()
