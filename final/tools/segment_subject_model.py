"""Local U2Net segmentation, refined at higher resolution with GrabCut.

Uses the neural model for foreground/background seeds so that similarly colored
clothing and backdrop do not require a global color threshold.
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('images',type=Path)
parser.add_argument('masks',type=Path)
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--green-turntable',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
os.environ['U2NET_HOME']=str(root/'tools/models/rembg')
os.environ['OMP_NUM_THREADS']='4'
from rembg import new_session
from PIL import Image
session=new_session('u2net',providers=['CPUExecutionProvider'])
cv2.setNumThreads(4)
args.masks.mkdir(parents=True,exist_ok=True)
args.review.mkdir(parents=True,exist_ok=True)
paths=sorted(args.images.glob('*.jpg'))
if args.limit: paths=paths[:args.limit]
tiles=[]
for path in paths:
    full=cv2.imdecode(np.fromfile(str(path),np.uint8),1)
    size=(full.shape[1],full.shape[0])
    image=cv2.resize(full,(1200,1600),interpolation=cv2.INTER_AREA)
    alpha=np.array(session.predict(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))[0])
    labels=np.where(alpha>100,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    labels[alpha<8]=cv2.GC_BGD
    core=cv2.erode((alpha>245).astype(np.uint8),np.ones((5,5),np.uint8))>0
    labels[core]=cv2.GC_FGD
    cv2.grabCut(image,labels,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
    mask=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    if args.green_turntable:
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        yy,xx=np.indices(mask.shape)
        screen=(hsv[:,:,0]>60)&(hsv[:,:,0]<110)&(hsv[:,:,1]>20)&(hsv[:,:,2]<160)
        edge=np.concatenate((screen[:,10:170],screen[:,1030:1190]),axis=1).mean(axis=1)
        rows=np.flatnonzero((edge>.6)&(np.arange(1600)>600))
        floor=int(rows[-1])
        blue=((hsv[:,:,0]>95)&(hsv[:,:,0]<125)&(hsv[:,:,1]>55)&(hsv[:,:,2]>80)&(yy>floor-100)).astype(np.uint8)
        blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        contours,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        base=np.zeros_like(blue)
        cv2.drawContours(base,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        probe=(mask[max(0,floor-60):floor-20]>0).mean(axis=0)>.5
        edges=np.diff(np.r_[0,probe,0].astype(int))
        starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
        windows=np.zeros(1200,bool)
        for left,right in zip(starts,ends):
            if right-left>=8:
                windows[max(0,left-4):min(1200,right+4)]=True
        bottom=int(np.flatnonzero(base.any(axis=1))[-1])
        lower=(yy>=floor-3)&(yy<=bottom)&windows[None,:]&(mask>0)
        mask=(((mask>0)&(yy<floor-3))|lower|(base>0)).astype(np.uint8)*255
    cv2.imencode('.png',cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST))[1].tofile(str(args.masks/(path.name+'.png')))
    preview=image.copy()
    preview[mask==0]=(225,225,225)
    tile=cv2.copyMakeBorder(cv2.resize(preview,(240,320)),23,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(tile,path.stem[-6:],(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
    tiles.append(tile)
    print(path.name,'foreground',round((mask>0).mean(),3),flush=True)
tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%5)])
cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]))[1].tofile(str(args.review/'model_masks_contact_sheet.jpg'))
