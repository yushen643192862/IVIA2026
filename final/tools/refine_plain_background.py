"""Refine Camera3 masks using the visible cardboard at both image margins."""
import argparse
from pathlib import Path
import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,default=0)
args=parser.parse_args()
cv2.setNumThreads(4)
root=Path(__file__).resolve().parents[1]
output=root/'Camera3_masks_refined'
output.mkdir(exist_ok=True)
tiles=[]
paths=sorted((root/'Camera3_masks').glob('*.png'))
if args.limit: paths=paths[:args.limit]
for i,path in enumerate(paths):
    im=cv2.imdecode(np.fromfile(str(root/'Camera3'/path.name[:-4]),np.uint8),1)
    original_size=(im.shape[1],im.shape[0])
    im=cv2.resize(im,(900,1200),interpolation=cv2.INTER_AREA)
    mask=cv2.imdecode(np.fromfile(str(path),np.uint8),0)
    mask=cv2.resize(mask,(900,1200),interpolation=cv2.INTER_NEAREST)
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
    # The subject is central; these bands contain only static background.
    sample=np.concatenate((lab[:,15:125],lab[:,790:885]),axis=1)
    background=cv2.GaussianBlur(np.median(sample,axis=1).astype(np.float32),(1,31),0)
    # Background and shaded skin overlap in color. Do not apply a global color
    # cutoff: locate the gap between bright leg interiors instead.
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    bg_value=np.median(np.concatenate((hsv[:,15:125,2],hsv[:,790:885,2]),axis=1),axis=1)
    yy=np.arange(1200)[:,None]
    blue=(hsv[:,:,0]>80)&(hsv[:,:,0]<125)&(hsv[:,:,1]>65)&(yy>790)&(mask>0)
    rows=np.flatnonzero(blue.sum(axis=1)>15)
    base_top=int(rows[0]) if len(rows) else 980
    skin=(hsv[:,:,2]>bg_value[:,None]+18)&(hsv[:,:,1]<85)&(hsv[:,:,0]<40)&(mask>0)
    for row in range(670,base_top):
        edges=np.diff(np.r_[0,skin[row],0].astype(int))
        starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
        segments=[(a,b) for a,b in zip(starts,ends) if b-a>=5 and a>180 and b<720]
        if len(segments)>=2:
            left,right=segments[0],segments[-1]
            if 8<right[0]-left[1]<250:
                a,b=left[1]+2,right[0]-2
                gap=mask[row,a:b]
                gap[hsv[row,a:b,2]>80]=0
    binary=cv2.morphologyEx((mask>0).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,components,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if n<2: raise ValueError('Empty segmentation')
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    # Preserve small disconnected hair tips; drop tiny background specks.
    keep_ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=40)
    keep_ids=keep_ids[keep_ids!=0]
    binary=np.isin(components,keep_ids).astype(np.uint8)*255
    # Horizontal border components can be large; remove components that do not
    # belong to the central subject and touch either background margin.
    for label in keep_ids:
        x,y,w,h,area=stats[label]
        if label!=largest and (x<140 or x+w>760): binary[components==label]=0
    cv2.imencode('.png',cv2.resize(binary,original_size,interpolation=cv2.INTER_NEAREST))[1].tofile(str(output/path.name))
    if i%5==0 or args.limit:
        im[binary==0]=(225,225,225)
        tiles.append(cv2.resize(im,(180,240)))
    print(path.name,flush=True)
if tiles:
    cv2.imencode('.jpg',np.hstack(tiles))[1].tofile(str(root/'outputs/camera3_scan_cuda/refined_masks_preview.jpg'))
