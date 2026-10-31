"""Camera4 green-screen segmentation, retaining opaque rotating geometry.

Transparent pedestal areas showing the stationary tabletop are excluded. The
floor boundary and leg windows are estimated separately for every photograph.
Review masks_contact_sheet.jpg before using these masks on other datasets.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('images',type=Path)
parser.add_argument('masks',type=Path)
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--initial-masks',type=Path,required=True)
args=parser.parse_args()
cv2.setNumThreads(4)
args.masks.mkdir(parents=True,exist_ok=True)
args.review.mkdir(parents=True,exist_ok=True)
tiles=[]
records=[]
for path in sorted(args.images.glob('*.jpg')):
    full=cv2.imdecode(np.fromfile(str(path),np.uint8),1)
    height,width=full.shape[:2]
    image=cv2.resize(full,(1200,1600),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    b,g,r=cv2.split(image.astype(float))
    green=((hsv[:,:,0]>=60)&(hsv[:,:,0]<=108)&(hsv[:,:,1]>35)
           &(hsv[:,:,2]<135)&(g>r*1.15)&(b<g*1.5))
    edge_fraction=np.concatenate((green[:,10:170],green[:,1030:1190]),axis=1).mean(axis=1)
    rows=np.flatnonzero((edge_fraction>.6)&(np.arange(1600)>600))
    if not len(rows): raise ValueError(f'Cannot locate green backdrop in {path}')
    floor=int(rows[-1])
    yy,xx=np.indices(green.shape)
    central=(xx>160)&(xx<1040)&(yy>60)
    initial=cv2.imdecode(np.fromfile(str(args.initial_masks/(path.name+'.png')),np.uint8),0)
    initial=cv2.resize(initial,(1200,1600),interpolation=cv2.INTER_NEAREST)>0
    # GrabCut preserves dark clothing better than a global chroma cutoff. Only
    # key the green leg gap below the shorts, where skin and tail colors differ.
    lower_green=((hsv[:,:,0]>=60)&(hsv[:,:,0]<=110)&(hsv[:,:,1]>15)
                 &(hsv[:,:,2]<175)&(yy>780)&(yy<floor))
    denim=(yy>580)&(yy<780)&(hsv[:,:,1]<45)&(hsv[:,:,2]>90)
    above=initial&(~lower_green)&((~green)|denim)&central&(yy<floor-4)
    # Find the leg cross-sections immediately above the backdrop's bottom.
    probe=above[max(0,floor-55):max(1,floor-15)].mean(axis=0)>.5
    edges=np.diff(np.r_[0,probe,0].astype(int))
    starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    leg_window=np.zeros(1200,bool)
    segments=[]
    for left,right in zip(starts,ends):
        if right-left>=8:
            leg_window[max(0,left-6):min(1200,right+6)]=True
            segments.append([int(left),int(right)])
    if not segments: raise ValueError(f'No legs above floor boundary in {path}')
    # The colored opaque graphic supplies rigid features. Filling its exterior
    # retains the white printed decoration without retaining wood seen through
    # the clear acrylic part of the pedestal.
    blue=((hsv[:,:,0]>95)&(hsv[:,:,0]<125)&(hsv[:,:,1]>55)
          &(hsv[:,:,2]>80)&(yy>floor-100)&central).astype(np.uint8)
    blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError(f'No opaque blue base in {path}')
    base=np.zeros_like(blue)
    contour=max(contours,key=cv2.contourArea)
    cv2.drawContours(base,[contour],-1,1,cv2.FILLED)
    wood=((hsv[:,:,0]>=5)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>60)&(hsv[:,:,2]>75))
    lower=((yy>=floor-4)&leg_window[None,:]&(~wood)&initial&central)
    # End leg windows at the pedestal; avoid any stationary material below it.
    bottom=int(np.max(np.flatnonzero(base.any(axis=1))))
    lower &= yy<=bottom
    candidate=above|lower|(base>0)
    # Keep sizable subject pieces, dropping isolated chalk/dust in the screen.
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    ids=[largest]
    mask=np.isin(labels,ids).astype(np.uint8)*255
    output=args.masks/(path.name+'.png')
    cv2.imencode('.png',cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST))[1].tofile(str(output))
    preview=image.copy()
    preview[mask==0]=(225,225,225)
    preview=cv2.resize(preview,(240,320))
    preview=cv2.copyMakeBorder(preview,23,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(preview,path.stem[-6:],(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
    tiles.append(preview)
    records.append({'image':path.name,'floor_row_at_1600':floor,'leg_windows':segments,
                    'foreground_fraction':float((mask>0).mean()),
                    'leg_gap_green_pixels_retained':int(np.count_nonzero((mask>0)&lower_green))})
    print(path.name,'floor',floor,'foreground',round((mask>0).mean(),3),flush=True)
tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%5)])
cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]))[1].tofile(str(args.review/'masks_contact_sheet.jpg'))
(args.review/'mask_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
