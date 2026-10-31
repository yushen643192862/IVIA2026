"""Compare unchanged-pose depth support in a manually checked upper-back ROI."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scanner.io import save_json,read_ply
from scanner.reconstruct import read_model

def read_depth(path):
    with path.open('rb') as stream:
        header=bytearray()
        while header.count(b'&')<3:
            char=stream.read(1)
            if not char: raise ValueError('Truncated depth file')
            header.extend(char)
        w,h,c=map(int,header.decode().strip('&').split('&'))
        return np.frombuffer(stream.read(),np.float32).reshape((w,h,c),order='F').transpose(1,0,2).squeeze()

parser=argparse.ArgumentParser()
parser.add_argument('baseline',type=Path)
parser.add_argument('candidate',type=Path)
parser.add_argument('--candidate-cloud',type=Path)
parser.add_argument('--output-prefix',default='back')
parser.add_argument('--depth-only',action='store_true',help='Compare available depth before point-cloud fusion finishes')
args=parser.parse_args()
name='IMG_20261024_194558.jpg'
original={c['file']:c for c in json.loads((args.baseline/'cameras.json').read_text(encoding='utf8'))}[name]
view={c['file']:c for c in read_model(args.baseline/'dense/model_text')}[name]
mask=cv2.imdecode(np.fromfile(original['object_mask'],np.uint8),0)
# A skin-only patch on this photograph, checked visually: below the neck and
# above the upper black strap. Coordinates are in the original 3072x4096 image.
roi=np.zeros_like(mask)
roi[950:1100,1450:1650]=255
region=np.zeros_like(mask)
region[750:1550,1100:1950]=255
roi[mask==0]=0
maps=cv2.initUndistortRectifyMap(np.array(original['K']),np.array(original['distortion']),
    np.eye(3),np.array(view['K']),tuple(view['size']),cv2.CV_32FC1)
roi=cv2.remap(roi,*maps,cv2.INTER_NEAREST)>0
region=cv2.remap(region,*maps,cv2.INTER_NEAREST)>0
object_mask=cv2.remap(mask,*maps,cv2.INTER_NEAREST)>0
y,x=np.nonzero(region)
box=(slice(y.min(),y.max()+1),slice(x.min(),x.max()+1))
image=cv2.imdecode(np.fromfile(str(args.baseline/'dense/images'/name),np.uint8),1)
panels=[]
report={'reference':name,'original_roi_xyxy':[1450,950,1650,1100],
    'roi_description':'skin-only upper back, below neck and above upper black strap',
    'roi_pixels':int(roi.sum()),'note':'Retained valid-depth fraction, not accuracy. Same cameras, masks, image pixels and resolution.'}
source=image.copy()
source[roi]=(source[roi]*.4+np.array([255,170,0])*.6).astype(np.uint8)
panels.append(('Skin ROI (blue)',source))
for label,folder in [('Baseline',args.baseline),('Back refinement',args.candidate)]:
    depth=read_depth(folder/'dense/stereo/depth_maps'/(name+'.geometric.bin'))
    valid=(depth>0)&np.isfinite(depth)
    report[label]={'upper_back_valid_fraction':float((valid&roi).sum()/roi.sum()),
        'foreground_valid_fraction':float((valid&object_mask).sum()/object_mask.sum())}
    overlay=image.copy()
    overlay[object_mask&~valid]=(overlay[object_mask&~valid]*.35+np.array([30,30,220])*.65).astype(np.uint8)
    overlay[object_mask&valid]=(overlay[object_mask&valid]*.5+np.array([40,190,40])*.5).astype(np.uint8)
    panels.append((label,overlay))
tiles=[]
for label,panel in panels:
    tile=cv2.resize(panel[box],(510,480))
    tile=cv2.copyMakeBorder(tile,45,40,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(tile,label,(15,30),0,.7,(25,25,25),2)
    cv2.putText(tile,'Green: valid depth; red: missing',(10,550),0,.5,(25,25,25),1)
    tiles.append(tile)
cv2.imencode('.png',np.hstack(tiles))[1].tofile(str(args.candidate/(args.output_prefix+'_depth_comparison.png')))
save_json(args.candidate/(args.output_prefix+'_depth_comparison.json'),report)
print(json.dumps(report,indent=2))
if args.depth_only: sys.exit(0)
cloud_panels=[]
for label,folder in [('Baseline back points',args.baseline),('Refined back points',args.candidate)]:
    cloud=folder/'dense_clean.ply'
    if not cloud.exists(): cloud=folder/'dense.ply'
    if folder==args.candidate and args.candidate_cloud: cloud=args.candidate_cloud
    points,_,attrs=read_ply(cloud,True)
    colors=np.column_stack([attrs[k] for k in ('red','green','blue')]).astype(np.uint8)
    normals=np.column_stack([attrs[k] for k in ('nx','ny','nz')])
    R=np.array(view['R']); t=np.array(view['t']); K=np.array(view['K'])
    xyz=points@R.T+t
    uv=(xyz@K.T)[:,:2]/xyz[:,2,None]
    xy=np.rint(np.where(np.isfinite(uv),uv,-1)).astype(int)
    facing=np.sum(normals*(np.array(view['center'])-points),axis=1)>0
    good=(xyz[:,2]>0)&facing&(xy[:,0]>=0)&(xy[:,0]<view['size'][0])&(xy[:,1]>=0)&(xy[:,1]<view['size'][1])
    ids=np.flatnonzero(good)
    depth=read_depth(folder/'dense/stereo/depth_maps'/(name+'.geometric.bin'))
    observed=depth[xy[ids,1],xy[ids,0]]
    ids=ids[(observed>0)&(np.abs(observed-xyz[ids,2])<.03*xyz[ids,2])]
    canvas=np.full((800,600,3),245,np.uint8)
    for i in ids[np.argsort(-xyz[ids,2])]:
        x,y=np.rint(uv[i]*[600/view['size'][0],800/view['size'][1]]).astype(int)
        cv2.circle(canvas,(int(x),int(y)),1,tuple(int(c) for c in colors[i,::-1]),-1)
    canvas=cv2.copyMakeBorder(canvas,45,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(canvas,label,(12,30),0,.65,(25,25,25),2)
    cloud_panels.append(canvas)
photo=cv2.copyMakeBorder(cv2.resize(image,(600,800)),45,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
cv2.putText(photo,'Original photograph',(12,30),0,.65,(25,25,25),2)
cv2.imencode('.png',np.hstack([photo,*cloud_panels]))[1].tofile(str(args.candidate/(args.output_prefix+'_points_comparison.png')))
