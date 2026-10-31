"""Report base planarity and proportions without claiming metric accuracy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scanner.io import read_ply,write_ply,save_json

parser=argparse.ArgumentParser()
parser.add_argument('folder',type=Path)
args=parser.parse_args()
root=args.folder
p,_,a=read_ply(root/'sparse.ply',True)
colors=np.column_stack([a[k] for k in ('red','green','blue')]).astype(np.uint8)
dist,_=cKDTree(p).query(p,k=9)
keep=dist[:,-1]<np.median(dist[:,-1])*5
core=p[keep]; rgb=colors[keep]
hsv=cv2.cvtColor(rgb[None,:,:],cv2.COLOR_RGB2HSV)[0]
blue=(hsv[:,0]>90)&(hsv[:,0]<125)&(hsv[:,1]>50)&(hsv[:,2]>90)
samples=core[blue]
rng=np.random.default_rng(7)
threshold=np.linalg.norm(np.percentile(core,95,axis=0)-np.percentile(core,5,axis=0))*.008
best=np.zeros(len(samples),bool)
for _ in range(500):
    tri=samples[rng.choice(len(samples),3,replace=False)]
    normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    length=np.linalg.norm(normal)
    if length<1e-9: continue
    normal/=length
    inside=np.abs((samples-tri[0])@normal)<threshold
    if inside.sum()>best.sum(): best=inside
base=samples[best]
center=np.mean(base,axis=0)
_,_,vt=np.linalg.svd(base-center,full_matrices=False)
normal=vt[-1]
body_height=(core-center)@normal
if abs(np.percentile(body_height,5))>abs(np.percentile(body_height,95)):
    normal=-normal
    body_height=-body_height
xy=(base-center)@vt[:2].T
boundary=cv2.convexHull(xy.astype(np.float32)).reshape(-1,2).astype(float)
extent=np.ptp(xy,axis=0).max()
best_circle=np.zeros(len(boundary),bool)
for _ in range(500):
    tri=boundary[rng.choice(len(boundary),3,replace=False)]
    matrix=2*(tri[1:]-tri[0])
    if abs(np.linalg.det(matrix))<1e-9: continue
    c=np.linalg.solve(matrix,np.sum(tri[1:]**2,axis=1)-np.sum(tri[0]**2))
    r=np.linalg.norm(tri[0]-c)
    if not .3*extent<r<1.5*extent: continue
    inside=np.abs(np.linalg.norm(boundary-c,axis=1)-r)<r*.04
    if inside.sum()>best_circle.sum(): best_circle=inside
rim=boundary[best_circle] if best_circle.sum()>=3 else boundary
fit=np.linalg.lstsq(np.column_stack((2*rim,np.ones(len(rim)))),np.sum(rim**2,axis=1),rcond=None)[0]
radius=float(np.sqrt(fit[2]+np.sum(fit[:2]**2)))
circle_error=np.abs(np.linalg.norm(rim-fit[:2],axis=1)-radius)
angles=np.sort(np.arctan2(rim[:,1]-fit[1],rim[:,0]-fit[0]))
coverage=float(360-np.degrees(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))))
report={'input_sparse_points':len(p),'local_density_retained_points':len(core),
    'blue_points':len(samples),'base_plane_inliers':len(base),
    'blue_plane_inlier_fraction':len(base)/len(samples),
    'base_plane_median_distance_sfm':float(np.median(np.abs((base-center)@normal))),
    'base_radius_sfm':radius,'base_circle_median_radial_error_fraction':float(np.median(circle_error)/radius),
    'base_circle_p95_radial_error_fraction':float(np.percentile(circle_error,95)/radius),
    'base_boundary_points':len(boundary),'base_circle_boundary_inliers':len(rim),
    'base_boundary_angular_coverage_degrees':coverage,
    'body_height_p95_over_base_diameter':float(np.percentile(body_height,95)/(2*radius)),
    'note':'Base color, plane selection and convex-hull circle RANSAC are heuristics. Circle residuals apply only to selected boundary inliers; they are not physical accuracy measurements. No metric ground truth.'}
save_json(root/'sparse_geometry_review.json',report)
write_ply(root/'sparse_review_core.ply',core,colors=rgb)
# Render the same retained samples in a base-aligned frame, keeping real scale
# across views. This preview omits statistical outliers, not raw reconstruction.
aligned=np.column_stack(((core-center)@vt[0],body_height,(core-center)@vt[1]))
canvas=np.full((850,1800,3),245,np.uint8)
for panel,(u,v,depth) in enumerate(((0,1,2),(2,1,0),(0,2,1))):
    uv=aligned[:,[u,v]].copy()
    if v==1: uv[:,1]*=-1
    lo,hi=np.percentile(uv,[0,100],axis=0)
    scale=min(530/max(hi[0]-lo[0],1e-6),720/max(hi[1]-lo[1],1e-6))
    pixel=(uv-(lo+hi)/2)*scale+[panel*600+300,430]
    for i in np.argsort(aligned[:,depth]):
        x,y=np.rint(pixel[i]).astype(int)
        cv2.circle(canvas,(int(x),int(y)),1,tuple(int(c) for c in rgb[i,::-1]),-1)
    cv2.putText(canvas,('Front/side','Side/front','Base plane')[panel],(panel*600+20,35),0,.7,(25,25,25),2)
cv2.imencode('.png',canvas)[1].tofile(str(root/'sparse_geometry_preview.png'))
print(json.dumps(report,indent=2))
