from pathlib import Path
import numpy as np
import cv2
from .io import write_ply,save_json
from .geometry import face_geometry


def look_at(center,target):
    center=np.asarray(center,float); forward=np.asarray(target,float)-center
    forward/=np.linalg.norm(forward); right=np.cross(forward,[0,0,1]); right/=np.linalg.norm(right)
    down=np.cross(forward,right); R=np.stack([right,down,forward])
    return {'R':R.tolist(),'t':(-R@center).tolist(),'center':center.tolist(),
            'K':[[650,0,400],[0,650,300],[0,0,1]],'size':[800,600],'distortion':[0]*5}


def make_demo(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    # Hollow cup profile: outer base -> outer wall -> lip -> inner wall -> inner floor.
    profile=[(.2,0),(1,0),(1,.8),(1,1.6),(1,2.2),(.78,2.2),(.78,1.6),(.78,.8),(.78,.25),(.2,.25)]
    n=32; vertices=np.array([[r*np.cos(a),r*np.sin(a),z] for r,z in profile for a in np.linspace(0,2*np.pi,n,endpoint=False)])
    faces=[]
    for j in range(len(profile)-1):
        for i in range(n):
            k=(i+1)%n; a=j*n+i; b=j*n+k; c=(j+1)*n+i; d=(j+1)*n+k
            faces.extend([[a,b,c],[b,d,c]])
    # Close base and inner floor with centers, leaving only an intentional damaged inner-wall patch.
    bottom=len(vertices); vertices=np.vstack([vertices,[0,0,0],[0,0,.25]])
    for i in range(n):
        k=(i+1)%n; faces.extend([[bottom,k,i],[bottom+1,9*n+i,9*n+k]])
    full=np.asarray(faces,int)
    centroids,normals,areas=face_geometry(vertices,full)
    remove=(np.arange(len(full))>=6*n*2)&(np.arange(len(full))<8*n*2)&(centroids[:,0]>.3)&(centroids[:,1]>.1)
    observed=full[~remove]
    # Dense evidence is deliberately absent on floor and sparse on an inner patch.
    rng=np.random.default_rng(7); points=[]
    for i,face in enumerate(full):
        center=centroids[i]
        if remove[i] or (.1<center[2]<.4 and np.linalg.norm(center[:2])<.8): continue
        count=2 if center[0]<-.3 and .3<center[2]<1.5 else 10
        u=rng.random((count,2)); flip=u.sum(axis=1)>1; u[flip]=1-u[flip]
        tri=vertices[face]; points.extend(tri[0]+u[:,0,None]*(tri[1]-tri[0])+u[:,1,None]*(tri[2]-tri[0]))
    write_ply(output/'mesh.ply',vertices,observed)
    write_ply(output/'dense.ply',np.asarray(points))
    write_ply(output/'sparse.ply',np.asarray(points)[::80])
    cameras=[]
    for i,a in enumerate([0,120,240]):
        c=look_at([5*np.cos(np.deg2rad(a)),5*np.sin(np.deg2rad(a)),1.2],[0,0,1.1])
        c['file']=f'synthetic_phone{i+1}.png'; cameras.append(c)
        image=np.full((600,800,3),245,np.uint8)
        uv,_=cv2.projectPoints(vertices,cv2.Rodrigues(np.asarray(c['R']))[0],np.asarray(c['t']),np.asarray(c['K']),np.zeros(5))
        uv=np.round(uv.reshape(-1,2)).astype(int)
        depth=(vertices@np.asarray(c['R']).T+np.asarray(c['t']))[observed,2].mean(axis=1)
        for fi in np.argsort(depth)[::-1]:
            shade=100+int(80*abs(normals[~remove][fi,2])); cv2.fillConvexPoly(image,uv[observed[fi]],(shade,shade,220))
        cv2.putText(image,'SYNTHETIC DEMO - NOT PHOTOGRAMMETRY',(10,25),0,.5,(0,0,0),1)
        cv2.imencode('.png',image)[1].tofile(str(output/c['file']))
    save_json(output/'cameras.json',cameras)
    config={'calibrated':False,'demo':True,'yaw_axis':[0,0,1],'yaw_pivot':[0,0,0],
            'pitch_axis':[1,0,0],'pitch_pivot':[0,0,1.1], 'platform_from_model':np.eye(4).tolist(),
            'yaw_candidates':list(range(0,360,45)),'pitch_candidates':[-40,0,40], 'pitch_limits':[-45,45],
            'current_pose':{'yaw':0,'pitch':0},'captured_poses':[{'yaw':0,'pitch':0}],
            'cameras':cameras,'fixture_mesh':None}
    save_json(output/'platform_demo.json',config)
    return output/'mesh.ply',output/'dense.ply',cameras,config
