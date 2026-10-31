import numpy as np
import cv2


def face_geometry(vertices,faces):
    tri=vertices[faces]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    lengths=np.linalg.norm(cross,axis=1)
    return tri.mean(axis=1),cross/np.maximum(lengths[:,None],1e-12),lengths/2


def ray_first_hit(origins,directions,vertices,faces,chunk=32):
    """Two-sided Moller-Trumbore intersections; no invented missing surfaces."""
    tri=vertices[faces]
    result=[]
    for start in range(0,len(origins),chunk):
        o=origins[start:start+chunk,None,:]; d=directions[start:start+chunk,None,:]
        closest=np.full(len(o),np.inf)
        for begin in range(0,len(tri),1024):
            block=tri[begin:begin+1024]; e1=block[:,1]-block[:,0]; e2=block[:,2]-block[:,0]
            h=np.cross(d,e2[None,:,:]); det=np.sum(e1[None,:,:]*h,axis=2)
            inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-10)
            s=o-block[None,:,0,:]; u=np.sum(s*h,axis=2)*inv
            q=np.cross(s,e1[None,:,:]); v=np.sum(d*q,axis=2)*inv
            t=np.sum(e2[None,:,:]*q,axis=2)*inv
            valid=(np.abs(det)>1e-10)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-8)
            closest=np.minimum(closest,np.min(np.where(valid,t,np.inf),axis=1))
        result.extend(closest.tolist())
    return np.asarray(result)


def visible(points,normals,camera,vertices,faces):
    R=np.asarray(camera['R']); t=np.asarray(camera['t']); K=np.asarray(camera['K'])
    C=-R.T@t; cp=points@R.T+t
    uv,_=cv2.projectPoints(points,np.asarray(cv2.Rodrigues(R)[0]),t,K,
                           np.asarray(camera.get('distortion',[0,0,0,0,0]),float))
    uv=uv.reshape(-1,2); w,h=camera['size']
    to_cam=C-points; distance=np.linalg.norm(to_cam,axis=1)
    toward=to_cam/np.maximum(distance[:,None],1e-12)
    # Mesh normals must be oriented consistently; two-sided incidences tolerate imported winding.
    incidence=np.abs(np.sum(normals*toward,axis=1))
    valid=(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&(incidence>.25)
    origins=np.repeat(C[None,:],len(points),axis=0); rays=-toward
    hit=ray_first_hit(origins,rays,vertices,faces)
    tolerance=max(np.linalg.norm(np.ptp(vertices,axis=0))*1e-4,1e-8)
    valid &= np.abs(hit-distance)<=tolerance
    return valid,toward,uv


def rotation(axis,angle):
    axis=np.asarray(axis,float); norm=np.linalg.norm(axis)
    if norm<1e-9: raise ValueError('Rotation axis cannot be zero')
    return cv2.Rodrigues(axis/norm*np.deg2rad(angle))[0]


def axis_transform(axis,pivot,angle):
    R=rotation(axis,angle); pivot=np.asarray(pivot,float)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=pivot-R@pivot
    return T


def platform_pose(config,yaw,pitch):
    # Inner pitch stage followed by outer yaw stage; pivots in the zero-pose platform frame.
    return axis_transform(config['yaw_axis'],config['yaw_pivot'],yaw) @ axis_transform(
        config['pitch_axis'],config['pitch_pivot'],pitch) @ np.asarray(config['platform_from_model'])


def transform(points,T):
    return points@T[:3,:3].T+T[:3,3]
