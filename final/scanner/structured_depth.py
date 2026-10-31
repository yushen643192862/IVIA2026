"""Metric camera/projector triangulation, with coordinates in the camera frame."""
import cv2
import numpy as np


def depth_gap_cutoff(depths,min_gap_mm=15.):
    """Find a large empty interval past the object's 95th depth percentile."""
    ordered = np.sort(np.asarray(depths,float))
    ordered = ordered[np.isfinite(ordered)]
    if len(ordered)<100:
        return None,None
    lower,upper = np.percentile(ordered,[95,99.95])
    gaps = np.diff(ordered)
    eligible = np.flatnonzero((ordered[:-1]>=lower)&(ordered[1:]<=upper)&(gaps>=min_gap_mm))
    if not len(eligible):
        return None,None
    index = eligible[np.argmax(gaps[eligible])]
    bounds = [float(ordered[index]),float(ordered[index+1])]
    return float(sum(bounds)/2),bounds


def triangulate(camera_pixels, projector_pixels, calibration, min_depth=50., max_depth=3000.,
                max_error=1.5, min_angle_deg=.25, diagnostics=None):
    camera_pixels = np.asarray(camera_pixels,np.float64).reshape(-1,2)
    projector_pixels = np.asarray(projector_pixels,np.float64).reshape(-1,2)
    if camera_pixels.shape!=projector_pixels.shape:
        raise ValueError('Camera/projector correspondence counts differ')
    kc = np.asarray(calibration['camera_matrix'],np.float64)
    kp = np.asarray(calibration['projector_matrix'],np.float64)
    dc = np.asarray(calibration['camera_distortion'],np.float64)
    dp = np.asarray(calibration['projector_distortion'],np.float64)
    rotation = np.asarray(calibration['R_camera_to_projector'],np.float64)
    translation = np.asarray(calibration['t_camera_to_projector_mm'],np.float64).reshape(3)
    if not all(np.isfinite(a).all() for a in (kc,kp,dc,dp,rotation,translation)):
        raise ValueError('Non-finite calibration')
    if kc.shape!=(3,3) or kp.shape!=(3,3) or rotation.shape!=(3,3) or np.linalg.norm(translation)<1e-6:
        raise ValueError('Invalid calibration geometry')
    count = len(camera_pixels)
    xyz = np.full((count,3),np.nan,np.float32)
    errors = np.full(count,np.inf,np.float32)
    valid = np.zeros(count,bool)
    stats = {'input_points':count,'finite_points':0,'positive_depth_both':0,
             'depth_range_pass':0,'reprojection_pass':0,'ray_angle_pass':0}
    depths,angles = [],[]
    p1 = np.column_stack((np.eye(3),np.zeros(3)))
    p2 = np.column_stack((rotation,translation))
    rvec = cv2.Rodrigues(rotation)[0]
    for start in range(0,count,100000):
        end = min(count,start+100000)
        cp,pp = camera_pixels[start:end],projector_pixels[start:end]
        cn = cv2.undistortPoints(cp[:,None,:],kc,dc).reshape(-1,2)
        pn = cv2.undistortPoints(pp[:,None,:],kp,dp).reshape(-1,2)
        homogeneous = cv2.triangulatePoints(p1,p2,cn.T,pn.T).T
        with np.errstate(divide='ignore',invalid='ignore'):
            points = homogeneous[:,:3]/homogeneous[:,3,None]
        finite = np.isfinite(points).all(axis=1)
        safe = np.where(finite[:,None],points,0.)
        in_projector = safe@rotation.T+translation
        camera_fit = cv2.projectPoints(safe,np.zeros(3),np.zeros(3),kc,dc)[0].reshape(-1,2)
        projector_fit = cv2.projectPoints(safe,rvec,translation,kp,dp)[0].reshape(-1,2)
        error = np.maximum(np.linalg.norm(camera_fit-cp,axis=1),np.linalg.norm(projector_fit-pp,axis=1))
        camera_ray = np.column_stack((cn,np.ones(len(cn))))
        projector_ray = np.column_stack((pn,np.ones(len(pn))))@rotation
        cosine = np.sum(camera_ray*projector_ray,axis=1)/(np.linalg.norm(camera_ray,axis=1)*np.linalg.norm(projector_ray,axis=1))
        angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
        good = finite & (points[:,2]>=min_depth) & (points[:,2]<=max_depth) & (in_projector[:,2]>0)
        good &= np.isfinite(error) & (error<=max_error) & (angle>=min_angle_deg)
        stats['finite_points'] += int(finite.sum())
        stats['positive_depth_both'] += int((finite&(points[:,2]>0)&(in_projector[:,2]>0)).sum())
        stats['depth_range_pass'] += int((finite&(points[:,2]>=min_depth)&(points[:,2]<=max_depth)).sum())
        stats['reprojection_pass'] += int((finite&np.isfinite(error)&(error<=max_error)).sum())
        stats['ray_angle_pass'] += int((finite&(angle>=min_angle_deg)).sum())
        if diagnostics is not None:
            depths.append(points[finite,2]); angles.append(angle[finite])
        xyz[start:end] = points
        errors[start:end] = error
        valid[start:end] = good
    if diagnostics is not None:
        def percentiles(values):
            values = np.asarray(values)
            values = values[np.isfinite(values)]
            return np.percentile(values,[0,25,50,75,100]).tolist() if len(values) else []
        stats.update(kept_points=int(valid.sum()),percentile_order=[0,25,50,75,100],
            depth_mm_percentiles=percentiles(np.concatenate(depths) if depths else []),
            reprojection_error_px_percentiles=percentiles(errors),
            reprojection_error_p95_px=float(np.percentile(errors[np.isfinite(errors)],95)) if np.isfinite(errors).any() else None,
            ray_angle_deg_percentiles=percentiles(np.concatenate(angles) if angles else []),
            note='Each pass count is independent, not a sequential rejection count.')
        diagnostics.update(stats)
    return xyz,valid,errors
