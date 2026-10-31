"""Robust multiscale point-to-plane refinement of an existing rigid alignment."""
import cv2
import numpy as np
from scipy.spatial import cKDTree


def moved(points,transform):
    return points@transform[:3,:3].T+transform[:3,3]


def surface_sample(points,voxel_mm,max_points,seed):
    cells = np.floor(points/voxel_mm).astype(np.int32)
    _,indices = np.unique(cells,axis=0,return_index=True)
    if len(indices)>max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices,max_points,replace=False)
    sample = points[indices]
    neighbors = cKDTree(points).query(sample,k=min(24,len(points)),workers=2)[1]
    neighbor_points = points[neighbors]
    centered = neighbor_points-neighbor_points.mean(axis=1,keepdims=True)
    covariance = np.einsum('nki,nkj->nij',centered,centered)
    normal = np.linalg.eigh(covariance)[1][:,:,0]
    normal[np.sum(normal*sample,axis=1)>0] *= -1
    return sample,normal


def correspond(source,source_normal,target,target_normal,transform,max_distance_mm,normal_cos=.65,target_tree=None):
    moved_source = moved(source,transform)
    distance,index = (target_tree if target_tree is not None else cKDTree(target)).query(moved_source,k=1,workers=2)
    transformed_normal = source_normal@transform[:3,:3].T
    compatibility = np.sum(transformed_normal*target_normal[index],axis=1)
    good = (distance<max_distance_mm)&(compatibility>normal_cos)
    ids = np.flatnonzero(good)
    if not len(ids):
        return ids,index,np.array([]),distance
    residual = np.sum((moved_source[ids]-target[index[ids]])*target_normal[index[ids]],axis=1)
    # Trim wrong nearest-neighbor matches on silhouettes and thin surfaces.
    bound = min(max_distance_mm*.55,float(np.percentile(np.abs(residual),80)))
    ids = ids[np.abs(residual)<=max(.25,bound)]
    residual = np.sum((moved_source[ids]-target[index[ids]])*target_normal[index[ids]],axis=1)
    return ids,index,residual,distance


def refine_pair(source_points,target_points,initial,seed=0,source_samples=None,target_samples=None):
    transform = initial.copy()
    stages = [(5.,10.,7),(2.,5.,7),(1.,2.5,9)]
    initial_quality = None
    final_data = None
    history = []
    for stage,(voxel,max_distance,iterations) in enumerate(stages):
        source,source_normal = source_samples[stage] if source_samples is not None else surface_sample(source_points,voxel,9000,seed+stage)
        target,target_normal = target_samples[stage] if target_samples is not None else surface_sample(target_points,voxel,14000,seed+stage+100)
        target_tree = cKDTree(target)
        for iteration in range(iterations):
            ids,index,residual,distance = correspond(source,source_normal,target,target_normal,transform,max_distance,target_tree=target_tree)
            if initial_quality is None:
                initial_quality = {'overlap_fraction':len(ids)/len(source),'median_plane_mm':float(np.median(np.abs(residual))) if len(residual) else None}
            if len(ids)<max(100,len(source)*.08):
                return None,{'accepted':False,'reason':'insufficient oriented surface overlap','stage_voxel_mm':voxel,
                             'correspondences':len(ids),'source_samples':len(source),'initial':initial_quality}
            moved_source = moved(source[ids],transform)
            center = np.mean(moved_source,axis=0)
            normals = target_normal[index[ids]]
            jacobian = np.column_stack((np.cross(moved_source-center,normals),normals))
            # A weak prior prevents degenerate planar patches from causing huge motion.
            regularizer = np.diag([1.5,1.5,1.5,.02,.02,.02])
            update = np.linalg.lstsq(np.vstack((jacobian,regularizer)),
                                      np.r_[-residual,np.zeros(6)],rcond=None)[0]
            rotation_step = update[:3]
            translation_step = update[3:]
            rotation_norm = np.linalg.norm(rotation_step)
            if rotation_norm>np.radians(2):
                rotation_step *= np.radians(2)/rotation_norm
            translation_norm = np.linalg.norm(translation_step)
            if translation_norm>3.:
                translation_step *= 3./translation_norm
            incremental = cv2.Rodrigues(rotation_step)[0]
            transform[:3,3] = incremental@(transform[:3,3]-center)+center+translation_step
            transform[:3,:3] = incremental@transform[:3,:3]
            if np.linalg.norm(rotation_step)<1e-5 and np.linalg.norm(translation_step)<.01:
                break
        ids,index,residual,distance = correspond(source,source_normal,target,target_normal,transform,max_distance,target_tree=target_tree)
        history.append({'voxel_mm':voxel,'overlap_fraction':len(ids)/len(source),
                        'median_plane_mm':float(np.median(np.abs(residual))) if len(residual) else None,
                        'iterations':iteration+1})
        final_data = (source,target,target_normal,ids,index,residual)
    source,target,target_normal,ids,index,residual = final_data
    if len(ids)<max(150,len(source)*.12):
        return None,{'accepted':False,'reason':'final overlap too small','history':history,'initial':initial_quality}
    median = float(np.median(np.abs(residual)))
    if median>1.:
        return None,{'accepted':False,'reason':'final surface distance exceeds 1 mm','history':history,'initial':initial_quality}
    # Evenly sample dense correspondences so one broad patch cannot dominate the graph.
    choice = np.linspace(0,len(ids)-1,min(100,len(ids)),dtype=int)
    selected = ids[choice]
    return (transform,source[selected],target[index[selected]],target_normal[index[selected]]),{
        'accepted':True,'initial':initial_quality,'history':history,
        'median_final_plane_mm':median,'inliers':len(ids)}
