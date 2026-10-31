"""Register rotated-object depth views with image features, metric RANSAC and a loop pose graph."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json,write_ply
from scanner.dense_registration import refine_pair,surface_sample
from tools.decode_capture_session import read_image


def rigid(source,target):
    a,b = source.mean(axis=0),target.mean(axis=0)
    u,_,vt = np.linalg.svd((source-a).T@(target-b))
    rotation = vt.T@u.T
    if np.linalg.det(rotation)<0:
        vt[-1] *= -1; rotation = vt.T@u.T
    transform = np.eye(4)
    transform[:3,:3] = rotation; transform[:3,3] = b-rotation@a
    return transform


def moved(points,pose):
    return points@pose[:3,:3].T+pose[:3,3]


def estimate(source,target,rng):
    best = np.zeros(len(source),bool)
    for _ in range(800):
        ids = rng.choice(len(source),3,replace=False)
        if np.linalg.norm(np.cross(source[ids[1]]-source[ids[0]],source[ids[2]]-source[ids[0]]))<1.:
            continue
        transform = rigid(source[ids],target[ids])
        keep = np.linalg.norm(moved(source,transform)-target,axis=1)<2.5
        if keep.sum()>best.sum():
            best = keep
    if best.sum()<15 or best.mean()<.15:
        return None,None
    transform = rigid(source[best],target[best])
    best = np.linalg.norm(moved(source,transform)-target,axis=1)<2.5
    return rigid(source[best],target[best]),best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--voxel-mm',type=float,default=.5)
    parser.add_argument('--dense-refine',action='store_true',help='Use multiscale point-to-plane ICP and dense pose graph constraints')
    parser.add_argument('--thigh-refine',action='store_true',help='Add validated local thigh ICP constraints to the dense pose graph')
    parser.add_argument('--pose-override',type=Path,help='Experimental externally estimated poses used for final fusion')
    args = parser.parse_args()
    if args.voxel_mm<=0:
        parser.error('Voxel size must be positive')
    if args.thigh_refine and not args.dense_refine:
        parser.error('--thigh-refine requires --dense-refine')
    summary = json.loads((args.depth_dir/'depth_summary.json').read_text(encoding='utf-8'))
    decoded = Path(summary['source'])
    masks = Path(summary['foreground_masks']) if summary.get('foreground_masks') else None
    if masks is None:
        raise ValueError('Foreground masks are required for rotating-object alignment')
    args.output.mkdir(parents=True,exist_ok=False)
    effective_masks = args.depth_dir/'effective_masks'
    if not effective_masks.is_dir() and summary.get('source_depth_directory'):
        effective_masks = Path(summary['source_depth_directory'])/'effective_masks'
    if not effective_masks.is_dir():
        effective_masks = masks
    names = [g['group'] for g in summary['groups']]
    sift = cv2.SIFT_create(nfeatures=7000,contrastThreshold=.015)
    views = []
    for name in names:
        print('Features',name,flush=True)
        color = read_image(decoded/'color_images'/(name+'.png'),cv2.IMREAD_COLOR)
        mask = read_image(masks/(name+'.png'))
        with np.load(args.depth_dir/name/'depth_mm.npz',allow_pickle=False) as data:
            xyz,pixels = data['xyz_mm'],data['pixels_xy']
        # Remove isolated 3D points before registration/normals.
        distance,_ = cKDTree(xyz).query(xyz,k=8,workers=2)
        keep = distance[:,-1]<4.
        xyz,pixels = xyz[keep],pixels[keep]
        if len(xyz)<100:
            raise ValueError(f'Too few points: {name}')
        keypoints,descriptors = sift.detectAndCompute(cv2.cvtColor(color,cv2.COLOR_BGR2GRAY),mask)
        xy = np.asarray([key.pt for key in keypoints],float)
        dist,ids = cKDTree(pixels).query(xy)
        supported = dist<=2.
        effective_mask = read_image(effective_masks/(name+'.png'))
        if effective_mask.shape != mask.shape:
            raise ValueError(f'Effective mask size differs: {name}')
        edge_distance = cv2.distanceTransform((effective_mask>0).astype(np.uint8),cv2.DIST_L2,3)
        views.append({'xyz':xyz,'pixels':pixels,'colors':color[pixels[:,1],pixels[:,0],::-1],
            'silhouette':edge_distance[pixels[:,1],pixels[:,0]]<=3.,
            'descriptors':descriptors[supported],'feature_xyz':xyz[ids[supported]]})
    rng = np.random.default_rng(7)
    matcher = cv2.BFMatcher()
    edges = []
    n = len(views)
    pairs = sorted({tuple(sorted((i,(i+gap)%n))) for i in range(n) for gap in (1,2,3) if (i+gap)%n!=i})
    for i,j in pairs:
        a,b = views[i],views[j]
        if min(len(a['descriptors']),len(b['descriptors']))<20:
            continue
        matches = [m for pair in matcher.knnMatch(b['descriptors'],a['descriptors'],k=2)
                   if len(pair)==2 for m,k in [pair] if m.distance<.7*k.distance]
        if len(matches)<15:
            continue
        source = np.array([b['feature_xyz'][m.queryIdx] for m in matches])
        target = np.array([a['feature_xyz'][m.trainIdx] for m in matches])
        transform,keep = estimate(source,target,rng)
        if transform is None:
            continue
        ids = np.flatnonzero(keep)
        ids = ids[np.linspace(0,len(ids)-1,min(50,len(ids)),dtype=int)]
        residual = np.linalg.norm(moved(source[keep],transform)-target[keep],axis=1)
        edges.append({'i':i,'j':j,'T':transform,'a':target[ids],'b':source[ids],
            'inliers':int(keep.sum()),'median_mm':float(np.median(residual))})
        print('Edge',names[i],names[j],'inliers',int(keep.sum()),'median mm',round(float(np.median(residual)),3),flush=True)
    save_json(args.output/'pairwise_registration.json',[{'i':names[e['i']],'j':names[e['j']],
        'inliers':e['inliers'],'median_mm':e['median_mm'],'j_to_i':e['T'].tolist()} for e in edges])
    poses = {0:np.eye(4)}
    while True:
        before = len(poses)
        for e in sorted(edges,key=lambda e:-e['inliers']):
            i,j,t = e['i'],e['j'],e['T']
            if i in poses and j not in poses: poses[j] = poses[i]@t
            elif j in poses and i not in poses: poses[i] = poses[j]@np.linalg.inv(t)
        if len(poses)==before: break
    missing = [names[i] for i in range(n) if i not in poses]
    if missing:
        save_json(args.output/'status.json',{'complete':False,'unregistered':missing})
        raise ValueError(f'Unregistered views: {missing}; refusing to merge disconnected clouds')
    def pack(pose):
        return np.r_[cv2.Rodrigues(pose[:3,:3])[0].ravel(),pose[:3,3]]
    def unpack(parameters):
        result = [np.eye(4)]
        for value in parameters.reshape(-1,6):
            pose = np.eye(4); pose[:3,:3] = cv2.Rodrigues(value[:3])[0]; pose[:3,3] = value[3:]; result.append(pose)
        return result
    initial = np.concatenate([pack(poses[i]) for i in range(1,n)])
    def objective(parameters):
        world = unpack(parameters)
        return np.concatenate([(moved(e['a'],world[e['i']])-moved(e['b'],world[e['j']])).ravel() for e in edges])
    rows = sum(len(e['a'])*3 for e in edges)
    sparse = lil_matrix((rows,len(initial)),dtype=int)
    start = 0
    for e in edges:
        length = len(e['a'])*3
        for node in (e['i'],e['j']):
            if node: sparse[start:start+length,(node-1)*6:node*6] = 1
        start += length
    print('Optimizing pose graph',len(edges),'edges',flush=True)
    fit = least_squares(objective,initial,jac_sparsity=sparse.tocsr(),loss='soft_l1',f_scale=1.,
        x_scale='jac',ftol=1e-4,xtol=1e-5,max_nfev=300,verbose=1)
    world = unpack(fit.x)
    residual = np.linalg.norm(objective(fit.x).reshape(-1,3),axis=1)
    save_json(args.output/'registration.json',{'poses':{name:pose.tolist() for name,pose in zip(names,world)},
        'median_correspondence_error_mm':float(np.median(residual)),
        'p95_correspondence_error_mm':float(np.percentile(residual,95)),
        'optimizer_success':bool(fit.success),'empirical_projector_offset_xy_px':summary.get('projector_offset_xy_px'),
        'coordinate_frame':'group_0001 camera; millimeters; empirical metric scale not independently validated'})
    if np.median(residual)>2.5 or np.percentile(residual,95)>5.:
        raise ValueError('Pose graph residual is too large; refusing to fuse')
    icp_accepted = 0
    if args.dense_refine:
        stages = (5.,2.,1.)
        samples = [[surface_sample(view['xyz'],voxel,9000,7+i*3+s)
                    for s,voxel in enumerate(stages)] for i,view in enumerate(views)]
        quality = []
        for edge in edges:
            i,j = edge['i'],edge['j']
            initial_pair = np.linalg.inv(world[i])@world[j]
            refined,diagnostic = refine_pair(views[j]['xyz'],views[i]['xyz'],initial_pair,
                seed=7+j,source_samples=samples[j],target_samples=samples[i])
            diagnostic.update({'i':names[i],'j':names[j]})
            if refined is not None:
                transform,source,target,target_normal = refined
                # Check that the refined transform still agrees with verified SIFT points.
                sparse_error = np.linalg.norm(moved(edge['b'],transform)-edge['a'],axis=1)
                diagnostic['sift_median_after_mm'] = float(np.median(sparse_error))
                if np.median(sparse_error)<=2.5 and np.percentile(sparse_error,95)<=5.:
                    edge['dense_a'],edge['dense_b'],edge['dense_normal'] = target,source,target_normal
                    icp_accepted += 1
                else:
                    diagnostic['accepted'] = False
                    diagnostic['reason'] = 'conflicts with verified SIFT correspondences'
            quality.append(diagnostic)
            print('ICP',names[i],names[j],diagnostic['accepted'],diagnostic.get('reason',''),flush=True)
        save_json(args.output/'icp_report.json',{'accepted_edges':icp_accepted,'tested_edges':len(edges),'edges':quality})
        if icp_accepted<max(10,n-1):
            raise ValueError(f'Only {icp_accepted} dense edges accepted; refusing unverified refinement')
        thigh_accepted = 0
        if args.thigh_refine:
            thigh_points = []
            thigh_counts = {}
            colored_points,sector_colors = [],[]
            for index,(view,pose) in enumerate(zip(views,world)):
                transformed = moved(view['xyz'],pose)
                rgb = view['colors'].astype(float)
                skin = ((rgb[:,0]>125)&(rgb[:,1]>100)&(rgb[:,2]>85)&
                    (rgb[:,0]>rgb[:,1]*1.02)&(rgb[:,1]>rgb[:,2]*.88)&
                    (rgb[:,0]<rgb[:,1]*1.65))
                region = ((transformed[:,0]>-25)&(transformed[:,0]<40)&
                    (transformed[:,1]>5)&(transformed[:,1]<70)&
                    (transformed[:,2]>790)&(transformed[:,2]<900))
                local = view['xyz'][skin&region]
                thigh_points.append(local)
                thigh_counts[names[index]]=len(local)
                sector = (0 if index<6 or index>=26 else
                    (1 if index<11 else (2 if index<21 else 3)))
                shade = ([245,60,60],[65,210,80],[60,115,245],[240,205,50])[sector]
                colored_points.append(moved(local[::3],pose))
                sector_colors.append(np.tile(shade,(len(local[::3]),1)))
            write_ply(args.output/'thigh_view_groups.ply',np.vstack(colored_points),
                colors=np.vstack(sector_colors).astype(np.uint8))
            save_json(args.output/'thigh_view_counts.json',{'points_per_view':thigh_counts,
                'colors':{'front':'red','first_side':'green','back':'blue','second_side':'yellow'},
                'roi_world_mm':{'x':[-25,40],'y':[5,70],'z':[790,900]},
                'note':'Skin-color heuristic and world-space box; includes some nearby clothing.'})
            thigh_quality=[]
            for edge in edges:
                i,j=edge['i'],edge['j']
                if min(abs(j-i),n-abs(j-i))!=1 or min(len(thigh_points[i]),len(thigh_points[j]))<300:
                    continue
                initial_pair=np.linalg.inv(world[i])@world[j]
                refined,diagnostic=refine_pair(thigh_points[j],thigh_points[i],initial_pair,seed=1000+i)
                diagnostic.update({'i':names[i],'j':names[j]})
                if refined is not None:
                    transform,source,target,target_normal=refined
                    probe=thigh_points[j][::max(1,len(thigh_points[j])//200)]
                    movement=np.linalg.norm(moved(probe,transform)-moved(probe,initial_pair),axis=1)
                    sparse_error=np.linalg.norm(moved(edge['b'],transform)-edge['a'],axis=1)
                    diagnostic['median_surface_motion_mm']=float(np.median(movement))
                    diagnostic['sift_median_after_mm']=float(np.median(sparse_error))
                    acceptable=(np.median(movement)<2. and
                        np.median(sparse_error)<2.5 and np.percentile(sparse_error,95)<5. and
                        diagnostic['median_final_plane_mm']<.8)
                    if acceptable:
                        edge['thigh_a'],edge['thigh_b'],edge['thigh_normal']=target,source,target_normal
                        thigh_accepted+=1
                    else:
                        diagnostic['accepted']=False
                        diagnostic['reason']='local movement, SIFT conflict or plane residual too large'
                thigh_quality.append(diagnostic)
                print('Thigh ICP',names[i],names[j],diagnostic['accepted'],diagnostic.get('reason',''),flush=True)
            save_json(args.output/'thigh_icp_report.json',{'accepted_edges':thigh_accepted,
                'tested_edges':len(thigh_quality),'edges':thigh_quality})
            if thigh_accepted<8:
                raise ValueError(f'Only {thigh_accepted} reliable thigh ICP edges; refusing local pose refinement')
        def dense_objective(parameters):
            refined_world = unpack(parameters)
            blocks=[]
            for edge in edges:
                i,j = edge['i'],edge['j']
                blocks.append((moved(edge['a'],refined_world[i])-moved(edge['b'],refined_world[j])).ravel())
                if 'dense_a' in edge:
                    delta=moved(edge['dense_a'],refined_world[i])-moved(edge['dense_b'],refined_world[j])
                    oriented=edge['dense_normal']@refined_world[i][:3,:3].T
                    blocks.append(np.einsum('ij,ij->i',delta,oriented))
                if 'thigh_a' in edge:
                    delta=moved(edge['thigh_a'],refined_world[i])-moved(edge['thigh_b'],refined_world[j])
                    oriented=edge['thigh_normal']@refined_world[i][:3,:3].T
                    blocks.append(1.5*np.einsum('ij,ij->i',delta,oriented))
            return np.concatenate(blocks)
        dense_rows=sum(3*len(e['a'])+len(e.get('dense_a',[]))+len(e.get('thigh_a',[])) for e in edges)
        graph_pattern=lil_matrix((dense_rows,len(fit.x)),dtype=int)
        start=0
        for edge in edges:
            length=3*len(edge['a'])+len(edge.get('dense_a',[]))+len(edge.get('thigh_a',[]))
            for node in (edge['i'],edge['j']):
                if node: graph_pattern[start:start+length,(node-1)*6:node*6]=1
            start+=length
        print('Optimizing dense pose graph',icp_accepted,'accepted edges',flush=True)
        refined_fit=least_squares(dense_objective,fit.x,jac_sparsity=graph_pattern.tocsr(),
            loss='soft_l1',f_scale=1.,x_scale='jac',ftol=1e-4,xtol=1e-5,max_nfev=200,verbose=1)
        refined_world=unpack(refined_fit.x)
        refined_sparse=np.concatenate([np.linalg.norm(moved(e['a'],refined_world[e['i']])-
            moved(e['b'],refined_world[e['j']]),axis=1) for e in edges])
        if np.median(refined_sparse)>2.5 or np.percentile(refined_sparse,95)>5.:
            raise ValueError('Dense optimization damaged sparse alignment; refusing to fuse')
        if args.thigh_refine:
            def thigh_plane_errors(poses):
                return np.concatenate([np.abs(np.einsum('ij,ij->i',
                    moved(e['thigh_a'],poses[e['i']])-moved(e['thigh_b'],poses[e['j']]),
                    e['thigh_normal']@poses[e['i']][:3,:3].T))
                    for e in edges if 'thigh_a' in e])
            before=thigh_plane_errors(world)
            after=thigh_plane_errors(refined_world)
            save_json(args.output/'thigh_global_report.json',{'accepted_local_edges':thigh_accepted,
                'median_before_mm':float(np.median(before)),'median_after_mm':float(np.median(after)),
                'p90_before_mm':float(np.percentile(before,90)),
                'p90_after_mm':float(np.percentile(after,90)),
                'sift_median_after_mm':float(np.median(refined_sparse))})
        world=refined_world
        save_json(args.output/'registration.json',{'poses':{name:pose.tolist() for name,pose in zip(names,world)},
            'median_correspondence_error_mm':float(np.median(refined_sparse)),
            'p95_correspondence_error_mm':float(np.percentile(refined_sparse,95)),
            'sparse_median_before_mm':float(np.median(residual)),
            'icp_accepted_edges':icp_accepted,'thigh_accepted_edges':thigh_accepted,
            'optimizer_success':bool(refined_fit.success),
            'empirical_projector_offset_xy_px':summary.get('projector_offset_xy_px'),
            'coordinate_frame':'group_0001 camera; millimeters; empirical metric scale not independently validated'})
    if args.pose_override:
        document=json.loads(args.pose_override.read_text(encoding='utf-8'))
        if set(document['poses']) != set(names):
            raise ValueError('Pose override does not contain exactly the depth groups')
        save_json(args.output/'registration_global_reference.json',
            json.loads((args.output/'registration.json').read_text(encoding='utf-8')))
        world=[np.asarray(document['poses'][name],float) for name in names]
        if any(pose.shape!=(4,4) or not np.isfinite(pose).all() for pose in world):
            raise ValueError('Invalid pose override')
        save_json(args.output/'registration.json',{'poses':document['poses'],
            'method':document.get('method','external pose override'),
            'pose_override_source':str(args.pose_override.resolve()),
            'experimental':True})
    points,colors,normals,view_ids,edge_flags = [],[],[],[],[]
    for index,(view,pose) in enumerate(zip(views,world)):
        xyz = view['xyz']
        # Normals are estimated per view and oriented towards that view's camera.
        _,neighbors = cKDTree(xyz).query(xyz,k=24,workers=2)
        normal = np.empty_like(xyz)
        for begin in range(0,len(xyz),10000):
            chunk = xyz[neighbors[begin:begin+10000]]
            centered = chunk-chunk.mean(axis=1,keepdims=True)
            covariance = np.einsum('nki,nkj->nij',centered,centered)
            normal[begin:begin+len(chunk)] = np.linalg.eigh(covariance)[1][:,:,0]
        normal[np.sum(normal*xyz,axis=1)>0] *= -1
        points.append(moved(xyz,pose)); colors.append(view['colors']); normals.append(normal@pose[:3,:3].T)
        view_ids.append(np.full(len(xyz),index,int))
        edge_flags.append(view['silhouette'])
    xyz,rgb,normal,source_view = np.vstack(points),np.vstack(colors),np.vstack(normals),np.concatenate(view_ids)
    edge_flags=np.concatenate(edge_flags)
    cells = np.floor(xyz/args.voxel_mm).astype(np.int64)
    _,first,inverse = np.unique(cells,axis=0,return_index=True,return_inverse=True)
    count = np.bincount(inverse)
    def average(values):
        return np.column_stack([np.bincount(inverse,weights=values[:,i])/count for i in range(3)])
    fused,colors,normal = average(xyz),average(rgb),average(normal)
    edge_fraction=np.bincount(inverse,weights=edge_flags)/count
    # Count unique contributing views per voxel, avoiding same-view density bias.
    pairs = np.unique(np.column_stack((inverse,source_view)),axis=0)
    support = np.bincount(pairs[:,0],minlength=len(count))
    view_bits=np.zeros(len(count),dtype=np.uint64)
    np.bitwise_or.at(view_bits,inverse,np.left_shift(np.uint64(1),source_view.astype(np.uint64)))
    primary_view=source_view[first]
    length = np.linalg.norm(normal,axis=1)
    keep = ((support>=2)|(args.dense_refine&(edge_fraction>=.5)))&(length>.25)
    fused,colors,normal = fused[keep],colors[keep],normal[keep]/length[keep,None]
    retained_support=support[keep]
    retained_bits=view_bits[keep]
    retained_primary=primary_view[keep]
    distance,_ = cKDTree(fused).query(fused,k=12,workers=2)
    keep = distance[:,-1]<3.
    fused,colors,normal = fused[keep],colors[keep],normal[keep]
    retained_support=retained_support[keep]
    retained_bits=retained_bits[keep]
    retained_primary=retained_primary[keep]
    if args.dense_refine:
        # Cross-view surface support may occupy neighboring voxels; reject incompatible
        # normals while allowing densely sampled silhouette points from one view.
        tree=cKDTree(fused)
        neighbor_distance,neighbor_index=tree.query(fused,k=min(16,len(fused)),workers=2)
        neighbor_normal=normal[neighbor_index]
        plane_distance=np.abs(np.einsum('nki,nki->nk',fused[:,None,:]-fused[neighbor_index],neighbor_normal))
        normal_dot=np.einsum('ni,nki->nk',normal,neighbor_normal)
        compatible=(neighbor_distance<2.)&(plane_distance<.75)&(normal_dot>.8)
        consistent=compatible.sum(axis=1)
        other_view=(retained_bits[neighbor_index]&~np.left_shift(np.uint64(1),retained_primary.astype(np.uint64))[:,None])!=0
        cross_view=(compatible&other_view).sum(axis=1)
        keep=(consistent>=5)&((cross_view>=2)|((retained_support==1)&(consistent>=9)))
        # Single-view voxels in this array were admitted only at the silhouette.
        fused,colors,normal=fused[keep],colors[keep],normal[keep]
    write_ply(args.output/'dense.ply',fused,colors=np.clip(colors,0,255).astype(np.uint8),normals=normal)
    save_json(args.output/'status.json',{'complete':True,'registered_views':n,'input_points':len(xyz),
        'fused_points':len(fused),'voxel_mm':args.voxel_mm,'minimum_supporting_views':2,
        'dense_refine':args.dense_refine,'icp_accepted_edges':icp_accepted,
        'thigh_refine':args.thigh_refine,
        'pose_override':str(args.pose_override.resolve()) if args.pose_override else None,
        'silhouette_mask_directory':str(effective_masks.resolve()),
        'mesh_complete':False,'empirical_compensation':summary.get('empirical_compensation',False)})
    print('Fused',len(fused),'points:',args.output/'dense.ply',flush=True)


if __name__=='__main__':
    main()
