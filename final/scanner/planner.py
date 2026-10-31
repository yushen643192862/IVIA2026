from pathlib import Path
import numpy as np
from .io import read_ply,save_json
from .geometry import platform_pose,transform,visible,ray_first_hit


def recommend(mesh_path,quality,config,output,top_k=4):
    vertices,faces=read_ply(mesh_path)
    if not config.get('cameras'): raise ValueError('Planner requires a fixed-camera rig configuration')
    T0=np.asarray(config['platform_from_model'],float)
    if T0.shape!=(4,4) or not np.allclose(T0[3],[0,0,0,1]): raise ValueError('Invalid platform_from_model')
    singular=np.linalg.svd(T0[:3,:3],compute_uv=False)
    if not np.allclose(singular,singular[0],rtol=1e-4) or singular[0]<=0 or np.linalg.det(T0[:3,:3])<=0:
        raise ValueError('platform_from_model must be a proper rigid transform with optional uniform positive scale')
    samples=quality['samples']; points=np.array([s['center'] for s in samples]); normals=np.array([s['normal'] for s in samples])
    weights=np.array([s['severity'] for s in samples])
    # Boundary-adjacent observed samples are a proxy; unseen geometry cannot be ray traced.
    for hole in quality['boundary_components']:
        distance=np.linalg.norm(points-np.array(hole['center']),axis=1)
        near=distance<quality['scale_bbox_diagonal']*.15
        weights[near]=np.maximum(weights[near],.6)
    targets=np.flatnonzero(weights>0)
    fixture=None
    if config.get('fixture_mesh'):
        fixture=read_ply(config['fixture_mesh'])
    current=config.get('current_pose',{'yaw':0,'pitch':0})
    captured=config.get('captured_poses',[])
    minimum,maximum=config.get('pitch_limits',[-45,45])
    candidates=[]
    for pitch in config['pitch_candidates']:
        if not minimum<=pitch<=maximum: continue
        for yaw in config['yaw_candidates']:
            if any(abs(((yaw-p['yaw']+180)%360)-180)<.1 and abs(pitch-p['pitch'])<.1 for p in captured): continue
            T=platform_pose(config,yaw,pitch)
            moved=transform(vertices,T); pp=transform(points,T)
            nn=normals@T[:3,:3].T; nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-12)
            count=np.zeros(len(points)); blocked=np.zeros(len(points))
            for camera in config['cameras']:
                valid,_,_=visible(pp,nn,camera,moved,faces)
                if fixture:
                    C=-np.asarray(camera['R']).T@np.asarray(camera['t'])
                    d=pp-C; distance=np.linalg.norm(d,axis=1)
                    hit=ray_first_hit(np.repeat(C[None,:],len(pp),axis=0),d/np.maximum(distance[:,None],1e-12),*fixture)
                    obstruction=hit<distance-1e-5
                    blocked+=valid&obstruction; valid&=~obstruction
                count+=valid
            gain=float(np.sum(weights*(count>0))); multi=float(np.sum(weights*(count>=2)))
            motion=abs(((yaw-current['yaw']+180)%360)-180)/180+abs(pitch-current['pitch'])/90
            total=max(float(weights.sum()),1e-12)
            score=(gain+.4*multi)/total-.05*motion
            candidates.append({'yaw':float(yaw),'pitch':float(pitch),'score':score,
                               'weighted_potential_coverage':gain,'weighted_multi_camera_coverage':multi,
                               'fixture_blocked_sample_count':int(np.count_nonzero(blocked)) if fixture else None,
                               'motion_cost':motion,'target_ids_visible':np.flatnonzero((count>0)&(weights>0)).tolist()})
    selected=[]; uncovered=set(targets.tolist())
    for _ in range(top_k):
        best=None; value=-np.inf
        for c in candidates:
            if c in selected: continue
            new=set(c['target_ids_visible'])&uncovered
            marginal=float(weights[list(new)].sum()) if new else 0.
            candidate_value=marginal/max(float(weights.sum()),1e-12)-.05*c['motion_cost']
            if candidate_value>value: value=candidate_value; best=c
        if best is None or not (set(best['target_ids_visible'])&uncovered): break
        selected.append(best); uncovered-=set(best['target_ids_visible'])
    result={'status':'calibrated_recommendations' if config.get('calibrated') else 'illustrative_only',
            'executable':bool(config.get('calibrated')),
            'coordinate_requirement':'Rig and axis pivots must share platform frame; platform_from_model must be measured',
            'target_sample_count':len(targets),'recommended':selected,
            'uncovered_target_count':len(uncovered),'candidates':sorted(candidates,key=lambda x:x['score'],reverse=True),
            'fixture_penalty_status':'modeled' if fixture else 'unknown; not assumed clear',
            'limitations':['Missing surfaces are unknown: boundary-adjacent samples are only proxies.',
                           'Visibility predicts potential coverage, not successful photogrammetric reconstruction.',
                           'No fixture collision, torque or cable-travel certification; platform motion remains manual.']}
    save_json(Path(output)/'recommended_poses.json',result)
    lines=['# 补拍建议','',f'状态：{result["status"]}。未标定配置仅用于演示。','',
           '| 顺序 | Yaw | Pitch | 预计质量目标可见权重 |','| --- | ---: | ---: | ---: |']
    for i,c in enumerate(selected,1): lines.append(f'| {i} | {c["yaw"]:.0f}° | {c["pitch"]:.0f}° | {c["weighted_potential_coverage"]:.3f} |')
    lines+=['','建议仅针对已知模型及边界邻域，不证明缺失处实际形状。',
            '没有夹具模型时，夹具遮挡状态为未知。禁止直接把演示姿态发送给电机。']
    (Path(output)/'recommendations.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
