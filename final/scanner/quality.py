import csv
from pathlib import Path
from collections import Counter,defaultdict
import cv2
import numpy as np
from .io import read_ply,write_ply,save_json
from .geometry import face_geometry,visible,ray_first_hit


def analyze(mesh_path,cloud_path,cameras,output,max_samples=1200):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    vertices,faces=read_ply(mesh_path); cloud,_=read_ply(cloud_path)
    if len(faces)==0 or len(cloud)<16: raise ValueError('Need triangle mesh and at least 16 dense points')
    centroids,normals,areas=face_geometry(vertices,faces)
    face_ids=np.linspace(0,len(faces)-1,min(max_samples,len(faces)),dtype=int)
    points=centroids[face_ids]; normals=normals[face_ids]
    scale=float(np.linalg.norm(np.ptp(vertices,axis=0)))
    if scale<=0: raise ValueError('Degenerate mesh')
    index=cv2.flann_Index(cloud.astype(np.float32),dict(algorithm=1,trees=4))
    _,dist2=index.knnSearch(points.astype(np.float32),16,params={'checks':64})
    nearest=np.sqrt(np.maximum(dist2[:,0],0)); spacing=np.sqrt(np.maximum(dist2[:,-1],0))
    # Absolute geometric evidence thresholds, scaled to object bounds, not millimeters.
    unsupported=nearest>scale*.025; sparse=spacing>scale*.09
    view_count=np.zeros(len(points),int); directions=[[] for _ in points]
    fixture_count=np.zeros(len(points),int); fixture_known=False
    for camera in cameras:
        valid,toward,uv=visible(points,normals,camera,vertices,faces)
        if camera.get('object_mask'):
            mask=cv2.imdecode(np.fromfile(str(camera['object_mask']),np.uint8),0)
            if mask is None or list(mask.shape[::-1])!=camera['size']: raise ValueError('Invalid observation mask')
            ids=np.flatnonzero(valid); xy=np.floor(uv[ids]).astype(int)
            valid[ids]&=mask[xy[:,1],xy[:,0]]>0
        if camera.get('fixture_mesh'):
            fixture_known=True; fv,ff=read_ply(camera['fixture_mesh'])
            C=-np.asarray(camera['R']).T@np.asarray(camera['t'])
            distance=np.linalg.norm(points-C,axis=1)
            hit=ray_first_hit(np.repeat(C[None,:],len(points),axis=0),-toward,fv,ff)
            blocked=valid&(hit<distance-scale*1e-4)
            fixture_count+=blocked; valid&=~blocked
        view_count+=valid
        for i in np.flatnonzero(valid): directions[i].append(toward[i])
    spread=np.zeros(len(points))
    for i,ds in enumerate(directions):
        if len(ds)>1:
            ds=np.asarray(ds); spread[i]=np.degrees(np.arccos(np.clip(np.min(ds@ds.T),-1,1)))
    low_views=(view_count<2)|(spread<10)
    severity=.55*unsupported+.25*sparse+.20*low_views
    if not cameras:
        severity=.7*unsupported+.3*sparse  # Unknown visibility is reported separately.
    counts=Counter(tuple(sorted((int(a),int(b)))) for face in faces for a,b in
                   [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])])
    boundaries=[e for e,n in counts.items() if n==1]
    adjacency=defaultdict(set)
    for a,b in boundaries: adjacency[a].add(b); adjacency[b].add(a)
    remaining=set(adjacency); holes=[]
    while remaining:
        component=set(); stack=[next(iter(remaining))]
        while stack:
            v=stack.pop()
            if v in component: continue
            component.add(v); stack.extend(adjacency[v]-component)
        remaining-=component
        closed=all(len(adjacency[v])==2 for v in component)
        ordered=[]
        if closed:
            first=next(iter(component)); previous=None; current=first
            while True:
                ordered.append(current)
                nxt=next(x for x in adjacency[current] if x!=previous)
                previous,current=current,nxt
                if current==first: break
        selected=vertices[list(component)]; center=selected.mean(axis=0)
        _,_,basis=np.linalg.svd(selected-center,full_matrices=False)
        area=None
        if closed and len(basis)>=2:
            xy=(vertices[ordered]-center)@basis[:2].T
            area=float(abs(np.dot(xy[:,0],np.roll(xy[:,1],1))-np.dot(xy[:,1],np.roll(xy[:,0],1)))/2)
        holes.append({'center':center.tolist(),'vertex_count':len(component),'closed_boundary_loop':closed,
                      'projected_area_estimate':area,'normal':basis[-1].tolist(),
                      'interpretation':'Open boundary; may be a missing surface or intentional opening'})
    colors=np.tile([170,170,170],(len(vertices),1))
    for i,fi in enumerate(face_ids):
        color=[230,55,45] if unsupported[i] else [240,190,40] if sparse[i] or (cameras and low_views[i]) else [45,190,80]
        if fixture_count[i]>0: color=[170,70,220]
        colors[faces[fi]]=color
    write_ply(output/'quality_colored.ply',vertices,faces,colors)
    preview=np.full((470,1200,3),245,np.uint8)
    for panel,(horizontal,vertical,depth_axis,label) in enumerate([(0,2,1,'FRONT'),(1,2,0,'SIDE'),(0,1,2,'TOP')]):
        xy=vertices[:,[horizontal,vertical]].copy(); low=xy.min(axis=0); span=np.maximum(np.ptp(xy,axis=0),1e-8)
        factor=min(340/span[0],330/span[1]); xy=(xy-low)*factor
        xy[:,0]+=panel*400+30; xy[:,1]=390-xy[:,1]
        polygons=np.round(xy).astype(np.int32)
        for fi in np.argsort(vertices[faces,depth_axis].mean(axis=1)):
            rgb=colors[faces[fi]].mean(axis=0)
            cv2.fillConvexPoly(preview,polygons[faces[fi]],tuple(float(x) for x in rgb[::-1]),cv2.LINE_AA)
        cv2.putText(preview,label,(panel*400+25,30),0,.8,(0,0,0),2)
    cv2.putText(preview,'Green: supported | Yellow: sparse/views | Red: unsupported | Gray: unsampled',(15,435),0,.55,(0,0,0),1)
    cv2.imencode('.png',preview)[1].tofile(str(output/'quality_preview.png'))
    if boundaries:
        bpoints=np.vstack([np.linspace(vertices[a],vertices[b],12) for a,b in boundaries])
        write_ply(output/'hole_boundaries.ply',bpoints,colors=np.tile([255,30,30],(len(bpoints),1)))
    else: write_ply(output/'hole_boundaries.ply',np.empty((0,3)))
    samples=[{'center':p.tolist(),'normal':n.tolist(),'severity':float(severity[i]),
              'nearest_dense_point':float(nearest[i]),'neighbor16_distance':float(spacing[i]),
              'visible_view_count':int(view_count[i]) if cameras else None,
              'view_spread_deg':float(spread[i]) if cameras else None,
              'fixture_blocked_views':int(fixture_count[i]) if fixture_known else None,
              'unsupported':bool(unsupported[i]),'sparse':bool(sparse[i])} for i,(p,n) in enumerate(zip(points,normals))]
    report={'mesh_vertices':len(vertices),'mesh_faces':len(faces),'dense_points':len(cloud),
            'sample_count':len(samples),'camera_count':len(cameras),'scale_bbox_diagonal':scale,
            'unsupported_fraction':float(np.mean(unsupported)),'sparse_fraction':float(np.mean(sparse)),
            'low_view_fraction':float(np.mean(low_views)) if cameras else None,
            'boundary_components':holes,'nonmanifold_edges':sum(n>2 for n in counts.values()),
            'fixture_occlusion':'Ray-tested with per-view registered fixture meshes' if fixture_known else 'Unknown: no registered per-view fixture geometry',
            'visibility_note':'Geometric visibility is potential coverage, not proof of reliable image/depth matching',
            'samples':samples}
    save_json(output/'quality_regions.json',report)
    with (output/'quality_report.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['sample','x','y','z','severity','nearest_point','neighbor16','views','spread_deg'])
        for i,s in enumerate(samples): writer.writerow([i,*s['center'],s['severity'],s['nearest_dense_point'],s['neighbor16_distance'],s['visible_view_count'],s['view_spread_deg']])
    return report
