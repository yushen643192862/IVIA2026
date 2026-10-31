import csv
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from .io import save_json,read_ply,write_ply
from .dense_masks import prepare_dense_masks,constrain_depth_maps


def quaternion_matrix(q):
    w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])


def read_model(folder):
    cameras={}
    for line in (folder/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        t=line.split(); cid=int(t[0]); model=t[1]; w,h=map(int,t[2:4]); p=list(map(float,t[4:]))
        if model in ['SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL']:
            fx=fy=p[0]; cx,cy=p[1:3]; d=[p[3] if len(p)>3 else 0,p[4] if len(p)>4 else 0,0,0,0]
        elif model in ['PINHOLE','OPENCV','FULL_OPENCV']:
            fx,fy,cx,cy=p[:4]; d=p[4:] if len(p)>4 else [0]*5
        else: raise ValueError(f'Unsupported projection model: {model}')
        cameras[cid]={'K':[[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1]],'distortion':d,'size':[w,h],'model':model}
    lines=(folder/'images.txt').read_text().splitlines(); views=[]; i=0
    while i<len(lines):
        line=lines[i]; i+=1
        if not line.strip() or line.startswith('#'): continue
        t=line.split(); camera=dict(cameras[int(t[8])]); R=quaternion_matrix(list(map(float,t[1:5])))
        tv=np.array(list(map(float,t[5:8])))
        camera.update({'file':' '.join(t[9:]),'R':R.tolist(),'t':tv.tolist(),'center':(-R.T@tv).tolist()})
        views.append(camera)
        i+=1  # Next line is the 2D observation record, possibly empty.
    return views


def reconstruct(images,output,executable='colmap',masks=None,static=False,sparse_only=False,high_detail=False,patch_window_radius=5,camera_params=None):
    images=Path(images).resolve(); output=Path(output).resolve()
    paths=sorted(p for p in images.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff'])
    if len(paths)<3: raise ValueError('Need at least three images; use phone-specific subfolders')
    if not static and masks is None:
        raise ValueError('Rotating objects require foreground masks. Supply --masks; --static-scene only applies to stationary scenes with moving cameras.')
    if masks:
        masks=Path(masks).resolve()
        for p in paths:
            expected=masks/(str(p.relative_to(images))+'.png')
            if not expected.is_file(): raise ValueError(f'Missing mask: {expected}')
            im=cv2.imdecode(np.fromfile(str(p),np.uint8),0)
            mask=cv2.imdecode(np.fromfile(str(expected),np.uint8),0)
            if im is None or mask is None or im.shape!=mask.shape or not np.any(mask>0):
                raise ValueError(f'Invalid or empty mask: {expected}')
    exe=shutil.which(executable)
    if exe is None and Path(executable).is_file(): exe=str(Path(executable).resolve())
    if exe is None: raise RuntimeError('COLMAP not installed/found. Set --colmap to the executable path; demo does not require COLMAP.')
    if (output/'database.db').exists(): raise ValueError('Output already contains a database. Use a new output directory to avoid mixing scans.')
    output.mkdir(parents=True,exist_ok=True); sparse=output/'sparse'; sparse.mkdir(exist_ok=True)
    logs=output/'logs'; logs.mkdir(exist_ok=True); history=[]
    def run(command,arguments):
        cmd=[exe,command,*map(str,arguments)]; history.append(cmd)
        save_json(logs/'commands.json',history)
        with (logs/(command+'.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        if completed.returncode: raise RuntimeError(f'{command} failed. Read {logs/(command+".log")}')
    def supported(command,modern,legacy):
        help_result=subprocess.run([exe,command,'-h'],capture_output=True,text=True,errors='replace')
        text=help_result.stdout+help_result.stderr
        if modern in text: return modern
        if legacy in text: return legacy
        raise RuntimeError(f'Cannot locate {modern}/{legacy} in installed COLMAP help')
    features=['--database_path',output/'database.db','--image_path',images,
              '--ImageReader.camera_model','SIMPLE_RADIAL','--ImageReader.single_camera_per_folder','1',
              supported('feature_extractor','--FeatureExtraction.use_gpu','--SiftExtraction.use_gpu'),'0']
    if high_detail:
        features[-1]='1'
        features+=['--FeatureExtraction.max_image_size','4096',
                   '--SiftExtraction.max_num_features','32768','--SiftExtraction.peak_threshold','0.0015']
    if masks: features+=['--ImageReader.mask_path',masks]
    if camera_params:
        values=np.asarray([float(v) for v in camera_params.split(',')])
        if len(values)!=4 or not np.isfinite(values).all() or values[0]<=0:
            raise ValueError('SIMPLE_RADIAL camera parameters must be f,cx,cy,k with positive focal length')
        features+=['--ImageReader.camera_params',camera_params]
    run('feature_extractor',features)
    matching=['--database_path',output/'database.db',
              supported('exhaustive_matcher','--FeatureMatching.use_gpu','--SiftMatching.use_gpu'),'1' if high_detail else '0']
    if high_detail: matching+=['--FeatureMatching.guided_matching','1']
    run('exhaustive_matcher',matching)
    run('mapper',['--database_path',output/'database.db','--image_path',images,'--output_path',sparse])
    models=list(sparse.glob('*/images.bin'))
    if not models: raise RuntimeError('No registered reconstruction. Check texture, masks, overlap and camera grouping.')
    model=max(models,key=lambda x:x.stat().st_size).parent
    txt=output/'sparse_text'; txt.mkdir(exist_ok=True)
    run('model_converter',['--input_path',model,'--output_path',txt,'--output_type','TXT'])
    cameras=read_model(txt); save_json(output/'cameras.json',cameras)
    if masks:
        for c in cameras: c['object_mask']=str(masks/(c['file']+'.png'))
        save_json(output/'cameras.json',cameras)
    point_errors=[]
    for line in (txt/'points3D.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        fields=line.split(); point_errors.append(float(fields[7]))
    save_json(output/'sparse_statistics.json',{'registered_images':len(cameras),'input_images':len(paths),
              'registration_fraction':len(cameras)/len(paths),'sparse_points':len(point_errors),
              'mean_point_reprojection_error_px':float(np.mean(point_errors)) if point_errors else None,
              'note':'Mean of COLMAP per-point errors, not a pooled RMSE'})
    run('model_converter',['--input_path',model,'--output_path',output/'sparse.ply','--output_type','PLY'])
    save_json(output/'reconstruction_status.json',{'input_images':len(paths),'registered_images':len(cameras),
              'submodels':len(models),'chosen_model':str(model),'coordinate_system':'SfM object frame, arbitrary scale',
              'dense_complete':False,'notes':'Different submodels are not merged; platform mapping remains uncalibrated.'})
    if sparse_only: return cameras
    return _finish_dense(images,output,masks,cameras,model,len(paths),len(models),run,patch_window_radius)


def resume_dense(images,output,executable='colmap',masks=None,patch_window_radius=5):
    """Continue from the exact sparse model previously exported for review."""
    images=Path(images).resolve(); output=Path(output).resolve()
    status=json.loads((output/'reconstruction_status.json').read_text(encoding='utf-8'))
    if status.get('dense_complete'):
        raise ValueError('Dense reconstruction already complete')
    model=Path(status['chosen_model'])
    cameras=json.loads((output/'cameras.json').read_text(encoding='utf-8'))
    if not (model/'images.bin').is_file() or not cameras:
        raise ValueError('No reviewed sparse model available')
    masks=Path(masks).resolve() if masks else None
    recorded={c.get('object_mask') for c in cameras}
    if recorded!={None}:
        if masks is None or any(str(masks/(c['file']+'.png'))!=c.get('object_mask') for c in cameras):
            raise ValueError('Resume must use the same masks as sparse reconstruction')
    elif masks is not None:
        raise ValueError('Sparse reconstruction did not use these masks')
    exe=shutil.which(executable)
    if exe is None and Path(executable).is_file(): exe=str(Path(executable).resolve())
    if exe is None: raise RuntimeError('COLMAP executable not found')
    logs=output/'logs'
    history=json.loads((logs/'commands.json').read_text(encoding='utf-8'))
    def run(command,arguments):
        cmd=[exe,command,*map(str,arguments)]; history.append(cmd)
        save_json(logs/'commands.json',history)
        with (logs/(command+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode: raise RuntimeError(f'{command} failed. Read {logs/(command+".log")}')
    return _finish_dense(images,output,masks,cameras,model,status['input_images'],status['submodels'],run,patch_window_radius)


def _finish_dense(images,output,masks,cameras,model,input_images,submodels,run,patch_window_radius=5):
    logs=output/'logs'
    dense=output/'dense'; dense.mkdir(exist_ok=True)
    run('image_undistorter',['--image_path',images,'--input_path',model,'--output_path',dense,
                           '--output_type','COLMAP','--max_image_size','1600'])
    foregrounds=None
    if masks:
        text=dense/'mask_model_text'; text.mkdir(exist_ok=True)
        run('model_converter',['--input_path',dense/'sparse','--output_path',text,'--output_type','TXT'])
        foregrounds=prepare_dense_masks(dense,cameras,read_model(text),patch_window_radius)
        # Photometric maps are constrained before geometric consistency uses
        # them as source depths. PatchMatch reuses these maps in the next call.
        run('patch_match_stereo',['--workspace_path',dense,'--workspace_format','COLMAP',
                '--PatchMatchStereo.geom_consistency','false','--PatchMatchStereo.window_radius',patch_window_radius])
        shutil.copyfile(logs/'patch_match_stereo.log',logs/'patch_match_stereo_photometric.log')
        constrain_depth_maps(dense,foregrounds,'photometric')
    run('patch_match_stereo',['--workspace_path',dense,'--workspace_format','COLMAP','--PatchMatchStereo.geom_consistency','true',
                             '--PatchMatchStereo.window_radius',patch_window_radius])
    if foregrounds is not None:
        constrain_depth_maps(dense,foregrounds,'geometric')
    run('stereo_fusion',['--workspace_path',dense,'--workspace_format','COLMAP','--input_type','geometric','--output_path',output/'dense_raw.ply'])
    return _export_surface(output,masks,cameras,input_images,submodels,run)


def _export_surface(output,masks,cameras,input_images,submodels,run,mesh=True):
    """Filter and mesh an existing fused cloud, preserving oriented normals."""
    logs=output/'logs'
    points,_,attributes=read_ply(output/'dense_raw.ply',with_attributes=True)
    if not all(k in attributes for k in ['nx','ny','nz']):
        raise RuntimeError('Fused PLY has no normals; Poisson requires oriented normals')
    normals=np.column_stack([attributes[k] for k in ['nx','ny','nz']])
    colors=np.column_stack([attributes[k] for k in ['red','green','blue']]) if all(k in attributes for k in ['red','green','blue']) else None
    if masks:
        votes=np.zeros(len(points),int)
        for c in cameras:
            mask=cv2.imdecode(np.fromfile(str(masks/(c['file']+'.png')),np.uint8),0)
            R=np.array(c['R']); t=np.array(c['t']); K=np.array(c['K'])
            for begin in range(0,len(points),20000):
                pp=points[begin:begin+20000]; uv,_=cv2.projectPoints(pp,cv2.Rodrigues(R)[0],t,K,np.array(c['distortion']))
                uv=uv.reshape(-1,2); finite=np.isfinite(uv).all(axis=1)
                xy=np.round(np.where(finite[:,None],uv,-1)).astype(int)
                valid=finite&((pp@R.T+t)[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
                ids=np.flatnonzero(valid); valid[ids]&=mask[xy[ids,1],xy[ids,0]]>0
                votes[begin:begin+len(pp)]+=valid
        keep=votes>=2; points=points[keep]; normals=normals[keep]
        if colors is not None: colors=colors[keep]
    if len(points)<16: raise RuntimeError('Too few foreground dense points to analyze')
    write_ply(output/'dense.ply',points,colors=colors,normals=normals)
    if not mesh: return cameras
    run('poisson_mesher',['--input_path',output/'dense.ply','--output_path',output/'mesh.ply'])
    mesh_vertices,mesh_faces=read_ply(output/'mesh.ply')
    meshing={'depth':13,'trim':10}
    if not len(mesh_faces) or len(mesh_vertices)<max(3,len(points)*.01):
        # Success may still mean empty or almost entirely removed geometry.
        shutil.copyfile(logs/'poisson_mesher.log',logs/'poisson_mesher_default.log')
        meshing={'depth':10,'trim':5}
        run('poisson_mesher',['--input_path',output/'dense.ply','--output_path',output/'mesh.ply',
                             '--PoissonMeshing.depth','10','--PoissonMeshing.trim','5'])
        mesh_vertices,mesh_faces=read_ply(output/'mesh.ply')
        if not len(mesh_vertices) or not len(mesh_faces):
            meshing={'depth':10,'trim':0}
            run('poisson_mesher',['--input_path',output/'dense.ply','--output_path',output/'mesh.ply',
                                 '--PoissonMeshing.depth','10','--PoissonMeshing.trim','0'])
            mesh_vertices,mesh_faces=read_ply(output/'mesh.ply')
    if not len(mesh_vertices) or not len(mesh_faces):
        raise RuntimeError('Poisson produced an empty mesh; dense.ply remains available')
    save_json(output/'reconstruction_status.json',{'input_images':input_images,'registered_images':len(cameras),
              'submodels':submodels,'dense_complete':True,'coordinate_system':'SfM object frame, arbitrary scale',
              'dense_points':len(points),'mesh_vertices':len(mesh_vertices),'mesh_faces':len(mesh_faces),'meshing':meshing,
              'notes':'Poisson may fill unseen surfaces; both meshing input and dense.ply are foreground-filtered. Review unsupported regions.'})
    return cameras
