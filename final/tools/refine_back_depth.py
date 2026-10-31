"""Controlled stereo experiment at unchanged camera geometry and resolution.

Keep baseline front-view depth maps and all camera poses; recompute only named
back references with wider matching windows and neighboring viewing directions.
Never overwrite the baseline reconstruction or reuse a partly filled target.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scanner.io import save_json
from scanner.reconstruct import _export_surface

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--colmap',type=Path,required=True)
parser.add_argument('--window-radius',type=int,default=9)
parser.add_argument('--all-views',action='store_true',help='Recompute every registered reference view')
parser.add_argument('--source-mode',choices=['baseline','neighbors'])
parser.add_argument('--filter-min-ncc',type=float)
parser.add_argument('--resume',action='store_true',help='Continue only when experiment settings match the recorded run')
args=parser.parse_args()
source_mode=args.source_mode or ('baseline' if args.all_views else 'neighbors')
minimum_ncc=args.filter_min_ncc if args.filter_min_ncc is not None else (.1 if args.all_views else .05)
source=args.source.resolve(); output=args.output.resolve()
if (output/'dense').exists() and not args.resume: raise ValueError('Use a new target directory or --resume with identical settings')
if not (source/'dense.ply').is_file(): raise ValueError('Baseline dense result is required')
if args.resume:
    recorded=json.loads((output/'stereo_experiment.json').read_text(encoding='utf8'))
    expected={'baseline':str(source),'all_views':args.all_views,'source_mode':source_mode,
              'window_radius':args.window_radius,'filter_min_ncc':minimum_ncc}
    if any(recorded.get(k)!=v for k,v in expected.items()):
        raise ValueError('Resume settings do not match recorded experiment; use a new target')
    if (source/'cameras.json').read_bytes()!=(output/'cameras.json').read_bytes():
        raise ValueError('Baseline cameras changed; cannot reuse cached depth')
    if json.loads((output/'reconstruction_status.json').read_text(encoding='utf8')).get('dense_complete'):
        print('Experiment already complete:',output)
        sys.exit(0)
output.mkdir(parents=True,exist_ok=True)
logs=output/'logs'; logs.mkdir(exist_ok=True)
cameras=json.loads((source/'cameras.json').read_text(encoding='utf8'))
by_name={c['file']:c for c in cameras}
suffixes=['194544','194548','194550','194555','194558','194601','194605','194607']
references=['IMG_20261024_'+suffix+'.jpg' for suffix in suffixes]
if args.all_views: references=sorted(by_name)
if not set(references)<=set(by_name): raise ValueError('Expected back images missing')
dense=output/'dense'
dense.mkdir(exist_ok=True)
if not args.resume:
    shutil.copytree(source/'dense/images',dense/'images')
    shutil.copytree(source/'dense/sparse',dense/'sparse')
for folder in ['depth_maps','normal_maps','consistency_graphs']:
    target=dense/'stereo'/folder
    target.mkdir(parents=True,exist_ok=True)
    for path in (source/'dense/stereo'/folder).glob('*'):
        if not args.resume and path.is_file() and not any(path.name.startswith(name+'.') for name in references):
            shutil.copy2(path,target/path.name)
if not args.resume:
    shutil.copy2(source/'dense/stereo/fusion.cfg',dense/'stereo/fusion.cfg')
    for name in ['cameras.json','sparse.ply','sparse_statistics.json']:
        shutil.copy2(source/name,output/name)
configuration=[]
lines=[]
baseline_lines=(source/'dense/stereo/patch-match.cfg').read_text(encoding='utf8').splitlines()
baseline_sources=dict(zip(baseline_lines[::2],baseline_lines[1::2]))
for name in references:
    if source_mode=='baseline':
        specification=baseline_sources[name]
        lines.extend([name,specification])
        configuration.append({'reference':name,'source_specification':specification})
        continue
    direction=np.array(by_name[name]['R'])[2]
    neighbors=[]
    for other in cameras:
        if other['file']==name: continue
        angle=float(np.degrees(np.arccos(np.clip(direction@np.array(other['R'])[2],-1,1))))
        if 3<=angle<=70: neighbors.append((angle,other['file']))
    neighbors.sort()
    selected=neighbors[:10]
    if len(selected)<3: raise ValueError(f'Too few neighboring source views: {name}')
    lines.extend([name,', '.join(other for _,other in selected)])
    configuration.append({'reference':name,'sources':[{'image':other,'view_angle_degrees':angle} for angle,other in selected]})
configuration_text='\n'.join(lines)+'\n'
if args.resume:
    if (dense/'stereo/patch-match.cfg').read_text(encoding='utf8')!=configuration_text:
        raise ValueError('Source-view configuration changed; cannot reuse cached depth')
else:
    (dense/'stereo/patch-match.cfg').write_text(configuration_text,encoding='utf8')
experiment={'baseline':str(source),'references':configuration,
    'all_views':args.all_views,'source_mode':source_mode,
    'window_radius':args.window_radius,'filter_min_ncc':minimum_ncc,'geometric_consistency':True,
    'resolution':'unchanged from baseline (maximum edge 1600)',
    'note':('Every registered view is recomputed. Camera poses, masks and source-view specifications are unchanged.'
            if args.all_views and source_mode=='baseline' else
            'Non-reference front/side depth maps are unchanged baseline data. Evaluate gains against baseline before selecting this result.')}
if not args.resume:
    save_json(output/'back_refinement.json',experiment)
    save_json(output/'stereo_experiment.json',experiment)
    save_json(output/'reconstruction_status.json',{'dense_complete':False,'input_images':len(cameras),
        'registered_images':len(cameras),'stage':'all_view_window_experiment' if args.all_views else 'back_depth_refinement',
        'baseline':str(source),'window_radius':args.window_radius,'window_size':2*args.window_radius+1})
history=json.loads((logs/'commands.json').read_text(encoding='utf8')) if args.resume else []
def run(command,arguments):
    cmd=[str(args.colmap.resolve()),command,*map(str,arguments)]
    history.append(cmd); save_json(logs/'commands.json',history)
    print('Running:',command,flush=True)
    with (logs/(command+'.log')).open('a' if args.resume else 'w',encoding='utf8') as stream:
        completed=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if completed.returncode: raise RuntimeError(f'{command} failed; see {logs}')
run('patch_match_stereo',['--workspace_path',dense,'--workspace_format','COLMAP',
    '--PatchMatchStereo.geom_consistency','true','--PatchMatchStereo.window_radius',args.window_radius,
    '--PatchMatchStereo.filter_min_ncc',minimum_ncc])
run('stereo_fusion',['--workspace_path',dense,'--workspace_format','COLMAP',
    '--input_type','geometric','--output_path',output/'dense_raw.ply'])
mask_dirs={str(Path(c['object_mask']).parent) for c in cameras}
if len(mask_dirs)!=1: raise ValueError('Expected one foreground mask directory')
_export_surface(output,Path(next(iter(mask_dirs))),cameras,len(cameras),1,run)
print('Stereo experiment complete:',output,flush=True)
