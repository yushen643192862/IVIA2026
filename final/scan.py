import argparse
import json
import importlib.util
import shutil
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description='Adaptive two-axis object scanner')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor'); p.add_argument('--colmap',default='colmap')
    p=sub.add_parser('demo'); p.add_argument('--output',type=Path,default=Path('outputs/demo')); p.add_argument('--max-samples',type=int,default=240)
    p=sub.add_parser('reconstruct'); p.add_argument('--images',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',type=Path); p.add_argument('--static-scene',action='store_true'); p.add_argument('--sparse-only',action='store_true'); p.add_argument('--colmap',default='colmap')
    p.add_argument('--high-detail',action='store_true',help='CUDA SIFT with a lower feature threshold and guided matching')
    p.add_argument('--camera-params',help='Initial SIMPLE_RADIAL parameters f,cx,cy,k (useful for images without EXIF)')
    p.add_argument('--patch-window-radius',type=int,default=5,help='Dense matching radius: 5 gives 11x11, 10 gives 21x21')
    p=sub.add_parser('resume-dense'); p.add_argument('--images',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',type=Path); p.add_argument('--colmap',default='colmap')
    p.add_argument('--patch-window-radius',type=int,default=5)
    p=sub.add_parser('run'); p.add_argument('--images',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masks',type=Path); p.add_argument('--static-scene',action='store_true'); p.add_argument('--colmap',default='colmap')
    p.add_argument('--platform',type=Path); p.add_argument('--max-samples',type=int,default=1200)
    p.add_argument('--patch-window-radius',type=int,default=5)
    p=sub.add_parser('analyze'); p.add_argument('--mesh',type=Path,required=True); p.add_argument('--cloud',type=Path,required=True)
    p.add_argument('--cameras',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-samples',type=int,default=1200)
    p=sub.add_parser('plan'); p.add_argument('--mesh',type=Path,required=True); p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--platform',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--top-k',type=int,default=4)
    args=parser.parse_args()
    if hasattr(args,'max_samples') and args.max_samples<1: parser.error('max-samples must be positive')
    if hasattr(args,'patch_window_radius') and args.patch_window_radius<1: parser.error('patch-window-radius must be positive')
    if args.command=='doctor':
        exe=shutil.which(args.colmap) or (str(Path(args.colmap).resolve()) if Path(args.colmap).is_file() else None)
        print(json.dumps({'python':sys.version.split()[0],'numpy':bool(importlib.util.find_spec('numpy')),
                          'opencv':bool(importlib.util.find_spec('cv2')),'colmap':exe,
                          'note':'Demo works without COLMAP; image reconstruction requires COLMAP and a supported dense backend.'},indent=2))
        return
    if args.command=='reconstruct':
        from scanner.reconstruct import reconstruct
        reconstruct(args.images,args.output,args.colmap,args.masks,args.static_scene,args.sparse_only,args.high_detail,args.patch_window_radius,camera_params=args.camera_params)
    elif args.command=='resume-dense':
        from scanner.reconstruct import resume_dense
        resume_dense(args.images,args.output,args.colmap,args.masks,args.patch_window_radius)
    elif args.command=='run':
        from scanner.reconstruct import reconstruct
        from scanner.quality import analyze
        from scanner.planner import recommend
        reconstruction=args.output/'reconstruction'
        cameras=reconstruct(args.images,reconstruction,args.colmap,args.masks,args.static_scene,patch_window_radius=args.patch_window_radius)
        report=analyze(reconstruction/'mesh.ply',reconstruction/'dense.ply',cameras,args.output/'quality',args.max_samples)
        if args.platform:
            config=load(args.platform)
            if config.get('fixture_mesh'): config['fixture_mesh']=str((args.platform.parent/config['fixture_mesh']).resolve())
            result=recommend(reconstruction/'mesh.ply',report,config,args.output/'planning')
            print('Planning:',result['status'])
        else:
            print('Reconstruction/analysis complete. Supply a calibrated --platform configuration for physical pose planning.')
    elif args.command=='analyze':
        from scanner.quality import analyze
        report=analyze(args.mesh,args.cloud,load(args.cameras) if args.cameras else [],args.output,args.max_samples)
        print('Analyzed samples:',report['sample_count'],'boundary components:',len(report['boundary_components']))
    elif args.command=='plan':
        from scanner.planner import recommend
        config=load(args.platform)
        if config.get('fixture_mesh'):
            config['fixture_mesh']=str((args.platform.parent/config['fixture_mesh']).resolve())
        result=recommend(args.mesh,load(args.quality),config,args.output,args.top_k)
        print(result['status'],'recommendations:',len(result['recommended']))
    elif args.command=='demo':
        from scanner.demo import make_demo
        from scanner.quality import analyze
        from scanner.planner import recommend
        mesh,cloud,cameras,config=make_demo(args.output)
        report=analyze(mesh,cloud,cameras,args.output/'quality',args.max_samples)
        result=recommend(mesh,report,config,args.output/'planning')
        print('SYNTHETIC DEMO completed:',args.output.resolve())
        print('Samples:',report['sample_count'],'boundaries:',len(report['boundary_components']),
              'unsupported fraction:',round(report['unsupported_fraction'],3))
        print('Recommendations:',[(r['yaw'],r['pitch']) for r in result['recommended']])
        print('These are illustrative poses, not calibrated motor commands.')


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as exc:
        print('ERROR:',exc,file=sys.stderr); sys.exit(1)
