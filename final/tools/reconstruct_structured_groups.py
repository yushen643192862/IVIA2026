"""Decode, measure and compare every captured angle; default contrast is 3 DN / 2%."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json
from tools.decode_capture_session import process as decode
from tools.reconstruct_structured_depth import process as measure
from tools.compare_structured_views import process as compare


def process(session,output,calibration,masks,reference_depth,bit_threshold=3.,relative_threshold=.02,
            white_threshold=20.,projector_offset=(-102.,-44.),limit=0):
    session,output = Path(session).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new output directory')
    if bit_threshold<=0 or white_threshold<=0 or not 0<relative_threshold<=1 or limit<0:
        raise ValueError('Invalid decoding parameters')
    output.mkdir(parents=True)
    parameters = {'session':str(session),'bit_threshold_dn':bit_threshold,
        'relative_threshold_fraction':relative_threshold,'white_threshold_dn':white_threshold,
        'projector_offset_xy_px':list(projector_offset),'calibration':str(Path(calibration).resolve()),
        'masks':str(Path(masks).resolve()),'reference_depth':str(Path(reference_depth).resolve()),'limit':limit,
        'note':'The camera/projector compensation is empirical. Old decoded maps, point clouds and model are not modified.'}
    save_json(output/'parameters.json',parameters)
    save_json(output/'status.json',{'complete':False,'phase':'decode'})
    decode(session,output/'decode',bit_threshold,white_threshold,relative_threshold,limit)
    save_json(output/'status.json',{'complete':False,'phase':'depth'})
    measure(output/'decode',calibration,output/'depth',masks=masks,
        projector_offset=projector_offset,reference_depth_dir=reference_depth)
    save_json(output/'status.json',{'complete':False,'phase':'comparisons'})
    compare(output/'depth',output/'comparisons')
    decoded = json.loads((output/'decode/decode_summary.json').read_text(encoding='utf-8'))
    depth = json.loads((output/'depth/depth_summary.json').read_text(encoding='utf-8'))
    results = [{'group':r['group'],'decoded_pixels':d['valid_pixels'],'cloud_points':r['points'],
        'far_background_points_removed':r['far_background_points_removed'],
        'depth_cutoff_mm':r['background_depth_gap']['cutoff_mm'],
        'accepted_fraction':r['points']/max(1,r['input_correspondences']),
        'reprojection_p95_px':r['diagnostics']['reprojection_error_p95_px']}
        for d,r in zip(decoded['groups'],depth['groups'])]
    summary = {'complete':True,'groups':len(results),'total_cloud_points':sum(r['cloud_points'] for r in results),
        'median_cloud_points':sorted(r['cloud_points'] for r in results)[len(results)//2],
        'parameters':parameters,'results':results,
        'next_step':'Review each angle before any new registration or mesh; shadow cutoffs come from per-group reference gaps.'}
    save_json(output/'summary.json',summary)
    save_json(output/'status.json',{'complete':True,'phase':'done','groups':len(results)})
    print('Complete:',len(results),'groups;',summary['total_cloud_points'],'cloud points. Open',output/'comparisons/index.html',flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',nargs='?',type=Path,default=ROOT/'outputs/structured_light/structured_capture/20261026_183358_297217')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--bit-threshold',type=float,default=3.,help='Absolute Gray-code pair contrast in DN; default 3')
    p.add_argument('--relative-threshold',type=float,default=.02,help='Fraction of local white-black contrast; default .02=2%%')
    p.add_argument('--white-threshold',type=float,default=20.)
    p.add_argument('--projector-offset',type=float,nargs=2,default=(-102.,-44.),metavar=('DX','DY'))
    p.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    p.add_argument('--masks-directory',type=Path,default=ROOT/'outputs/structured_light/structured_subject_green_only_v2/masks')
    p.add_argument('--reference-depth-dir',type=Path,default=ROOT/'outputs/structured_light/structured_depth_green_only_v2')
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    process(args.session,args.output,args.calibration,args.masks_directory,args.reference_depth_dir,
        args.bit_threshold,args.relative_threshold,args.white_threshold,args.projector_offset,args.limit)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
