"""Check one fixed projector offset on every decoded view before cloud export."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json
from scanner.structured_depth import triangulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('decoded_dir',type=Path)
    p.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    p.add_argument('--offset',type=float,nargs=2,default=(-102.,-44.))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    summary = json.loads((args.decoded_dir/'decode_summary.json').read_text(encoding='utf-8'))
    c = json.loads(args.calibration.read_text(encoding='utf-8'))
    x,y,w,h = summary['crop_roi_xywh']
    rows = []
    for g in summary['groups']:
        name = g['group']
        with np.load(args.decoded_dir/name/'decoded.npz',allow_pickle=False) as a:
            yy,xx = np.nonzero(a['valid'])
            pp = np.column_stack((a['projector_x'][yy,xx],a['projector_y'][yy,xx]))+args.offset
        diag = {}
        xyz,keep,errors = triangulate(np.column_stack((xx+x,yy+y)),pp,c,diagnostics=diag)
        finite = errors[np.isfinite(errors)]
        fraction = float(keep.mean()) if len(keep) else 0.
        p95 = float(np.percentile(finite,95)) if len(finite) else None
        passed = fraction>=.95 and p95 is not None and p95<=1.5
        row = {'group':name,'passed':passed,'kept_fraction':fraction,'diagnostics':diag}
        rows.append(row)
        print(f'{name}: pass={passed}, kept={fraction:.2%}, reprojection p95={p95}',flush=True)
    report = {'all_groups_passed':all(r['passed'] for r in rows),'projector_offset_xy_px':args.offset,
        'criteria':'At least 95% of decoded correspondences pass all geometric checks; unfiltered reprojection p95 <=1.5 px',
        'empirical':True,'independent_metric_accuracy_validated':False,'groups':rows}
    save_json(args.output,report)
    print('All groups passed:',report['all_groups_passed'],flush=True)


if __name__=='__main__':
    main()
