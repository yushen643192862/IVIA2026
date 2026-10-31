"""Audit coordinate offsets on existing correspondences; never apply fitted corrections."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json


def epipolar_residual(camera_pixels,projector_pixels,calibration):
    kc,kp = (np.asarray(calibration[name],float) for name in ('camera_matrix','projector_matrix'))
    dc,dp = (np.asarray(calibration[name],float) for name in ('camera_distortion','projector_distortion'))
    rotation = np.asarray(calibration['R_camera_to_projector'],float)
    tx,ty,tz = calibration['t_camera_to_projector_mm']
    essential = np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])@rotation
    a = cv2.undistortPoints(np.asarray(camera_pixels,float)[:,None,:],kc,dc).reshape(-1,2)
    b = cv2.undistortPoints(np.asarray(projector_pixels,float)[:,None,:],kp,dp).reshape(-1,2)
    line = np.column_stack((a,np.ones(len(a))))@essential.T
    denominator = np.sqrt((line[:,0]/kp[0,0])**2+(line[:,1]/kp[1,1])**2)
    return np.sum(np.column_stack((b,np.ones(len(b))))*line,axis=1)/np.maximum(denominator,1e-12)


def fit_offset(camera_pixels,projector_pixels,calibration,side):
    offset = np.zeros(2)
    condition = None
    for _ in range(20):
        cp,pp = (camera_pixels+offset,projector_pixels) if side=='camera' else (camera_pixels,projector_pixels+offset)
        error = epipolar_residual(cp,pp,calibration)
        columns = []
        for step in np.eye(2)*.1:
            shifted = epipolar_residual(cp+step,pp,calibration) if side=='camera' else epipolar_residual(cp,pp+step,calibration)
            columns.append((shifted-error)/.1)
        jacobian = np.column_stack(columns)
        condition = float(np.linalg.cond(jacobian))
        delta = np.linalg.lstsq(jacobian,-error,rcond=None)[0]
        offset += delta
        if np.linalg.norm(delta)<1e-5:
            break
    return offset,condition


def metrics(cp,pp,calibration):
    error = np.abs(epipolar_residual(cp,pp,calibration))
    return {'median_projector_epipolar_px':float(np.median(error)),
            'p95_projector_epipolar_px':float(np.percentile(error,95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decoded_dir',type=Path)
    parser.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new JSON filename')
    summary = json.loads((args.decoded_dir/'decode_summary.json').read_text(encoding='utf-8'))
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    x,y,w,h = summary['crop_roi_xywh']
    groups = summary['groups']
    chosen = [groups[i]['group'] for i in sorted({0,len(groups)//2,len(groups)-1})]
    report = {'crop_roi_xywh':[x,y,w,h],'groups':[],
        'note':'Epipolar distance in ideal undistorted projector pixels is not the two-view triangulation reprojection error. Fitted offsets are hypotheses only; calibration, crop metadata and decoded maps are unchanged.'}
    fitted = {}
    for name in chosen:
        with np.load(args.decoded_dir/name/'decoded.npz',allow_pickle=False) as data:
            yy,xx = np.nonzero(data['valid'])
            if not len(xx):
                raise ValueError(f'No valid correspondences: {name}')
            ids = np.linspace(0,len(xx)-1,min(3000,len(xx)),dtype=int)
            xx,yy = xx[ids],yy[ids]
            pp = np.column_stack((data['projector_x'][yy,xx],data['projector_y'][yy,xx])).astype(float)
        cp = np.column_stack((xx+x,yy+y)).astype(float)
        record = {'group':name,'baseline':metrics(cp,pp,calibration),'offset_hypotheses':{}}
        for side in ('camera','projector'):
            delta,condition = fit_offset(cp,pp,calibration,side)
            if side not in fitted:
                fitted[side] = delta
            shared = fitted[side]
            a,b = (cp+shared,pp) if side=='camera' else (cp,pp+shared)
            record['offset_hypotheses'][side] = {'independent_fit_offset_xy':delta.tolist(),
                'fit_condition_number':condition,'first_group_offset_xy':shared.tolist(),
                'using_first_group_offset':metrics(a,b,calibration)}
        record['crop_hypotheses'] = {label:metrics(cp+delta,pp,calibration) for label,delta in
            [('missing_crop_offset',[-x,-y]),('double_crop_offset',[x,y]),('swapped_crop_offsets',[y-x,x-y])]}
        report['groups'].append(record)
        print(name,json.dumps(record),flush=True)
    save_json(args.output,report)
    print('Saved audit:',args.output)


if __name__=='__main__':
    main()
