"""Controlled subset calibration experiments with common held-out images."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.calibration.read_text(encoding='utf-8'))
    cols, rows = data['board_inner_corners']
    size = tuple(data['image_size_wh'])
    obj = np.zeros((cols*rows,3),np.float32)
    obj[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)*data['square_size']
    corners, metrics = [], []
    for v in data['views']:
        path = args.images/v['file']
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[::-1] != size:
            raise ValueError(f'Invalid image: {path}')
        ok, pts = cv2.findChessboardCorners(image,(cols,rows),cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            raise RuntimeError(f'Corner detection failed: {path}')
        pts = pts.reshape(-1,1,2)
        cv2.cornerSubPix(image,pts,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,50,.001))
        corners.append(pts)
        center = pts.reshape(-1,2).mean(axis=0)/np.array(size)
        hull = cv2.convexHull(pts.reshape(-1,2))
        R = np.array(v['rotation_matrix'])
        tilt = float(np.degrees(np.arccos(np.clip(abs(R[2,2]),0,1))))
        metrics.append({'file':v['file'], 'center_x':float(center[0]),'center_y':float(center[1]),
                        'center_distance':float(np.linalg.norm(center-.5)),
                        'board_area_fraction':float(cv2.contourArea(hull)/(size[0]*size[1])),
                        'tilt_deg':tilt})
    if len(corners) != 19:
        raise ValueError('This experiment plan expects the current 19-image dataset')
    # Fixed shared validation set. These images are never used to fit subset intrinsics.
    test = [2,6,11,16]
    pool = [i for i in range(19) if i not in test]
    rng = np.random.default_rng(20261010)
    jobs = []
    for n in [5,10,15]:
        for repeat in range(10 if n<15 else 1):
            jobs.append(('image_count',str(n),repeat,sorted(rng.choice(pool,n,replace=False).tolist())))
    by_center = sorted(pool,key=lambda i:metrics[i]['center_distance'])
    by_angle = sorted(pool,key=lambda i:metrics[i]['tilt_deg'])
    jobs += [('coverage','central',0,by_center[:6]),('coverage','peripheral',0,by_center[-6:]),
             ('angle','low_tilt',0,by_angle[:6]),('angle','high_tilt',0,by_angle[-6:])]
    results = []
    for factor,group,repeat,indices in jobs:
        rms,K,dist,_,_ = cv2.calibrateCamera([obj.copy() for _ in indices],
                                            [corners[i] for i in indices],size,None,None)
        errors=[]
        for i in test:
            ok,rv,tv = cv2.solvePnP(obj,corners[i],K,dist)
            if not ok:
                raise RuntimeError('Validation pose estimation failed')
            projected,_ = cv2.projectPoints(obj,rv,tv,K,dist)
            e = np.linalg.norm(projected.reshape(-1,2)-corners[i].reshape(-1,2),axis=1)
            errors.extend(e.tolist())
        results.append({'factor':factor,'group':group,'repeat':repeat,'train_count':len(indices),
                        'train_rmse_px':float(rms),'validation_rmse_px':float(np.sqrt(np.mean(np.square(errors)))),
                        'fx':float(K[0,0]),'fy':float(K[1,1]),'cx':float(K[0,2]),'cy':float(K[1,2]),
                        'mean_tilt_deg':float(np.mean([metrics[i]['tilt_deg'] for i in indices])),
                        'mean_center_distance':float(np.mean([metrics[i]['center_distance'] for i in indices])),
                        'mean_board_area_fraction':float(np.mean([metrics[i]['board_area_fraction'] for i in indices])),
                        'training_files':[metrics[i]['file'] for i in indices],
                        'validation_files':[metrics[i]['file'] for i in test],
                        'camera_matrix':K.tolist(),'distortion':dist.ravel().tolist()})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'experiments.json').write_text(json.dumps({'seed':20261010,'image_metrics':metrics,'runs':results},indent=2),encoding='utf-8')
    keys=[k for k in results[0] if k not in ['camera_matrix','distortion']]
    with (args.output/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader()
        for r in results:
            writer.writerow({k: ';'.join(r[k]) if isinstance(r[k],list) else r[k] for k in keys})
    summary=[]
    for factor,group in dict.fromkeys((r['factor'],r['group']) for r in results):
        runs=[r for r in results if r['factor']==factor and r['group']==group]
        summary.append({'factor':factor,'group':group,'runs':len(runs),'n':runs[0]['train_count'],
                        'train_mean':float(np.mean([r['train_rmse_px'] for r in runs])),
                        'validation_mean':float(np.mean([r['validation_rmse_px'] for r in runs])),
                        'validation_std':float(np.std([r['validation_rmse_px'] for r in runs])),
                        'fx_std':float(np.std([r['fx'] for r in runs])),
                        'tilt_mean':float(np.mean([r['mean_tilt_deg'] for r in runs])),
                        'center_distance_mean':float(np.mean([r['mean_center_distance'] for r in runs])),
                        'area_mean':float(np.mean([r['mean_board_area_fraction'] for r in runs]))})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
