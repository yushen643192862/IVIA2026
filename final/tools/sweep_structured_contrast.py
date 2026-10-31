"""Controlled group-one contrast sweep, preserving masks, geometry and original outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json
from scanner.structured_decode import decode_graycode
from scanner.structured_depth import triangulate,depth_gap_cutoff
from tools.decode_capture_session import read_image,save_image
from tools.reconstruct_structured_depth import write_cloud
from tools.compare_structured_views import splat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,default=ROOT/'outputs/structured_light/structured_capture/20261026_183358_297217')
    p.add_argument('--mask',type=Path,default=ROOT/'outputs/structured_light/structured_subject_green_only_v2/masks/group_0001.png')
    p.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    p.add_argument('--thresholds',nargs='+',type=float,default=[8,7,6,5,4,3,2,1])
    p.add_argument('--relative-threshold',type=float,default=.05)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    thresholds = sorted(set(args.thresholds),reverse=True)
    if thresholds[0]!=8 or min(thresholds)<=0 or not 0<=args.relative_threshold<=1:
        p.error('Include baseline 8 as the largest threshold; thresholds must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    group = args.capture/'group_0001'
    record = json.loads((group/'group.json').read_text(encoding='utf-8'))
    if not record['complete']:
        raise ValueError('Incomplete capture')
    c = json.loads(args.calibration.read_text(encoding='utf-8'))
    lookup = {f['pattern']['name']:f for f in record['frames']}
    print('Reading raw measurement frames once ...',flush=True)
    def reader(pattern):
        return read_image(group/lookup[pattern['name']]['raw_file'])
    projector = record['projector']
    decoded = decode_graycode([f['pattern'] for f in record['frames']],reader,
        (projector['width'],projector['height']),bit_threshold=0,white_threshold=20,relative_threshold=0)
    color_record = next(f for f in record['frames'] if f['pattern'].get('purpose')=='color')
    color = read_image(group/color_record['file'],cv2.IMREAD_COLOR)
    mask = read_image(args.mask)>127
    if mask.shape!=decoded['valid'].shape:
        raise ValueError('Mask and decode dimensions differ')
    contrast = decoded['minimum_pair_contrast']
    illumination = decoded['illumination']
    x0,y0,w,h = record['crop_roi_xywh']
    baseline_valid = None
    baseline_tree = None
    frame = None
    far_cutoff = None
    gap_bounds = None
    reports,tiles,html_rows = [],[],[]
    panel_w,panel_h = 480,round(h*480/w)
    for threshold in thresholds:
        print('Threshold',threshold,flush=True)
        selected = decoded['valid']&mask&(contrast>=np.maximum(threshold,illumination*args.relative_threshold))
        yy,xx = np.nonzero(selected)
        pp = np.column_stack((decoded['projector_x'][yy,xx],decoded['projector_y'][yy,xx]))+[-102.,-44.]
        diag = {}
        xyz,keep,errors = triangulate(np.column_stack((xx+x0,yy+y0)),pp,c,diagnostics=diag)
        if far_cutoff is None and threshold==8:
            far_cutoff,gap_bounds = depth_gap_cutoff(xyz[keep,2])
            if far_cutoff is None:
                raise ValueError('No safe depth gap found in baseline; do not guess a background cutoff')
            print('Distant-background depth gap:',gap_bounds,'cutoff:',far_cutoff,flush=True)
        far = keep&(xyz[:,2]>far_cutoff)
        keep &= ~far
        valid = np.zeros(mask.shape,bool);valid[yy[keep],xx[keep]] = True
        if baseline_valid is None:
            baseline_valid = valid.copy()
            baseline_tree = cKDTree(xyz[keep])
        new = keep&(~baseline_valid[yy,xx])
        nearest_new = baseline_tree.query(xyz[new])[0] if new.any() else np.array([])
        folder = args.output/f'contrast_{threshold:g}';folder.mkdir()
        write_cloud(folder/'cloud.ply',xyz[keep],color[yy[keep],xx[keep],::-1])
        depth = np.full(mask.shape,np.nan,np.float32);depth[yy[keep],xx[keep]] = xyz[keep,2]
        np.savez_compressed(folder/'depth_mm.npz',depth_mm=depth,valid=valid)
        save_image(folder/'valid_mask.png',valid.astype(np.uint8)*255)
        bgr = color[yy[keep],xx[keep]]
        front = splat(np.column_stack((xx[keep],yy[keep]))*panel_w/w,xyz[keep,2],bgr,panel_w,panel_h)
        angle = np.radians(35.)
        rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        if frame is None:
            center = np.median(xyz[keep],axis=0)
            rotated = (xyz[keep]-center)@rotation.T
            low,high = np.percentile(rotated[:,:2],[0,100],axis=0)
            scale = min((panel_w-30)/max(high[0]-low[0],1e-6),(panel_h-30)/max(high[1]-low[1],1e-6))
            frame = center,(low+high)/2,scale
        center,middle,scale = frame
        rotated = (xyz[keep]-center)@rotation.T
        oblique = splat((rotated[:,:2]-middle)*scale+[panel_w/2,panel_h/2],rotated[:,2],bgr,panel_w,panel_h)
        added = np.full_like(color,32)
        added[baseline_valid] = (130,130,130)
        added[valid&~baseline_valid] = (0,255,0)
        rejected = np.zeros(mask.shape,bool);rejected[yy[~keep],xx[~keep]]=True
        added[rejected] = (0,0,255)
        added[yy[far],xx[far]] = (255,200,0)
        panels = []
        for im,label in [(cv2.resize(color,(panel_w,panel_h)),'White-light photo'),(front,'Measured cloud: camera view'),(oblique,'Measured cloud: fixed 35 deg view'),(cv2.resize(added,(panel_w,panel_h),interpolation=cv2.INTER_NEAREST),'Gray baseline / green added / red rejected')]:
            im=cv2.copyMakeBorder(im,38,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
            cv2.putText(im,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
            panels.append(im)
        strip = np.hstack(panels)
        strip=cv2.copyMakeBorder(strip,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(strip,f'Contrast {threshold:g} DN | relative {args.relative_threshold:.0%} | Z<={far_cutoff:.1f} mm | {int(keep.sum()):,} points | +{int(new.sum()):,} vs baseline',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1,cv2.LINE_AA)
        save_image(folder/'comparison.jpg',strip)
        tiles.append(cv2.resize(strip,(1440,round(strip.shape[0]*.75))))
        neighbors = cKDTree(xyz[keep]).query(xyz[keep],k=8,workers=2)[0][:,-1]
        report = {'bit_threshold':threshold,'relative_threshold':args.relative_threshold,'white_threshold':20,
            'decoded_foreground_pixels':len(keep),'kept_points':int(keep.sum()),'added_points':int(new.sum()),
            'far_background_points_removed':int(far.sum()),'far_depth_cutoff_mm':far_cutoff,
            'foreground_pixel_coverage':float(keep.sum()/mask.sum()),
            'geometry_pass_fraction':float(keep.mean()),
            'unfiltered_reprojection_p95_px':float(np.percentile(errors[np.isfinite(errors)],95)),
            'kept_reprojection_median_px':float(np.median(errors[keep])),
            'kept_reprojection_p95_px':float(np.percentile(errors[keep],95)),
            'isolated_fraction_8th_neighbor_over_4mm':float(np.mean(neighbors>4.)),
            'added_distance_to_baseline_p95_mm':float(np.percentile(nearest_new,95)) if new.any() else None,
            'relative_floor_dominant_fraction':float(np.mean((illumination*args.relative_threshold>=threshold)[mask]))}
        save_json(folder/'quality.json',report);reports.append(report)
        print(json.dumps(report),flush=True)
        html_rows.append(f'<section><h2>Contrast {threshold:g} DN — {report["kept_points"]:,} points (+{report["added_points"]:,})</h2><p>Geometry pass {report["geometry_pass_fraction"]:.2%}; reprojection p95 {report["unfiltered_reprojection_p95_px"]:.3f} px</p><a href="contrast_{threshold:g}/comparison.jpg"><img src="contrast_{threshold:g}/comparison.jpg"></a></section>')
    save_image(args.output/'gradient_contact.jpg',np.vstack(tiles))
    save_json(args.output/'summary.json',{'group':'group_0001','projector_offset_xy_px':[-102,-44],
        'mask':str(args.mask.resolve()),'results':reports,
        'relative_threshold':args.relative_threshold,'far_depth_cutoff_mm':far_cutoff,'baseline_far_depth_gap_mm':gap_bounds,
        'note':'Only absolute pair-contrast threshold varies. Effective threshold=max(absolute,relative_fraction*white-black). Geometry limits unchanged. The baseline depth gap removes distant points after triangulation. No smoothing/filling. Lower error alone does not prove metric accuracy.'})
    with (args.output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)
    page='<!doctype html><meta charset="utf-8"><title>Group 1 对比度阈值梯度</title><style>body{font:16px sans-serif;margin:24px;background:#eee}section{background:white;padding:12px;margin:20px 0}img{width:100%}</style><h1>Group 1 对比度阈值梯度</h1><p>绝对阈值8→1，其他解码与几何参数固定。用原阈值数据里的深度空档筛掉远处暗背景。左起：白光照片、同视角点云、统一35°视角点云、新增点。灰色=原有点，绿色=新增点，红色=几何过滤，青色=远处背景。</p>'+''.join(html_rows)
    (args.output/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    main()
