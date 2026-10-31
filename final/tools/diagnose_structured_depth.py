"""Read a sample of existing decoded points and print geometry diagnostics; no reconstruction exports."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.structured_depth import triangulate
from tools.decode_capture_session import read_image,save_image
from scanner.io import save_json


def comparison_regions(color,foreground_mask=None):
    """Conservative color grouping for diagnostics, not a subject segmentation."""
    hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(color.astype(np.float32))
    green = (hsv[:,:,0]>=35)&(hsv[:,:,0]<=85)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=30)
    green &= (g>r*1.25)&(g>b*1.15)
    kernel = np.ones((7,7),np.uint8)
    background = cv2.erode(green.astype(np.uint8),kernel)>0
    candidate = cv2.erode((~green).astype(np.uint8),kernel)>0
    if foreground_mask is not None:
        if foreground_mask.shape!=green.shape:
            raise ValueError('Foreground mask must match cropped color image size')
        candidate = cv2.erode((foreground_mask>127).astype(np.uint8),kernel)>0
        background &= foreground_mask<=127
    return {'green_background':background,
            'subject_mask_interior' if foreground_mask is not None else 'non_green_candidate':candidate}


def diagnose_region(valid,px,py,offset,calibration,sample):
    yy,xx = np.nonzero(valid)
    count = len(xx)
    selection = np.linspace(0,count-1,min(count,sample),dtype=int)
    xx,yy = xx[selection],yy[selection]
    pp = np.column_stack((px[yy,xx],py[yy,xx]))
    report = {'decoded_pixels_in_region':count}
    triangulate(np.column_stack((xx+offset[0],yy+offset[1])),pp,calibration,diagnostics=report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('decoded_dir',type=Path)
    p.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    p.add_argument('--group',default='group_0001')
    p.add_argument('--sample',type=int,default=5000)
    p.add_argument('--compare-green',action='store_true',help='Compare definite green background and non-green candidates separately')
    p.add_argument('--foreground-mask',type=Path,help='Optional reviewed crop-size binary subject mask')
    p.add_argument('--output',type=Path,help='New directory for diagnostic JSON and region overlay')
    args = p.parse_args()
    if args.foreground_mask and not args.compare_green:
        p.error('--foreground-mask requires --compare-green')
    if args.output and args.output.exists():
        p.error('Output exists; use a new diagnostic directory')
    if args.sample<1 or Path(args.group).name!=args.group:
        p.error('Invalid sample count or group name')
    summary = json.loads((args.decoded_dir/'decode_summary.json').read_text(encoding='utf-8'))
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    x,y,w,h = summary['crop_roi_xywh']
    with np.load(args.decoded_dir/args.group/'decoded.npz',allow_pickle=False) as data:
        if data['valid'].shape!=(h,w):
            raise ValueError('Decoded shape does not match crop metadata')
        valid = data['valid'].astype(bool)
        px,py = data['projector_x'],data['projector_y']
    if px.shape!=valid.shape or py.shape!=valid.shape:
        raise ValueError('Projector coordinate map shape mismatch')
    valid &= np.isfinite(px)&np.isfinite(py)&(px>=0)&(py>=0)
    report = {'group':args.group,'crop_roi_xywh':[x,y,w,h]}
    report['all_decoded'] = diagnose_region(valid,px,py,(x,y),calibration,args.sample)
    overlay = None
    if args.compare_green:
        color = read_image(args.decoded_dir/'color_images'/(args.group+'.png'),cv2.IMREAD_COLOR)
        if color.shape[:2]!=valid.shape:
            raise ValueError('Color image size differs from decoded maps')
        mask = read_image(args.foreground_mask) if args.foreground_mask else None
        regions = comparison_regions(color,mask)
        report['regions'] = {name:diagnose_region(valid&region,px,py,(x,y),calibration,args.sample)
                             for name,region in regions.items()}
        report['note'] = 'Non-green candidate is NOT a reviewed subject mask; it can contain table, stand and other background. Boundary band is excluded. Color grouping only affects this diagnostic.'
        overlay = (color*.35).astype(np.uint8)
        for (name,region),tint in zip(regions.items(),[(0,255,0),(255,0,255)]):
            overlay[region] = (color[region]*.5+np.asarray(tint)*.5).astype(np.uint8)
        cv2.putText(overlay,'Green: screen candidate; magenta: non-green / supplied mask',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        save_json(args.output/'diagnostics.json',report)
        if overlay is not None:
            save_image(args.output/'regions_overlay.png',overlay)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
