"""Convert decoded Gray-code correspondences into metric depth and per-view point clouds."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.structured_depth import triangulate,depth_gap_cutoff
from scanner.io import save_json
from tools.decode_capture_session import read_image,save_image


def write_cloud(path,xyz,rgb):
    vertices = np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for index,axis in enumerate(('x','y','z')):
        vertices[axis] = xyz[:,index]
    for index,channel in enumerate(('red','green','blue')):
        vertices[channel] = rgb[:,index]
    header = (f'ply\nformat binary_little_endian 1.0\ncomment units millimeters; camera coordinates\nelement vertex {len(xyz)}\n'
        'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('xb') as stream:
        stream.write(header.encode('ascii')); stream.write(vertices.tobytes())


def process(decoded_dir,calibration_path,output,masks=None,min_depth=50.,max_depth=3000.,max_error=1.5,limit=0,projector_offset=(0.,0.),reference_depth_dir=None,mask_inset_px=6.):
    decoded_dir,calibration_path,output = Path(decoded_dir).resolve(),Path(calibration_path).resolve(),Path(output).resolve()
    summary = json.loads((decoded_dir/'decode_summary.json').read_text(encoding='utf-8'))
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    for key in ('serial','Width','Height','OffsetX','OffsetY'):
        if summary['camera'].get(key)!=calibration['camera'].get(key):
            raise ValueError(f'Calibration camera geometry differs: {key}')
    for key in ('width','height'):
        if summary['projector'][key]!=calibration['projector'][key]:
            raise ValueError(f'Calibration projector resolution differs: {key}')
    x0,y0,width,height = summary['crop_roi_xywh']
    if min(x0,y0)<0 or min(width,height)<1 or x0+width>summary['camera']['Width'] or y0+height>summary['camera']['Height']:
        raise ValueError('Crop is outside full camera image')
    groups = summary['groups'][:limit] if limit else summary['groups']
    if not groups:
        raise ValueError('No decoded groups')
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    output.mkdir(parents=True)
    if masks:
        (output/'effective_masks').mkdir()
    save_json(output/'calibration_used.json',calibration)
    save_json(output/'coordinate_compensation.json',{'projector_offset_xy_px':list(projector_offset),
        'empirical':bool(np.any(projector_offset)),'independently_validated':False,
        'convention':'projector_used = decoded_projector + offset; original maps/calibration unchanged'})
    reference_cutoffs = {}
    if reference_depth_dir is not None:
        reference_depth_dir = Path(reference_depth_dir).resolve()
        for group in groups:
            name = group['group']
            with np.load(reference_depth_dir/name/'depth_mm.npz',allow_pickle=False) as reference:
                reference_z = reference['depth_mm']
                if reference_z.shape!=(height,width):
                    raise ValueError(f'Reference depth size mismatch: {name}')
                cutoff,gap = depth_gap_cutoff(reference_z)
                if cutoff is None:
                    raise ValueError(f'No safe distant-background depth gap in {name}; refusing an arbitrary cutoff')
                reference_cutoffs[name] = {'cutoff_mm':cutoff,'empty_gap_mm':gap}
        save_json(output/'background_depth_cutoffs.json',{'source':str(reference_depth_dir),'groups':reference_cutoffs,
            'method':'midpoint of largest >=15 mm depth gap above reference 95th percentile'})
    records = []
    save_json(output/'status.json',{'complete':False,'groups_done':0,'groups_total':len(groups)})
    for group in groups:
        name = group['group']
        if Path(name).name!=name or not name.startswith('group_'):
            raise ValueError('Invalid group name')
        print(f'Triangulating {name} ...',flush=True)
        with np.load(decoded_dir/name/'decoded.npz',allow_pickle=False) as data:
            valid = data['valid'].astype(bool)
            px,py = data['projector_x'],data['projector_y']
        if valid.shape!=(height,width) or px.shape!=valid.shape or py.shape!=valid.shape:
            raise ValueError(f'Decoded size mismatch: {name}')
        valid &= np.isfinite(px)&np.isfinite(py)&(px>=0)&(py>=0)
        valid &= (px<summary['projector']['width'])&(py<summary['projector']['height'])
        mask_inset_removed = 0
        if masks:
            path = Path(masks)/(name+'.png')
            mask = read_image(path)
            if mask.shape!=valid.shape:
                raise ValueError(f'Mask size differs; no automatic resizing: {path}')
            foreground = (mask>127).astype(np.uint8)
            if mask_inset_px:
                distance_to_background = cv2.distanceTransform(foreground,cv2.DIST_L2,5)
                interior = distance_to_background>mask_inset_px
                mask_inset_removed = int(np.count_nonzero(valid & (foreground>0) & ~interior))
                valid &= interior
            else:
                valid &= foreground>0
                interior = foreground>0
            save_image(output/'effective_masks'/(name+'.png'),interior.astype(np.uint8)*255)
        yy,xx = np.nonzero(valid)
        camera_pixels = np.column_stack((xx+x0,yy+y0))
        projector_pixels = np.column_stack((px[yy,xx],py[yy,xx]))
        projector_pixels = projector_pixels+np.asarray(projector_offset)
        diagnostics = {}
        points,keep,errors = triangulate(camera_pixels,projector_pixels,calibration,min_depth,max_depth,max_error,diagnostics=diagnostics)
        far_removed = 0
        if reference_cutoffs:
            far = keep&(points[:,2]>reference_cutoffs[name]['cutoff_mm'])
            far_removed = int(far.sum())
            keep &= ~far
        if not keep.any():
            folder = output/name; folder.mkdir()
            save_json(folder/'failure_diagnostics.json',diagnostics)
            save_json(output/'status.json',{'complete':False,'failed':True,'failed_group':name,
                'groups_done':len(records),'groups_total':len(groups),'diagnostics':diagnostics,
                'error':'No points passed geometry checks; no empty cloud exported. Check calibration/capture geometry compatibility.'})
            raise ValueError(f'{name}: zero valid points; see {folder / "failure_diagnostics.json"}. Reprojection pass: {diagnostics["reprojection_pass"]}/{len(keep)}')
        color = read_image(decoded_dir/'color_images'/(name+'.png'),cv2.IMREAD_COLOR)
        if color.shape[:2]!=valid.shape:
            raise ValueError(f'Color image size mismatch: {name}')
        folder = output/name; folder.mkdir()
        depth = np.full(valid.shape,np.nan,np.float32)
        depth[yy[keep],xx[keep]] = points[keep,2]
        np.savez_compressed(folder/'depth_mm.npz',depth_mm=depth,valid=np.isfinite(depth),
            xyz_mm=points[keep],pixels_xy=np.column_stack((xx[keep],yy[keep])),reprojection_error_px=errors[keep])
        write_cloud(folder/'cloud.ply',points[keep],color[yy[keep],xx[keep],::-1])
        preview = np.zeros((*valid.shape,3),np.uint8)
        if keep.any():
            low,high = np.percentile(points[keep,2],[2,98])
            scaled = np.zeros(valid.shape,np.uint8)
            scaled[yy[keep],xx[keep]] = np.clip((points[keep,2]-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8)
            preview = cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
            preview[~np.isfinite(depth)] = 0
        else:
            low,high = None,None
        save_image(folder/'depth_preview.png',preview)
        record = {'group':name,'input_correspondences':len(keep),'points':int(keep.sum()),
            'far_background_points_removed':far_removed,'background_depth_gap':reference_cutoffs.get(name),
            'projector_offset_xy_px':list(projector_offset),'diagnostics':diagnostics,
            'median_reprojection_error_px':float(np.median(errors[keep])) if keep.any() else None,
            'preview_depth_limits_mm':[low,high],'foreground_mask_applied':masks is not None,
            'mask_inset_px':mask_inset_px if masks else 0.,'mask_inset_correspondences_removed':mask_inset_removed,
            'coordinate_frame':'camera; x right, y down, z forward; millimeters'}
        save_json(folder/'quality.json',record)
        records.append(record)
        save_json(output/'status.json',{'complete':False,'groups_done':len(records),'groups_total':len(groups)})
        print(f"  kept {record['points']}/{len(keep)} points",flush=True)
    result = {'complete':True,'source':str(decoded_dir),'calibration_source':str(calibration_path),
        'reference_depth_source':str(reference_depth_dir) if reference_depth_dir else None,
        'projector_offset_xy_px':list(projector_offset),'empirical_compensation':bool(np.any(projector_offset)),
        'groups':records,'foreground_masks':str(Path(masks).resolve()) if masks else None,
        'min_depth_mm':min_depth,'max_depth_mm':max_depth,'max_reprojection_error_px':max_error,
        'mask_inset_px':mask_inset_px if masks else 0.,
        'calibration_provisional':calibration.get('provisional',True),
        'registered':False,'mesh_complete':False,
        'note':'Independent per-view camera-frame clouds. Rotated-object views are not aligned; do not concatenate as a model. Without masks, backgrounds remain.'}
    save_json(output/'depth_summary.json',result)
    save_json(output/'status.json',{'complete':True,'groups_done':len(records),'groups_total':len(groups)})
    print('Saved per-view depth and clouds:',output,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decoded_dir',type=Path)
    parser.add_argument('--calibration',type=Path,default=ROOT/'outputs/structured_light/calibration/current/calibration.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--masks-directory',type=Path,help='Binary crop-size foreground masks named group_0001.png, etc.')
    parser.add_argument('--mask-inset-px',type=float,default=6.,help='Discard depth pixels within this distance of the mask background (default: 6 px)')
    parser.add_argument('--min-depth-mm',type=float,default=50.)
    parser.add_argument('--max-depth-mm',type=float,default=3000.)
    parser.add_argument('--max-reprojection-error',type=float,default=1.5)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--projector-offset',type=float,nargs=2,default=(0.,0.),metavar=('DX','DY'),help='Explicit empirical offset in projector pixels; original maps remain unchanged')
    parser.add_argument('--reference-depth-dir',type=Path,help='Previous conservative depth maps to derive a separate distant-background cutoff for each group')
    args = parser.parse_args()
    if not 0<args.min_depth_mm<args.max_depth_mm or args.max_reprojection_error<=0 or args.limit<0 or args.mask_inset_px<0:
        parser.error('Invalid depth/error/limit parameters')
    try:
        process(args.decoded_dir,args.calibration,args.output,args.masks_directory,
            args.min_depth_mm,args.max_depth_mm,args.max_reprojection_error,args.limit,args.projector_offset,args.reference_depth_dir,args.mask_inset_px)
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
