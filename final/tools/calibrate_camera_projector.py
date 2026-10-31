"""Decode printed-board captures and estimate a preliminary camera/projector calibration."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.structured_decode import decode_graycode
from tools.decode_capture_session import read_image, save_image
from tools.capture_structured_light import save_json


def projector_corners(decoded, corners, radius=30):
    """Fit local camera-to-projector homographies on the planar printed board."""
    height,width = decoded['valid'].shape
    projected,indices,errors = [],[],[]
    for index,(cx,cy) in enumerate(corners):
        x0,x1 = max(0,int(cx)-radius),min(width,int(cx)+radius+1)
        y0,y1 = max(0,int(cy)-radius),min(height,int(cy)+radius+1)
        yy,xx = np.nonzero(decoded['valid'][y0:y1,x0:x1])
        if len(xx)<30:
            continue
        xx,yy = xx+x0,yy+y0
        # Bound memory/work while retaining spatially distributed samples.
        step = max(1,len(xx)//3000)
        xx,yy = xx[::step],yy[::step]
        camera = np.column_stack((xx,yy)).astype(np.float32)
        projector = np.column_stack((decoded['projector_x'][yy,xx],decoded['projector_y'][yy,xx])).astype(np.float32)
        homography,inliers = cv2.findHomography(camera,projector,cv2.RANSAC,1.5)
        if homography is None or inliers is None:
            continue
        good = inliers.ravel().astype(bool)
        if good.sum()<20 or good.mean()<.5:
            continue
        # Reject extrapolation from a one-sided patch: the corner must lie inside its support.
        hull = cv2.convexHull(camera[good])
        if cv2.pointPolygonTest(hull,(float(cx),float(cy)),False)<0:
            continue
        estimated = cv2.perspectiveTransform(camera[good,None,:],homography).reshape(-1,2)
        rms = float(np.sqrt(np.mean(np.sum((estimated-projector[good])**2,axis=1))))
        target = cv2.perspectiveTransform(np.array([[[cx,cy]]],np.float32),homography)[0,0]
        if not np.isfinite(target).all() or rms>1.:
            continue
        projected.append(target); indices.append(index); errors.append(rms)
    return np.asarray(indices,int),np.asarray(projected,np.float32).reshape(-1,2),errors


def solve(objects,cameras,projectors,camera_size,projector_size):
    # Three planar views cannot support a rich distortion model robustly.
    flags = cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K3
    cam_rms,kc,dc,cr,ct = cv2.calibrateCamera(objects,cameras,camera_size,None,None,flags=flags)
    proj_rms,kp,dp,pr,pt = cv2.calibrateCamera(objects,projectors,projector_size,None,None,flags=flags)
    stereo = cv2.stereoCalibrate(objects,cameras,projectors,kc,dc,kp,dp,camera_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,100,1e-7))
    rms,kc,dc,kp,dp,rotation,translation,essential,fundamental = stereo
    if not all(np.isfinite(v).all() for v in (kc,dc,kp,dp,rotation,translation)):
        raise ValueError('Non-finite calibration; acquire more diverse board poses')
    views = []
    normals = []
    for obj,cam,proj,r,t,rp,tp in zip(objects,cameras,projectors,cr,ct,pr,pt):
        cam_fit = cv2.projectPoints(obj,r,t,kc,dc)[0].reshape(-1,2)
        proj_fit = cv2.projectPoints(obj,rp,tp,kp,dp)[0].reshape(-1,2)
        rotation_board = cv2.Rodrigues(r)[0]
        normal = rotation_board[:,2]
        normals.append(normal)
        views.append({'camera_rms_px':float(np.sqrt(np.mean(np.sum((cam_fit-cam)**2,axis=1)))),
                      'projector_rms_px':float(np.sqrt(np.mean(np.sum((proj_fit-proj)**2,axis=1)))),
                      'board_normal_camera':normal.tolist(),'board_translation_camera_mm':t.ravel().tolist()})
    angles = [float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),-1,1))))
              for i,a in enumerate(normals) for b in normals[i+1:]]
    return {'camera_matrix':kc.tolist(),'camera_distortion':dc.ravel().tolist(),
        'projector_matrix':kp.tolist(),'projector_distortion':dp.ravel().tolist(),
        'R_camera_to_projector':rotation.tolist(),'t_camera_to_projector_mm':translation.ravel().tolist(),
        'transform_convention':'X_projector = R_camera_to_projector @ X_camera + t_camera_to_projector_mm',
        'baseline_mm':float(np.linalg.norm(translation)),'camera_rms_px':float(cam_rms),
        'projector_rms_px':float(proj_rms),'stereo_rms_px':float(rms),'views':views,
        'maximum_board_normal_separation_deg':max(angles,default=0.),
        'distortion_model':'OpenCV radial k1,k2 estimated; tangential p1,p2 and k3 fixed zero'}


def process(folder,output,threshold=8.,radius=30):
    folder,output = Path(folder).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new output directory')
    groups = []
    for path in sorted(folder.glob('pose_*/group.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('complete') and record['saved_images']==record['expected_images']==len(record['frames']):
            groups.append((path.parent,record))
    if len(groups)<3:
        raise ValueError(f'Need at least 3 complete poses; found {len(groups)}')
    output.mkdir(parents=True)
    objects,cameras,projectors,reports = [],[],[],[]
    common = None
    for group,record in groups:
        print(f'Decoding {group.name} ...',flush=True)
        config = (record['camera'],record['projector'],record['calibration_board']['board'])
        if common is not None and config!=common:
            raise ValueError('Camera/projector/board configuration changed between poses')
        common = config
        camera,projector,board = config
        size = (camera['Width'],camera['Height'])
        if record['crop_roi_xywh']!=[0,0,*size] or record['saved_image_size']!=list(size):
            raise ValueError('Calibration requires full camera images')
        frames = record['frames']
        lookup = {f['pattern']['name']:f for f in frames}
        def reader(pattern):
            name = lookup[pattern['name']].get('raw_file')
            if not name:
                raise ValueError('Raw measurement image missing')
            path = (group/name).resolve()
            if not path.is_relative_to(group):
                raise ValueError('Raw image path is outside capture folder')
            image = read_image(path)
            if image.shape[::-1]!=size:
                raise ValueError('Raw image size mismatch')
            return image
        decoded = decode_graycode([f['pattern'] for f in frames],reader,
            (projector['width'],projector['height']),
            next((f['pattern']['cell_size'] for f in frames if f['pattern'].get('role')=='graycode'),1),threshold)
        corners = np.asarray(record['calibration_board']['corners_before_xy'],np.float32)
        indices,projected,errors = projector_corners(decoded,corners,radius)
        destination = output/group.name; destination.mkdir()
        save_image(destination/'valid_mask.png',decoded['valid'].astype(np.uint8)*255)
        report = {'pose':group.name,'usable_corners':len(indices),'total_corners':len(corners),
            'corner_indices':indices.tolist(),'projector_corners_xy':projected.tolist(),
            'local_homography_rms_px':errors,'used':len(indices)>=12}
        save_json(destination/'correspondences.json',report)
        reports.append(report)
        print(f'  usable corners: {len(indices)}/{len(corners)}',flush=True)
        if report['used']:
            objects.append(np.asarray(record['calibration_board']['object_points_mm'],np.float32)[indices])
            cameras.append(corners[indices]); projectors.append(projected)
    save_json(output/'decode_report.json',{'poses':reports})
    if len(objects)<3:
        raise ValueError('Fewer than 3 poses have >=12 reliable projector corners; inspect decode_report.json. Do not lower thresholds just to obtain a result.')
    camera,projector,board = common
    result = solve(objects,cameras,projectors,(camera['Width'],camera['Height']),
                   (projector['width'],projector['height']))
    result.update(camera=camera,projector=projector,board=board,poses_used=[r['pose'] for r in reports if r['used']],
        provisional=True,validated_for_depth=False,
        note='Training reprojection errors are not independent accuracy validation. Check baseline and held-out board measurements before depth use; acquire 15-20 varied poses.')
    save_json(output/'calibration.json',result)
    print(f"Saved {output/'calibration.json'}\nCamera RMS: {result['camera_rms_px']:.3f} px; projector RMS: {result['projector_rms_px']:.3f} px; stereo RMS: {result['stereo_rms_px']:.3f} px\nPRELIMINARY, not independently validated for depth",flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration_folder',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--bit-threshold',type=float,default=8.)
    parser.add_argument('--corner-radius',type=int,default=30)
    args = parser.parse_args()
    if args.bit_threshold<=0 or args.corner_radius<5:
        parser.error('Invalid confidence threshold or corner radius')
    try:
        process(args.calibration_folder,args.output,args.bit_threshold,args.corner_radius)
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
