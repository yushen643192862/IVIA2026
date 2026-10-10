"""Calibrate a fixed camera from chessboard images (OpenCV + NumPy)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def save_image(path, image):
    ok, data = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    data.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('calibration_results'))
    parser.add_argument('--cols', type=int, default=10, help='Inner corner columns')
    parser.add_argument('--rows', type=int, default=7, help='Inner corner rows')
    parser.add_argument('--square-size', type=float, default=1.0,
                        help='Physical square edge length; default 1 = square units')
    parser.add_argument('--unit', default='square', help='Translation unit, e.g. mm')
    args = parser.parse_args()
    if args.square_size <= 0 or min(args.cols, args.rows) < 2:
        parser.error('Square size must be positive and corner counts at least 2')
    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    obj = np.zeros((args.cols * args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    obj *= args.square_size
    objects, corners, accepted, rejected = [], [], [], []
    size = None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'corners').mkdir(exist_ok=True)
    for path in paths:
        im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            rejected.append({'file': path.name, 'reason': 'Unreadable image'})
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if size is not None and current_size != size:
            raise ValueError(f'Image resolution differs: {path}')
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, pts = cv2.findChessboardCorners(gray, (args.cols, args.rows), flags)
        if not ok:
            rejected.append({'file': path.name, 'reason': 'Full chessboard not detected'})
            continue
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))
        size = current_size
        objects.append(obj.copy())
        corners.append(pts)
        accepted.append(path)
        cv2.drawChessboardCorners(im, (args.cols, args.rows), pts, True)
        save_image(args.output / 'corners' / (path.stem + '.jpg'), im)
    if len(accepted) < 3:
        raise RuntimeError(f'Only {len(accepted)} valid images; need at least 3 diverse views')
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, corners, size, None, None)
    views = []
    squared_error = 0.0
    point_count = 0
    for path, op, ip, rv, tv in zip(accepted, objects, corners, rvecs, tvecs):
        projected, _ = cv2.projectPoints(op, rv, tv, K, dist)
        residual = ip.reshape(-1, 2) - projected.reshape(-1, 2)
        errors = np.linalg.norm(residual, axis=1)
        squared_error += float(np.sum(errors ** 2))
        point_count += len(errors)
        R, _ = cv2.Rodrigues(rv)
        views.append({'file': path.name, 'rmse_px': float(np.sqrt(np.mean(errors ** 2))),
                      'mean_error_px': float(errors.mean()), 'max_error_px': float(errors.max()),
                      'rvec': rv.ravel().tolist(), 'rotation_matrix': R.tolist(),
                      'translation': tv.ravel().tolist(),
                      'camera_center_in_board_coordinates': (-R.T @ tv).ravel().tolist()})
    result = {'image_size_wh': list(size), 'board_inner_corners': [args.cols, args.rows],
              'square_size': args.square_size, 'translation_unit': args.unit,
              'coordinate_convention': 'X_camera = R * X_board + t; board origin at first detected inner corner, X along columns, Y along rows, Z by right hand rule',
              'camera_matrix': K.tolist(), 'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
              'distortion_coefficients': dist.ravel().tolist(), 'opencv_rms_px': float(rms),
              'recomputed_rmse_px': float(np.sqrt(squared_error / point_count)),
              'accepted_count': len(accepted), 'rejected': rejected, 'views': views}
    (args.output / 'calibration.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez(args.output / 'calibration.npz', camera_matrix=K, distortion=dist,
             rvecs=np.array(rvecs), tvecs=np.array(tvecs), image_size=np.array(size))
    lines = ['file,rmse_px,mean_error_px,max_error_px']
    lines += [f"{v['file']},{v['rmse_px']:.6f},{v['mean_error_px']:.6f},{v['max_error_px']:.6f}" for v in views]
    (args.output / 'per_image_errors.csv').write_text('\n'.join(lines), encoding='utf-8-sig')
    first = cv2.imdecode(np.fromfile(str(accepted[0]), np.uint8), cv2.IMREAD_COLOR)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    save_image(args.output / 'undistorted_example.jpg', cv2.undistort(first, K, dist, None, new_K))
    print(f'Accepted: {len(accepted)}; rejected: {len(rejected)}')
    print('Camera matrix:\n', K)
    print('Distortion [k1,k2,p1,p2,k3]:', dist.ravel())
    print(f'Reprojection RMSE: {rms:.6f} pixels')
    print('Worst views:', [(v['file'], round(v['rmse_px'], 4)) for v in sorted(views, key=lambda v: v['rmse_px'], reverse=True)[:5]])
    print('Output:', args.output.resolve())


if __name__ == '__main__':
    main()
