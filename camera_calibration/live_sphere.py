"""Live chessboard sphere AR using existing camera calibration, without recalibration."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


class BaslerCapture:
    """pypylon continuous acquisition with explicit grab-buffer release."""
    def __init__(self, index, serial, width, height, exposure, gain):
        try:
            from pypylon import pylon, genicam
        except ImportError as exc:
            raise RuntimeError('Basler input requires pypylon: pip install pypylon') from exc
        self.pylon = pylon
        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if serial:
            devices = [d for d in devices if d.GetSerialNumber() == serial]
            index = 0
        if index < 0 or index >= len(devices):
            raise RuntimeError('No matching Basler camera connected')
        self.camera = pylon.InstantCamera(factory.CreateDevice(devices[index]))
        try:
            self.camera.Open()
            nodes = self.camera.GetNodeMap()

            def set_optional(name, value):
                node = nodes.GetNode(name)
                if node is not None and genicam.IsWritable(node):
                    node.SetValue(value)

            # Preserve sensor ROI/binning and existing settings; do not load defaults.
            # Exact capture dimensions are required to avoid guessing ROI intrinsics.
            actual = (int(self.camera.Width.GetValue()), int(self.camera.Height.GetValue()))
            if actual != (width, height):
                raise RuntimeError(f'Basler resolution {actual} differs from requested '
                                   f'{width}x{height}. Set the calibrated ROI in pylon Viewer first.')
            set_optional('AcquisitionMode', 'Continuous')
            selector = nodes.GetNode('TriggerSelector')
            if selector is not None and genicam.IsWritable(selector):
                for entry in selector.GetEntries():
                    if genicam.IsAvailable(entry):
                        selector.SetValue(entry.GetSymbolic())
                        set_optional('TriggerMode', 'Off')
            else:
                set_optional('TriggerMode', 'Off')
            set_optional('ExposureMode', 'Timed')
            set_optional('ExposureAuto', 'Off')
            set_optional('GainAuto', 'Off')
            set_optional('AcquisitionFrameRateEnable', False)
            for name, value in [('ExposureTime', exposure), ('Gain', gain)]:
                if value is not None:
                    node = nodes.GetNode(name)
                    if node is None or not genicam.IsWritable(node):
                        raise RuntimeError(f'Camera does not support writable {name}')
                    node.SetValue(value)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            info = self.camera.GetDeviceInfo()
            print('Basler:', info.GetModelName(), 'serial:', info.GetSerialNumber(),
                  'resolution:', actual)
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            self.release()
            raise

    def read(self):
        result = self.camera.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f'Basler grab failed: {result.ErrorDescription}')
            converted = self.converter.Convert(result)
            try:
                return True, converted.GetArray().copy()
            finally:
                converted.Release()
        finally:
            result.Release()

    def release(self):
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        finally:
            if self.camera.IsOpen():
                self.camera.Close()


def sphere_mesh(radius):
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 41), np.linspace(0, np.pi, 25))
    normals = np.column_stack(((np.sin(phi)*np.cos(theta)).ravel(),
                              (np.sin(phi)*np.sin(theta)).ravel(), np.cos(phi).ravel()))
    faces = []
    for j in range(24):
        for i in range(40):
            a = j*41+i
            faces.extend([(a,a+1,a+41), (a+1,a+42,a+41)])
    return radius*normals, normals, np.array(faces)


class SphereAR:
    def __init__(self, calibration, radius):
        self.data = calibration
        self.K = np.array(calibration['camera_matrix'], dtype=float)
        self.dist = np.array(calibration['distortion_coefficients'], dtype=float)
        self.cols, self.rows = calibration['board_inner_corners']
        self.s = calibration['square_size']
        self.radius = radius*self.s
        self.offsets, self.normals, self.faces = sphere_mesh(self.radius)
        self.object_points = np.zeros((self.cols*self.rows, 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)*self.s

    def process(self, frame):
        w, h = frame.shape[1], frame.shape[0]
        original_w, original_h = self.data['image_size_wh']
        if abs(w/h-original_w/original_h) > 0.01:
            raise ValueError(f'Frame {w}x{h} has a different aspect ratio from calibration '
                             f'{original_w}x{original_h}. Use the calibrated camera mode.')
        K = self.K.copy()
        K[0, :] *= w/original_w
        K[1, :] *= h/original_h
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1., 960/w)
        small = cv2.resize(gray, (round(w*scale), round(h*scale)))
        ok, corners = cv2.findChessboardCorners(
            small, (self.cols, self.rows), cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            return frame, 'Chessboard not found', None
        corners = corners.reshape(-1, 1, 2)
        corners[:, :, 0] *= w/small.shape[1]
        corners[:, :, 1] *= h/small.shape[0]
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
        ok, rv, tv = cv2.solvePnP(self.object_points, corners, K, self.dist,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return frame, 'Pose estimation failed', None
        projected_board, _ = cv2.projectPoints(self.object_points, rv, tv, K, self.dist)
        residual = projected_board.reshape(-1,2)-corners.reshape(-1,2)
        error = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        R, _ = cv2.Rodrigues(rv)
        C = (-R.T@tv).ravel()
        side = 1 if C[2] > 0 else -1
        center = np.array([(self.cols-1)*self.s/2, (self.rows-1)*self.s/2, side*self.radius])
        vertices = self.offsets+center
        camera_points = vertices@R.T+tv.ravel()
        if np.any(camera_points[:, 2] <= 0):
            return frame, 'Invalid pose', None
        projected, _ = cv2.projectPoints(vertices, rv, tv, K, self.dist)
        projected = projected.reshape(-1,2)
        faces = self.faces
        positions = camera_points[faces].mean(axis=1)
        normal = self.normals[faces].mean(axis=1)@R.T
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        eye = -positions/np.linalg.norm(positions, axis=1, keepdims=True)
        visible = np.sum(normal*eye, axis=1)>0
        light = np.array([-.4,-.6,-1.]); light /= np.linalg.norm(light)
        diffuse = np.maximum(normal@light, 0)
        halfway = eye+light
        halfway /= np.maximum(np.linalg.norm(halfway, axis=1, keepdims=True), 1e-12)
        specular = np.maximum(np.sum(normal*halfway, axis=1), 0)**40
        colors = np.clip(np.array([230.,125.,35.])*(.22+.78*diffuse[:,None])+
                         110*specular[:,None], 0,255)
        for fi in np.argsort(positions[:,2])[::-1]:
            if visible[fi]:
                polygon = np.round(projected[faces[fi]]).astype(np.int32)
                cv2.fillConvexPoly(frame, polygon, tuple(float(x) for x in colors[fi]), cv2.LINE_AA)
        return frame, f'Tracking | RMSE {error:.2f} px', error


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=base/'results/calibration.json')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['basler','auto','dshow','msmf'], default='basler')
    parser.add_argument('--serial', help='Basler serial number, recommended with multiple cameras')
    parser.add_argument('--exposure-us', type=float, help='Fixed exposure in microseconds; otherwise retain camera value')
    parser.add_argument('--gain-db', type=float, help='Fixed gain; otherwise retain camera value')
    parser.add_argument('--radius', type=float, default=.5, help='Radius in square units')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--check-images', type=Path, help='Offline validation instead of opening camera')
    parser.add_argument('--output', type=Path, default=base/'results/live_demo')
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error('Radius must be positive')
    data = json.loads(args.calibration.read_text(encoding='utf-8'))
    ar = SphereAR(data, args.radius)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.check_images:
        results = []
        for view in data['views']:
            path = args.check_images/view['file']
            frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f'Cannot read {path}')
            frame, status, error = ar.process(frame)
            cv2.imencode('.jpg', frame)[1].tofile(str(args.output/(path.stem+'_live_check.jpg')))
            results.append({'file': path.name, 'status': status, 'rmse_px': error})
        (args.output/'offline_validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'Tracked {sum(r["rmse_px"] is not None for r in results)}/{len(results)} images')
        return
    if args.backend == 'basler':
        capture = BaslerCapture(args.camera, args.serial,
                                args.width or data['image_size_wh'][0],
                                args.height or data['image_size_wh'][1],
                                args.exposure_us, args.gain_db)
    else:
        backend = {'auto':cv2.CAP_ANY, 'dshow':cv2.CAP_DSHOW, 'msmf':cv2.CAP_MSMF}[args.backend]
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError('Cannot open camera; try another --camera index or backend')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width or data['image_size_wh'][0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height or data['image_size_wh'][1])
    writer = None
    previous = time.perf_counter()
    try:
        cv2.namedWindow('Sphere AR - Q quit, S snapshot, R record', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sphere AR - Q quit, S snapshot, R record', 1100,825)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Failed to read camera frame')
            rendered, status, _ = ar.process(frame)
            now = time.perf_counter()
            fps = 1/max(now-previous, 1e-6); previous = now
            cv2.putText(rendered, f'{status} | {fps:.1f} FPS', (15,35),
                        cv2.FONT_HERSHEY_SIMPLEX,.75,(0,220,0),2)
            if writer is not None:
                cv2.putText(rendered,'REC',(15,70),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
                writer.write(rendered)
            cv2.imshow('Sphere AR - Q quit, S snapshot, R record', rendered)
            key = cv2.waitKey(1)&0xff
            if key in [ord('q'),27]:
                break
            if key == ord('s'):
                cv2.imencode('.jpg',rendered)[1].tofile(str(args.output/f'snapshot_{time.time_ns()}.jpg'))
            if key == ord('r'):
                if writer is not None:
                    writer.release(); writer = None
                else:
                    path = args.output/f'demo_{time.time_ns()}.avi'
                    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),
                                             15,(rendered.shape[1],rendered.shape[0]))
                    if not writer.isOpened():
                        writer.release(); writer = None
                        print('Recording unavailable; snapshots still work')
            if cv2.getWindowProperty('Sphere AR - Q quit, S snapshot, R record',cv2.WND_PROP_VISIBLE)<1:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
