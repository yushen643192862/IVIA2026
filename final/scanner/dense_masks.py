"""Foreground constraints in the undistorted COLMAP dense workspace."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .io import save_json


def read_stereo_array(path):
    """Read COLMAP's width&height&channels& + float32 matrix format."""
    with Path(path).open('rb') as stream:
        header = bytearray()
        while header.count(b'&') < 3:
            char = stream.read(1)
            if not char or len(header) > 100:
                raise ValueError(f'Invalid stereo header: {path}')
            header.extend(char)
        width, height, channels = map(int, header.decode('ascii').strip('&').split('&'))
        values = np.frombuffer(stream.read(), dtype='<f4')
    if min(width, height, channels) < 1 or values.size != width*height*channels:
        raise ValueError(f'Invalid stereo array dimensions: {path}')
    return values.reshape((width, height, channels), order='F').transpose(1, 0, 2).copy()


def write_stereo_array(path, array):
    array = np.asarray(array, dtype='<f4')
    if array.ndim != 3:
        raise ValueError('Stereo array must have shape (height, width, channels)')
    height, width, channels = array.shape
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(f'{width}&{height}&{channels}&'.encode('ascii'))
        stream.write(array.transpose(1, 0, 2).ravel(order='F').tobytes())
    temporary.replace(path)


def prepare_dense_masks(dense, cameras, views, window_radius):
    """Remap original masks; mask dense images without touching source photos."""
    dense = Path(dense)
    original = {camera['file']: camera for camera in cameras}
    records = []
    for view in views:
        camera = original[view['file']]
        mask_path = Path(camera['object_mask'])
        records.append({'file': view['file'], 'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                        'original_K': camera['K'], 'distortion': camera['distortion'],
                        'dense_K': view['K'], 'dense_size': view['size']})
    configuration = {'version': 1, 'window_radius': window_radius, 'views': records}
    manifest = dense/'foreground_config.json'
    cached = any((dense/'stereo/depth_maps').rglob('*.bin'))
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != configuration:
            raise ValueError('Dense foreground settings changed; use a new output directory')
    elif cached:
        raise ValueError('Existing depth maps were not foreground-constrained; use a new output directory')

    result = {}
    for view in views:
        name = view['file']
        camera = original[name]
        mask = cv2.imdecode(np.fromfile(camera['object_mask'], np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or list(mask.shape[::-1]) != camera['size']:
            raise ValueError(f'Invalid original mask: {name}')
        maps = cv2.initUndistortRectifyMap(np.asarray(camera['K'], float),
                    np.asarray(camera['distortion'], float), np.eye(3),
                    np.asarray(view['K'], float), tuple(view['size']), cv2.CV_32FC1)
        foreground = cv2.remap(mask, *maps, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if not foreground.any():
            raise ValueError(f'Empty undistorted mask: {name}')
        destination = dense/'masks'/(name+'.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', foreground.astype(np.uint8)*255)[1].tofile(str(destination))

        image_path = dense/'images'/name
        backup = dense/'images_unmasked'/name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not manifest.exists() or not backup.exists():
            shutil.copyfile(image_path, backup)
        image = cv2.imdecode(np.fromfile(str(backup), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != foreground.shape:
            raise ValueError(f'Undistorted image/mask size mismatch: {name}')
        image[~foreground] = 0
        extension = image_path.suffix.lower()
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 100] if extension in ('.jpg', '.jpeg') else []
        ok, encoded = cv2.imencode(extension, image, parameters)
        if not ok:
            raise ValueError(f'Cannot encode masked dense image: {name}')
        encoded.tofile(str(image_path))
        result[name] = foreground
    save_json(manifest, configuration)
    return result


def constrain_depth_maps(dense, foregrounds, kind):
    """Invalidate background depth and normal samples before downstream use."""
    dense = Path(dense)
    report = []
    for name, foreground in foregrounds.items():
        depth_path = dense/'stereo/depth_maps'/(name+'.'+kind+'.bin')
        normal_path = dense/'stereo/normal_maps'/(name+'.'+kind+'.bin')
        depth = read_stereo_array(depth_path)
        normal = read_stereo_array(normal_path)
        if depth.shape != (*foreground.shape, 1) or normal.shape != (*foreground.shape, 3):
            raise ValueError(f'Depth/normal/mask size mismatch: {name}')
        valid = foreground & np.isfinite(depth[:,:,0]) & (depth[:,:,0] > 0)
        removed = int(np.count_nonzero((depth[:,:,0] > 0) & ~foreground))
        depth[~valid] = 0
        normal[~valid] = 0
        write_stereo_array(depth_path, depth)
        write_stereo_array(normal_path, normal)
        report.append({'image': name, 'foreground_pixels': int(foreground.sum()),
                       'background_depths_removed': removed, 'retained_depths': int(valid.sum())})
    save_json(dense/f'foreground_{kind}_report.json', report)
    return report
