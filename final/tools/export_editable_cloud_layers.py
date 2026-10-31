"""Export each registered capture as an independently movable colored PLY layer."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.decode_capture_session import read_image

VERTEX = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def write_binary_ply(path, xyz, rgb):
    vertex = np.empty(len(xyz), dtype=VERTEX)
    for i, axis in enumerate(('x', 'y', 'z')):
        vertex[axis] = xyz[:, i]
    for i, axis in enumerate(('red', 'green', 'blue')):
        vertex[axis] = rgb[:, i]
    header = (f'ply\nformat binary_little_endian 1.0\n'
              'comment unit millimeter; common world frame is group_0001 camera frame\n'
              f'element vertex {len(vertex)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    decoded = Path(summary['source'])
    poses = json.loads(args.registration.read_text(encoding='utf-8'))['poses']
    names = [item['group'] for item in summary['groups']]
    if set(names) != set(poses):
        raise ValueError('Depth groups and registration poses differ')
    args.output.mkdir(parents=True)
    files = []
    for name in names:
        with np.load(args.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as depth:
            xyz = depth['xyz_mm'][::args.stride]
            pixels = depth['pixels_xy'][::args.stride].astype(int)
        image = read_image(decoded / 'color_images' / f'{name}.png', cv2.IMREAD_COLOR)
        if np.any(pixels < 0) or np.any(pixels[:, 0] >= image.shape[1]) or np.any(pixels[:, 1] >= image.shape[0]):
            raise ValueError(f'Pixel coordinates out of bounds: {name}')
        rgb = image[pixels[:, 1], pixels[:, 0], ::-1]
        pose = np.asarray(poses[name], dtype=float)
        world = xyz @ pose[:3, :3].T + pose[:3, 3]
        destination = args.output / f'{name}.ply'
        write_binary_ply(destination, world, rgb)
        files.append({'group': name, 'file': destination.name, 'points': len(world)})
        print(f'{name}: {len(world)} points', flush=True)
    save_json(args.output / 'manifest.json', {
        'coordinate_frame': 'group_0001 camera frame', 'units': 'millimeters',
        'depth_dir': str(args.depth_dir.resolve()),
        'registration': str(args.registration.resolve()),
        'stride': args.stride, 'files': files,
        'note': 'Each PLY is an independent layer in one shared coordinate frame. Moving a layer changes all points from that group.'})


if __name__ == '__main__':
    main()
