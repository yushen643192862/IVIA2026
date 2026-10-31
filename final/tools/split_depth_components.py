"""Split each depth view into spatially connected point-cloud components."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.decode_capture_session import read_image
from tools.export_editable_cloud_layers import write_binary_ply


def label_points(xyz, voxel_mm):
    cells = np.floor(xyz / voxel_mm).astype(np.int32)
    origin = cells.min(axis=0)
    local = cells - origin
    shape = tuple((local.max(axis=0) + 1).tolist())
    if np.prod(shape, dtype=np.int64) > 60_000_000:
        raise ValueError(f'Voxel label grid too large: {shape}; increase --voxel-mm')
    occupancy = np.zeros(shape, bool)
    occupancy[tuple(local.T)] = True
    labels, _ = ndimage.label(occupancy, structure=np.ones((3, 3, 3), bool))
    point_labels = labels[tuple(local.T)]
    return point_labels, shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-mm', type=float, default=2.)
    parser.add_argument('--min-component-points', type=int, default=100)
    parser.add_argument('--groups', nargs='*', help='Optional group names, e.g. group_0001 group_0016')
    args = parser.parse_args()
    if args.voxel_mm <= 0 or args.min_component_points <= 0:
        parser.error('Voxel size and minimum points must be positive')
    if args.output.exists():
        parser.error('Output already exists')
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    decoded = Path(summary['source'])
    names = [row['group'] for row in summary['groups']]
    if args.groups:
        missing = set(args.groups) - set(names)
        if missing:
            parser.error(f'Unknown groups: {sorted(missing)}')
        names = [name for name in names if name in args.groups]
    poses = json.loads(args.registration.read_text(encoding='utf-8'))['poses']
    args.output.mkdir(parents=True)
    report = {'voxel_mm': args.voxel_mm, 'min_component_points': args.min_component_points,
              'coordinate_frame': 'group_0001 camera frame', 'units': 'millimeters',
              'groups': []}
    for name in names:
        with np.load(args.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
            xyz, pixels = data['xyz_mm'], data['pixels_xy'].astype(int)
        image = read_image(decoded / 'color_images' / f'{name}.png', cv2.IMREAD_COLOR)
        rgb = image[pixels[:, 1], pixels[:, 0], ::-1]
        labels, grid_shape = label_points(xyz, args.voxel_mm)
        count = np.bincount(labels)
        order = np.argsort(count[1:])[::-1] + 1
        pose = np.asarray(poses[name], float)
        world = xyz @ pose[:3, :3].T + pose[:3, 3]
        group_dir = args.output / name
        group_dir.mkdir()
        components = []
        small = np.ones(len(world), bool)
        for rank, component_id in enumerate(order, 1):
            if count[component_id] < args.min_component_points:
                break
            selector = labels == component_id
            small &= ~selector
            component = world[selector]
            filename = f'component_{rank:03d}.ply'
            write_binary_ply(group_dir / filename, component, rgb[selector])
            components.append({'file': filename, 'rank': rank, 'points': int(count[component_id]),
                               'bounds_min_mm': component.min(axis=0).tolist(),
                               'bounds_max_mm': component.max(axis=0).tolist()})
        if small.any():
            write_binary_ply(group_dir / 'small_fragments.ply', world[small], rgb[small])
        row = {'group': name, 'total_points': len(world), 'voxel_grid_shape': grid_shape,
               'all_component_count': int(len(order)), 'exported_components': components,
               'small_fragment_points': int(small.sum())}
        report['groups'].append(row)
        print(name, 'components', len(order), 'exported', len(components),
              'largest', components[0]['points'] if components else 0, flush=True)
    save_json(args.output / 'components.json', report)


if __name__ == '__main__':
    main()
