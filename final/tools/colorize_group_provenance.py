"""Color every registered depth view uniquely to inspect point-cloud provenance."""
import argparse
import colorsys
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json


VERTEX = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('group_id', 'u1')])


def color_for_group(index):
    hue = ((index * 137.508) % 360.) / 360.
    saturation = (.83, .67, .92)[index % 3]
    value = (.99, .86)[index % 2]
    return np.rint(np.asarray(colorsys.hsv_to_rgb(hue, saturation, value)) * 255).astype(np.uint8)


def header(count):
    return (f'ply\nformat binary_little_endian 1.0\n'
            'comment coordinates: group_0001 camera frame, units: millimeters\n'
            'comment group_id records the original capture group; colors are diagnostic\n'
            f'element vertex {count}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property uchar group_id\nend_header\n').encode('ascii')


def make_vertices(points, color, group_id):
    vertices = np.empty(len(points), dtype=VERTEX)
    for index, axis in enumerate(('x', 'y', 'z')):
        vertices[axis] = points[:, index]
    for index, channel in enumerate(('red', 'green', 'blue')):
        vertices[channel] = color[index]
    vertices['group_id'] = group_id
    return vertices


def write_one(path, vertices):
    with path.open('wb') as stream:
        stream.write(header(len(vertices)))
        stream.write(vertices.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=3, help='Keep every Nth raw depth point (default: 3)')
    parser.add_argument('--merged-only', action='store_true')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('Stride must be positive')
    if args.output.exists():
        parser.error('Output directory already exists')
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    registration = json.loads(args.registration.read_text(encoding='utf-8'))
    names = [item['group'] for item in summary['groups']]
    if set(names) != set(registration['poses']):
        raise ValueError('Pose groups do not match depth groups')
    args.output.mkdir(parents=True)
    vertices_by_group = []
    legend = []
    for index, name in enumerate(names, start=1):
        with np.load(args.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
            points = data['xyz_mm'][::args.stride]
        pose = np.asarray(registration['poses'][name], float)
        world = points @ pose[:3, :3].T + pose[:3, 3]
        color = color_for_group(index)
        vertices = make_vertices(world, color, index)
        vertices_by_group.append(vertices)
        if not args.merged_only:
            write_one(args.output / f'{name}.ply', vertices)
        hex_color = '#%02X%02X%02X' % tuple(int(x) for x in color)
        legend.append({'group': name, 'group_id': index, 'hex': hex_color,
                       'rgb': color.tolist(), 'points': len(vertices)})
        print(name, len(vertices), hex_color, flush=True)
    total = sum(len(vertices) for vertices in vertices_by_group)
    with (args.output / 'all_groups_colored.ply').open('wb') as stream:
        stream.write(header(total))
        for vertices in vertices_by_group:
            stream.write(vertices.tobytes())
    save_json(args.output / 'group_colors.json', {'registration': str(args.registration.resolve()),
        'depth_directory': str(args.depth_dir.resolve()), 'stride': args.stride,
        'total_points': total, 'groups': legend,
        'note': 'These are registered source points before voxel fusion. Each color identifies one original capture group.'})
    cells = []
    for item in legend:
        row = (item['group_id'] - 1) // 8
        column = (item['group_id'] - 1) % 8
        x, y = 20 + column * 160, 35 + row * 54
        cells.append(f'<rect x="{x}" y="{y}" width="28" height="28" fill="{item["hex"]}" stroke="#333"/>')
        cells.append(f'<text x="{x+38}" y="{y+20}" font-size="16" font-family="Arial">{item["group"]}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="270" viewBox="0 0 1300 270">'
           '<rect width="1300" height="270" fill="white"/>' + ''.join(cells) + '</svg>')
    (args.output / 'group_colors.svg').write_text(svg, encoding='utf-8')
    print('Saved', total, 'points to', args.output / 'all_groups_colored.ply')


if __name__ == '__main__':
    main()
