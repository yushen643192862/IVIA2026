"""Compare per-view depth, fused cloud, and Poisson mesh in selected object regions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.export_editable_cloud_layers import write_binary_ply
from tools.fuse_component_assembly import read_binary_vertices

REGIONS = {
    'leg_and_nearby': (np.array([-25., 5., 790.]), np.array([40., 70., 900.])),
    'handheld_staff': (np.array([-105., -80., 800.]), np.array([-20., -5., 865.])),
}


def xyz_rgb(vertices):
    xyz = np.column_stack([vertices[name] for name in ('x', 'y', 'z')])
    rgb = np.column_stack([vertices[name] for name in ('red', 'green', 'blue')])
    return xyz, rgb


def region(points, colors, bounds):
    lo, hi = bounds
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep], colors[keep]


def roughness(points, seed):
    if len(points) < 100:
        return {'sampled': len(points), 'eligible': 0}
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(points), min(5000, len(points)), replace=False)
    distance, nearest = cKDTree(points).query(points[ids], k=min(32, len(points)), workers=2)
    sigma, radius = [], []
    for d, neighbors in zip(distance, nearest):
        local = points[neighbors[d < 3.]]
        if len(local) < 8:
            continue
        local = local - local.mean(axis=0)
        eigen = np.linalg.eigvalsh(local.T @ local / len(local))
        sigma.append(np.sqrt(max(0., eigen[0])))
        radius.append(float(d[min(len(d) - 1, 17)]))
    if not sigma:
        return {'sampled': len(ids), 'eligible': 0}
    return {'sampled': len(ids), 'eligible': len(sigma),
            'plane_sigma_mm_p50': float(np.percentile(sigma, 50)),
            'plane_sigma_mm_p90': float(np.percentile(sigma, 90)),
            'neighbor18_distance_mm_p50': float(np.percentile(radius, 50))}


def overlap(a, b):
    if len(a) < 100 or len(b) < 100:
        return {'points_a': len(a), 'points_b': len(b), 'comparable': False}
    a = a[::max(1, len(a) // 10000)]
    b = b[::max(1, len(b) // 10000)]
    ab = cKDTree(b).query(a, k=1, workers=2)[0]
    ba = cKDTree(a).query(b, k=1, workers=2)[0]
    return {'points_a': len(a), 'points_b': len(b), 'comparable': True,
            'a_within_1mm': float(np.mean(ab < 1.)),
            'a_within_3mm': float(np.mean(ab < 3.)),
            'b_within_1mm': float(np.mean(ba < 1.)),
            'b_within_3mm': float(np.mean(ba < 3.)),
            'a_distance_mm_p50': float(np.median(ab)),
            'b_distance_mm_p50': float(np.median(ba))}


def mesh_vertices(path):
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('value', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    with path.open('rb') as stream:
        count = None
        while True:
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            if line == 'end_header':
                break
        if count is None:
            raise ValueError('Missing vertex count')
        return np.fromfile(stream, dtype=dtype, count=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assembly_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    args.output.mkdir(parents=True)
    fused_xyz, fused_rgb = xyz_rgb(read_binary_vertices(args.assembly_dir / 'fused_components.ply'))
    mesh_xyz, mesh_rgb = xyz_rgb(mesh_vertices(args.assembly_dir / 'poisson_mesh_v1/mesh.ply'))
    # Read every assembled group once; all are already in the common world frame.
    groups = {}
    for group in range(1, 32):
        name = f'group_{group:04d}'
        groups[name] = xyz_rgb(read_binary_vertices(args.assembly_dir / f'{name}_assembled.ply'))
    result = {'regions': {}}
    for name, bounds in REGIONS.items():
        folder = args.output / name
        folder.mkdir()
        per_group = {}
        cropped = {}
        for group, (xyz, rgb) in groups.items():
            points, colors = region(xyz, rgb, bounds)
            cropped[group] = points
            per_group[group] = {'points': len(points), 'roughness': roughness(points, int(group[-4:]))}
            if group in ('group_0001', 'group_0016'):
                write_binary_ply(folder / f'{group}.ply', points, colors)
        fused_points, fused_colors = region(fused_xyz, fused_rgb, bounds)
        mesh_points, mesh_colors = region(mesh_xyz, mesh_rgb, bounds)
        write_binary_ply(folder / 'fused.ply', fused_points, fused_colors)
        write_binary_ply(folder / 'mesh_vertices.ply', mesh_points, mesh_colors)
        if len(mesh_points) and len(fused_points):
            mesh_distance = cKDTree(fused_points).query(mesh_points[::max(1, len(mesh_points) // 20000)],
                                                    k=1, workers=2)[0]
            mesh_support = {'sampled': len(mesh_distance),
                            'within_1mm': float(np.mean(mesh_distance < 1.)),
                            'over_2mm': float(np.mean(mesh_distance > 2.)),
                            'distance_mm_p50': float(np.percentile(mesh_distance, 50)),
                            'distance_mm_p90': float(np.percentile(mesh_distance, 90))}
        else:
            mesh_support = {'sampled': 0}
        pairs = {}
        for a, b in ((1, 2), (15, 16), (16, 17), (1, 31), (8, 9), (24, 25)):
            pairs[f'{a:02d}-{b:02d}'] = overlap(cropped[f'group_{a:04d}'], cropped[f'group_{b:04d}'])
        result['regions'][name] = {'bounds_mm': [bounds[0].tolist(), bounds[1].tolist()],
                                    'groups': per_group, 'fused': {'points': len(fused_points),
                                        'roughness': roughness(fused_points, 104)},
                                    'mesh': {'vertices': len(mesh_points),
                                        'roughness': roughness(mesh_points, 105),
                                        'distance_to_fused': mesh_support},
                                    'group_pairs': pairs}
    save_json(args.output / 'report.json', result)
    print(json.dumps({name: {'fused_points': row['fused']['points'],
                             'mesh_vertices': row['mesh']['vertices'],
                             'fused_roughness': row['fused']['roughness'],
                             'mesh_support': row['mesh']['distance_to_fused'],
                             'group_pairs': row['group_pairs']}
                      for name, row in result['regions'].items()}, indent=2))


if __name__ == '__main__':
    main()
