"""Transfer oriented normals from a nearby reference cloud and run COLMAP Poisson meshing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import read_ply, save_json
from tools.fuse_component_assembly import read_binary_vertices


def ply_counts(path):
    count = {}
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Invalid mesh PLY')
        for raw in stream:
            line = raw.decode('ascii').strip()
            if line.startswith('element '):
                _, name, number = line.split()
                count[name] = int(number)
            elif line == 'end_header':
                break
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cloud', type=Path)
    parser.add_argument('--normal-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--colmap', type=Path, default=ROOT / 'tools/colmap-4.2.1-cuda/bin/colmap.exe')
    parser.add_argument('--max-normal-distance-mm', type=float, default=2.)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--trim', type=float, default=5.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    args.output.mkdir(parents=True)
    raw = read_binary_vertices(args.cloud)
    xyz = np.column_stack([raw[key] for key in ('x', 'y', 'z')])
    rgb = np.column_stack([raw[key] for key in ('red', 'green', 'blue')])
    reference, _, attrs = read_ply(args.normal_reference, with_attributes=True)
    if any(key not in attrs for key in ('nx', 'ny', 'nz')):
        raise ValueError('Reference PLY has no oriented normals')
    reference_normal = np.column_stack([attrs[key] for key in ('nx', 'ny', 'nz')])
    distance, index = cKDTree(reference).query(xyz, k=1, workers=2)
    normal = reference_normal[index]
    length = np.linalg.norm(normal, axis=1)
    keep = (distance <= args.max_normal_distance_mm) & (length > .5)
    if keep.mean() < .95:
        raise ValueError(f'Only {keep.mean():.1%} points have a nearby reliable normal')
    xyz, rgb, normal = xyz[keep], rgb[keep], normal[keep] / length[keep, None]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                      ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
    vertices = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate(('x', 'y', 'z')):
        vertices[name] = xyz[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = rgb[:, i]
    for i, name in enumerate(('nx', 'ny', 'nz')):
        vertices[name] = normal[:, i]
    prepared = args.output / 'oriented_input.ply'
    with prepared.open('wb') as stream:
        stream.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property float nx\nproperty float ny\nproperty float nz\nend_header\n').encode('ascii'))
        stream.write(vertices.tobytes())
    mesh = args.output / 'mesh.ply'
    command = [str(args.colmap.resolve()), 'poisson_mesher', '--input_path', str(prepared.resolve()),
               '--output_path', str(mesh.resolve()), '--PoissonMeshing.depth', str(args.depth),
               '--PoissonMeshing.trim', str(args.trim)]
    with (args.output / 'poisson.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not mesh.exists():
        raise RuntimeError(f'Poisson meshing failed with exit code {completed.returncode}; see poisson.log')
    counts = ply_counts(mesh)
    result = {'source_cloud': str(args.cloud.resolve()),
              'normal_reference': str(args.normal_reference.resolve()),
              'input_points': len(raw), 'oriented_points': len(xyz),
              'normal_transfer_within_1mm_fraction': float(np.mean(distance < 1.)),
              'normal_transfer_within_2mm_fraction': float(np.mean(distance < 2.)),
              'mesh_vertices': counts.get('vertex', 0), 'mesh_faces': counts.get('face', 0),
              'depth': args.depth, 'trim': args.trim, 'command': command,
              'note': 'Experimental Poisson mesh. Normals were transferred from the baseline cloud; unsupported regions may be invented.'}
    save_json(args.output / 'status.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
