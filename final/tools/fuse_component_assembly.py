"""Voxel-fuse a component assembly while counting independent capture views."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.export_editable_cloud_layers import VERTEX, write_binary_ply


def read_binary_vertices(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY: {path}')
        while stream.readline().strip() != b'end_header':
            pass
        return np.fromfile(stream, dtype=VERTEX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assembly_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-mm', type=float, default=.5)
    parser.add_argument('--min-views', type=int, default=2)
    args = parser.parse_args()
    if args.voxel_mm <= 0 or args.min_views < 1:
        parser.error('Voxel size and minimum views must be positive')
    if args.output.exists():
        parser.error('Output already exists')
    report = json.loads((args.assembly_dir / 'component_registration.json').read_text(encoding='utf-8'))
    chunks, group_ids = [], []
    for i, group in enumerate(report['groups']):
        vertices = read_binary_vertices(args.assembly_dir / f"{group['group']}_assembled.ply")
        if len(vertices) != group['total_points']:
            raise ValueError(f"Point count mismatch for {group['group']}")
        chunks.append(vertices)
        group_ids.append(np.full(len(vertices), i, np.uint8))
    vertices = np.concatenate(chunks)
    source = np.concatenate(group_ids)
    xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
    rgb = np.column_stack([vertices[key] for key in ('red', 'green', 'blue')])
    cell = np.floor(xyz / args.voxel_mm).astype(np.int32)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    voxels = len(counts)
    pairs = np.unique(np.column_stack((inverse, source)), axis=0)
    support = np.bincount(pairs[:, 0], minlength=voxels)
    def mean(array):
        return np.column_stack([np.bincount(inverse, weights=array[:, j]) / counts for j in range(3)])
    centers = mean(xyz)
    colors = np.rint(mean(rgb)).clip(0, 255).astype(np.uint8)
    keep = support >= args.min_views
    centers, colors = centers[keep], colors[keep]
    if len(centers) >= 12:
        distance = cKDTree(centers).query(centers, k=12, workers=2)[0][:, -1]
        dense = distance < 3.
        centers, colors = centers[dense], colors[dense]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(args.output, centers, colors)
    save_json(args.output.with_suffix('.json'), {
        'source': str(args.assembly_dir.resolve()), 'input_points': len(xyz),
        'output_points': len(centers), 'voxel_mm': args.voxel_mm,
        'min_views': args.min_views, 'note': 'Diagnostic component fusion; no surface-normal consistency filter.'})
    print(f'Fused {len(xyz)} source points into {len(centers)} points', flush=True)


if __name__ == '__main__':
    main()
