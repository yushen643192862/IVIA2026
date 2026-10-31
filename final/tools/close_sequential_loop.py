"""Distribute a measured first/last loop correction across sequential poses."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequential_registration', type=Path)
    parser.add_argument('--pairwise', type=Path, required=True)
    parser.add_argument('--depth-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.sequential_registration.read_text(encoding='utf-8'))
    names = sorted(document['poses'])
    poses = [np.asarray(document['poses'][name], float) for name in names]
    pairs = json.loads(args.pairwise.read_text(encoding='utf-8'))
    first, last = names[0], names[-1]
    matches = [edge for edge in pairs if edge['i'] == first and edge['j'] == last]
    if len(matches) != 1 or matches[0]['inliers'] < 100:
        raise ValueError('No reliable first/last loop edge')
    measured_last = np.asarray(matches[0]['j_to_i'], float)
    correction = measured_last @ np.linalg.inv(poses[-1])
    generator = logm(correction)
    if np.max(np.abs(np.imag(generator))) > 1e-6:
        raise ValueError('Complex loop correction')
    generator = np.real(generator)
    closed = [expm(generator * (index / (len(poses) - 1))) @ pose
              for index, pose in enumerate(poses)]
    source = args.depth_dir
    with np.load(source / first / 'depth_mm.npz', allow_pickle=False) as data:
        target = data['xyz_mm']
    with np.load(source / last / 'depth_mm.npz', allow_pickle=False) as data:
        cloud = data['xyz_mm'][::max(1, len(data['xyz_mm']) // 5000)]
    def overlap(pose):
        moved = cloud @ pose[:3, :3].T + pose[:3, 3]
        distances = cKDTree(target).query(moved, k=1, workers=2)[0]
        near = distances < 5.
        return {'within_5mm_fraction': float(np.mean(near)),
                'median_distance_within_5mm': float(np.median(distances[near])) if near.any() else None}
    report = {'method': 'sequential nominal angle + silhouette + ICP; measured first/last loop correction distributed over steps',
              'poses': {name: pose.tolist() for name, pose in zip(names, closed)},
              'loop_edge_inliers': matches[0]['inliers'],
              'first_last_before': overlap(poses[-1]),
              'first_last_after': overlap(closed[-1]),
              'source_registration': str(args.sequential_registration.resolve()),
              'experimental': True}
    save_json(args.output, report)
    print('First/last:', report['first_last_before'], '->', report['first_last_after'])


if __name__ == '__main__':
    main()
