"""Test bounded per-view depth scale/offset against held-out structured-light overlaps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.dense_registration import surface_sample
from scanner.io import save_json


def get_overlap(a, b, i, j, poses, reference_depth, limit, seed):
    """Fixed nearest-surface correspondences, with linearized ray-depth coefficients."""
    pa, na = a
    pb, nb = b
    ta, tb = poses[i], poses[j]
    wa = pa @ ta[:3, :3].T + ta[:3, 3]
    wb = pb @ tb[:3, :3].T + tb[:3, 3]
    normal_a = na @ ta[:3, :3].T
    normal_b = nb @ tb[:3, :3].T
    distance, index = cKDTree(wa).query(wb, k=1, workers=2)
    agreement = np.einsum('ij,ij->i', normal_b, normal_a[index])
    good = (distance < 4.) & (agreement > .65)
    ids = np.flatnonzero(good)
    if len(ids) < 80:
        return None
    plane = np.einsum('ij,ij->i', wb[ids] - wa[index[ids]], normal_a[index[ids]])
    ids = ids[np.abs(plane) < 2.]
    if len(ids) < 80:
        return None
    if len(ids) > limit:
        ids = np.random.default_rng(seed).choice(ids, limit, replace=False)
    target = index[ids]
    normal = normal_a[target]
    residual = np.einsum('ij,ij->i', wb[ids] - wa[target], normal)
    # Moving a camera-space point along its own ray by one depth millimeter.
    ray_b = (pb[ids] / pb[ids, 2, None]) @ tb[:3, :3].T
    ray_a = (pa[target] / pa[target, 2, None]) @ ta[:3, :3].T
    coeff_b = np.einsum('ij,ij->i', ray_b, normal)
    coeff_a = -np.einsum('ij,ij->i', ray_a, normal)
    return {'i': i, 'j': j, 'residual': residual,
            'coef_i': coeff_a, 'coef_j': coeff_b,
            'zi': pa[target, 2] - reference_depth,
            'zj': pb[ids, 2] - reference_depth,
            'candidates': len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-depth-mm', type=float, default=850.)
    parser.add_argument('--max-scale-change', type=float, default=.02)
    parser.add_argument('--max-offset-mm', type=float, default=3.)
    parser.add_argument('--prior-weight', type=float, default=1., help='Multiplier on depth-change prior; smaller values permit larger corrections')
    args = parser.parse_args()
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    names = [item['group'] for item in summary['groups']]
    registration = json.loads(args.registration.read_text(encoding='utf-8'))
    poses = [np.asarray(registration['poses'][name], float) for name in names]
    n = len(names)
    samples = []
    for k, name in enumerate(names):
        with np.load(args.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
            xyz = data['xyz_mm']
        samples.append(surface_sample(xyz, 2., 5000, 100 + k))
        print('Sample', name, len(samples[-1][0]), flush=True)
    train, heldout = [], []
    for gap, destination, maximum in ((1, train, 350), (2, heldout, 1000)):
        for i in range(n):
            j = (i + gap) % n
            edge = get_overlap(samples[i], samples[j], i, j, poses,
                               args.reference_depth_mm, maximum, 900 + i + gap * n)
            if edge is not None:
                destination.append(edge)
    if len(train) < n - 2 or len(heldout) < n // 2:
        raise ValueError(f'Insufficient overlaps: train={len(train)}, heldout={len(heldout)}')

    def design(edges):
        row_count = sum(len(edge['residual']) for edge in edges)
        matrix = np.zeros((row_count, 2 * (n - 1)), np.float64)
        residual = np.empty(row_count)
        start = 0
        for edge in edges:
            count = len(edge['residual'])
            section = slice(start, start + count)
            residual[section] = edge['residual']
            for node, coeff, z in ((edge['i'], edge['coef_i'], edge['zi']),
                                   (edge['j'], edge['coef_j'], edge['zj'])):
                if node:
                    column = 2 * (node - 1)
                    matrix[section, column] = coeff * z
                    matrix[section, column + 1] = coeff
            start += count
        return matrix, residual

    a_train, r_train = design(train)
    a_test, r_test = design(heldout)
    # Give each adjacent pair similar total weight, regardless of point density.
    weights = np.concatenate([np.full(len(edge['residual']), 1 / np.sqrt(len(edge['residual'])))
                              for edge in train])
    prior = np.tile([25., .33], n - 1) * args.prior_weight  # 2% scale and 1.5 mm offset cost ~0.5 mm.
    def objective(parameters):
        corrected = (r_train + a_train @ parameters) * weights
        return np.r_[corrected, prior * parameters]
    bounds = np.tile([args.max_scale_change, args.max_offset_mm], n - 1)
    result = least_squares(objective, np.zeros(len(bounds)),
                           bounds=(-bounds, bounds), loss='soft_l1', f_scale=.1,
                           max_nfev=100, x_scale='jac')
    parameters = result.x
    def scores(matrix, original):
        before = np.abs(original)
        after = np.abs(original + matrix @ parameters)
        return {'points': len(before), 'median_before_mm': float(np.median(before)),
                'median_after_mm': float(np.median(after)),
                'p90_before_mm': float(np.percentile(before, 90)),
                'p90_after_mm': float(np.percentile(after, 90)),
                'mean_before_mm': float(np.mean(before)),
                'mean_after_mm': float(np.mean(after))}
    shifts = [{'group': names[0], 'scale': 1., 'offset_mm': 0.}]
    shifts.extend({'group': name, 'scale': float(1 + parameters[2 * (i - 1)]),
                   'offset_mm': float(parameters[2 * (i - 1) + 1])}
                  for i, name in enumerate(names[1:], 1))
    report = {'reference_depth_mm': args.reference_depth_mm,
              'prior_weight': args.prior_weight,
              'training_edges': len(train), 'heldout_edges': len(heldout),
              'training': scores(a_train, r_train), 'heldout_gap2': scores(a_test, r_test),
              'per_group': shifts, 'optimizer_success': bool(result.success),
              'note': 'Fixed poses and fixed correspondences; held-out gap-2 views were excluded from fitting. This is a diagnostic, not a validated depth correction.'}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'depth_correction_trial.json', report)
    print(json.dumps({key: report[key] for key in ('training_edges', 'heldout_edges', 'training', 'heldout_gap2')}, indent=2))


if __name__ == '__main__':
    main()
