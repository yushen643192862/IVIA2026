"""Apply a reviewed small per-view depth correction without changing source depths."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.depth_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    trial = json.loads(args.trial.read_text(encoding='utf-8'))
    summary = json.loads((source / 'depth_summary.json').read_text(encoding='utf-8'))
    groups = [item['group'] for item in summary['groups']]
    corrections = {item['group']: item for item in trial['per_group']}
    if set(groups) != set(corrections):
        raise ValueError('Trial groups do not match depth groups')
    if not trial.get('optimizer_success'):
        raise ValueError('The trial optimizer did not succeed')
    reference = float(trial['reference_depth_mm'])
    output.mkdir(parents=True)
    records = []
    for name in groups:
        record = corrections[name]
        scale, offset = float(record['scale']), float(record['offset_mm'])
        if abs(scale - 1.) > .020001 or abs(offset) > 3.00001:
            raise ValueError(f'Correction exceeds the trial limits: {name}')
        with np.load(source / name / 'depth_mm.npz', allow_pickle=False) as data:
            xyz = data['xyz_mm']
            pixels = data['pixels_xy']
            depth = data['depth_mm']
            reprojection = data['reprojection_error_px']
        new_depth = reference + scale * (xyz[:, 2] - reference) + offset
        if np.any(new_depth <= 0):
            raise ValueError(f'Nonpositive corrected depth: {name}')
        corrected_xyz = xyz * (new_depth / xyz[:, 2])[:, None]
        corrected_map = depth.copy()
        finite = np.isfinite(corrected_map)
        corrected_map[finite] = reference + scale * (corrected_map[finite] - reference) + offset
        folder = output / name
        folder.mkdir()
        np.savez_compressed(folder / 'depth_mm.npz', depth_mm=corrected_map,
                            valid=finite, xyz_mm=corrected_xyz,
                            pixels_xy=pixels, reprojection_error_px=reprojection)
        records.append({'group': name, 'points': len(xyz), 'scale': scale,
                        'offset_mm': offset,
                        'median_absolute_depth_change_mm': float(np.median(np.abs(new_depth - xyz[:, 2]))),
                        'max_absolute_depth_change_mm': float(np.max(np.abs(new_depth - xyz[:, 2])))})
        print('Corrected', name, len(xyz), 'points', flush=True)
    derived = copy.deepcopy(summary)
    derived['source_depth_directory'] = str(source)
    derived['depth_correction_trial'] = str(args.trial.resolve())
    derived['depth_correction_reference_mm'] = reference
    derived['depth_correction_experimental'] = True
    derived['note'] = ('Experimental bounded per-view ray-depth correction. '
                       'Original triangulation reprojection errors are retained for traceability '
                       'and do not validate corrected points. Re-run registration before use.')
    save_json(output / 'depth_summary.json', derived)
    save_json(output / 'correction_applied.json', {'source': str(source),
        'trial': str(args.trial.resolve()), 'groups': records,
        'heldout_test': trial['heldout_gap2'], 'experimental': True})


if __name__ == '__main__':
    main()
