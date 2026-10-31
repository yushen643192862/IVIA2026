"""Keep only the structured-light data chain for the current Poisson mesh."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs' / 'structured_light'
DEPTH = BASE / 'structured_depth_mask_inset6_v1'

REMOVE_TOP = (
    'structured_depth_mask_inset2_v1',
    'structured_depth_mask_inset10_v1',
    'structured_depth_mask_inset16_v1',
    'structured_depth_mask_inset6_relative_trial_v1',
    'structured_fused_green_only_v3',
)
KEEP_DEPTH = {
    'effective_masks', 'fusion_edge_recovery_v1', 'fusion_sequential_closed_v1',
    'sequential_registration_v1', 'component_four_step_v3',
    'background_depth_cutoffs.json', 'calibration_used.json',
    'coordinate_compensation.json', 'depth_summary.json', 'status.json',
    *(f'group_{i:04d}' for i in range(1, 32)),
}
KEEP_GROUPS = {'decode', 'parameters.json'}
KEEP_FUSION_REFERENCE = {'registration.json'}
KEEP_FUSION_BASELINE = {
    'dense.ply', 'registration.json', 'pairwise_registration.json',
    'status.json', 'icp_report.json', 'cloud_preview.png',
}


def safe(path):
    if path.is_symlink():
        raise ValueError(f'Refusing symlink: {path}')
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(BASE.resolve()):
        raise ValueError(f'Path escapes structured_light: {path}')
    return resolved


def plans():
    result = [BASE / name for name in REMOVE_TOP]
    result += [p for p in DEPTH.iterdir() if p.name not in KEEP_DEPTH]
    groups = BASE / 'structured_groups_3_2pct_v1'
    result += [p for p in groups.iterdir() if p.name not in KEEP_GROUPS]
    result += [p for p in (DEPTH / 'fusion_edge_recovery_v1').iterdir()
               if p.name not in KEEP_FUSION_REFERENCE]
    result += [p for p in (DEPTH / 'fusion_sequential_closed_v1').iterdir()
               if p.name not in KEEP_FUSION_BASELINE]
    result += [p for p in (BASE / 'structured_subject_green_only_v2').iterdir()
               if p.name not in {'masks', 'report.json'}]
    for i in range(1, 32):
        folder = DEPTH / f'group_{i:04d}'
        result += [p for p in folder.iterdir() if p.name not in {'depth_mm.npz', 'quality.json'}]
    mesh_chain = DEPTH / 'component_four_step_v3'
    result += [mesh_chain / name for name in ('all_components_assembled.ply',)
               if (mesh_chain / name).exists()]
    mesh_output = mesh_chain / 'poisson_mesh_v1'
    result += [mesh_output / name for name in ('mesh_vertices_preview.ply',)
               if (mesh_output / name).exists()]
    return sorted((safe(p) for p in result if p.exists()), key=str)


def bytes_and_files(path):
    if path.is_file():
        return path.stat().st_size, 1
    total, count = 0, 0
    for child in path.rglob('*'):
        if child.is_symlink():
            raise ValueError(f'Refusing symlink inside {path}: {child}')
        if child.is_file():
            total += child.stat().st_size
            count += 1
    return total, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    required = [
        BASE / 'structured_capture/20261026_183358_297217',
        BASE / 'calibration/current/calibration.json',
        BASE / 'structured_decode_20261026_210453/decode_summary.json',
        BASE / 'structured_subject_green_only_v2/masks',
        BASE / 'structured_groups_3_2pct_v1/decode/decode_summary.json',
        DEPTH / 'depth_summary.json',
        DEPTH / 'group_0001/depth_mm.npz',
        DEPTH / 'group_0031/depth_mm.npz',
        DEPTH / 'sequential_registration_v1/registration_closed.json',
        DEPTH / 'fusion_sequential_closed_v1/dense.ply',
        DEPTH / 'component_four_step_v3/fused_components.ply',
        DEPTH / 'component_four_step_v3/poisson_mesh_v1/mesh.ply',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Mesh chain is incomplete: {missing}')
    targets = plans()
    metrics = [bytes_and_files(path) for path in targets]
    total_bytes = sum(item[0] for item in metrics)
    total_files = sum(item[1] for item in metrics)
    print(f'Planned removal: {len(targets)} paths, {total_files} files, {total_bytes / 1024**3:.2f} GiB')
    for path, (size, count) in zip(targets, metrics):
        print(f'{path.relative_to(BASE)}  ({count} files, {size / 1024**2:.1f} MiB)')
    if not args.apply:
        return
    audit = {'kept_mesh': str(required[-1].resolve()),
             'removed': [{'path': str(path.relative_to(BASE)), 'files': count, 'bytes': size}
                         for path, (size, count) in zip(targets, metrics)],
             'removed_bytes': total_bytes, 'removed_files': total_files}
    for path in targets:
        safe(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print('REMOVED', path.relative_to(BASE), flush=True)
    (BASE / 'current_mesh_cleanup_manifest.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
