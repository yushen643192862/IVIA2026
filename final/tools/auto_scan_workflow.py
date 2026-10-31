"""36-angle structured-light scan: capture, reconstruct, and propose gap retakes.

Run from final/: python tools/auto_scan_workflow.py capture --run-dir outputs/new_scan
Then:           python tools/auto_scan_workflow.py reconstruct --run-dir outputs/new_scan
The motor adapter is a user-supplied Python file exporting rotate(deg) and
slide(mm). A moved camera/projector requires a new calibrated extrinsic pose.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.decode_capture_session import read_image


def run_command(*items):
    command = [sys.executable, *map(str, items)]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def completed_groups(session):
    result = []
    for folder in sorted(session.glob('group_*')):
        if not folder.is_dir():
            continue
        file = folder / 'group.json'
        if file.is_file() and json.loads(file.read_text(encoding='utf-8')).get('complete'):
            result.append(folder)
    return result


def inspect_coverage(depth_dir, output, step_degrees, min_area=100, margin_px=12):
    """Report interior depth gaps in each view; this is a candidate list, not proof of a 3D hole."""
    masks = depth_dir / 'effective_masks'
    rows = []
    for index, folder in enumerate(sorted(depth_dir.glob('group_*'))):
        source = folder / 'depth_mm.npz'
        mask_file = masks / (folder.name + '.png')
        if not source.is_file() or not mask_file.is_file():
            continue
        with np.load(source, allow_pickle=False) as data:
            depth = data['depth_mm']
        mask = read_image(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError(f'Invalid effective mask: {mask_file}')
        interior = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5) >= margin_px
        missing = interior & ~(np.isfinite(depth) & (depth > 0))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
        holes = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                holes.append({'area_px': area, 'center_xy_px': centers[label].round(1).tolist()})
        holes.sort(key=lambda item: -item['area_px'])
        rows.append({'group': folder.name, 'nominal_angle_deg': index * step_degrees,
                     'interior_mask_pixels': int(interior.sum()),
                     'missing_depth_fraction': float(missing.sum() / max(1, interior.sum())),
                     'hole_candidates': holes[:10]})
    rows.sort(key=lambda item: -item['missing_depth_fraction'])
    report = {'method': '2D interior depth-gap screening', 'groups': rows,
              'retake_candidates': [row for row in rows if row['hole_candidates']][:12],
              'warning': 'Occlusion, dark material, and a wrong mask can produce these gaps. '
                         'A 2D gap is not proof that the fused 3D surface has a hole. '
                         'Moving the camera/projector slider requires a calibrated pose and registration before fusion.'}
    save_json(output, report)
    return report


def capture(args):
    if not args.motion_adapter:
        raise ValueError('Capture requires --motion-adapter')
    session = args.run_dir / 'capture'
    command = [ROOT / 'tools/capture_structured_light.py', '--motion-adapter', args.motion_adapter,
               '--auto-groups', 36, '--angle-step', 10, '--display', args.display]
    command += ['--resume-session', session] if session.is_dir() else ['--output', session]
    if args.serial:
        command += ['--serial', args.serial]
    run_command(*command)
    groups = completed_groups(session)
    if len(groups) != 36:
        raise RuntimeError(f'Capture stopped with {len(groups)}/36 complete groups; resume after checking the log')
    save_json(args.run_dir / 'capture_complete.json', {'groups': 36, 'angle_step_degrees': 10,
        'first_angle_degrees': 0, 'last_angle_degrees': 350, 'session': str(session.resolve())})


def reconstruct(args):
    session = args.run_dir / 'capture'
    if len(completed_groups(session)) != 36:
        raise ValueError('Exactly 36 complete groups are required')
    if not args.calibration.is_file():
        raise FileNotFoundError(args.calibration)
    decode = args.run_dir / 'decode'
    masks = args.run_dir / 'segmentation'
    depth = args.run_dir / 'depth'
    fused = args.run_dir / 'fusion'
    if not decode.exists():
        run_command(ROOT / 'tools/decode_capture_session.py', session, '--output', decode,
                    '--bit-threshold', args.bit_threshold, '--relative-threshold', args.relative_threshold)
    if not masks.exists():
        run_command(ROOT / 'tools/segment_structured_subject.py', decode / 'color_images', '--output', masks)
    if not depth.exists():
        run_command(ROOT / 'tools/reconstruct_structured_depth.py', decode, '--calibration', args.calibration,
                    '--masks-directory', masks / 'masks', '--mask-inset-px', args.mask_inset_px,
                    '--projector-offset', *args.projector_offset, '--output', depth)
    if not fused.exists():
        run_command(ROOT / 'tools/fuse_structured_clouds.py', depth, '--output', fused,
                    '--voxel-mm', args.voxel_mm, '--dense-refine')
    report = inspect_coverage(depth, args.run_dir / 'gap_candidates.json', 10)
    save_json(args.run_dir / 'workflow_status.json', {'stage': 'fused', 'cloud': str(fused / 'dense.ply'),
              'gap_candidate_groups': len(report['retake_candidates']),
              'note': 'Inspect fused cloud and gap candidates before any slider retake.'})
    print('Cloud:', fused / 'dense.ply', flush=True)
    print('Gap candidates:', args.run_dir / 'gap_candidates.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['capture', 'reconstruct', 'inspect'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--motion-adapter', type=Path)
    parser.add_argument('--display', type=int, default=2)
    parser.add_argument('--serial')
    parser.add_argument('--calibration', type=Path,
                        default=ROOT / 'outputs/structured_light/calibration/current/calibration.json')
    parser.add_argument('--bit-threshold', type=float, default=3.)
    parser.add_argument('--relative-threshold', type=float, default=.02)
    parser.add_argument('--mask-inset-px', type=float, default=6.)
    parser.add_argument('--projector-offset', type=float, nargs=2, default=(0., 0.))
    parser.add_argument('--voxel-mm', type=float, default=.5)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.stage == 'capture':
        args.run_dir.mkdir(parents=True, exist_ok=True)
        capture(args)
    elif args.stage == 'reconstruct':
        reconstruct(args)
    else:
        inspect_coverage(args.run_dir / 'depth', args.run_dir / 'gap_candidates.json', 10)


if __name__ == '__main__':
    main()
