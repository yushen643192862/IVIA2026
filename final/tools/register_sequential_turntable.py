"""Experimental sequential turntable registration: nominal angle, silhouette search, then ICP."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.dense_registration import refine_pair, moved
from scanner.io import save_json
from tools.decode_capture_session import read_image


def turn_about(axis, center, angle):
    rotation = cv2.Rodrigues(axis * angle)[0]
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = center - rotation @ center
    return result


def axis_from_poses(poses):
    axes = []
    for first, second in zip(poses[:-1], poses[1:]):
        relative = second @ np.linalg.inv(first)
        vector = cv2.Rodrigues(relative[:3, :3])[0].ravel()
        if np.linalg.norm(vector) < np.radians(2):
            continue
        axis = vector / np.linalg.norm(vector)
        if axes and axis @ axes[0] < 0:
            axis = -axis
        axes.append(axis)
    axis = np.mean(axes, axis=0)
    axis /= np.linalg.norm(axis)
    matrix = np.concatenate([np.eye(3) - pose[:3, :3] for pose in poses[1:]])
    values = np.concatenate([pose[:3, 3] for pose in poses[1:]])
    center = np.linalg.lstsq(matrix, values, rcond=None)[0]
    return axis, center


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--axis-reference', type=Path, required=True,
                        help='Existing registration used only to estimate rotation axis and center')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'outputs/structured_light/calibration/current/calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-degrees', type=float, default=9.)
    args = parser.parse_args()
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    names = [item['group'] for item in summary['groups']]
    baseline = json.loads(args.axis_reference.read_text(encoding='utf-8'))
    reference_poses = [np.asarray(baseline['poses'][name], float) for name in names]
    axis, center = axis_from_poses(reference_poses)
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    decode = json.loads((Path(summary['source']) / 'decode_summary.json').read_text(encoding='utf-8'))
    x0, y0, width, height = decode['crop_roi_xywh']
    K = np.asarray(calibration['camera_matrix'], float)
    distortion = np.asarray(calibration['camera_distortion'], float)
    mask_dir = args.depth_dir / 'effective_masks'
    if not mask_dir.is_dir() and summary.get('source_depth_directory'):
        mask_dir = Path(summary['source_depth_directory']) / 'effective_masks'
    if not mask_dir.is_dir():
        mask_dir = Path(summary['foreground_masks'])
    clouds = []
    for name in names:
        with np.load(args.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
            clouds.append(data['xyz_mm'])
    nominal = np.radians(360. / (len(names) - 1))
    offsets = np.radians(np.arange(-args.search_degrees, args.search_degrees + 1e-6, 3.))
    poses = [np.eye(4)]
    steps = []
    for j in range(1, len(names)):
        i = j - 1
        previous = poses[i]
        target = clouds[i]
        source = clouds[j]
        sampled_target = target[::max(1, len(target) // 5000)]
        sampled_source = source[::max(1, len(source) // 5000)]
        tree = cKDTree(sampled_target)
        mask = read_image(mask_dir / (names[j] + '.png'))
        if mask.shape != (height, width):
            raise ValueError(f'Mask dimension mismatch: {names[j]}')
        outside = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)

        def score(candidate):
            relative = np.linalg.inv(previous) @ candidate
            transformed = moved(sampled_source, relative)
            distance = tree.query(transformed, k=1, workers=2)[0]
            overlap = distance < 5.
            overlap_fraction = float(np.mean(overlap))
            if overlap.sum() < 100:
                surface = 10.
            else:
                surface = float(np.median(distance[overlap]))
            # Project the previous view into the candidate current camera.
            camera = moved(sampled_target, np.linalg.inv(candidate) @ previous)
            projected = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)
            x = np.rint(projected[:, 0] - x0).astype(int)
            y = np.rint(projected[:, 1] - y0).astype(int)
            inside = (camera[:, 2] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            penalty = np.full(len(x), 10., float)
            penalty[inside] = np.minimum(outside[y[inside], x[inside]], 10.)
            silhouette = float(np.mean(penalty))
            return surface + 3. * (1. - overlap_fraction) + .12 * silhouette, overlap_fraction, silhouette

        candidates = []
        for offset in offsets:
            angle = nominal + offset
            predicted = turn_about(axis, center, angle) @ previous
            result = score(predicted)
            candidates.append((result[0], angle, predicted, result[1], result[2]))
        _, angle, initial, initial_overlap, initial_silhouette = min(candidates, key=lambda item: item[0])
        local_initial = np.linalg.inv(previous) @ initial
        refined, diagnostic = refine_pair(source, target, local_initial, seed=100 + j)
        if refined is not None:
            local_transform = refined[0]
            adjusted = previous @ local_transform
            adjusted_score, adjusted_overlap, adjusted_silhouette = score(adjusted)
            initial_score, _, _ = score(initial)
            use_icp = adjusted_score <= initial_score + .25 and adjusted_overlap >= initial_overlap - .05
            next_pose = adjusted if use_icp else initial
        else:
            use_icp = False
            next_pose = initial
            adjusted_overlap, adjusted_silhouette = initial_overlap, initial_silhouette
        poses.append(next_pose)
        steps.append({'from': names[i], 'to': names[j],
                      'coarse_angle_deg': float(np.degrees(angle)),
                      'theoretical_angle_deg': float(np.degrees(nominal)),
                      'coarse_overlap_fraction': initial_overlap,
                      'coarse_silhouette_outside_px': initial_silhouette,
                      'used_icp': bool(use_icp),
                      'final_overlap_fraction': adjusted_overlap if use_icp else initial_overlap,
                      'final_silhouette_outside_px': adjusted_silhouette if use_icp else initial_silhouette,
                      'icp': diagnostic})
        print(names[i], '->', names[j], 'angle', round(np.degrees(angle), 1),
              'overlap', round(steps[-1]['final_overlap_fraction'], 3),
              'ICP', use_icp, flush=True)
    last = poses[-1]
    # The first and last frames need not be identical; report their best 3D overlap instead.
    distance = cKDTree(clouds[0]).query(moved(clouds[-1][::max(1,len(clouds[-1])//5000)], last),
                                      k=1, workers=2)[0]
    near = distance < 5.
    closure = {'overlap_fraction': float(np.mean(near)),
               'median_nearest_mm': float(np.median(distance[near])) if near.any() else None}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'registration.json', {'poses': {name: pose.tolist() for name, pose in zip(names, poses)},
              'method': 'sequential nominal turntable angle + mask silhouette search + adjacent-view ICP',
              'axis_from': str(args.axis_reference.resolve()),
              'axis': axis.tolist(), 'center_mm': center.tolist(),
              'nominal_step_deg': float(np.degrees(nominal)), 'steps': steps,
              'first_last_overlap': closure,
              'experimental': True})
    print('First/last overlap:', closure, flush=True)


if __name__ == '__main__':
    main()
