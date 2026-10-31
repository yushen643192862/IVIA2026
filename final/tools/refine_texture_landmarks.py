"""Refine registered depth views using distinctive textured 3D landmarks."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scanner.io import save_json
from tools.decode_capture_session import read_image


def moved(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def rigid(source, target):
    left, right = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - left).T @ (target - right))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = right - rotation @ left
    return pose


def landmarks(name, depth_dir, color_dir, mask_dir, sift, maximum):
    with np.load(depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
        xyz, pixels = data['xyz_mm'], data['pixels_xy']
    image = read_image(color_dir / f'{name}.png', cv2.IMREAD_GRAYSCALE)
    mask = read_image(mask_dir / f'{name}.png')
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    if descriptors is None:
        return {'xyz': xyz, 'feature_xyz': np.empty((0, 3)), 'descriptors': np.empty((0, 128), np.float32)}
    positions = np.array([point.pt for point in keypoints], float)
    distances, indices = cKDTree(pixels).query(positions)
    valid = np.flatnonzero(distances <= 2.)
    valid = valid[np.argsort([-keypoints[i].response for i in valid])[:maximum]]
    return {'xyz': xyz, 'feature_xyz': xyz[indices[valid]], 'descriptors': descriptors[valid]}


def robust_pose(source, target, initial, rng, iterations, threshold):
    if len(source) < 20:
        return None, None
    best = np.empty(0, int)
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        triangle = np.linalg.norm(np.cross(source[sample[1]] - source[sample[0]],
                                           source[sample[2]] - source[sample[0]]))
        if triangle < 50.:
            continue
        candidate = rigid(source[sample], target[sample])
        residual = np.linalg.norm(moved(source, candidate) - target, axis=1)
        inliers = np.flatnonzero(residual < threshold)
        if len(inliers) > len(best):
            best = inliers
    if len(best) < 20:
        return None, None
    candidate = rigid(source[best], target[best])
    residual = np.linalg.norm(moved(source, candidate) - target, axis=1)
    best = np.flatnonzero(residual < threshold)
    if len(best) < 20:
        return None, None
    return rigid(source[best], target[best]), best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--initial-registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-features', type=int, default=1200)
    parser.add_argument('--max-pose-motion-mm', type=float, default=5.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    source = Path(summary['source'])
    names = [row['group'] for row in summary['groups']]
    initial_doc = json.loads(args.initial_registration.read_text(encoding='utf-8'))
    initial_poses = {name: np.asarray(initial_doc['poses'][name], float) for name in names}
    mask_dir = args.depth_dir / 'effective_masks'
    if not mask_dir.is_dir():
        mask_dir = Path(summary['foreground_masks'])
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.015)
    data = [landmarks(name, args.depth_dir, source / 'color_images', mask_dir, sift, args.max_features)
            for name in names]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(157)
    refined = {names[0]: initial_poses[names[0]]}
    report = []
    for i in range(1, len(names)):
        name = names[i]
        view = data[i]
        # Recent views share physical surface; first views provide a potential loop anchor.
        references = list(range(max(0, i - 5), i))
        if i >= len(names) - 5:
            references = sorted(set(references + list(range(min(4, i)))))
        model_xyz, model_desc, model_groups = [], [], []
        for j in references:
            if len(data[j]['descriptors']) == 0:
                continue
            model_xyz.append(moved(data[j]['feature_xyz'], refined[names[j]]))
            model_desc.append(data[j]['descriptors'])
            model_groups.extend([names[j]] * len(data[j]['descriptors']))
        outcome = {'group': name, 'reference_groups': [names[j] for j in references], 'accepted': False}
        if not model_desc or len(view['descriptors']) < 20:
            outcome['reason'] = 'too few depth-supported texture features'
            refined[name] = initial_poses[name]
            report.append(outcome)
            continue
        targets = np.vstack(model_xyz)
        descriptors = np.vstack(model_desc)
        matches = []
        for pair in matcher.knnMatch(view['descriptors'], descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance:
                matches.append(pair[0])
        # Initial geometry rejects appearance matches to unrelated repeated patterns.
        src = np.array([view['feature_xyz'][m.queryIdx] for m in matches]).reshape(-1, 3)
        dst = np.array([targets[m.trainIdx] for m in matches]).reshape(-1, 3)
        distance_before = np.linalg.norm(moved(src, initial_poses[name]) - dst, axis=1)
        near = distance_before < 12.
        src, dst = src[near], dst[near]
        outcome['ratio_matches'] = len(matches)
        outcome['geometry_matches'] = len(src)
        if len(src) < 25:
            outcome['reason'] = 'too few geometrically plausible correspondences'
            refined[name] = initial_poses[name]
            report.append(outcome)
            print(name, outcome['reason'], flush=True)
            continue
        pose, inliers = robust_pose(src, dst, initial_poses[name], rng, 450, 2.)
        if pose is None:
            outcome['reason'] = 'no consistent rigid 3D landmark transform'
        else:
            spread = np.linalg.svd(dst[inliers] - dst[inliers].mean(axis=0), compute_uv=False)
            motion = np.linalg.norm(moved(view['xyz'][::max(1, len(view['xyz']) // 1000)], pose) -
                                    moved(view['xyz'][::max(1, len(view['xyz']) // 1000)], initial_poses[name]), axis=1)
            before = np.linalg.norm(moved(src[inliers], initial_poses[name]) - dst[inliers], axis=1)
            after = np.linalg.norm(moved(src[inliers], pose) - dst[inliers], axis=1)
            outcome.update({'inliers': len(inliers), 'inlier_fraction': len(inliers) / len(src),
                            'landmark_spread_mm': spread.tolist(),
                            'median_motion_mm': float(np.median(motion)),
                            'median_landmark_before_mm': float(np.median(before)),
                            'median_landmark_after_mm': float(np.median(after)),
                            'supporting_groups': sorted(set(np.asarray(model_groups)[[matches[k].trainIdx for k in np.flatnonzero(near)[inliers]]]))})
            # A distinctive patch must occupy real 3D extent; never permit large global moves.
            if len(inliers) < 30 or len(inliers) / len(src) < .25 or spread[1] < 8. or len(outcome['supporting_groups']) < 2:
                outcome['reason'] = 'landmarks too sparse, confined to one patch, or seen by only one model view'
            elif np.median(motion) > args.max_pose_motion_mm:
                outcome['reason'] = 'pose correction exceeds motion guard'
            elif np.median(after) >= np.median(before) - .15:
                outcome['reason'] = 'no material landmark improvement'
            else:
                # Check the complete depth surface against recent registered views.
                current = moved(view['xyz'][::20], initial_poses[name])
                changed = moved(view['xyz'][::20], pose)
                recent = np.vstack([moved(data[j]['xyz'][::20], refined[names[j]])
                                    for j in references if i - j <= 5])
                tree = cKDTree(recent)
                before_dist = tree.query(current)[0]
                after_dist = tree.query(changed)[0]
                baseline_overlap = float(np.mean(before_dist < 5.))
                new_overlap = float(np.mean(after_dist < 5.))
                outcome.update({'surface_overlap_before': baseline_overlap,
                                'surface_overlap_after': new_overlap})
                if new_overlap < baseline_overlap - .02:
                    outcome['reason'] = 'whole-surface overlap regressed'
                else:
                    outcome['accepted'] = True
        refined[name] = pose if outcome['accepted'] else initial_poses[name]
        report.append(outcome)
        print(name, 'accepted' if outcome['accepted'] else outcome.get('reason'),
              outcome.get('inliers', 0), flush=True)
    args.output.mkdir(parents=True)
    save_json(args.output / 'registration.json', {
        'poses': {name: refined[name].tolist() for name in names},
        'method': 'initial poses plus validated distinctive texture 3D landmarks against registered model',
        'source_registration': str(args.initial_registration.resolve()), 'experimental': True})
    save_json(args.output / 'texture_report.json', {
        'groups_total': len(names), 'groups_refined': sum(row['accepted'] for row in report),
        'max_pose_motion_mm': args.max_pose_motion_mm, 'groups': report})


if __name__ == '__main__':
    main()
