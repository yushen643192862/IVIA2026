"""Incrementally align disconnected dense components using shared textured 3D points."""
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
from tools.export_editable_cloud_layers import VERTEX, write_binary_ply
from tools.refine_texture_landmarks import moved, robust_pose
from tools.split_depth_components import label_points
from tools.photometric_component_alignment import PhotoReferences, small_adjustment


def correct_edge_colors(xyz, pixels, rgb, mask, edge_px=6):
    """Borrow color only from a nearby, depth-consistent interior patch."""
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    pixel_distance = distance[pixels[:, 1], pixels[:, 0]]
    edge_ids = np.flatnonzero(pixel_distance <= edge_px)
    interior_ids = np.flatnonzero(pixel_distance >= edge_px + 8)
    corrected = rgb.copy()
    result = {'edge_candidates': len(edge_ids), 'interior_points': len(interior_ids),
              'recolored_points': 0}
    if len(edge_ids) == 0 or len(interior_ids) < 12:
        return corrected, result
    tree = cKDTree(pixels[interior_ids])
    nearest_distance, nearest_index = tree.query(pixels[edge_ids], k=12, distance_upper_bound=18.)
    for row, point_id in enumerate(edge_ids):
        valid = np.isfinite(nearest_distance[row])
        if valid.sum() < 8 or nearest_distance[row, valid].min() > 12.:
            continue
        neighbors = interior_ids[nearest_index[row, valid]]
        depths = xyz[neighbors, 2]
        median_depth = np.median(depths)
        depth_gap = xyz[point_id, 2] - median_depth
        if not (6. <= depth_gap <= 25.) or np.median(np.abs(depths - median_depth)) > 1.5:
            continue
        color = rgb[neighbors].astype(np.float32)
        middle = np.median(color, axis=0)
        if np.percentile(np.linalg.norm(color - middle, axis=1), 80) > 14.:
            continue
        if np.linalg.norm(rgb[point_id].astype(float) - middle) < 12.:
            continue
        corrected[point_id] = np.rint(middle).astype(np.uint8)
        result['recolored_points'] += 1
    return corrected, result


def label_color_spatial(xyz, rgb, voxel_mm):
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv.T.astype(int)
    category = np.where(v < 65, 0, np.where(s < 70, 1, 2 + ((h + 22) % 180) // 45))
    labels = np.zeros(len(xyz), np.int32)
    offset = 0
    grids = {}
    for value in np.unique(category):
        selected = category == value
        part, shape = label_points(xyz[selected], voxel_mm)
        labels[selected] = part + offset
        offset += int(part.max())
        grids[int(value)] = shape
    return labels, grids


def load_view(name, depth_dir, color_dir, mask_dir, sift, voxel_mm):
    with np.load(depth_dir / name / 'depth_mm.npz', allow_pickle=False) as depth:
        xyz = depth['xyz_mm']
        pixels = depth['pixels_xy'].astype(int)
    image = read_image(color_dir / f'{name}.png', cv2.IMREAD_COLOR)
    rgb = image[pixels[:, 1], pixels[:, 0], ::-1]
    mask = read_image(mask_dir / f'{name}.png')
    rgb, edge_report = correct_edge_colors(xyz, pixels, rgb, mask)
    labels, grid_shape = label_color_spatial(xyz, rgb, voxel_mm)
    unique, counts = np.unique(labels, return_counts=True)
    ordering = unique[np.argsort(counts)[::-1]]
    rank_by_label = np.zeros(int(unique.max()) + 1, np.int32)
    rank_by_label[ordering] = np.arange(1, len(ordering) + 1)
    rank = rank_by_label[labels]
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)
        point_ids = np.empty(0, int)
    else:
        positions = np.array([point.pt for point in keypoints], float)
        distances, nearest = cKDTree(pixels).query(positions)
        valid = np.flatnonzero(distances <= 2.)
        valid = valid[np.argsort([-keypoints[i].response for i in valid])[:1500]]
        descriptors, point_ids = descriptors[valid], nearest[valid]
    return {'xyz': xyz, 'rgb': rgb, 'rank': rank, 'counts': counts[np.argsort(counts)[::-1]],
            'descriptors': descriptors, 'feature_points': xyz[point_ids],
            'feature_rank': rank[point_ids], 'grid_shape': grid_shape,
            'edge_color_report': edge_report}


def vertices(xyz, rgb):
    result = np.empty(len(xyz), dtype=VERTEX)
    for i, key in enumerate(('x', 'y', 'z')):
        result[key] = xyz[:, i]
    for i, key in enumerate(('red', 'green', 'blue')):
        result[key] = rgb[:, i]
    return result


def header(count):
    return (f'ply\nformat binary_little_endian 1.0\n'
            'comment unit millimeter; group_0001 camera coordinate frame\n'
            f'element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', type=Path)
    parser.add_argument('--initial-registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-mm', type=float, default=3.)
    parser.add_argument('--min-component-points', type=int, default=500)
    parser.add_argument('--max-component-motion-mm', type=float, default=3.)
    parser.add_argument('--max-groups', type=int, default=0, help='0 processes every group')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    if args.voxel_mm <= 0 or args.min_component_points <= 0:
        parser.error('Voxel size and component size must be positive')
    summary = json.loads((args.depth_dir / 'depth_summary.json').read_text(encoding='utf-8'))
    names = [row['group'] for row in summary['groups']]
    if args.max_groups:
        names = names[:args.max_groups]
    decoded = Path(summary['source'])
    decode_report = json.loads((decoded / 'decode_summary.json').read_text(encoding='utf-8'))
    calibration = json.loads(Path(summary['calibration_source']).read_text(encoding='utf-8'))
    mask_dir = args.depth_dir / 'effective_masks'
    if not mask_dir.is_dir():
        mask_dir = Path(summary['foreground_masks'])
    initial = json.loads(args.initial_registration.read_text(encoding='utf-8'))['poses']
    photos = PhotoReferences(names, args.depth_dir, decoded / 'color_images', mask_dir,
                             initial, calibration['camera_matrix'], calibration['camera_distortion'],
                             decode_report['crop_roi_xywh'][:2])
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.015)
    rng = np.random.default_rng(1703)
    args.output.mkdir(parents=True)
    # Store prior landmark positions with the accepted component transform.
    model_desc, model_xyz, model_group = [], [], []
    reports = []
    point_files = []
    total_points = 0
    for index, name in enumerate(names):
        view = load_view(name, args.depth_dir, decoded / 'color_images', mask_dir, sift, args.voxel_mm)
        pose = np.asarray(initial[name], float)
        group_dir = args.output / name
        group_dir.mkdir()
        world = moved(view['xyz'], pose)
        feature_world = moved(view['feature_points'], pose)
        group_report = {'group': name, 'total_points': len(world),
                        'voxel_grid_shape': view['grid_shape'],
                        'edge_color_report': view['edge_color_report'],
                        'dense_component_count': int(np.sum(view['counts'] >= args.min_component_points)),
                        'components': []}
        # Match once against the entire already assembled model, then assign matches by component.
        if model_desc and len(view['descriptors']):
            training = np.vstack(model_desc)
            target = np.vstack(model_xyz)
            source_groups = np.concatenate(model_group)
            pairs = matcher.knnMatch(view['descriptors'], training, k=min(12, len(training)))
            matches = []
            match_support = {}
            for pair in pairs:
                if len(pair) < 2:
                    continue
                first = pair[0]
                location = target[first.trainIdx]
                distinct = next((m for m in pair[1:]
                                 if np.linalg.norm(target[m.trainIdx] - location) > 5.), None)
                if distinct is None or first.distance >= .7 * distinct.distance:
                    continue
                matches.append(first)
                match_support[first.queryIdx] = sorted({int(source_groups[m.trainIdx]) for m in pair
                    if np.linalg.norm(target[m.trainIdx] - location) < 3. and
                    m.distance <= first.distance * 1.3 + 10.})
        else:
            target = np.empty((0, 3))
            source_groups = np.empty(0, int)
            matches = []
            match_support = {}
        accepted = 0
        photo_accepted = 0
        for component_rank, count in enumerate(view['counts'], 1):
            if count < args.min_component_points:
                break
            selector = view['rank'] == component_rank
            feature_selector = view['feature_rank'] == component_rank
            feature_ids = np.flatnonzero(feature_selector)
            component_matches = [m for m in matches if feature_selector[m.queryIdx]]
            src = np.array([view['feature_points'][m.queryIdx] for m in component_matches]).reshape(-1, 3)
            dst = np.array([target[m.trainIdx] for m in component_matches]).reshape(-1, 3)
            baseline_distance = np.linalg.norm(moved(src, pose) - dst, axis=1)
            plausible = baseline_distance < 12.
            src, dst = src[plausible], dst[plausible]
            used_matches = [m for m, ok in zip(component_matches, plausible) if ok]
            row = {'rank': component_rank, 'points': int(count), 'texture_features': len(feature_ids),
                   'plausible_matches': len(src), 'accepted': False}
            transform = pose
            if index == 0:
                row['reason'] = 'fixed base group'
            elif len(src) < 15:
                row['reason'] = 'too few cross-view textured 3D matches'
            else:
                candidate, inliers = robust_pose(src, dst, pose, rng, 300, 2.)
                if candidate is None or len(inliers) < 20:
                    row['reason'] = 'no reliable rigid 3D correspondence'
                else:
                    supporters = sorted({group for k in inliers
                        for group in match_support[used_matches[k].queryIdx]})
                    spread = np.linalg.svd(dst[inliers] - dst[inliers].mean(axis=0), compute_uv=False)
                    probe = view['xyz'][selector][::max(1, int(count) // 300)]
                    motion = float(np.median(np.linalg.norm(moved(probe, candidate) - moved(probe, pose), axis=1)))
                    before = float(np.median(np.linalg.norm(moved(src[inliers], pose) - dst[inliers], axis=1)))
                    after = float(np.median(np.linalg.norm(moved(src[inliers], candidate) - dst[inliers], axis=1)))
                    row.update({'inliers': len(inliers), 'support_groups': supporters,
                                'feature_spread_mm': spread.tolist(), 'motion_mm': motion,
                                'landmark_before_mm': before, 'landmark_after_mm': after})
                    if len(supporters) < 2 or spread[1] < 5.:
                        row['reason'] = 'feature support is limited to one view or a narrow patch'
                    elif motion > args.max_component_motion_mm:
                        row['reason'] = 'component movement exceeds guard'
                    elif after >= before - .15:
                        row['reason'] = 'texture alignment does not materially improve'
                    else:
                        transform = candidate
                        row['accepted'] = True
                        row['method'] = 'texture_3d'
                        accepted += 1
            if index and not row['accepted']:
                candidate, photo_report = small_adjustment(
                    view['xyz'][selector], view['rgb'][selector], pose, photos,
                    range(max(0, index - 5), index))
                row['photo_reprojection'] = photo_report
                if candidate is not None:
                    transform = candidate
                    row['accepted'] = True
                    row['method'] = 'multi_view_photos'
                    if 'reason' in row:
                        row['texture_rejection_reason'] = row.pop('reason')
                    row['reason'] = 'accepted by multiple depth-consistent white-light views'
                    photo_accepted += 1
            world[selector] = moved(view['xyz'][selector], transform)
            feature_world[feature_selector] = moved(view['feature_points'][feature_selector], transform)
            filename = f'component_{component_rank:03d}.ply'
            write_binary_ply(group_dir / filename, world[selector], view['rgb'][selector])
            row['file'] = filename
            row['transform'] = transform.tolist()
            group_report['components'].append(row)
        # Components below threshold remain at their original group pose.
        fragment_selector = view['rank'] > len(group_report['components'])
        if fragment_selector.any():
            write_binary_ply(group_dir / 'small_fragments.ply', world[fragment_selector], view['rgb'][fragment_selector])
        group_report['small_fragment_points'] = int(fragment_selector.sum())
        group_report['accepted_component_count'] = accepted
        group_report['accepted_photo_component_count'] = photo_accepted
        reports.append(group_report)
        model_desc.append(view['descriptors'])
        model_xyz.append(feature_world)
        model_group.append(np.full(len(feature_world), index + 1, int))
        filename = args.output / f'{name}_assembled.ply'
        write_binary_ply(filename, world, view['rgb'])
        point_files.append(filename)
        total_points += len(world)
        print(name, 'dense components', len(group_report['components']),
              'texture', accepted, 'photos', photo_accepted, flush=True)
    # This unfiltered diagnostic intentionally retains every original depth point.
    with (args.output / 'all_components_assembled.ply').open('wb') as stream:
        stream.write(header(total_points))
        for path in point_files:
            with path.open('rb') as source_file:
                while source_file.readline().strip() != b'end_header':
                    pass
                while chunk := source_file.read(1024 * 1024):
                    stream.write(chunk)
    save_json(args.output / 'component_registration.json', {
        'method': 'group_0001 fixed; each later color-spatial component aligned by 3D texture, then multi-view photos when validated',
        'initial_registration': str(args.initial_registration.resolve()),
        'voxel_mm': args.voxel_mm, 'min_component_points': args.min_component_points,
        'max_component_motion_mm': args.max_component_motion_mm,
        'groups': reports, 'total_points': total_points,
        'note': 'Diagnostic unfiltered assembly; unsupported components keep the initial group pose.'})


if __name__ == '__main__':
    main()
