"""Conservative small component alignment using multiple original white-light views."""
import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from tools.decode_capture_session import read_image


class PhotoReferences:
    def __init__(self, names, depth_dir, color_dir, mask_dir, poses, camera_matrix, distortion, crop_xy):
        self.names = names
        self.depth_dir = depth_dir
        self.color_dir = color_dir
        self.mask_dir = mask_dir
        self.poses = poses
        self.K = np.asarray(camera_matrix, float)
        self.distortion = np.asarray(distortion, float)
        self.crop_xy = np.asarray(crop_xy, float)
        self.cache = {}

    def get(self, index):
        if index in self.cache:
            return self.cache[index]
        name = self.names[index]
        image = read_image(self.color_dir / f'{name}.png', cv2.IMREAD_COLOR)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        mask = read_image(self.mask_dir / f'{name}.png') > 0
        with np.load(self.depth_dir / name / 'depth_mm.npz', allow_pickle=False) as data:
            xyz = data['xyz_mm']
            pixels = data['pixels_xy'].astype(int)
        depth = np.zeros(mask.shape, np.float32)
        depth[pixels[:, 1], pixels[:, 0]] = xyz[:, 2]
        distance, indices = ndimage.distance_transform_edt(depth == 0, return_indices=True)
        nearest = depth[tuple(indices)]
        pose = np.asarray(self.poses[name], float)
        item = {'lab': lab, 'mask': mask, 'depth': nearest, 'depth_distance': distance,
                'inverse_pose': np.linalg.inv(pose)}
        self.cache[index] = item
        while len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        return item

    def project(self, world, reference):
        camera = world @ reference['inverse_pose'][:3, :3].T + reference['inverse_pose'][:3, 3]
        full = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), self.K, self.distortion)[0].reshape(-1, 2)
        return full - self.crop_xy, camera[:, 2]


def small_adjustment(points, source_rgb, component_pose, references, indices,
                     max_samples=350, max_translation_mm=2., max_rotation_deg=.6):
    """Return transform and diagnostics; reject weak or occluded photometric evidence."""
    report = {'attempted': True, 'accepted': False, 'reference_groups': []}
    if len(points) < 300 or len(indices) < 2:
        report['reason'] = 'component too small or fewer than two prior views'
        return None, report
    choice = np.linspace(0, len(points) - 1, min(max_samples, len(points)), dtype=int)
    local = points[choice]
    source_lab = cv2.cvtColor(source_rgb[choice].reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    if np.linalg.norm(np.std(source_lab, axis=0)) < 10.:
        report['reason'] = 'surface has too little color variation'
        return None, report
    baseline = local @ component_pose[:3, :3].T + component_pose[:3, 3]
    center = np.median(baseline, axis=0)
    visible = []
    for index in indices:
        ref = references.get(index)
        xy, z = references.project(baseline, ref)
        ix = np.rint(xy[:, 0]).astype(int)
        iy = np.rint(xy[:, 1]).astype(int)
        h, w = ref['mask'].shape
        inside = (z > 0) & (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
        good = np.flatnonzero(inside)
        good = good[ref['mask'][iy[good], ix[good]] &
                    (ref['depth_distance'][iy[good], ix[good]] <= 2.) &
                    (np.abs(ref['depth'][iy[good], ix[good]] - z[good]) <= 5.)]
        if len(good) >= 30:
            visible.append((index, good, ref))
            report['reference_groups'].append(references.names[index])
    if len(visible) < 2 or sum(len(item[1]) for item in visible) < 100:
        report['reason'] = 'not visibly depth-consistent in two original photographs'
        return None, report

    def evaluate(parameters):
        rotation = cv2.Rodrigues(parameters[:3])[0]
        shifted = (baseline - center) @ rotation.T + center + parameters[3:]
        all_errors = []
        view_errors = []
        valid_count = 0
        for _, ids, ref in visible:
            xy, z = references.project(shifted[ids], ref)
            ix = np.rint(xy[:, 0]).astype(int)
            iy = np.rint(xy[:, 1]).astype(int)
            h, w = ref['mask'].shape
            inside = (z > 0) & (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
            valid = np.flatnonzero(inside)
            valid = valid[ref['mask'][iy[valid], ix[valid]] &
                          (ref['depth_distance'][iy[valid], ix[valid]] <= 2.) &
                          (np.abs(ref['depth'][iy[valid], ix[valid]] - z[valid]) <= 6.)]
            error = np.full(len(ids), 35., float)
            if len(valid):
                target_lab = ref['lab'][iy[valid], ix[valid]].astype(float)
                delta = target_lab - source_lab[ids[valid]]
                error[valid] = np.sqrt(.15 * delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
                valid_count += len(valid)
            all_errors.append(np.clip(error, 0, 35.))
            view_errors.append(float(np.mean(np.clip(error, 0, 35.))))
        score = float(np.mean(np.concatenate(all_errors)))
        return score, valid_count, view_errors

    start = np.zeros(6)
    before, baseline_valid, before_views = evaluate(start)
    angle = np.radians(max_rotation_deg)
    bounds = [(-angle, angle)] * 3 + [(-max_translation_mm, max_translation_mm)] * 3
    fit = minimize(lambda value: evaluate(value)[0], start, method='Powell', bounds=bounds,
                   options={'maxiter': 12, 'maxfev': 90, 'xtol': .03, 'ftol': .015})
    after, final_valid, after_views = evaluate(fit.x)
    rotation = cv2.Rodrigues(fit.x[:3])[0]
    moved_points = (baseline - center) @ rotation.T + center + fit.x[3:]
    motion = float(np.median(np.linalg.norm(moved_points - baseline, axis=1)))
    report.update({'samples': len(local), 'visible_samples': baseline_valid,
                   'error_before': before, 'error_after': after,
                   'error_before_by_view': before_views, 'error_after_by_view': after_views,
                   'motion_mm': motion, 'optimizer_evaluations': int(fit.nfev)})
    improvements = np.asarray(before_views) - np.asarray(after_views)
    if (after > before * .9 or before - after < 1.5 or final_valid < baseline_valid * .9 or
            motion > 2. or np.sum(improvements >= 1.) < 2 or np.min(improvements) < -.5):
        report['reason'] = 'multi-view photo agreement did not improve safely'
        return None, report
    adjustment = np.eye(4)
    adjustment[:3, :3] = rotation
    adjustment[:3, 3] = center - rotation @ center + fit.x[3:]
    report['accepted'] = True
    return adjustment @ component_pose, report
