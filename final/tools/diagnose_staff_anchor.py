"""Check whether the hand-held staff can anchor front/back depth registration."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.decode_capture_session import read_image
from scanner.io import write_ply

DEPTH = ROOT / 'outputs/structured_light/structured_depth_mask_inset6_v1'
COLORS = ROOT / 'outputs/structured_light/structured_groups_3_2pct_v1/decode/color_images'
REG = DEPTH / 'fusion_sequential_closed_v1/registration.json'
OUT = DEPTH / 'staff_anchor_diagnostic'


def load(group, stride=1):
    name = f'group_{group:04d}'
    with np.load(DEPTH / name / 'depth_mm.npz') as data:
        xyz = data['xyz_mm'][::stride]
        pixels = data['pixels_xy'][::stride].astype(int)
    return xyz, pixels


def world(group, points, poses):
    pose = np.asarray(poses[f'group_{group:04d}'], float)
    return points @ pose[:3, :3].T + pose[:3, 3]


def candidate(group, poses):
    xyz, pixels = load(group)
    image = read_image(COLORS / f'group_{group:04d}.png', cv2.IMREAD_COLOR)
    rgb = image[pixels[:, 1], pixels[:, 0], ::-1].astype(float)
    x, y = pixels.T
    red = (rgb[:, 0] > 70) & (rgb[:, 0] > 1.16 * rgb[:, 1]) & (rgb[:, 0] > 1.08 * rgb[:, 2])
    if group == 1:
        area = (x > 250) & (x < 570) & (abs(y - (0.65 * x + 135)) < 12)
    else:
        area = (x > 600) & (x < 930) & (abs(y - (-0.33 * x + 760)) < 12)
    return world(group, xyz[red & area], poses), pixels[red & area]


def score(a, b, offset):
    d = cKDTree(a).query(b + offset)[0]
    return float(np.mean(np.clip(d, 0, 8))), float(np.mean(d < 5)), float(np.median(d))


def seam(a, b, shift):
    d = cKDTree(a).query(b + shift)[0]
    return {'within_5mm': round(float(np.mean(d < 5)), 4),
            'median_mm': round(float(np.median(d)), 2),
            'p25_mm': round(float(np.percentile(d, 25)), 2)}


def main():
    poses = json.loads(REG.read_text(encoding='utf-8'))['poses']
    front, front_pixels = candidate(1, poses)
    back, back_pixels = candidate(16, poses)
    # Corresponding shaft side is expected left of the subject; reject red outfit elsewhere.
    front = front[front[:, 0] < -30]
    back = back[back[:, 0] < -30]
    zero = np.zeros(3)
    baseline = score(front, back, zero)
    best = (float('inf'), None, None)
    for dx in np.arange(-30, 31, 5):
        for dy in np.arange(-30, 31, 5):
            for dz in np.arange(-40, 11, 5):
                t = np.array([dx, dy, dz], float)
                s = score(front, back, t)
                if s[0] < best[0]:
                    best = (s[0], t, s)
    coarse = best[1]
    for dx in np.arange(coarse[0] - 4, coarse[0] + 5, 1):
        for dy in np.arange(coarse[1] - 4, coarse[1] + 5, 1):
            for dz in np.arange(coarse[2] - 4, coarse[2] + 5, 1):
                t = np.array([dx, dy, dz], float)
                s = score(front, back, t)
                if s[0] < best[0]:
                    best = (s[0], t, s)
    shift = best[1]
    seams = {}
    for a, b in ((10, 12), (11, 12), (21, 22), (21, 23)):
        p = world(a, load(a, 20)[0], poses)
        q = world(b, load(b, 20)[0], poses)
        # For second seam, block 12..21 is first rather than second.
        if a == 21:
            p, q = q, p
        seams[f'{a:02d}-{b:02d}'] = {'before': seam(p, q, zero), 'after': seam(p, q, shift)}
    OUT.mkdir(exist_ok=True)
    write_ply(OUT / 'staff_candidates_before.ply', np.vstack([front, back]),
              colors=np.vstack([np.tile([255, 20, 20], (len(front), 1)), np.tile([20, 160, 255], (len(back), 1))]))
    write_ply(OUT / 'staff_candidates_shifted.ply', np.vstack([front, back + shift]),
              colors=np.vstack([np.tile([255, 20, 20], (len(front), 1)), np.tile([20, 160, 255], (len(back), 1))]))
    result = {'front_candidates': len(front), 'back_candidates': len(back),
              'candidate_note': 'Color and image-line heuristic; not verified point correspondences.',
              'baseline_staff': {'clipped_mean_mm': baseline[0], 'within_5mm': baseline[1], 'median_mm': baseline[2]},
              'best_shift_back_mm': shift.tolist(), 'shift_length_mm': float(np.linalg.norm(shift)),
              'shifted_staff': {'clipped_mean_mm': best[2][0], 'within_5mm': best[2][1], 'median_mm': best[2][2]},
              'seams': seams}
    # Show which original views contribute points around the hand-held staff.
    roi_min = np.array([-105., -80., 800.])
    roi_max = np.array([-20., -5., 865.])
    by_group = []
    roi_points = []
    roi_colors = []
    for group in range(1, 32):
        sample = world(group, load(group, 5)[0], poses)
        selected = sample[np.all((sample >= roi_min) & (sample <= roi_max), axis=1)]
        hue = (group * 137.508) % 360
        import colorsys
        color = np.rint(np.asarray(colorsys.hsv_to_rgb(hue / 360, .9, 1)) * 255).astype(np.uint8)
        by_group.append({'group': group, 'points_stride5': len(selected), 'rgb': color.tolist()})
        if len(selected):
            roi_points.append(selected)
            roi_colors.append(np.tile(color, (len(selected), 1)))
    write_ply(OUT / 'staff_region_by_group.ply', np.vstack(roi_points),
              colors=np.vstack(roi_colors))
    result['staff_region_box_mm'] = [roi_min.tolist(), roi_max.tolist()]
    result['staff_region_points_by_group'] = by_group
    # Moving only the rear-facing view also moves its arm and torso points.
    local_checks = {}
    for other in (14, 15, 17, 18):
        fixed = world(other, load(other, 12)[0], poses)
        moving = world(16, load(16, 12)[0], poses)
        local_checks[f'16-{other:02d}'] = {
            'whole_before': seam(fixed, moving, zero),
            'whole_after': seam(fixed, moving, shift),
        }
        body_min = np.array([-20., -75., 790.])
        body_max = np.array([90., 90., 920.])
        fixed_body = fixed[np.all((fixed >= body_min) & (fixed <= body_max), axis=1)]
        moving_body = moving[np.all((moving >= body_min) & (moving <= body_max), axis=1)]
        if len(fixed_body) and len(moving_body):
            local_checks[f'16-{other:02d}']['body_before'] = seam(fixed_body, moving_body, zero)
            local_checks[f'16-{other:02d}']['body_after'] = seam(fixed_body, moving_body, shift)
    result['single_group_shift_checks'] = local_checks
    (OUT / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
