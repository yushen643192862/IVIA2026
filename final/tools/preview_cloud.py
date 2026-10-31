"""Render three orthographic views of a colored PLY point cloud."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scanner.io import read_ply

parser = argparse.ArgumentParser()
parser.add_argument('cloud', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--blue-base', action='store_true', help='Orient a figure with a blue pedestal upright')
args = parser.parse_args()
points, _, attrs = read_ply(args.cloud, with_attributes=True)
if not len(points):
    raise ValueError('Empty point cloud')
colors = np.column_stack([attrs[k] for k in ('red', 'green', 'blue')]).astype(np.uint8)
_, axes = np.linalg.eigh(np.cov(points.T))
aligned = (points - np.median(points, axis=0)) @ axes[:, ::-1]
# Orient the height axis so that its narrower end is above the wide base.
low = aligned[aligned[:, 0] < np.percentile(aligned[:, 0], 15), 1:]
high = aligned[aligned[:, 0] > np.percentile(aligned[:, 0], 85), 1:]
if np.ptp(low, axis=0).sum() > np.ptp(high, axis=0).sum():
    aligned[:, 0] *= -1
if args.blue_base:
    blue = (colors[:, 2].astype(float) > colors[:, 0] * 1.2) & (colors[:, 1].astype(float) > colors[:, 0] * 1.1)
    if np.any(blue) and np.median(aligned[blue, 0]) < 0:
        aligned[:, 0] *= -1
canvas = np.full((850, 1800, 3), 245, np.uint8)
for panel, (a, b) in enumerate(((1, 0), (2, 0), (1, 2))):
    uv = aligned[:, [a, b]]
    lo, hi = np.percentile(uv, [0, 100], axis=0)
    scale = min(540 / max(hi[0] - lo[0], 1e-9), 720 / max(hi[1] - lo[1], 1e-9))
    xy = (uv - (lo + hi) / 2) * scale + [panel * 600 + 300, 430]
    for i in np.argsort(aligned[:, 3-a-b]):
        x, y = np.rint(xy[i]).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 1, tuple(int(c) for c in colors[i, ::-1]), -1)
    cv2.putText(canvas, f'PCA view {panel+1}', (panel*600+20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2)
cv2.imencode('.png', canvas)[1].tofile(str(args.output))
print('Points:', len(points), 'PCA extents:', np.ptp(aligned, axis=0))
