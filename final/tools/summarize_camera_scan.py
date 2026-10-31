"""Validate this scan and render orthographic previews of its measured points."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scanner.io import read_ply, save_json

root = Path(__file__).resolve().parents[1] / 'outputs/camera_scan_cuda'
points, _, attrs = read_ply(root / 'dense.ply', with_attributes=True)
vertices, faces = read_ply(root / 'mesh.ply')
if not len(points) or not len(vertices) or not len(faces):
    raise RuntimeError('Empty reconstruction artifact')
colors = np.column_stack([attrs[k] for k in ('red', 'green', 'blue')]).astype(np.uint8)
centered = points - np.median(points, axis=0)
_, axes = np.linalg.eigh(np.cov(centered.T))
aligned = centered @ axes[:, ::-1]
canvas = np.full((850, 1800, 3), 245, np.uint8)
for panel, (a, b) in enumerate(((1, 0), (2, 0), (1, 2))):
    uv = aligned[:, [a, b]]
    lo, hi = np.percentile(uv, [0, 100], axis=0)
    scale = min(540 / max(hi[0]-lo[0], 1e-9), 720 / max(hi[1]-lo[1], 1e-9))
    xy = (uv - (lo+hi)/2) * scale + [panel*600+300, 430]
    for i in np.argsort(aligned[:, 3-a-b]):
        x, y = np.rint(xy[i]).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 1, tuple(int(c) for c in colors[i, ::-1]), -1)
    cv2.putText(canvas, f'PCA view {panel+1}', (panel*600+20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (30,30,30), 2)
cv2.imencode('.png', canvas)[1].tofile(str(root / 'dense_preview.png'))
save_json(root / 'reconstruction_status.json', {
    'input_images': 10, 'registered_images': 7, 'submodels': 1,
    'dense_complete': True, 'dense_points': len(points),
    'mesh_vertices': len(vertices), 'mesh_faces': len(faces),
    'coordinate_system': 'SfM object frame, arbitrary scale',
    'colmap': '4.2.1 CUDA',
    'meshing': {'depth': 10, 'trim': 0},
    'notes': 'Default trim=10 produced an empty mesh. Untrimmed Poisson surface includes extrapolated regions; dense.ply contains measured foreground points. Seven of ten views registered. No physical scale calibration.'
})
print('Foreground points:', len(points), 'mesh vertices:', len(vertices), 'faces:', len(faces))
