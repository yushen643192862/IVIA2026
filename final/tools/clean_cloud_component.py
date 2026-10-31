"""Export the largest locally connected point component, retaining raw data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scanner.io import read_ply, write_ply, save_json

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
args = parser.parse_args()
p, _, attributes = read_ply(args.folder / 'dense.ply', with_attributes=True)
data = np.ascontiguousarray(p, dtype=np.float32)
index = cv2.flann_Index(data, dict(algorithm=1, trees=4))
neighbors, distances = index.knnSearch(data, 12, params={'checks':64})
robust_extent = np.linalg.norm(np.percentile(p, 95, axis=0)-np.percentile(p, 5, axis=0))
radius = min(float(np.median(np.sqrt(distances[:, -1])) * 4), float(robust_extent * .015))
parent = np.arange(len(p))
sizes = np.ones(len(p), dtype=int)
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for i in range(len(p)):
    for j in neighbors[i, distances[i] < radius**2]:
        a, b = find(i), find(int(j))
        if a != b:
            if sizes[a] < sizes[b]: a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
roots = np.array([find(i) for i in range(len(p))])
ids, counts = np.unique(roots, return_counts=True)
keep = roots == ids[counts.argmax()]
if keep.sum() < .3 * len(p):
    raise RuntimeError('Largest component too small for automatic cleanup; inspect raw cloud')
colors = np.column_stack([attributes[k] for k in ('red', 'green', 'blue')])
normals = np.column_stack([attributes[k] for k in ('nx', 'ny', 'nz')])
write_ply(args.folder/'dense_clean.ply', p[keep], colors=colors[keep], normals=normals[keep])
save_json(args.folder/'cleanup.json', {'method':'largest connected component in 12-neighbor graph',
    'radius_sfm_units':radius, 'input_points':len(p), 'retained_points':int(keep.sum()),
    'note':'Disconnected legitimate details can be removed. Original dense.ply is retained for comparison.'})
print('Input:', len(p), 'retained:', keep.sum(), 'radius:', radius)
