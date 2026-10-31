"""Remove low-saturation table visible below this figure's shorts.

This is specific to Camera(1), whose skin is warmer than the gray table and
whose blue pedestal supplies a per-image lower limit. Preserve original masks.
"""
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
target = root / 'Camera1_masks_refined'
target.mkdir(exist_ok=True)
tiles = []
for index, path in enumerate(sorted((root / 'Camera(1)').glob('*.jpg'))):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
    mask = cv2.imdecode(np.fromfile(str(root / 'Camera1_masks' / (path.name+'.png')), np.uint8), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yy = np.arange(image.shape[0])[:, None]
    blue = (hsv[:, :, 0] > 80) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 65) & (yy > 2850) & (mask > 0)
    rows = np.flatnonzero(blue.sum(axis=1) > 50)
    base_top = int(rows[0]) if len(rows) else 3150
    # Only remove pixels between two detected leg segments. Applying a global
    # saturation cutoff would also erase shaded skin and highlights.
    skin = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 27)
            & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] > 70) & (mask > 0))
    for row in range(2300, base_top):
        line = skin[row].astype(np.uint8)
        edges = np.diff(np.r_[0, line, 0].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        segments = [(a, b) for a, b in zip(starts, ends) if b-a >= 14 and a > 850 and b < 2350]
        if len(segments) >= 2:
            left, right = segments[0], segments[-1]
            if 15 < right[0]-left[1] < 650:
                gap = mask[row, left[1]:right[0]]
                gap[hsv[row, left[1]:right[0], 1] < 43] = 0
    cv2.imencode('.png', mask)[1].tofile(str(target / (path.name+'.png')))
    if index % 4 == 0:
        image[mask == 0] = (225, 225, 225)
        tiles.append(cv2.resize(image, (180, 240)))
cv2.imencode('.jpg', np.hstack(tiles))[1].tofile(str(root/'outputs/camera1_scan_cuda/refined_masks_preview.jpg'))
