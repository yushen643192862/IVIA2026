"""Export calibrated camera poses and chessboard as colored PLY point clouds."""
import argparse
import json
from pathlib import Path

import numpy as np


def write_ply(path, points, colors):
    with path.open('w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.calibration.read_text(encoding='utf-8'))
    output = args.output or args.calibration.parent
    output.mkdir(parents=True, exist_ok=True)
    cols, rows = data['board_inner_corners']
    s = data['square_size']
    # First inner corner is (0,0); include the surrounding row/column of squares.
    x, y = np.meshgrid(np.linspace(-s, cols*s, (cols+1)*24+1),
                       np.linspace(-s, rows*s, (rows+1)*24+1))
    board = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    dark = ((np.floor(x.ravel()/s).astype(int) + np.floor(y.ravel()/s).astype(int)) % 2) == 0
    board_colors = np.repeat(np.where(dark, 45, 230)[:, None], 3, axis=1)
    centers = np.array([v['camera_center_in_board_coordinates'] for v in data['views']])
    center_colors = np.tile([255, 40, 40], (len(centers), 1))
    points, colors = [board, centers], [board_colors, center_colors]
    K = np.array(data['camera_matrix'])
    w, h = data['image_size_wh']
    uv = np.array([[0,0,1], [w-1,0,1], [w-1,h-1,1], [0,h-1,1]])
    # Short pinhole frusta indicate camera orientation, not a reconstructed surface.
    rays = (np.linalg.inv(K) @ uv.T).T * (2*s)
    table = ['index,file,x,y,z']
    for i, (v, center) in enumerate(zip(data['views'], centers), 1):
        R = np.array(v['rotation_matrix'])
        frame = center + rays @ R
        tint = np.array([40, 130, 255])
        for a,b in [(center,p) for p in frame] + [(frame[j],frame[(j+1)%4]) for j in range(4)]:
            line = np.linspace(a,b,100)
            points.append(line)
            colors.append(np.tile(tint,(len(line),1)))
        # A small red sphere makes each camera center visible in point-cloud viewers.
        theta, phi = np.meshgrid(np.linspace(0,2*np.pi,24),np.linspace(0,np.pi,12))
        sphere = center + .12*s*np.column_stack((
            (np.sin(phi)*np.cos(theta)).ravel(),
            (np.sin(phi)*np.sin(theta)).ravel(),np.cos(phi).ravel()))
        points.append(sphere)
        colors.append(np.tile([255,40,40],(len(sphere),1)))
        table.append(f'{i},{v["file"]},{center[0]},{center[1]},{center[2]}')
    write_ply(output/'calibration_scene.ply',np.vstack(points),np.vstack(colors))
    write_ply(output/'camera_centers.ply',centers,center_colors)
    write_ply(output/'chessboard.ply',board,board_colors)
    (output/'camera_centers.csv').write_text('\n'.join(table),encoding='utf-8-sig')
    print(f'Exported {len(centers)} camera centers and chessboard to {output.resolve()}')


if __name__ == '__main__':
    main()
