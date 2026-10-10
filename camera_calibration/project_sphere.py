"""Project a shaded sphere tangent to the calibrated chessboard onto each image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def export_sphere_ply(output, data, radius):
    """Save a triangle mesh in the first view's chessboard coordinate system."""
    cols, rows = data['board_inner_corners']
    s = data['square_size']
    side = 1 if data['views'][0]['camera_center_in_board_coordinates'][2] > 0 else -1
    center = np.array([(cols-1)*s/2, (rows-1)*s/2, side*radius])
    segments, rings = 80, 48
    points = [center + [0, 0, radius]]
    for j in range(1, rings):
        phi = np.pi*j/rings
        for i in range(segments):
            theta = 2*np.pi*i/segments
            points.append(center + radius*np.array([
                np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]))
    bottom = len(points)
    points.append(center + [0, 0, -radius])
    faces = []
    for i in range(segments):
        nxt = (i+1) % segments
        faces.append([0, 1+i, 1+nxt])
        for j in range(rings-2):
            a, b = 1+j*segments+i, 1+j*segments+nxt
            faces.extend([[a, a+segments, b], [b, a+segments, b+segments]])
        a = 1+(rings-2)*segments
        faces.append([a+i, bottom, a+nxt])
    vertices = np.array(points)
    faces = np.array(faces)
    cross = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                     vertices[faces[:,2]]-vertices[faces[:,0]])
    inward = np.sum(cross*(vertices[faces].mean(axis=1)-center), axis=1) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    colors = np.tile([35, 125, 230], (len(points), 1)).tolist()  # PLY uses RGB.

    def write(name, mesh_points, mesh_colors, mesh_faces):
        with (output/name).open('w', encoding='ascii') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'comment unit {data["translation_unit"]}\n')
            f.write('comment board coordinates of first calibration view\n')
            f.write(f'element vertex {len(mesh_points)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write(f'element face {len(mesh_faces)}\n')
            f.write('property list uchar int vertex_indices\nend_header\n')
            for p, c in zip(mesh_points, mesh_colors):
                f.write(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}\n')
            for a, b, c in mesh_faces:
                f.write(f'3 {a} {b} {c}\n')

    all_faces = faces.tolist()
    # Separate square vertices retain crisp black/white boundaries.
    for y in range(-1, rows):
        for x in range(-1, cols):
            start = len(points)
            points.extend([[x*s,y*s,0], [(x+1)*s,y*s,0],
                           [(x+1)*s,(y+1)*s,0], [x*s,(y+1)*s,0]])
            value = 45 if (x+y) % 2 == 0 else 230
            colors.extend([[value]*3 for _ in range(4)])
            triangles = [[start,start+1,start+2], [start,start+2,start+3]]
            if side < 0:
                triangles = [t[::-1] for t in triangles]
            all_faces.extend(triangles)
    def add_tube(a, b, color, thickness=0.025):
        direction = b-a
        direction /= np.linalg.norm(direction)
        helper = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([0., 1., 0.])
        u = np.cross(direction, helper)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        start = len(points)
        for end in [a,b]:
            for angle in np.linspace(0,2*np.pi,8,endpoint=False):
                points.append(end+thickness*s*(np.cos(angle)*u+np.sin(angle)*v))
                colors.append(color)
        for j in range(8):
            n = (j+1)%8
            all_faces.extend([[start+j,start+n,start+8+j],
                              [start+n,start+8+n,start+8+j]])

    K = np.array(data['camera_matrix'])
    w,h = data['image_size_wh']
    uv = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])
    rays = (np.linalg.inv(K)@uv.T).T*(2*s)
    for view in data['views']:
        R = np.array(view['rotation_matrix'])
        C = np.array(view['camera_center_in_board_coordinates'])
        frame = C+rays@R
        # Red octahedron at each camera center; blue frustum; green optical axis.
        start = len(points)
        points.extend(C+.12*s*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]))
        colors.extend([[255,40,40]]*6)
        for tip in [4,5]:
            for a,b in [(0,2),(2,1),(1,3),(3,0)]:
                all_faces.append([start+tip,start+a,start+b] if tip==4 else [start+tip,start+b,start+a])
        for a,b in [(C,p) for p in frame]+[(frame[j],frame[(j+1)%4]) for j in range(4)]:
            add_tube(a,b,[40,130,255])
        add_tube(C,C+np.array([0.,0.,3*s])@R,[40,210,70],.035)
    write('calibration_scene_with_sphere.ply', points, colors, all_faces)
    print('Saved calibration_scene_with_sphere.ply: sphere, board, camera centers and directions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=0.5, help='Radius in square edge lengths (default: 0.5)')
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error('Radius must be positive')
    data = json.loads(args.calibration.read_text(encoding='utf-8'))
    K = np.array(data['camera_matrix'])
    dist = np.array(data['distortion_coefficients'])
    cols, rows = data['board_inner_corners']
    s = data['square_size']
    radius = args.radius*s
    args.output.mkdir(parents=True, exist_ok=True)
    if not data['views']:
        raise ValueError('Calibration contains no camera poses')
    export_sphere_ply(args.output, data, radius)
    thumbnails = []
    for view in data['views']:
        source = args.images/view['file']
        image = cv2.imdecode(np.fromfile(str(source), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Cannot read {source}')
        if list(image.shape[1::-1]) != data['image_size_wh']:
            raise ValueError(f'Image resolution differs: {source}')
        rv = np.array(view['rvec'], dtype=float)
        tv = np.array(view['translation'], dtype=float)
        R = np.array(view['rotation_matrix'])
        # Choose the side of the board facing the camera, including flipped poses.
        side = 1 if view['camera_center_in_board_coordinates'][2] > 0 else -1
        center = np.array([(cols-1)*s/2, (rows-1)*s/2, side*radius])
        longitude = np.linspace(0, 2*np.pi, 81)
        latitude = np.linspace(0, np.pi, 49)
        theta, phi = np.meshgrid(longitude, latitude)
        normals = np.column_stack(((np.sin(phi)*np.cos(theta)).ravel(),
                                   (np.sin(phi)*np.sin(theta)).ravel(), np.cos(phi).ravel()))
        vertices = center+radius*normals
        faces = []
        for j in range(48):
            for i in range(80):
                a = j*81+i
                faces.extend([(a,a+1,a+81), (a+1,a+82,a+81)])
        faces = np.array(faces)
        camera_vertices = vertices@R.T+tv
        if np.any(camera_vertices[:,2] <= 0):
            raise ValueError(f'Sphere crosses camera plane: {source}')
        projected, _ = cv2.projectPoints(vertices, rv, tv, K, dist)
        projected = projected.reshape(-1,2)
        depth = camera_vertices[faces,2].mean(axis=1)
        face_normal = normals[faces].mean(axis=1)@R.T
        face_normal /= np.maximum(np.linalg.norm(face_normal,axis=1,keepdims=True),1e-12)
        face_position = camera_vertices[faces].mean(axis=1)
        eye = -face_position/np.linalg.norm(face_position,axis=1,keepdims=True)
        visible = np.sum(face_normal*eye,axis=1)>0
        light = np.array([-.4,-.6,-1.0]); light /= np.linalg.norm(light)
        diffuse = np.maximum(face_normal@light,0)
        halfway = eye+light
        halfway /= np.maximum(np.linalg.norm(halfway,axis=1,keepdims=True),1e-12)
        specular = np.maximum(np.sum(face_normal*halfway,axis=1),0)**40
        base_bgr = np.array([230.,125.,35.])
        shading = np.clip(base_bgr[None,:]*(.22+.78*diffuse[:,None])+110*specular[:,None],0,255)
        for fi in np.argsort(depth)[::-1]:
            if visible[fi]:
                polygon = np.round(projected[faces[fi]]).astype(np.int32)
                cv2.fillConvexPoly(image,polygon,tuple(float(x) for x in shading[fi]),lineType=cv2.LINE_AA)
        ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        encoded.tofile(str(args.output/(source.stem+'_sphere.jpg')))
        thumb = cv2.resize(image,(480,360))
        cv2.putText(thumb,source.stem[-8:],(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        thumbnails.append(thumb)
    sheet = np.full(((len(thumbnails)+3)//4*360,4*480,3),235,np.uint8)
    for i,thumb in enumerate(thumbnails):
        sheet[i//4*360:(i//4+1)*360,i%4*480:(i%4+1)*480] = thumb
    cv2.imencode('.jpg',sheet)[1].tofile(str(args.output/'sphere_gallery.jpg'))
    print(f'Projected sphere into {len(thumbnails)} images: {args.output.resolve()}')


if __name__ == '__main__':
    main()
