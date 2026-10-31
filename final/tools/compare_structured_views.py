"""Export white-light photos alongside actual per-view PLY point-cloud renders."""
import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import read_ply,save_json
from tools.decode_capture_session import read_image,save_image


def splat(xy,depth,colors,width,height):
    canvas = np.full((height,width,3),32,np.uint8)
    xy = np.rint(xy).astype(int)
    valid = np.isfinite(depth)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(depth[ids])]
    flat = xy[ids,1]*width+xy[ids,0]
    _,first = np.unique(flat,return_index=True)
    ids = ids[first]
    canvas[xy[ids,1],xy[ids,0]] = colors[ids]
    return canvas


def render(points,bgr,calibration,crop,panel_width=480):
    x,y,w,h = crop
    panel_height = round(h*panel_width/w)
    uv = cv2.projectPoints(points,np.zeros(3),np.zeros(3),
        np.asarray(calibration['camera_matrix']),np.asarray(calibration['camera_distortion']))[0].reshape(-1,2)
    uv = (uv-[x,y])*panel_width/w
    front = splat(uv,points[:,2],bgr,panel_width,panel_height)
    center = np.median(points,axis=0)
    angle = np.radians(35.)
    rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    rotated = (points-center)@rotation.T
    low,high = rotated[:,:2].min(axis=0),rotated[:,:2].max(axis=0)
    scale = min((panel_width-30)/max(high[0]-low[0],1e-6),(panel_height-30)/max(high[1]-low[1],1e-6))
    xy = (rotated[:,:2]-(low+high)/2)*scale+[panel_width/2,panel_height/2]
    oblique = splat(xy,rotated[:,2],bgr,panel_width,panel_height)
    return front,oblique


def process(depth_dir,output):
    depth_dir,output = Path(depth_dir),Path(output)
    summary = json.loads((depth_dir/'depth_summary.json').read_text(encoding='utf-8'))
    decoded = Path(summary['source'])
    calibration = json.loads((depth_dir/'calibration_used.json').read_text(encoding='utf-8'))
    decode_summary = json.loads((decoded/'decode_summary.json').read_text(encoding='utf-8'))
    crop = decode_summary['crop_roi_xywh']
    output.mkdir(parents=True,exist_ok=False)
    tiles,records,links = [],[],[]
    for record in summary['groups']:
        name = record['group']
        photo_path = decoded/'color_images'/(name+'.png')
        cloud_path = depth_dir/name/'cloud.ply'
        photo = read_image(photo_path,cv2.IMREAD_COLOR)
        points,_,attrs = read_ply(cloud_path,with_attributes=True)
        if not len(points):
            raise ValueError(f'Empty cloud: {name}')
        bgr = np.column_stack([attrs[c] for c in ('blue','green','red')]).astype(np.uint8)
        front,oblique = render(points,bgr,calibration,crop)
        original = cv2.resize(photo,(front.shape[1],front.shape[0]),interpolation=cv2.INTER_AREA)
        panels = []
        for image,label in [(original,'White-light photo'),(front,'Point cloud: camera view'),(oblique,'Point cloud: rotated 35 deg')]:
            image = cv2.copyMakeBorder(image,40,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
            cv2.putText(image,label,(10,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,20,20),1,cv2.LINE_AA)
            panels.append(image)
        strip = np.hstack(panels)
        strip = cv2.copyMakeBorder(strip,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(strip,f'{name} | {len(points):,} measured points | no surface filling',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1,cv2.LINE_AA)
        save_image(output/(name+'.jpg'),strip)
        tiles.append(cv2.resize(strip,(480,round(strip.shape[0]/3)),interpolation=cv2.INTER_AREA))
        links.append(f'<figure><a href="{name}.jpg"><img loading="lazy" src="{name}.jpg" alt="{name}"></a><figcaption>{name} · {len(points):,} points · <a href="{html.escape(cloud_path.resolve().as_uri())}">cloud.ply</a></figcaption></figure>')
        records.append({'group':name,'photo':str(photo_path.resolve()),'cloud':str(cloud_path.resolve()),'points':len(points),'comparison':name+'.jpg'})
        print(name,len(points),flush=True)
    if not tiles:
        raise ValueError('No views')
    tiles += [np.full_like(tiles[0],255) for _ in range((-len(tiles))%3)]
    save_image(output/'all_groups_contact.jpg',np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    page = '<!doctype html><meta charset="utf-8"><title>照片与点云对比</title><style>body{font:16px sans-serif;margin:24px;background:#eee;color:#222}figure{margin:20px 0;background:white;padding:10px}img{width:100%;height:auto}figcaption{padding:8px}</style><h1>31组照片与点云对比</h1><p>从左到右：原始白光照片、相机视角点云、旋转35°点云。黑色空隙表示没有点；未填孔，未生成曲面。点击图片查看大图。斜视图独立居中缩放。</p>'+''.join(links)
    (output/'index.html').write_text(page,encoding='utf-8')
    save_json(output/'manifest.json',{'groups':records,'point_rendering':'One pixel per projected point; nearest depth wins; no splat dilation or hole filling','empirical_projector_offset_xy_px':summary.get('projector_offset_xy_px')})
    print('Comparisons:',output,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('depth_dir',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    process(args.depth_dir,args.output)


if __name__=='__main__':
    main()
