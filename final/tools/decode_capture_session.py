"""Audit and decode a captured Gray-code session into projector-coordinate maps."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.io import save_json
from scanner.structured_decode import decode_graycode


def read_image(path, flag=cv2.IMREAD_GRAYSCALE):
    image = cv2.imdecode(np.fromfile(str(path),np.uint8),flag)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def save_image(path, image):
    path = Path(path)
    ok,encoded = cv2.imencode(path.suffix,image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    encoded.tofile(str(path))


def process(session, output, bit_threshold=3., white_threshold=20., relative_threshold=.02, limit=0):
    session,output = Path(session).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing captures and decodes are not overwritten')
    config = json.loads((session/'session.json').read_text(encoding='utf-8'))
    groups = sorted(p for p in session.glob('group_*') if p.is_dir())
    if limit:
        groups = groups[:limit]
    if not groups:
        raise ValueError('No capture groups')
    output.mkdir(parents=True)
    (output/'color_images').mkdir()
    rows,tiles = [],[]
    common_camera = None
    common_roi = None
    for group in groups:
        metadata = json.loads((group/'group.json').read_text(encoding='utf-8'))
        frames = metadata['frames']
        if not metadata.get('complete') or metadata.get('saved_images')!=metadata['expected_images'] or len(frames)!=metadata['expected_images']:
            raise ValueError(f'Incomplete capture group: {group.name}')
        if metadata['projector']!=config['projector']:
            raise ValueError(f'Projector configuration changed in {group.name}')
        if [f['pattern'] for f in frames]!=config['frames']:
            raise ValueError(f'Pattern sequence changed in {group.name}')
        if common_camera is None:
            common_camera = metadata['camera']; common_roi = metadata['crop_roi_xywh']
        if metadata['camera']!=common_camera or metadata['crop_roi_xywh']!=common_roi:
            raise ValueError(f'Camera settings/crop changed in {group.name}')
        lookup = {f['pattern']['name']:f for f in frames}
        raw_paths = {}
        for frame in frames:
            if frame['pattern'].get('purpose')=='color':
                continue
            name = frame.get('raw_file')
            if not name:
                raise ValueError(f'Measurement raw path missing: {group.name}/{frame["file"]}')
            path = (group/name).resolve()
            if not path.is_relative_to(group.resolve()) or not path.is_file():
                raise ValueError(f'Invalid raw path: {path}')
            raw_paths[frame['pattern']['name']] = path
        projector = config['projector']
        def reader(pattern):
            image = read_image(raw_paths[pattern['name']])
            if list(image.shape[::-1])!=metadata['saved_image_size']:
                raise ValueError(f'Image/crop size mismatch: {group.name}')
            return image
        decoded = decode_graycode([f['pattern'] for f in frames],reader,
                    (projector['width'],projector['height']),config.get('cell_size',1),
                    bit_threshold,white_threshold,relative_threshold)
        folder = output/group.name; folder.mkdir()
        np.savez_compressed(folder/'decoded.npz',**{k:v for k,v in decoded.items() if k!='bit_statistics'})
        valid = decoded['valid']
        save_image(folder/'valid_mask.png',valid.astype(np.uint8)*255)
        color_frame = next(f for f in frames if f['pattern'].get('purpose')=='color')
        color = read_image(group/color_frame['file'],cv2.IMREAD_COLOR)
        shutil.copyfile(group/color_frame['file'],output/'color_images'/(group.name+'.png'))
        panels = [color]
        for axis,size in [('x',projector['width']),('y',projector['height'])]:
            value = np.clip(decoded['projector_'+axis]/max(1,size-1)*255,0,255).astype(np.uint8)
            panel = cv2.applyColorMap(value,cv2.COLORMAP_TURBO)
            panel[~valid] = 0
            panels.append(panel)
        preview = np.hstack([cv2.resize(p,(420,428),interpolation=cv2.INTER_AREA) for p in panels])
        for index,label in enumerate(['White-light color','Projector X (not depth)','Projector Y (not depth)']):
            cv2.putText(preview,label,(index*420+10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        save_image(folder/'decode_preview.jpg',preview)
        confidence = cv2.applyColorMap(np.clip(decoded['minimum_pair_contrast']*4,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
        save_image(folder/'confidence.png',confidence)
        report = {'group':group.name,'valid_pixels':int(valid.sum()),'image_pixels':int(valid.size),
            'valid_fraction_full_crop':float(valid.mean()),
            'median_min_pair_contrast_valid':float(np.median(decoded['minimum_pair_contrast'][valid])) if valid.any() else 0.,
            'median_complement_error_valid':float(np.median(decoded['maximum_complement_error'][valid])) if valid.any() else 0.,
            'bit_threshold':bit_threshold,'white_threshold':white_threshold,'relative_threshold':relative_threshold,
            'note':'Projector correspondences, not depth. Coverage includes background until a foreground mask is applied.'}
        save_json(folder/'quality.json',report)
        with (folder/'bit_quality.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(decoded['bit_statistics'][0]))
            writer.writeheader(); writer.writerows(decoded['bit_statistics'])
        rows.append(report)
        overlay = color.copy()
        overlay[~valid] = (overlay[~valid]*.2).astype(np.uint8)
        tile = cv2.resize(overlay,(252,257),interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile,22,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
        cv2.putText(tile,group.name,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        tiles.append(tile)
        print(group.name,'valid',round(valid.mean()*100,2),'%',flush=True)
    tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%6)])
    save_image(output/'decode_contact.jpg',np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
    save_json(output/'decode_summary.json',{'source':str(session),'groups':rows,'camera':common_camera,
        'bit_threshold':bit_threshold,'white_threshold':white_threshold,'relative_threshold':relative_threshold,
        'projector':config['projector'],'crop_roi_xywh':common_roi,
        'median_valid_fraction_full_crop':float(np.median([r['valid_fraction_full_crop'] for r in rows])),
        'calibrated_depth_complete':False,
        'required_for_depth':'Confirmed camera/projector K, distortion, camera-to-projector R/t; no arbitrary historical calibration is assumed.'})
    print('Decoded groups:',len(rows),'Output:',output,flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bit-threshold',type=float,default=3.)
    p.add_argument('--white-threshold',type=float,default=20.)
    p.add_argument('--relative-threshold',type=float,default=.02)
    p.add_argument('--limit',type=int,default=0)
    a = p.parse_args()
    process(a.session,a.output,a.bit_threshold,a.white_threshold,a.relative_threshold,a.limit)


if __name__=='__main__':
    main()
