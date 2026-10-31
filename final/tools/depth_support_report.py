"""Measure retained depth inside masks remapped to undistorted image pixels.

This measures support/coverage, not reconstruction accuracy or completeness.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scanner.reconstruct import read_model

def read_depth(path):
    with path.open('rb') as stream:
        header=bytearray()
        while header.count(b'&')<3:
            char=stream.read(1)
            if not char: raise ValueError('Truncated depth header')
            header.extend(char)
        w,h,c=map(int,header.decode().strip('&').split('&'))
        return np.frombuffer(stream.read(),np.float32).reshape((w,h,c),order='F').transpose(1,0,2).squeeze()

parser=argparse.ArgumentParser()
parser.add_argument('folder',type=Path)
parser.add_argument('--colmap',type=Path,required=True)
args=parser.parse_args()
root=args.folder.resolve()
model=root/'dense/model_text'
model.mkdir(exist_ok=True)
subprocess.run([str(args.colmap.resolve()),'model_converter','--input_path',str(root/'dense/sparse'),
                '--output_path',str(model),'--output_type','TXT'],check=True)
original={c['file']:c for c in json.loads((root/'cameras.json').read_text(encoding='utf8'))}
rows=[]
for view in read_model(model):
    camera=original[view['file']]
    mask=cv2.imdecode(np.fromfile(camera['object_mask'],np.uint8),0)
    maps=cv2.initUndistortRectifyMap(np.array(camera['K']),np.array(camera['distortion']),
            np.eye(3),np.array(view['K']),tuple(view['size']),cv2.CV_32FC1)
    foreground=cv2.remap(mask,*maps,cv2.INTER_NEAREST,borderValue=0)>0
    row={'image':view['file'],'foreground_pixels':int(foreground.sum())}
    for kind in ('photometric','geometric'):
        depth=read_depth(root/'dense/stereo/depth_maps'/(view['file']+'.'+kind+'.bin'))
        if depth.shape!=foreground.shape: raise ValueError('Depth/image resolution mismatch')
        valid=(depth>0)&np.isfinite(depth)&foreground
        row[kind+'_valid_pixels']=int(valid.sum())
        row[kind+'_valid_fraction']=float(valid.sum()/max(1,foreground.sum()))
    rows.append(row)
with (root/'depth_support.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Views:',len(rows),'median geometric foreground support:',round(float(np.median([r['geometric_valid_fraction'] for r in rows])),3))
