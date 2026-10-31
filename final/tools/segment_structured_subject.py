"""Direct green-color subtraction, including internal gaps and green base decorations."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.decode_capture_session import read_image,save_image
from scanner.io import save_json


def segment(color):
    hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(color.astype(float))
    # OpenCV hue uses 0..179. Include dim and weakly saturated green edges.
    green = (hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=25)
    green &= (g>r*1.08)&(g>b*1.03)
    mask = (~green).astype(np.uint8)*255
    return mask,{'method':'direct_green_subtraction','foreground_fraction':float((mask>0).mean()),
        'green_fraction':float(green.mean()),'hue_range_opencv':[35,90],
        'minimum_saturation':25,'green_over_red':1.08,'green_over_blue':1.03,
        'note':'All non-green pixels retained, including non-green background/table. No component filtering, erosion or hole filling. Green grass is removed intentionally.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('color_images',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    masks = args.output/'masks'; masks.mkdir()
    masked = args.output/'masked_images'; masked.mkdir()
    tiles,records = [],[]
    for path in sorted(args.color_images.glob('group_*.png')):
        color = read_image(path,cv2.IMREAD_COLOR)
        mask,report = segment(color)
        save_image(masks/path.name,mask)
        black = color.copy(); black[mask==0] = 0
        save_image(masked/path.name,black)
        preview = color.copy(); preview[mask==0] = (230,230,230)
        tile = cv2.resize(preview,(220,224))
        tile = cv2.copyMakeBorder(tile,24,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(tile,path.stem,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        tiles.append(tile); records.append({'group':path.stem,**report})
        print(path.stem,report,flush=True)
    if not tiles:
        raise ValueError('No color frames')
    tiles += [np.full_like(tiles[0],255) for _ in range((-len(tiles))%6)]
    save_image(args.output/'masks_contact.jpg',np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
    save_json(args.output/'report.json',records)


if __name__=='__main__':
    main()
