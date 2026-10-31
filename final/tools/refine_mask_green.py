"""Remove confidently identified green-screen pixels from existing full-size masks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def refine(image, initial):
    if image is None or initial is None or image.shape[:2] != initial.shape:
        raise ValueError('Image and mask must have the same size')
    h,w = initial.shape
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(image.astype(np.float32))
    rim = np.zeros((h,w),bool)
    rim[:,:w//5] = True
    rim[:,-w//5:] = True
    sample = rim&(initial==0)&(hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)
    sample &= (hsv[:,:,1]>45)&(g>r*1.2)&(g>b*1.1)
    if np.count_nonzero(sample)<h*w*.01:
        raise ValueError('Could not sample enough green screen outside the old mask')
    hue = float(np.median(hsv[:,:,0][sample]))
    saturation = float(np.median(hsv[:,:,1][sample]))
    brightness = float(np.median(hsv[:,:,2][sample]))
    # Require BOTH strong green/red and green/blue contrast. Turquoise hair,
    # cap edges and cool reflections overlap in hue but not in channel ratios.
    red_ratio = max(2.2,float(np.percentile(g[sample]/np.maximum(r[sample],1),10))*.78)
    blue_ratio = max(1.42,float(np.percentile(g[sample]/np.maximum(b[sample],1),10))*.94)
    screen = (np.abs(hsv[:,:,0].astype(np.float32)-hue)<=16)
    screen &= (hsv[:,:,1]>max(42,saturation*.42))
    screen &= (hsv[:,:,2]>max(35,brightness*.32))
    screen &= (g>r*red_ratio)&(g>b*blue_ratio)
    removed = (initial>0)&screen
    result = initial.copy()
    result[removed] = 0
    return result, removed, {'sample_hue':hue,'green_over_red':red_ratio,
                             'green_over_blue':blue_ratio,'removed_pixels':int(removed.sum()),
                             'original_foreground_pixels':int(np.count_nonzero(initial)),
                             'result_foreground_pixels':int(np.count_nonzero(result))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images',type=Path)
    p.add_argument('initial_masks',type=Path)
    p.add_argument('output_masks',type=Path)
    p.add_argument('--review',type=Path,required=True)
    a = p.parse_args()
    if a.output_masks.resolve()==a.initial_masks.resolve():
        p.error('Use a new output directory to preserve the original masks')
    a.output_masks.mkdir(parents=True,exist_ok=True)
    a.review.mkdir(parents=True,exist_ok=True)
    records, tiles = [], []
    paths = sorted(a.images.glob('*.jpg'))
    for path in paths:
        original = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        mask_path = a.initial_masks/(path.name+'.png')
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        initial = cv2.imdecode(np.fromfile(str(mask_path),np.uint8),cv2.IMREAD_GRAYSCALE)
        result,removed,report = refine(original,initial)
        cv2.imencode('.png',result)[1].tofile(str(a.output_masks/(path.name+'.png')))
        view = original.copy(); view[result==0] = 0
        tile = cv2.resize(view,(240,320),interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile,22,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
        cv2.putText(tile,path.stem[-6:],(6,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        tiles.append(tile)
        records.append({'image':path.name,**report})
        print(path.name,'removed',report['removed_pixels'],flush=True)
    if not tiles:
        p.error('No JPG images found')
    tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%7)])
    cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+7]) for i in range(0,len(tiles),7)]))[1].tofile(str(a.review/'refined_masks_contact.jpg'))
    (a.review/'refinement_report.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Saved',len(records),'masks',flush=True)


if __name__=='__main__':
    main()
