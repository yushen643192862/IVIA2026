"""Segment a green-screen capture from image colors, without a neural model.

Outputs full-resolution masks and review previews in separate directories.
The largest centrally located non-screen component is treated as the subject.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def segment(image):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.float32))
    # Sample unambiguous screen pixels from the outer fifth of the photograph.
    rim = np.zeros((height,width),np.uint8)
    strip = max(20,width//5)
    rim[:, :strip] = 1
    rim[:, -strip:] = 1
    chroma = (g>r*1.22)&(g>b*1.10)&(hsv[:,:,1]>45)&(hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)
    sample = chroma & (rim>0)
    if sample.sum()<height*width*.02:
        raise ValueError('Too few green-screen samples at image sides')
    hue = float(np.median(hsv[:,:,0][sample]))
    saturation = float(np.median(hsv[:,:,1][sample]))
    value = float(np.median(hsv[:,:,2][sample]))
    # Hue alone overlaps turquoise hair/cap edges. A real green screen has
    # substantially more green than BOTH red and blue in this capture.
    green_over_red = max(2.2,float(np.percentile(g[sample]/np.maximum(r[sample],1),10))*.78)
    green_over_blue = max(1.42,float(np.percentile(g[sample]/np.maximum(b[sample],1),10))*.94)
    # Strong green pixels count as background even inside closed hair/tail gaps.
    screen = (np.abs(hsv[:,:,0].astype(np.float32)-hue)<=16)&(hsv[:,:,1]>max(42,saturation*.42))
    screen &= (g>r*green_over_red)&(g>b*green_over_blue)
    screen &= hsv[:,:,2]>max(35,value*.32)
    candidate = (~screen).astype(np.uint8)
    # Remove isolated salt noise before deciding which component is the figure.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate,8)
    central = (centroids[:,0]>width*.2)&(centroids[:,0]<width*.8)
    central &= (centroids[:,1]>height*.05)&(centroids[:,1]<height*.85)
    central[0] = False
    eligible = np.flatnonzero(central & (stats[:,cv2.CC_STAT_AREA]>height*width*.001))
    if not len(eligible):
        raise ValueError('Could not locate a central foreground component')
    largest = eligible[np.argmax(stats[eligible,cv2.CC_STAT_AREA])]
    # Nearby disconnected thin pieces (hair, shoes, tail) may be genuine.
    main = labels==largest
    x,y,w,h,_ = stats[largest]
    pad = max(12,int(max(width,height)*.025))
    left, top = max(0,x-pad), max(0,y-pad)
    right, bottom = min(width,x+w+pad), min(height,y+h+pad)
    # Bounding boxes select thin nearby components in one pass. A per-label
    # full-resolution scan would be much too slow on 12-megapixel photographs.
    component_ids = np.arange(1,count)
    boxes = stats[1:]
    near = (boxes[:,cv2.CC_STAT_LEFT]>=left)&(boxes[:,cv2.CC_STAT_TOP]>=top)
    near &= (boxes[:,cv2.CC_STAT_LEFT]+boxes[:,cv2.CC_STAT_WIDTH]<=right)
    near &= (boxes[:,cv2.CC_STAT_TOP]+boxes[:,cv2.CC_STAT_HEIGHT]<=bottom)
    near &= boxes[:,cv2.CC_STAT_AREA]>=24
    selected = component_ids[near]
    keep = np.isin(labels,selected) | main
    mask = keep.astype(np.uint8)*255
    # The blue turntable is a large, separate color patch below the figure.
    # Locate it from color and area, then exclude its blue surface and the
    # stationary apparatus below it. Keep the shoes immediately above/at rim.
    yy = np.arange(height)[:,None]
    blue = ((hsv[:,:,0]>=88)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>60)
            &(hsv[:,:,2]>75)&(yy>height*.40)).astype(np.uint8)
    nblue, blue_labels, blue_stats, _ = cv2.connectedComponentsWithStats(blue,8)
    base = None
    if nblue>1:
        ids = np.flatnonzero(blue_stats[:,cv2.CC_STAT_AREA]>height*width*.002)
        ids = ids[ids!=0]
        if len(ids):
            base = ids[np.argmax(blue_stats[ids,cv2.CC_STAT_AREA])]
            base_top = int(blue_stats[base,cv2.CC_STAT_TOP])
            mask[(blue_labels==base)] = 0
            cutoff = base_top+int(height*.04)
            # In elevated views the figure extends below the visible rim.
            # Keep pixels near warm, lightly saturated limb colors instead
            # of cutting off everything below the blue disk.
            warm = (mask>0)&(hsv[:,:,1]<110)&(hsv[:,:,2]>100)
            warm &= (r>g*.85)&(r>b*1.05)
            disk_ratio = float(blue_stats[base,cv2.CC_STAT_WIDTH]/blue_stats[base,cv2.CC_STAT_HEIGHT])
            if disk_ratio<2.0 and np.count_nonzero(warm[cutoff:])>1500:
                vicinity = cv2.distanceTransform((~warm).astype(np.uint8),cv2.DIST_L2,3)<85
                mask[cutoff:] *= vicinity[cutoff:]
            else:
                mask[cutoff:] = 0
    return mask, {'screen_hue':hue,'screen_saturation':saturation,'screen_value':value,
                  'green_over_red':green_over_red,'green_over_blue':green_over_blue,
                  'screen_fraction':float(screen.mean()),'foreground_fraction':float((mask>0).mean()),
                  'selected_component_area':int(stats[largest,cv2.CC_STAT_AREA]),
                  'base_detected':base is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images',type=Path)
    p.add_argument('masks',type=Path)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    args = p.parse_args()
    if args.masks.resolve()==args.images.resolve():
        p.error('Masks must be separate from input photographs')
    args.masks.mkdir(parents=True,exist_ok=True)
    args.review.mkdir(parents=True,exist_ok=True)
    paths = sorted(args.images.glob('*.jpg'))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        p.error('No JPG photographs found')
    tiles, records = [], []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read {path}')
        mask, report = segment(image)
        cv2.imencode('.png',mask)[1].tofile(str(args.masks/(path.name+'.png')))
        small = cv2.resize(image,(240,320),interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask,(240,320),interpolation=cv2.INTER_NEAREST)>0
        small[~small_mask] = 0
        tile = cv2.copyMakeBorder(small,22,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
        cv2.putText(tile,path.stem[-6:],(6,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        tiles.append(tile)
        records.append({'image':path.name,**report})
        print(path.name,'foreground',round(report['foreground_fraction'],4),flush=True)
    tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%7)])
    cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+7]) for i in range(0,len(tiles),7)]))[1].tofile(str(args.review/'green_masks_contact.jpg'))
    (args.review/'green_masks_report.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Saved',len(records),'full-resolution masks',flush=True)


if __name__=='__main__':
    main()
