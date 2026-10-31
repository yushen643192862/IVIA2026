"""Reviewable green-screen cleanup after model masks; retain opaque blue base."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from export_masked_images import save_masked_image

p = argparse.ArgumentParser()
p.add_argument('images', type=Path)
p.add_argument('initial', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--review', type=Path, required=True)
p.add_argument('--subject-only', action='store_true')
p.add_argument('--elevated-from', help='First elevated-view filename stem in this capture (optional)')
p.add_argument('--masked-images', type=Path, help='Full-resolution black-background PNG directory')
a = p.parse_args()
masked_images = a.masked_images or a.output.with_name(a.output.name+'_images')
a.output.mkdir(parents=True, exist_ok=True)
a.review.mkdir(parents=True, exist_ok=True)
tiles, records = [], []
for path in sorted(a.images.glob('*.jpg')):
    full = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
    h, w = full.shape[:2]
    im = cv2.resize(full, (900, 1200))
    mask = cv2.imdecode(np.fromfile(str(a.initial / (path.name+'.png')), np.uint8), 0)
    mask = cv2.resize(mask, (900, 1200), interpolation=cv2.INTER_NEAREST)>0
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(im.astype(float))
    green = (hsv[:,:,0]>35)&(hsv[:,:,0]<88)&(hsv[:,:,1]>65)&(g>r*1.2)&(g>b*1.12)
    if not a.subject_only:
        mask &= ~green
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])
    subject = labels == largest
    ys, xs = np.where(subject)
    yy, xx = np.indices(mask.shape)
    # Search only close to the feet, so blue hair and the cap cannot be a base.
    blue = ((hsv[:,:,0]>=88)&(hsv[:,:,0]<125)&(hsv[:,:,1]>55)
            &(hsv[:,:,2]>90)&(yy>ys.max()-65)&(yy<ys.max()+150)
            &(xx>xs.min()-100)&(xx<xs.max()+100)).astype(np.uint8)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base = np.zeros_like(blue)
    area = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area>200:
            cv2.drawContours(base, [contour], -1, 1, cv2.FILLED)
    result = (mask | (base>0)).astype(np.uint8)*255
    if a.subject_only:
        # Cyan pedestal is only removed below the torso, preserving blue hair.
        lower_blue = (hsv[:,:,0]>=85)&(hsv[:,:,0]<125)&(hsv[:,:,1]>55)&(yy>ys.min()+.7*(ys.max()-ys.min()))
        candidate = mask & ~lower_blue
        # Elevated views include the pedestal in the model prediction. Key
        # chromatic pedestal pixels below the hips while preserving warm skin.
        elevated = bool(a.elevated_from and path.stem >= a.elevated_from)
        if elevated:
            low = yy>ys.min()+.48*(ys.max()-ys.min())
            chroma_base = (hsv[:,:,0]>35)&(hsv[:,:,0]<125)&(hsv[:,:,1]>35)
            candidate = mask & ~(low & chroma_base)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
        keep = np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>40)
        keep = keep[keep!=0]
        if elevated:
            main = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
            distance = cv2.distanceTransform((labels!=main).astype(np.uint8),cv2.DIST_L2,3)
            keep = [k for k in keep if k==main or distance[labels==k].min()<5]
        result = np.isin(labels,keep).astype(np.uint8)*255
    full_mask = cv2.resize(result,(w,h),interpolation=cv2.INTER_NEAREST)
    cv2.imencode('.png',full_mask)[1].tofile(str(a.output/(path.name+'.png')))
    save_masked_image(full,full_mask,masked_images/(path.name+'.png'))
    preview = im.copy()
    preview[result==0] = (220,220,220)
    tile = cv2.copyMakeBorder(cv2.resize(preview,(225,300)),24,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.putText(tile,path.stem[-6:],(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    tiles.append(tile)
    records.append({'image':path.name,'foreground_fraction':float((result>0).mean()),'base_area_at_900x1200':area})
tiles.extend([np.full_like(tiles[0],245) for _ in range((-len(tiles))%7)])
cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+7]) for i in range(0,len(tiles),7)]))[1].tofile(str(a.review/'refined_masks_contact.jpg'))
(a.review/'mask_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Masks:',len(records),'Missing base:',sum(r['base_area_at_900x1200']<=200 for r in records))
print('Masked images:',masked_images)
