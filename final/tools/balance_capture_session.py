"""Apply saved white-balance gains to existing exports, preserving measurement originals."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.capture import apply_white_balance
from tools.capture_structured_light import save_image, save_json


def balance_group(group, fallback_gains):
    path = group/'group.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data.get('complete') or len(data['frames'])!=data['expected_images']:
        raise ValueError(f'Group is not complete: {group}')
    gains = np.asarray(data.get('white_balance_bgr',fallback_gains),float)
    if gains.shape!=(3,) or not np.isfinite(gains).all() or np.any(gains<=0):
        raise ValueError(f'Invalid gains: {group}')
    backup = group/'group_before_white_balance.json'
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(path.read_bytes())
    changed = 0
    for frame in data['frames']:
        name = frame['file']
        if Path(name).name!=name:
            raise ValueError(f'Expected a filename within group: {name}')
        color = frame['pattern'].get('purpose')=='color'
        if frame.get('software_white_balance_applied'):
            # Existing color exports must not be balanced a second time.
            if not color and not frame.get('raw_file'):
                raise ValueError(f'Balanced measurement has no raw backup: {name}')
            continue
        image_path = group/name
        raw_file = 'raw/'+name
        raw_path = group/raw_file
        raw_path.parent.mkdir(exist_ok=True)
        if not raw_path.exists():
            with raw_path.open('xb') as stream:
                stream.write(image_path.read_bytes())
        image = cv2.imdecode(np.fromfile(str(raw_path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read {raw_path}')
        result = apply_white_balance(image,gains)
        temporary = group/(name+'.'+uuid.uuid4().hex+'.tmp')
        save_image(temporary,result)
        temporary.replace(image_path)
        frame['software_white_balance_applied'] = True
        frame['raw_file'] = raw_file
        save_json(path,data)
        changed += 1
    data['measurement_image_source'] = 'raw/; root PNGs are white-balanced viewing/color exports'
    save_json(path,data)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    args = parser.parse_args()
    profile_path = args.session/'white_balance.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    total = 0
    for group in sorted(args.session.glob('group_*')):
        if group.is_dir() and (group/'group.json').exists():
            count = balance_group(group,profile['gains_bgr'])
            total += count
            print(group.name,'corrected',count,flush=True)
    profile['applied_to'] = 'all root PNG exports and previews; raw/ preserves unbalanced measurement patterns and references'
    save_json(profile_path,profile)
    print('Corrected',total,'images; measurement originals retained in each group/raw/',flush=True)


if __name__=='__main__':
    main()
