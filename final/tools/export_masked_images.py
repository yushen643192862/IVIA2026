"""Apply existing binary masks and save full-resolution, black-background PNGs."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def save_masked_image(image, mask, output):
    output = Path(output)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError(f'Invalid image or mask dimensions: {output.name}')
    result = image.copy()
    result[mask == 0] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', result)
    if not ok:
        raise RuntimeError(f'Could not encode {output}')
    encoded.tofile(str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    images, masks, output = (x.resolve() for x in (args.images, args.masks, args.output))
    if output == images or output.is_relative_to(images) or output == masks or output.is_relative_to(masks):
        p.error('Output must be separate from input images and masks')
    paths = sorted(x for x in images.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
    if not paths:
        p.error('No input images found')
    for path in paths:
        relative = path.relative_to(images)
        mask_path = masks/(str(relative)+'.png')
        if not mask_path.is_file():
            raise FileNotFoundError(f'Missing mask: {mask_path}')
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_path),np.uint8),cv2.IMREAD_GRAYSCALE)
        save_masked_image(image,mask,output/(str(relative)+'.png'))
    print(f'Saved {len(paths)} masked images: {output}',flush=True)


if __name__ == '__main__':
    main()
