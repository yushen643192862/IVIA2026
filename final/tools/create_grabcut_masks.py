"""Create foreground masks for a fixed-camera turntable sequence.

The rectangle is deliberately wider than the carton. GrabCut refines it using
the background surrounding the object; inspect the generated masks before use
on a different scene.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--processing-size", type=int, default=0,
                        help="Optional maximum edge for segmentation; masks retain original resolution")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rect", nargs=4, type=int, default=(1020, 780, 1100, 3000),
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    cv2.setNumThreads(max(1, args.threads))
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(args.images.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {image_path}")
        x, y, width, height = args.rect
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError(f"Rectangle is outside {image_path}: {args.rect}")
        original_size = (image.shape[1], image.shape[0])
        if args.processing_size > 0 and max(image.shape[:2]) > args.processing_size:
            scale = args.processing_size / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            x, y, width, height = [int(round(v * scale)) for v in (x, y, width, height)]
        labels = np.zeros(image.shape[:2], np.uint8)
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, labels, (x, y, width, height), background, foreground, 8,
                    cv2.GC_INIT_WITH_RECT)
        mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if mask.shape[::-1] != original_size:
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        destination = args.output / f"{image_path.name}.png"
        cv2.imencode(".png", mask)[1].tofile(str(destination))
        print(destination, flush=True)


if __name__ == "__main__":
    main()
