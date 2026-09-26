"""Create an exposure/gain contact sheet from e<exposure>g<gain>.bmp files.

Run: python tupian.py
Optional: python tupian.py --input PATH --output PATH.png --cell-width 320
Dependencies: opencv-python, numpy (same as the acquisition script).
"""
import argparse
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from data_paths import latest_scan

# None selects the newest nonempty scan folder / saves inside that folder.
IMAGE_FOLDER = None
OUTPUT_PATH = None
CELL_WIDTH = 240
FILENAME_PATTERN = re.compile(r"e(\d+)g(\d+)\.bmp", re.IGNORECASE)


def latest_scan_folder():
    return latest_scan(Path(__file__).resolve().parent)


def read_image(path):
    # imread is unreliable for Chinese paths on some OpenCV Windows builds.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path.name}")
    return image


def put_text(canvas, text, x, y, scale=0.6, color=(40, 40, 40)):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def merge_images_adaptive_grid(input_folder, output_path, cell_width=CELL_WIDTH):
    input_folder, output_path = Path(input_folder), Path(output_path)
    if cell_width < 120:
        raise ValueError("cell_width must be at least 120 for readable labels.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output must have a .png extension.")
    grid_data = {}
    for path in sorted(input_folder.iterdir()):
        match = FILENAME_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            key = tuple(map(int, match.groups()))
            if key in grid_data:
                raise ValueError(f"Duplicate exposure/gain pair: {path.name}")
            grid_data[key] = path
    if not grid_data:
        raise ValueError(f"No e<exposure>g<gain>.bmp images in {input_folder}")

    sorted_e = sorted({e for e, g in grid_data})
    sorted_g = sorted({g for e, g in grid_data})
    first_image = None
    for path in grid_data.values():
        try:
            first_image = read_image(path)
            break
        except (OSError, ValueError, cv2.error):
            continue
    if first_image is None:
        raise ValueError("None of the input images could be decoded.")
    height, width = first_image.shape[:2]
    cell_height = max(1, round(cell_width * height / width))
    del first_image

    gap, left, top, footer = 4, 110, 100, 40
    rows, cols = len(sorted_g), len(sorted_e)
    canvas_width = left + cols * (cell_width + gap) + gap
    canvas_height = top + rows * (cell_height + gap) + footer
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    put_text(canvas, "Exposure / Gain comparison", 15, 30, 0.85)
    put_text(canvas, "Columns: exposure (us) | Rows: gain (dB) | Original brightness",
             15, 57, 0.6)
    put_text(canvas, "Gain (dB)", 10, top - 14, 0.55)
    for col, exposure in enumerate(sorted_e):
        x = left + col * (cell_width + gap)
        put_text(canvas, f"e={exposure}", x + 5, top - 14, 0.55)
    for row, gain in enumerate(sorted_g):
        y = top + row * (cell_height + gap)
        put_text(canvas, f"g={gain}", 18, y + cell_height // 2, 0.7)

    total = rows * cols
    success = missing = failed = 0
    print(f"Input: {input_folder}", flush=True)
    print(f"Grid: {rows} gain rows x {cols} exposure columns; "
          f"output: {canvas_width} x {canvas_height}", flush=True)
    for row, gain in enumerate(sorted_g):
        for col, exposure in enumerate(sorted_e):
            x = left + col * (cell_width + gap)
            y = top + row * (cell_height + gap)
            path = grid_data.get((exposure, gain))
            cell = canvas[y:y + cell_height, x:x + cell_width]
            if path is None:
                cell[:] = (210, 210, 210)
                put_text(canvas, "MISSING", x + 10, y + cell_height // 2)
                missing += 1
            else:
                try:
                    image = read_image(path)
                    src_h, src_w = image.shape[:2]
                    scale = min(cell_width / src_w, cell_height / src_h)
                    new_w = max(1, min(cell_width, round(src_w * scale)))
                    new_h = max(1, min(cell_height, round(src_h * scale)))
                    thumb = cv2.resize(
                        image, (new_w, new_h),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                    )
                    dx, dy = (cell_width - new_w) // 2, (cell_height - new_h) // 2
                    cell[:] = (230, 230, 230)
                    cell[dy:dy + new_h, dx:dx + new_w] = thumb
                    success += 1
                except (OSError, ValueError, cv2.error) as exc:
                    cell[:] = (200, 200, 240)
                    put_text(canvas, "READ ERROR", x + 10, y + cell_height // 2)
                    print(f"Warning: {path.name}: {exc}", flush=True)
                    failed += 1
            processed = success + missing + failed
            if processed % 48 == 0 or processed == total:
                print(f"[{processed}/{total}] placed={success}, "
                      f"missing={missing}, failed={failed}", flush=True)

    put_text(canvas, f"Images: {success} | Missing: {missing} | Read errors: {failed}",
             15, canvas_height - 13, 0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    with output_path.open("wb") as output:
        output.write(encoded.tobytes())
    print(f"Saved: {output_path}", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=IMAGE_FOLDER)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--cell-width", type=int, default=CELL_WIDTH)
    args = parser.parse_args()
    try:
        folder = args.input if args.input is not None else latest_scan_folder()
        output = args.output if args.output is not None else folder / "exposure_gain_grid.png"
        merge_images_adaptive_grid(folder, output, args.cell_width)
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
