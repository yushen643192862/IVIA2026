"""提取固定白墙 ROI 的单图噪声估计值和平均亮度，不进行模型拟合。

运行：python extract_noise.py
指定数据：python extract_noise.py --input "采集文件夹"
修改原图 ROI：python extract_noise.py --roi 682 415 889 830

ROI 坐标顺序为 x0 y0 x1 y1，右边界和下边界不包含在内。
默认 ROI 适用于本次 2592x1944 的静止场景；更换场景后应重新检查。
依赖：numpy、opencv-python（与采集脚本相同）。

noise_std_gray = std(gray_roi - GaussianBlur(gray_roi))，ddof=1。
先转 float32 再转换灰度和相减；统计时去掉半个滤波核宽的边界。
这是高频残差的空间噪声代理值，包含残余纹理及相机处理的影响，
不是严格的传感器时间噪声测量，也没有作滤波器噪声增益校正。
所有图片使用同一原图 ROI 和同一平滑参数；不缩放、不调整亮度。
保留全部记录，fit_candidate 仅作为拟合前的筛选提示。
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from data_paths import latest_scan


DEFAULT_ROI = (682, 415, 889, 830)
DEFAULT_IMAGE_SIZE = (2592, 1944)  # width, height
IMAGE_PATTERN = re.compile(r"e\d+g\d+\.bmp", re.IGNORECASE)
METRIC_FIELDS = [
    'image_width', 'image_height', 'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1',
    'gaussian_kernel', 'gaussian_sigma', 'noise_pixel_count',
    'image_mean_gray', 'roi_mean_gray', 'roi_spatial_std_gray',
    'noise_std_gray', 'noise_variance_gray', 'residual_mean_gray',
    'roi_near_black_fraction', 'roi_near_white_fraction',
    'roi_channel_saturation_fraction', 'fit_candidate', 'quality_flags',
    'status', 'error',
]


def latest_scan_folder():
    return latest_scan(Path(__file__).resolve().parent)


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path.name}')
    return image


def extract_metrics(image, roi, kernel=5, sigma=1.0,
                    min_mean=5.0, max_mean=250.0, max_clip_fraction=0.01):
    """计算单张原图的指标；噪声标准差单位为灰度级，方差为灰度级平方。"""
    if kernel < 3 or kernel % 2 == 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Use an odd kernel >= 3 and a positive finite sigma.')
    if not 0 <= min_mean < max_mean <= 255 or not 0 <= max_clip_fraction <= 1:
        raise ValueError('Invalid brightness or clipping thresholds.')
    height, width = image.shape[:2]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'ROI {roi} is outside image size {width}x{height}.')
    if min(x1 - x0, y1 - y0) <= kernel:
        raise ValueError('ROI must be larger than the Gaussian kernel.')

    region = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(region.astype(np.float32), cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=sigma,
                              sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    radius = kernel // 2
    residual = (gray - smooth)[radius:-radius, radius:-radius]
    variance = float(np.var(residual, dtype=np.float64, ddof=1))
    mean_gray = float(np.mean(gray, dtype=np.float64))
    # 亮度与噪声均在 0..255 灰度尺度上；全图均值用通道均值线性组合。
    mean_b, mean_g, mean_r, _ = cv2.mean(image)
    black = float(np.mean(gray <= 1.0))
    white = float(np.mean(gray >= 254.0))
    saturated = float(np.mean(np.any(region >= 254, axis=2)))
    flags = []
    if mean_gray < min_mean:
        flags.append('low_roi_mean')
    if mean_gray > max_mean:
        flags.append('high_roi_mean')
    if black > max_clip_fraction:
        flags.append('near_black_pixels')
    if white > max_clip_fraction:
        flags.append('near_white_pixels')
    if saturated > max_clip_fraction:
        flags.append('channel_saturation')

    return {
        'image_width': width, 'image_height': height,
        'roi_x0': x0, 'roi_y0': y0, 'roi_x1': x1, 'roi_y1': y1,
        'gaussian_kernel': kernel, 'gaussian_sigma': sigma,
        'noise_pixel_count': residual.size,
        'image_mean_gray': 0.114 * mean_b + 0.587 * mean_g + 0.299 * mean_r,
        'roi_mean_gray': mean_gray,
        # 原 ROI 标准差会混入光照渐变，仅保留作参考，不作为噪声指标。
        'roi_spatial_std_gray': float(np.std(gray, dtype=np.float64, ddof=1)),
        'noise_std_gray': math.sqrt(variance),
        'noise_variance_gray': variance,
        'residual_mean_gray': float(np.mean(residual, dtype=np.float64)),
        'roi_near_black_fraction': black,
        'roi_near_white_fraction': white,
        'roi_channel_saturation_fraction': saturated,
        'fit_candidate': int(not flags), 'quality_flags': ';'.join(flags),
        'status': 'ok', 'error': '',
    }


def save_roi_preview(image, roi, output):
    """仅预览缩放，指标计算始终使用原图。"""
    scale = min(1.0, 1254 / image.shape[1])
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = (round(value * scale) for value in roi)
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(preview, 'Noise / brightness ROI', (x0, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    ok, data = cv2.imencode('.png', preview)
    if not ok:
        raise RuntimeError('Could not encode the ROI preview.')
    output.write_bytes(data.tobytes())


def run_extraction(folder, output, roi=DEFAULT_ROI, kernel=5, sigma=1.0,
                   min_mean=5.0, max_mean=250.0, max_clip_fraction=0.01,
                   expected_size=DEFAULT_IMAGE_SIZE):
    folder, output = Path(folder), Path(output)
    parameters_path = folder / 'scan_parameters.csv'
    with parameters_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        source_fields = reader.fieldnames or []
        required = {'filename', 'actual_exposure_us', 'actual_gain_db'}
        if not required.issubset(source_fields):
            raise ValueError(f'Metadata CSV must contain {sorted(required)}')
        records = list(reader)
    if not records:
        raise ValueError('Metadata CSV is empty.')
    if len({row['filename'] for row in records}) != len(records):
        raise ValueError('Duplicate filenames in metadata CSV.')
    if output.suffix.lower() != '.csv' or output.resolve() == parameters_path.resolve():
        raise ValueError('Choose a .csv output separate from scan_parameters.csv.')
    output.parent.mkdir(parents=True, exist_ok=True)
    count_ok = count_failed = count_candidates = 0
    preview_score = float('inf')
    preview_image = None
    preview_filename = None
    print(f'Input: {folder}', flush=True)
    print(f'ROI (x0,y0,x1,y1): {roi}; kernel={kernel}, sigma={sigma}', flush=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = source_fields + [field for field in METRIC_FIELDS if field not in source_fields]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, metadata in enumerate(records, 1):
            row = dict(metadata)
            try:
                if not IMAGE_PATTERN.fullmatch(metadata['filename']):
                    raise ValueError('Expected filename e<exposure>g<gain>.bmp.')
                for field in ('actual_exposure_us', 'actual_gain_db'):
                    if not math.isfinite(float(metadata[field])):
                        raise ValueError(f'Invalid {field}')
                image = read_image(folder / metadata['filename'])
                if (image.shape[1], image.shape[0]) != tuple(expected_size):
                    raise ValueError(f'Expected image size {expected_size}; '
                                     'check ROI before using a different resolution.')
                metrics = extract_metrics(image, roi, kernel, sigma, min_mean,
                                          max_mean, max_clip_fraction)
                row.update(metrics)
                count_ok += 1
                count_candidates += metrics['fit_candidate']
                # 保存一张中等亮度图片的 ROI 位置预览，便于人工核对。
                score = abs(metrics['roi_mean_gray'] - 110)
                if score < preview_score:
                    preview_score = score
                    preview_image = image.copy()
                    preview_filename = metadata['filename']
            except (OSError, ValueError, cv2.error) as exc:
                row.update(status='error', error=str(exc), fit_candidate=0,
                           quality_flags='read_or_parameter_error')
                count_failed += 1
                print(f"Warning: {metadata['filename']}: {exc}", flush=True)
            writer.writerow(row)
            if index % 48 == 0 or index == len(records):
                stream.flush()
                print(f'[{index}/{len(records)}] ok={count_ok}, '
                      f'candidates={count_candidates}, errors={count_failed}', flush=True)

    preview_path = output.with_name(output.stem + '_roi_preview.png')
    if preview_image is not None:
        save_roi_preview(preview_image, roi, preview_path)
    settings = {
        'source_csv': str(parameters_path.resolve()), 'output_csv': str(output.resolve()),
        'roi_xyxy_exclusive': list(roi), 'expected_image_size_wh': list(expected_size),
        'gaussian_kernel': kernel, 'gaussian_sigma': sigma,
        'grayscale': '0.114 B + 0.587 G + 0.299 R, float32, scale 0..255',
        'noise_method': 'std of ROI minus GaussianBlur; ddof=1; kernel//2 border excluded',
        'interpretation': 'Single-image high-frequency spatial residual proxy; not temporal sensor noise. No filter-gain correction.',
        'brightness': 'roi_mean_gray: full ROI mean; image_mean_gray: whole-image mean',
        'fit_candidate_rule': {
            'roi_mean_range_inclusive': [min_mean, max_mean],
            'near_black_gray_le': 1.0, 'near_white_gray_ge': 254.0,
            'channel_saturation_any_bgr_ge': 254,
            'max_fraction_inclusive': max_clip_fraction,
            'note': 'Heuristic quality flags only; all records retained. Texture is not automatically screened.',
        },
        'total_records': len(records), 'successful': count_ok,
        'errors': count_failed, 'fit_candidates': count_candidates,
        'preview_source_filename': preview_filename,
        'noise_units': 'gray levels (std), gray levels squared (variance)',
    }
    output.with_name(output.stem + '_settings.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {output}', flush=True)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, help='Folder containing scan_parameters.csv and BMPs')
    parser.add_argument('--output', type=Path, help='Output CSV path')
    parser.add_argument('--roi', nargs=4, type=int, default=DEFAULT_ROI,
                        metavar=('X0', 'Y0', 'X1', 'Y1'))
    parser.add_argument('--image-size', nargs=2, type=int, default=DEFAULT_IMAGE_SIZE,
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--min-mean', type=float, default=5.0)
    parser.add_argument('--max-mean', type=float, default=250.0)
    parser.add_argument('--max-clip-fraction', type=float, default=0.01)
    args = parser.parse_args()
    try:
        if args.kernel < 3 or args.kernel % 2 == 0 or not math.isfinite(args.sigma) or args.sigma <= 0:
            raise ValueError('Use an odd --kernel >= 3 and a positive finite --sigma.')
        if not 0 <= args.min_mean < args.max_mean <= 255 or not 0 <= args.max_clip_fraction <= 1:
            raise ValueError('Invalid brightness or clipping thresholds.')
        folder = args.input if args.input is not None else latest_scan_folder()
        output = args.output if args.output is not None else folder / 'noise_analysis' / 'noise_features.csv'
        settings = run_extraction(folder, output, tuple(args.roi), args.kernel, args.sigma,
                                  args.min_mean, args.max_mean, args.max_clip_fraction,
                                  tuple(args.image_size))
        return 1 if settings['errors'] else 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
