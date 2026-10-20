"""Generate exact projector patterns with NumPy; no image synthesis required."""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--period', type=int, default=128,
                        help='Full black-white or sinusoidal period in projector pixels')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'patterns')
    args = parser.parse_args()
    w, h, period = args.width, args.height, args.period
    if min(w,h) < 2 or period < 4 or period % 2:
        parser.error('Dimensions must exceed 1; period must be even and at least 4')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    x, y = np.arange(w), np.arange(h)
    patterns = [
        ('01_binary_horizontal.png', np.broadcast_to(((y//(period//2))%2*255)[:,None], (h,w)).astype(np.uint8)),
        ('02_binary_vertical.png', np.broadcast_to((x//(period//2))%2*255, (h,w)).astype(np.uint8)),
    ]
    bits = (w-1).bit_length()
    gray = x ^ (x >> 1)
    for bit in reversed(range(bits)):
        plane = (((gray >> bit)&1)*255).astype(np.uint8)
        plane = np.broadcast_to(plane,(h,w)).copy()
        patterns.append((f'gray_x_bit{bit:02d}.png', plane))
        patterns.append((f'gray_x_bit{bit:02d}_inverse.png', 255-plane))
    patterns.append(('gray_reference_white.png', np.full((h,w),255,np.uint8)))
    patterns.append(('gray_reference_black.png', np.zeros((h,w),np.uint8)))
    for i, degrees in enumerate((0,90,180,270)):
        intensity = np.rint(127.5+127.5*np.cos(2*np.pi*x/period+np.deg2rad(degrees))).astype(np.uint8)
        patterns.append((f'{5+i:02d}_phase_{degrees:03d}.png', np.broadcast_to(intensity,(h,w)).copy()))
    overview = np.full((((len(patterns)+3)//4)*230,4*400,3),35,np.uint8)
    for i, (name, im) in enumerate(patterns):
        ok, encoded = cv.imencode('.png',im)
        if not ok:
            raise RuntimeError(name)
        encoded.tofile(str(out/name))
        tile = cv.cvtColor(cv.resize(im,(384,190),interpolation=cv.INTER_AREA),cv.COLOR_GRAY2BGR)
        row,col = divmod(i,4)
        overview[row*230+32:row*230+222,col*400+8:col*400+392]=tile
        cv.putText(overview,name,(col*400+8,row*230+22),0,.43,(240,240,240),1)
    cv.imencode('.jpg',overview)[1].tofile(str(out/'preview.jpg'))
    metadata = dict(width=w,height=h,period_pixels=period,binary_stripe_width_pixels=period//2,
        gray_code=f'{bits}-bit reflected Gray code of projector column x; MSB first, each followed by inverse',
        gray_bits=bits, gray_axis='x', gray_unique_columns=w,
        phase_degrees=[0,90,180,270],phase_orientation='vertical stripes, intensity varies with x',
        phase_formula='round(127.5 + 127.5*cos(2*pi*x/period + phase))',
        files=[name for name,_ in patterns])
    (out/'patterns.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (out/'README.md').write_text(
        f'# 投影图案（{len(patterns)} 张）\n\n'
        f'分辨率 {w}×{h}，8 位灰度 PNG。请按原始尺寸全屏显示；preview.jpg 仅作总览。\n\n'
        f'- 01：横向黑白条纹，条宽 {period//2} 像素。\n'
        f'- 02：纵向黑白条纹，条宽 {period//2} 像素。\n'
        f'- gray_x_bit：完整 {bits} 位格雷码，编码每一列 x=0…{w-1}。'
        f'从 bit{bits-1:02d} 到 bit00 投影，每位后紧接同名 inverse 反色图，共 {2*bits} 张。\n'
        '- gray_reference_white/black：全白、全黑参考图，用于识别照明与阴影范围。\n'
        f'- 05–08：纵向正弦条纹，周期 {period} 像素，相位 0°、90°、180°、270°。\n\n'
        '格雷码采用 G(x)=x XOR (x>>1)，每一列有唯一编码；只编码横坐标，未编码行坐标。'
        '这可用于已校正水平双目的对应匹配。拍到最低位条纹需要足够的投影与相机分辨能力。'
        '单相机-投影仪若需完整二维像素对应，还需增加 y 方向编码或使用已标定极线约束。'
        '四步相移恢复包裹相位，结合解码后的列坐标可辅助消除周期歧义；仍需几何标定。\n\n'
        '依次投影，每次稳定后拍摄左右图，设备和物体保持静止；'
        '不要把总览图当投影输入。相移可受投影仪 gamma、过曝及表面反光影响。\n\n'
        '重新生成：`python lab5/generate_patterns.py`。'
        '可用 --width、--height 匹配投影仪原生输出，用 --period 调整完整周期。\n',encoding='utf-8')
    print(f'Generated {len(patterns)} PNG patterns at {w}x{h}: {out}')


if __name__ == '__main__':
    main()
