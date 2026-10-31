"""Generate an exact-size A4 landscape chessboard as vector PDF and SVG."""
import argparse
import json
from pathlib import Path


def generate(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    page_w,page_h = 297.,210.
    columns,rows,square = 10,7,20.
    left,top = (page_w-columns*square)/2,(page_h-rows*square)/2
    title = 'CAMERA - PROJECTOR CALIBRATION TARGET'
    subtitle = 'Inner corners: 9 x 6 | Squares: 10 x 7 | Square edge: 20 mm'
    note = 'PRINT A4 LANDSCAPE / ACTUAL SIZE 100% / NO FIT-TO-PAGE'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">',
           '<rect width="297" height="210" fill="white"/>',
           f'<text x="148.5" y="13" font-family="sans-serif" font-size="5" text-anchor="middle">{title}</text>',
           f'<text x="148.5" y="23" font-family="sans-serif" font-size="3.8" text-anchor="middle">{subtitle}</text>']
    pt = 72/25.4
    commands = ['0 g']

    def pdf_text(text,x,y,size):
        escaped = text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
        commands.append(f'BT /F1 {size*pt:.6f} Tf {x*pt:.6f} {(page_h-y)*pt:.6f} Td ({escaped}) Tj ET')

    pdf_text(title,63,13,5)
    pdf_text(subtitle,48.5,23,3.8)
    for y in range(rows):
        for x in range(columns):
            if (x+y)%2==0:
                xpos,ypos = left+x*square,top+y*square
                svg.append(f'<rect x="{xpos}" y="{ypos}" width="20" height="20" fill="black"/>')
                commands.append(f'{xpos*pt:.6f} {(page_h-ypos-square)*pt:.6f} {square*pt:.6f} {square*pt:.6f} re f')
    start,end,y = 98.5,198.5,184.
    svg.append(f'<path d="M {start} {y} H {end} M {start} {y-2} V {y+2} M {end} {y-2} V {y+2}" fill="none" stroke="black" stroke-width="0.3"/>')
    commands.append(f'{.3*pt:.6f} w {start*pt:.6f} {(page_h-y)*pt:.6f} m {end*pt:.6f} {(page_h-y)*pt:.6f} l S')
    for x in (start,end):
        commands.append(f'{x*pt:.6f} {(page_h-y+2)*pt:.6f} m {x*pt:.6f} {(page_h-y-2)*pt:.6f} l S')
    for text,baseline in [('100 mm - verify with a ruler after printing',193), (note,203)]:
        svg.append(f'<text x="148.5" y="{baseline}" font-family="sans-serif" font-size="3.2" text-anchor="middle">{text}</text>')
    pdf_text('100 mm - verify with a ruler after printing',112,193,3.2)
    pdf_text(note,53,203,3.2)
    svg.append('</svg>')
    stem = 'chessboard_9x6_20mm_A4'
    (output/(stem+'.svg')).write_text('\n'.join(svg),encoding='utf-8')
    content = '\n'.join(commands).encode('ascii')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w*pt:.6f} {page_h*pt:.6f}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>'.encode(),
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'\nendstream']
    document = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for number,obj in enumerate(objects,1):
        offsets.append(len(document))
        document.extend(f'{number} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref = len(document)
    document.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        document.extend(f'{offset:010d} 00000 n \n'.encode())
    document.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    (output/(stem+'.pdf')).write_bytes(document)
    (output/'board.json').write_text(json.dumps({'inner_corners':[9,6],'squares':[10,7],
        'square_size_mm':20,'board_size_mm':[200,140],'paper_size_mm':[297,210],
        'print_scale':'100% actual size; verify printed dimensions with a ruler',
        'note':'Use measured printed dimensions if print scale differs; attach to a flat matte rigid support.'},indent=2),encoding='utf-8')
    print(output/(stem+'.pdf'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/structured_light/calibration_board'))
    generate(parser.parse_args().output)
