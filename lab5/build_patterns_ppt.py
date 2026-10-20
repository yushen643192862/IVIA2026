"""Build a full-screen projection PPTX using standard-library Open XML."""
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
PATTERNS = ROOT / 'patterns'
OUTPUT = ROOT / 'projection_patterns.pptx'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
PREFIX = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
GROUP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
         '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
         '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


def relationships(entries):
    return f'<Relationships xmlns="{REL}">' + ''.join(
        f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>'
        for rid, kind, target in entries) + '</Relationships>'


def main():
    names = ['gray_reference_black.png', 'gray_reference_white.png']
    for bit in range(10,-1,-1):
        names.extend([f'gray_x_bit{bit:02d}.png', f'gray_x_bit{bit:02d}_inverse.png'])
    names += ['01_binary_horizontal.png', '02_binary_vertical.png']
    names += [f'{5+i:02d}_phase_{degrees:03d}.png' for i,degrees in enumerate((0,90,180,270))]
    for name in names:
        if not (PATTERNS/name).is_file():
            raise FileNotFoundError(PATTERNS/name)
    width, height = 12192000, 6858000
    types = ['<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>']
    overrides = [('presentation.xml','presentation.main'), ('slideMasters/slideMaster1.xml','slideMaster'),
                 ('slideLayouts/slideLayout1.xml','slideLayout')]
    overrides += [(f'slides/slide{i}.xml','slide') for i in range(1,len(names)+1)]
    for part, kind in overrides:
        types.append(f'<Override PartName="/ppt/{part}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml"/>')
    types.append('<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>')
    types.append('</Types>')
    with ZipFile(OUTPUT,'w',ZIP_DEFLATED) as z:
        def xml(name, content):
            z.writestr(name,PREFIX+content)
        xml('[Content_Types].xml',''.join(types))
        xml('_rels/.rels',relationships([('rId1','officeDocument','ppt/presentation.xml')]))
        slides = ''.join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1,len(names)+1))
        xml('ppt/presentation.xml', f'<p:presentation xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'<p:sldIdLst>{slides}</p:sldIdLst><p:sldSz cx="{width}" cy="{height}" type="screen16x9"/>'
            '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
        xml('ppt/_rels/presentation.xml.rels',relationships(
            [('rId1','slideMaster','slideMasters/slideMaster1.xml')] +
            [(f'rId{i+1}','slide',f'slides/slide{i}.xml') for i in range(1,len(names)+1)]))
        color_map = ('<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
                     'accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" '
                     'hlink="hlink" tx1="dk1" tx2="dk2"/>')
        xml('ppt/slideMasters/slideMaster1.xml', f'<p:sldMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}">'
            f'<p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{color_map}'
            '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
            '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
        xml('ppt/slideMasters/_rels/slideMaster1.xml.rels',relationships([
            ('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
        xml('ppt/slideLayouts/slideLayout1.xml', f'<p:sldLayout xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" type="blank" preserve="1">'
            f'<p:cSld name="Blank"><p:spTree>{GROUP}</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
        xml('ppt/slideLayouts/_rels/slideLayout1.xml.rels',relationships([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
        colors = dict(dk1='000000',lt1='FFFFFF',dk2='222222',lt2='EEEEEE',accent1='4472C4',
                      accent2='ED7D31',accent3='A5A5A5',accent4='FFC000',accent5='5B9BD5',
                      accent6='70AD47',hlink='0563C1',folHlink='954F72')
        scheme = ''.join(f'<a:{key}><a:srgbClr val="{value}"/></a:{key}>' for key,value in colors.items())
        xml('ppt/theme/theme1.xml',f'<a:theme xmlns:a="{A}" name="Projection">'
            f'<a:themeElements><a:clrScheme name="Projection">{scheme}</a:clrScheme>'
            '<a:fontScheme name="Projection"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/>'
            '<a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/>'
            '<a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Projection">'
            '<a:fillStyleLst>' + '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3 +
            '</a:fillStyleLst><a:lnStyleLst>' +
            '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'*3 +
            '</a:lnStyleLst><a:effectStyleLst>' + '<a:effectStyle><a:effectLst/></a:effectStyle>'*3 +
            '</a:effectStyleLst><a:bgFillStyleLst>' + '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3 +
            '</a:bgFillStyleLst>'
            '</a:fmtScheme></a:themeElements></a:theme>')
        for i,name in enumerate(names,1):
            z.write(PATTERNS/name,f'ppt/media/image{i}.png')
            xml(f'ppt/slides/slide{i}.xml',f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}">'
                f'<p:cSld name="{escape(name)}"><p:spTree>{GROUP}<p:pic><p:nvPicPr>'
                f'<p:cNvPr id="2" name="{escape(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/>'
                '</p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/>'
                '<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>'
                '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic></p:spTree></p:cSld>'
                '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr><p:transition advClick="1"/></p:sld>')
            xml(f'ppt/slides/_rels/slide{i}.xml.rels',relationships([
                ('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image',f'../media/image{i}.png')]))
    (ROOT/'projection_slide_order.txt').write_text('\n'.join(f'{i:02d}  {name}' for i,name in enumerate(names,1)),encoding='utf-8')
    print(f'Created {len(names)} slides: {OUTPUT}')


if __name__ == '__main__':
    main()
