"""Append axis-swapped pictures to the user's edited PPTX, preserving originals."""
import copy
import posixpath
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
NS = {'p':P,'a':A,'r':R}
for prefix, uri in NS.items():
    ET.register_namespace(prefix,uri)


def xml(tree):
    return ET.tostring(tree,encoding='utf-8',xml_declaration=True)


def resolve(part, target):
    return posixpath.normpath(posixpath.join(posixpath.dirname(part),target)) if not target.startswith('/') else target.lstrip('/')


def relationships_path(part):
    return posixpath.join(posixpath.dirname(part),'_rels',posixpath.basename(part)+'.rels')


def main():
    source = ROOT/'projection_patterns.pptx'
    output = ROOT/'projection_patterns_vertical_horizontal.pptx'
    images_dir = ROOT/'horizontal_from_ppt'
    images_dir.mkdir(exist_ok=True)
    with ZipFile(source) as z:
        contents = {n:z.read(n) for n in z.namelist()}
    presentation = ET.fromstring(contents['ppt/presentation.xml'])
    relations = ET.fromstring(contents['ppt/_rels/presentation.xml.rels'])
    types = ET.fromstring(contents['[Content_Types].xml'])
    slide_list = presentation.find('p:sldIdLst',NS)
    originals = list(slide_list)
    relation_map = {r.get('Id'):r.get('Target') for r in relations}
    max_slide = max(int(re.search(r'slide(\d+)\.xml$',n).group(1)) for n in contents
                    if re.fullmatch(r'ppt/slides/slide\d+\.xml',n))
    next_id = max(int(s.get('id')) for s in originals)+1
    ids = {r.get('Id') for r in relations}
    order, previews = [], []
    for index, original in enumerate(originals,1):
        old_part = resolve('ppt/presentation.xml',relation_map[original.get('{'+R+'}id')])
        slide = ET.fromstring(contents[old_part])
        rels = ET.fromstring(contents[relationships_path(old_part)])
        pictures = slide.findall('.//p:pic',NS)
        if len(pictures)!=1:
            raise ValueError(f'Slide {index} must contain exactly one picture')
        picture = pictures[0]
        blip = picture.find('p:blipFill/a:blip',NS)
        image_id = blip.get('{'+R+'}embed')
        image_rel = next(r for r in rels if r.get('Id')==image_id)
        path = resolve(old_part,image_rel.get('Target'))
        image = cv.imdecode(np.frombuffer(contents[path],np.uint8),cv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Cannot decode image on slide {index}')
        height,width = image.shape[:2]
        interpolation = cv.INTER_NEAREST if len(np.unique(image))<=2 else cv.INTER_LINEAR
        # Swap x/y while retaining full slide coverage and normalized pattern phase.
        horizontal = cv.resize(cv.transpose(image),(width,height),interpolation=interpolation)
        ok, encoded = cv.imencode('.png',horizontal)
        if not ok:
            raise RuntimeError('PNG encode failed')
        filename = f'horizontal_{index:02d}.png'
        (images_dir/filename).write_bytes(encoded.tobytes())
        media_part = f'ppt/media/added_{filename}'
        contents[media_part] = encoded.tobytes()
        image_rel.set('Target','../media/'+posixpath.basename(media_part))
        properties = picture.find('p:nvPicPr/p:cNvPr',NS)
        original_name = properties.get('name','')
        properties.set('name','Horizontal '+original_name)
        common = slide.find('p:cSld',NS)
        common.set('name','Horizontal '+original_name)
        new_part = f'ppt/slides/slide{max_slide+index}.xml'
        contents[new_part] = xml(slide)
        contents[relationships_path(new_part)] = xml(rels)
        rid = f'rIdHorizontal{index}'
        while rid in ids:
            rid += '_'
        ids.add(rid)
        ET.SubElement(relations,'{'+PKG+'}Relationship',Id=rid,Type=R+'/slide',
                      Target='slides/'+posixpath.basename(new_part))
        ET.SubElement(slide_list,'{'+P+'}sldId',{'id':str(next_id+index-1),'{'+R+'}id':rid})
        ET.SubElement(types,'{'+CT+'}Override',PartName='/'+new_part,
                      ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
        order.append(f'{index:02d} original: {original_name}')
        order.append(f'{len(originals)+index:02d} horizontal: {original_name}')
        thumb = cv.resize(horizontal,(384,216),interpolation=cv.INTER_AREA)
        if thumb.ndim==2:
            thumb=cv.cvtColor(thumb,cv.COLOR_GRAY2BGR)
        elif thumb.shape[2]==4:
            thumb=cv.cvtColor(thumb,cv.COLOR_BGRA2BGR)
        previews.append(thumb)
    if not any(t.get('Extension')=='png' for t in types):
        ET.SubElement(types,'{'+CT+'}Default',Extension='png',ContentType='image/png')
    contents['ppt/presentation.xml'] = xml(presentation)
    contents['ppt/_rels/presentation.xml.rels'] = xml(relations)
    contents['[Content_Types].xml'] = xml(types)
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for name,data in contents.items():
            z.writestr(name,data)
    order.sort(key=lambda line:int(line.split()[0]))
    (ROOT/'projection_vertical_horizontal_order.txt').write_text('\n'.join(order),encoding='utf-8')
    sheet = np.full((((len(previews)+3)//4)*250,4*400,3),30,np.uint8)
    for i,thumb in enumerate(previews):
        row,col=divmod(i,4)
        sheet[row*250+30:row*250+246,col*400+8:col*400+392]=thumb
        cv.putText(sheet,f'Slide {len(originals)+i+1}: horizontal',(col*400+8,row*250+22),0,.5,(255,255,255),1)
    cv.imencode('.jpg',sheet)[1].tofile(str(images_dir/'preview.jpg'))
    print(f'Preserved {len(originals)} slides and appended {len(originals)} horizontal slides: {output}')


if __name__=='__main__':
    main()
