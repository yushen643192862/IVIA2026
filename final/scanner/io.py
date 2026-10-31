import json
import struct
from pathlib import Path
import numpy as np


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_ply(path, vertices, faces=None, colors=None, normals=None):
    vertices = np.asarray(vertices).reshape(-1, 3)
    faces = np.asarray(faces if faces is not None else [], dtype=int).reshape(-1, 3)
    colors = np.asarray(colors if colors is not None else np.tile([180,180,180], (len(vertices),1)), dtype=int)
    with Path(path).open('w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n')
        f.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n')
        if normals is not None: f.write('property float nx\nproperty float ny\nproperty float nz\n')
        if len(faces):
            f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\n')
        f.write('end_header\n')
        for i,(v,c) in enumerate(zip(vertices, colors)):
            line=' '.join(f'{x:.8g}' for x in v)+' '+ ' '.join(str(int(x)) for x in c)
            if normals is not None: line+=' '+' '.join(f'{x:.8g}' for x in normals[i])
            f.write(line+'\n')
        for a,b,c in faces:
            f.write(f'3 {a} {b} {c}\n')


def read_ply(path,with_attributes=False):
    """Read standard ASCII or binary PLY vertices and polygon faces."""
    types = {'char':'b','int8':'b','uchar':'B','uint8':'B','short':'h','int16':'h',
             'ushort':'H','uint16':'H','int':'i','int32':'i','uint':'I','uint32':'I',
             'float':'f','float32':'f','double':'d','float64':'d'}
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        elements = []; fmt = None
        while True:
            line = f.readline()
            if not line: raise ValueError('Truncated PLY header')
            tokens = line.decode('ascii').split()
            if not tokens: continue
            if tokens[0] == 'format': fmt = tokens[1]
            if tokens[0] == 'element': elements.append([tokens[1],int(tokens[2]),[]])
            if tokens[0] == 'property': elements[-1][2].append(tokens[1:])
            if tokens[0] == 'end_header': break
        if fmt not in ['ascii','binary_little_endian','binary_big_endian']:
            raise ValueError('Unsupported PLY encoding')
        endian = '<' if fmt == 'binary_little_endian' else '>'
        vertices, faces, attributes = [], [], []
        def scalar(kind):
            code = endian+types[kind]; length = struct.calcsize(code)
            raw = f.read(length)
            if len(raw) != length: raise ValueError('Truncated PLY body')
            return struct.unpack(code,raw)[0]
        for name,count,props in elements:
            for _ in range(count):
                row={}; stream=iter(f.readline().decode().split()) if fmt=='ascii' else None
                for prop in props:
                    if prop[0]=='list':
                        n=int(next(stream)) if stream is not None else scalar(prop[1])
                        values=[int(next(stream)) if stream is not None else scalar(prop[2]) for _ in range(n)]
                        row[prop[3]]=values
                    else:
                        row[prop[1]]=float(next(stream)) if stream is not None else scalar(prop[0])
                if name=='vertex':
                    vertices.append([row['x'],row['y'],row['z']]); attributes.append(row)
                if name=='face':
                    polygon=row.get('vertex_indices',row.get('vertex_index',[]))
                    faces.extend([[polygon[0],polygon[i],polygon[i+1]] for i in range(1,len(polygon)-1)])
    v=np.asarray(vertices,float).reshape(-1,3); faces=np.asarray(faces,int).reshape(-1,3)
    if not np.isfinite(v).all() or (len(faces) and (faces.min()<0 or faces.max()>=len(v))):
        raise ValueError('Invalid PLY coordinates/indices')
    if with_attributes:
        attrs={k:np.asarray([r[k] for r in attributes]) for k in (attributes[0] if attributes else {})
               if not isinstance(attributes[0][k],list)}
        return v,faces,attrs
    return v,faces
