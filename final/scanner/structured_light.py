"""Deterministic monochrome projection patterns; no camera or GUI dependency."""
import numpy as np


def capture_sequence_graycode(width, height, level=220, cell_size=1):
    """Complete row/column Gray code, dense to sparse, then raw refs and color."""
    if min(width,height,cell_size)<1 or not 32<=level<=255:
        raise ValueError('Positive dimensions/cell size and projector level 32..255 required')
    frames = []
    # Y varies across horizontal stripes; X varies across vertical stripes.
    for axis,length,orientation in [('y',height,'horizontal'),('x',width,'vertical')]:
        cells = (length+cell_size-1)//cell_size
        for bit in range((cells-1).bit_length()):
            for inverse in (False,True):
                frames.append({'index':len(frames),'role':'graycode','purpose':'coordinate',
                    'axis':axis,'orientation':orientation,'bit':bit,'inverse':inverse,
                    'cell_size':cell_size,'level':level,
                    'name':f'{len(frames)+1:02d}_{orientation}_bit{bit:02d}'+('_inverse' if inverse else '')})
    for role,purpose,label in [('black','shadow_reference','black_reference'),
                                ('white','shadow_reference','white_reference'),
                                ('white','color','color_white')]:
        frames.append({'index':len(frames),'role':role,'purpose':purpose,'level':level,
                       'name':f'{len(frames)+1:02d}_{label}'})
    return frames


def capture_sequence_11(level=220, fine_cycles=16):
    """Two five-step phase frequencies plus one uniform-white color frame.

    The coarse half-cycle stays away from the 0/2pi discontinuity. This is not
    a truncated Gray-code sequence; decoding requires phase calibration.
    """
    if not 32<=level<=255 or fine_cycles<1:
        raise ValueError('Projector level must be 32..255 and fine cycles positive')
    frames = []
    for frequency,cycles,offset in [('coarse',.5,np.pi/2),('fine',float(fine_cycles),0.)]:
        for step in range(5):
            frames.append({'index':len(frames),'role':'phase','frequency':frequency,
                'cycles':cycles,'phase_offset':float(offset),'step':step,
                'shift':float(2*np.pi*step/5),'level':level,
                'name':f'{len(frames)+1:02d}_{frequency}_phase{step}'})
    frames.append({'index':10,'role':'white','level':level,'name':'11_white'})
    return frames


def render_capture_pattern(frame, width, height):
    if min(width,height)<1:
        raise ValueError('Projection dimensions must be positive')
    if frame['role'] in ('white','black'):
        return np.full((height,width),frame['level'] if frame['role']=='white' else 0,np.uint8)
    if frame['role']=='graycode':
        bits = render_pattern(frame,width,height,frame['cell_size'])
        return (bits>0).astype(np.uint8)*frame['level']
    if frame['role']!='phase':
        raise ValueError('Expected a phase or uniform-white frame')
    position = (np.arange(width,dtype=np.float64)+.5)/width
    phase = 2*np.pi*frame['cycles']*position+frame['phase_offset']+frame['shift']
    level = frame['level']
    stripe = np.rint(level*.5+level*.45*np.cos(phase)).clip(0,255).astype(np.uint8)
    return np.broadcast_to(stripe[None,:],(height,width)).copy()


def pattern_sequence(width, height, mode='graycode', cell_size=1):
    if min(width, height, cell_size) < 1:
        raise ValueError('Width, height and cell size must be positive')
    if mode not in ('graycode', 'random', 'checkerboard', 'white', 'black'):
        raise ValueError(f'Unknown pattern mode: {mode}')
    frames = []

    def add(role, **extra):
        frames.append({'index': len(frames), 'role': role, **extra})

    if mode == 'graycode':
        add('white')
        add('black')
        for axis, length in (('x', width), ('y', height)):
            cells = (length+cell_size-1)//cell_size
            for bit in reversed(range((cells-1).bit_length())):
                add('graycode', axis=axis, bit=bit, inverse=False)
                add('graycode', axis=axis, bit=bit, inverse=True)
    else:
        add(mode)
    for frame in frames:
        label = frame['role']
        if frame['role'] == 'graycode':
            label = f"{frame['axis']}_bit{frame['bit']:02d}"+('_inverse' if frame['inverse'] else '')
        frame['name'] = f"{frame['index']:03d}_{label}"
    return frames


def render_pattern(frame, width, height, cell_size=1, seed=0):
    if min(width, height, cell_size) < 1:
        raise ValueError('Width, height and cell size must be positive')
    role = frame['role']
    if role in ('white', 'black'):
        return np.full((height, width), 255 if role=='white' else 0, np.uint8)
    xx = np.arange(width, dtype=np.uint32)//cell_size
    yy = np.arange(height, dtype=np.uint32)//cell_size
    if role == 'graycode':
        values = xx if frame['axis']=='x' else yy
        gray = values ^ (values >> 1)
        stripe = (((gray >> frame['bit']) & 1)*255).astype(np.uint8)
        if frame['inverse']:
            stripe = 255-stripe
        return np.broadcast_to(stripe[None,:] if frame['axis']=='x' else stripe[:,None],
                               (height,width)).copy()
    if role == 'checkerboard':
        return (((yy[:,None]+xx[None,:]) & 1)*255).astype(np.uint8)
    if role == 'random':
        rng = np.random.default_rng(seed)
        cells = rng.integers(0,2,size=(int(yy[-1])+1,int(xx[-1])+1),dtype=np.uint8)*255
        return cells[yy[:,None],xx[None,:]]
    raise ValueError(f'Unknown role: {role}')
