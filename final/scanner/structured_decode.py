"""Decode per-pixel camera-to-projector correspondences, not uncalibrated depth."""
import math
import numpy as np


def decode_graycode(patterns, read_image, projector_size, cell_size=1,
                    bit_threshold=8., white_threshold=20., relative_threshold=.05):
    width,height = map(int,projector_size)
    if min(width,height,cell_size)<1 or min(bit_threshold,white_threshold,relative_threshold)<0:
        raise ValueError('Invalid projector dimensions or confidence thresholds')
    references = {p['role']:p for p in patterns if p.get('purpose')=='shadow_reference'}
    if set(references)!={'white','black'}:
        raise ValueError('Exactly one raw black and white reference required')
    if sum(p.get('purpose')=='shadow_reference' for p in patterns)!=2:
        raise ValueError('Duplicate shadow references')
    white = np.asarray(read_image(references['white']),np.float32)
    black = np.asarray(read_image(references['black']),np.float32)
    if white.ndim!=2 or white.shape!=black.shape:
        raise ValueError('References must be same-size grayscale images')
    illumination = white-black
    illuminated = (illumination>=white_threshold)&(white<250)
    required = np.maximum(bit_threshold,np.maximum(illumination,0)*relative_threshold)
    valid = illuminated.copy()
    minimum = np.full(white.shape,np.inf,np.float32)
    complement = np.zeros(white.shape,np.float32)
    coordinates = {}
    statistics = []
    for axis,length in [('y',height),('x',width)]:
        cells = math.ceil(length/cell_size)
        bits = (cells-1).bit_length()
        selected = [p for p in patterns if p.get('role')=='graycode' and p.get('axis')==axis]
        keys = [(int(p['bit']),bool(p['inverse'])) for p in selected]
        expected = {(bit,inverse) for bit in range(bits) for inverse in (False,True)}
        if len(keys)!=len(expected) or set(keys)!=expected:
            raise ValueError(f'Incomplete or duplicate Gray-code bits on axis {axis}')
        lookup = dict(zip(keys,selected))
        gray = np.zeros(white.shape,np.uint32)
        for bit in range(bits):
            first = np.asarray(read_image(lookup[(bit,False)]),np.float32)
            second = np.asarray(read_image(lookup[(bit,True)]),np.float32)
            if first.shape!=white.shape or second.shape!=white.shape:
                raise ValueError('Pattern/reference size mismatch')
            difference = first-second
            contrast = np.abs(difference)
            reliable = contrast>=required
            valid &= reliable
            minimum = np.minimum(minimum,contrast)
            complement = np.maximum(complement,np.abs(first+second-white-black))
            gray |= (difference>0).astype(np.uint32)<<bit
            statistics.append({'axis':axis,'bit':bit,'orientation':'horizontal' if axis=='y' else 'vertical',
                'illuminated_pixels':int(illuminated.sum()),
                'reliable_fraction_in_illuminated':float(np.mean(reliable[illuminated])) if illuminated.any() else 0.,
                'median_pair_contrast':float(np.median(contrast[illuminated])) if illuminated.any() else 0.})
        binary = gray.copy()
        shift = 1
        while shift<32:
            binary ^= binary>>shift
            shift *= 2
        valid &= binary<cells
        # Use the center of the actual last cell if dimensions aren't divisible.
        start = binary.astype(np.float32)*cell_size
        coordinates[axis] = (start+np.minimum(start+cell_size,length)-1)*.5
    minimum[~np.isfinite(minimum)] = 0
    for array in coordinates.values():
        array[~valid] = -1
    return {'projector_x':coordinates['x'],'projector_y':coordinates['y'],'valid':valid,
            'illumination':illumination,'minimum_pair_contrast':minimum,
            'maximum_complement_error':complement,'bit_statistics':statistics}
