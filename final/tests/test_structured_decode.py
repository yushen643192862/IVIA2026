import unittest
import numpy as np

from scanner.structured_light import capture_sequence_graycode,render_capture_pattern
from scanner.structured_decode import decode_graycode


class StructuredDecodeTests(unittest.TestCase):
    def test_coordinates_survive_ambient_light_and_missing_pixels_are_invalid(self):
        width,height = 13,9
        patterns = capture_sequence_graycode(width,height,220)
        images = {p['name']:render_capture_pattern(p,width,height).astype(float)*.6+30 for p in patterns}
        def reader(p):
            image = images[p['name']].copy()
            image[0,0] = 40  # shadow: no difference for any projected pattern
            if p.get('axis')=='x' and p.get('bit')==0:
                image[1,1] = 80  # ambiguous first bit, even though refs are lit
            return image
        result = decode_graycode(patterns,reader,(width,height))
        valid = result['valid']
        self.assertFalse(valid[0,0]); self.assertFalse(valid[1,1])
        self.assertEqual(int(valid.sum()),width*height-2)
        yy,xx = np.indices((height,width))
        np.testing.assert_array_equal(result['projector_x'][valid],xx[valid])
        np.testing.assert_array_equal(result['projector_y'][valid],yy[valid])
        self.assertEqual(result['projector_x'][0,0],-1)

    def test_incomplete_bits_are_rejected(self):
        patterns = capture_sequence_graycode(13,9)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            decode_graycode(patterns[1:],lambda p:render_capture_pattern(p,13,9),(13,9))


if __name__=='__main__':
    unittest.main()
