import unittest
import numpy as np

from scanner.structured_light import pattern_sequence, render_pattern, capture_sequence_graycode, render_capture_pattern


class StructuredLightTests(unittest.TestCase):
    def test_capture_graycode_is_complete_dense_to_sparse_with_raw_references(self):
        full = capture_sequence_graycode(1920,1080)
        self.assertEqual(len(full),47)
        self.assertEqual(sum(f.get('orientation')=='horizontal' for f in full),22)
        self.assertEqual(sum(f.get('orientation')=='vertical' for f in full),22)
        self.assertEqual([f['role'] for f in full[-3:]],['black','white','white'])
        self.assertEqual(full[-1]['purpose'],'color')
        self.assertNotEqual(full[-2]['purpose'],'color')
        for cell in (1,3):
            width,height = 13,9
            frames = capture_sequence_graycode(width,height,220,cell)
            coded = [f for f in frames if f['role']=='graycode']
            self.assertEqual(coded[0]['axis'],'y')
            for axis in ('y','x'):
                selected = [f for f in coded if f['axis']==axis]
                self.assertEqual([f['bit'] for f in selected[::2]],list(range(len(selected)//2)))
                gray = np.zeros((height,width),np.uint32)
                for first,second in zip(selected[::2],selected[1::2]):
                    a = render_capture_pattern(first,width,height)
                    b = render_capture_pattern(second,width,height)
                    np.testing.assert_array_equal(a.astype(int)+b,220)
                    gray |= (a>110).astype(np.uint32)<<first['bit']
                binary = gray.copy()
                for shift in (1,2,4,8,16):
                    binary ^= binary>>shift
                expected = (np.arange(height)//cell)[:,None] if axis=='y' else np.arange(width)//cell
                np.testing.assert_array_equal(binary,np.broadcast_to(expected,(height,width)))

    def test_graycode_decodes_every_pixel_and_inverse_on_non_power_of_two_size(self):
        for width,height,cell in [(13,9,1),(13,9,3),(1,1,1)]:
            frames = pattern_sequence(width,height,cell_size=cell)
            self.assertEqual(frames[0]['role'],'white')
            self.assertEqual(frames[1]['role'],'black')
            self.assertTrue(np.all(render_pattern(frames[0],width,height,cell)==255))
            self.assertTrue(np.all(render_pattern(frames[1],width,height,cell)==0))
            for first,second in zip(frames[2::2],frames[3::2]):
                a = render_pattern(first,width,height,cell)
                b = render_pattern(second,width,height,cell)
                self.assertEqual(a.shape,(height,width))
                np.testing.assert_array_equal(a.astype(int)+b,255)
            for axis in ('x','y'):
                gray = np.zeros((height,width),np.uint32)
                for frame in frames:
                    if frame.get('axis')==axis and not frame['inverse']:
                        gray |= (render_pattern(frame,width,height,cell)>127).astype(np.uint32)<<frame['bit']
                decoded = gray.copy()
                shift = 1
                while shift<32:
                    decoded ^= decoded>>shift
                    shift *= 2
                expected = np.arange(width)//cell if axis=='x' else (np.arange(height)//cell)[:,None]
                np.testing.assert_array_equal(decoded,np.broadcast_to(expected,(height,width)))

    def test_random_pattern_is_repeatable_and_cells_are_not_resampled(self):
        frame = pattern_sequence(15,11,'random',4)[0]
        first = render_pattern(frame,15,11,4,42)
        second = render_pattern(frame,15,11,4,42)
        np.testing.assert_array_equal(first,second)
        self.assertTrue(set(np.unique(first)).issubset({0,255}))
        for y in range(0,11,4):
            for x in range(0,15,4):
                self.assertTrue(np.all(first[y:y+4,x:x+4]==first[y,x]))


if __name__=='__main__':
    unittest.main()
