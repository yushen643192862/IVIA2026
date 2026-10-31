import json
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np

from scanner.dense_masks import (read_stereo_array, write_stereo_array,
                                 prepare_dense_masks, constrain_depth_maps)


class DenseMaskTests(unittest.TestCase):
    def test_colmap_channel_layout_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'normal.bin'
            path.write_bytes(b'3&2&3&'+struct.pack('<18f',*range(18)))
            array = read_stereo_array(path)
            np.testing.assert_array_equal(array[1,2], [5,11,17])
            copy = path.with_name('copy.bin')
            write_stereo_array(copy,array)
            self.assertEqual(copy.read_bytes(),path.read_bytes())
            path.write_bytes(b'3&2&3&'+b'\x00'*4)
            with self.assertRaisesRegex(ValueError,'dimensions'):
                read_stereo_array(path)

    def test_undistorted_masks_background_depth_and_cache_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dense = root/'dense'
            (dense/'images/nested').mkdir(parents=True)
            image_path = dense/'images/nested/view.png'
            image = np.full((3,3,3),120,np.uint8)
            cv2.imwrite(str(image_path),image)
            mask_path = root/'mask.png'
            mask = np.zeros((5,7),np.uint8)
            mask[2,3] = 255
            cv2.imwrite(str(mask_path),mask)
            camera = {'file':'nested/view.png','size':[7,5],
                      'K':[[4,0,3],[0,4,2],[0,0,1]],
                      'distortion':[.2,0,0,0,0],'object_mask':str(mask_path)}
            view = {'file':camera['file'],'size':[3,3],
                    'K':[[2,0,1],[0,2,1],[0,0,1]]}
            foregrounds = prepare_dense_masks(dense,[camera],[view],10)
            foreground = foregrounds[camera['file']]
            self.assertEqual(int(foreground.sum()),1)
            self.assertTrue(foreground[1,1])
            masked = cv2.imread(str(image_path))
            self.assertTrue(np.all(masked[~foreground]==0))
            np.testing.assert_array_equal(masked[1,1],image[1,1])
            for kind in ('photometric','geometric'):
                depth_path = dense/'stereo/depth_maps'/(camera['file']+'.'+kind+'.bin')
                normal_path = dense/'stereo/normal_maps'/(camera['file']+'.'+kind+'.bin')
                depth_path.parent.mkdir(parents=True,exist_ok=True)
                normal_path.parent.mkdir(parents=True,exist_ok=True)
                depth = np.full((3,3,1),2,np.float32)
                normal = np.ones((3,3,3),np.float32)
                write_stereo_array(depth_path,depth)
                write_stereo_array(normal_path,normal)
                report = constrain_depth_maps(dense,foregrounds,kind)
                self.assertEqual(report[0]['background_depths_removed'],8)
                actual = read_stereo_array(depth_path)
                self.assertEqual(actual[1,1,0],2)
                self.assertTrue(np.all(actual[~foreground]==0))
                self.assertTrue(np.all(read_stereo_array(normal_path)[~foreground]==0))
            # The identical workspace can resume, but changed masks or settings
            # cannot silently reuse depth estimates from another experiment.
            prepare_dense_masks(dense,[camera],[view],10)
            with self.assertRaisesRegex(ValueError,'settings changed'):
                prepare_dense_masks(dense,[camera],[view],5)
            cv2.imwrite(str(mask_path),np.full_like(mask,255))
            with self.assertRaisesRegex(ValueError,'settings changed'):
                prepare_dense_masks(dense,[camera],[view],10)
            (dense/'foreground_config.json').unlink()
            with self.assertRaisesRegex(ValueError,'not foreground-constrained'):
                prepare_dense_masks(dense,[camera],[view],10)


if __name__=='__main__':
    unittest.main()
