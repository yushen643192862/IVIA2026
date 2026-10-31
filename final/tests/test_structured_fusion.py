import unittest

import cv2
import numpy as np

from tools.fuse_structured_clouds import rigid,moved,estimate


class FusionTests(unittest.TestCase):
    def test_rigid_registration_and_outlier_rejection(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(80,3))*30
        pose = np.eye(4)
        pose[:3,:3] = cv2.Rodrigues(np.array([.1,.3,-.2]))[0]
        pose[:3,3] = [4.,-8.,20.]
        target = moved(source,pose)
        np.testing.assert_allclose(rigid(source,target),pose,atol=1e-10)
        corrupted = target.copy(); corrupted[:25] += rng.normal(size=(25,3))*100
        fitted,keep = estimate(source,corrupted,rng)
        self.assertEqual(int(keep.sum()),55)
        np.testing.assert_allclose(fitted,pose,atol=1e-10)


if __name__=='__main__':
    unittest.main()
