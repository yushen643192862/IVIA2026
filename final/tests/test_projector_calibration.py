import unittest

import cv2
import numpy as np

from tools.calibrate_camera_projector import projector_corners, solve


class ProjectorCalibrationTests(unittest.TestCase):
    def test_local_homography_recovers_corner_without_sampling_black_corner(self):
        y,x = np.indices((100,100),dtype=np.float32)
        valid = ((x+y)%3==0)
        valid[48:53,48:53] = False
        decoded = {'valid':valid,'projector_x':2*x+.2*y+10,'projector_y':.1*x+1.5*y+20}
        indices,projected,errors = projector_corners(decoded,np.array([[50.,50.]],np.float32),20)
        np.testing.assert_array_equal(indices,[0])
        np.testing.assert_allclose(projected,[[120,100]],atol=.01)
        self.assertLess(errors[0],.01)
        decoded['valid'][:] = False
        self.assertEqual(len(projector_corners(decoded,np.array([[50.,50.]]))[0]),0)

    def test_synthetic_three_pose_calibration_recovers_known_baseline(self):
        obj = np.zeros((54,3),np.float32)
        obj[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*20
        kc = np.array([[1400.,0,1296],[0,1400,972],[0,0,1]])
        kp = np.array([[1200.,0,960],[0,1200,540],[0,0,1]])
        baseline = np.array([0.,120.,0.])
        objects,cameras,projectors = [],[],[]
        for r,t in [([.1,.25,.03],[-80,-50,650]),([-.3,.1,-.05],[-60,-70,740]),([.2,-.3,.08],[-90,-30,600])]:
            r,t = np.array(r,dtype=float),np.array(t,dtype=float)
            objects.append(obj.copy())
            cameras.append(cv2.projectPoints(obj,r,t,kc,None)[0].reshape(-1,2))
            projectors.append(cv2.projectPoints(obj,r,t+baseline,kp,None)[0].reshape(-1,2))
        result = solve(objects,cameras,projectors,(2592,1944),(1920,1080))
        self.assertLess(result['stereo_rms_px'],.01)
        self.assertAlmostEqual(result['baseline_mm'],120.,delta=.1)
        np.testing.assert_allclose(result['t_camera_to_projector_mm'],baseline,atol=.1)


if __name__=='__main__':
    unittest.main()
