import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from scanner.structured_depth import triangulate,depth_gap_cutoff
from scanner.io import read_ply,save_json
from tools.reconstruct_structured_depth import process
from tools.decode_capture_session import save_image
from tools.diagnose_structured_depth import comparison_regions
from tools.project_patterns import projector_canvas_geometry
from tools.segment_structured_subject import segment


def calibration():
    return {'camera_matrix':[[800.,0,320],[0,820.,240],[0,0,1]],
        'projector_matrix':[[700.,0,300],[0,710.,220],[0,0,1]],
        'camera_distortion':[-.1,.02,.001,-.001,0],
        'projector_distortion':[-.05,.01,0,0,0],
        'R_camera_to_projector':cv2.Rodrigues(np.array([.01,.02,.005]))[0].tolist(),
        't_camera_to_projector_mm':[0.,100.,0.],
        'camera':{'serial':'test','Width':640,'Height':480,'OffsetX':0,'OffsetY':0},
        'projector':{'width':640,'height':480}}


class StructuredDepthTests(unittest.TestCase):
    def test_distant_depth_gap_separates_subject_from_shadow_background(self):
        near = np.linspace(760,945,2000)
        far = np.linspace(1022,1040,30)
        cutoff,bounds = depth_gap_cutoff(np.r_[near,far])
        self.assertGreater(cutoff,945)
        self.assertLess(cutoff,1022)
        self.assertEqual(bounds,[945.,1022.])
        self.assertEqual(depth_gap_cutoff(np.arange(100.,300.))[0],None)

    def test_direct_green_mask_preserves_isolated_parts_and_internal_holes(self):
        color = np.full((30,30,3),(20,150,20),np.uint8)
        color[5:25,5:25] = (100,80,180)
        color[10:15,10:15] = (20,150,20)
        color[2,2] = (100,80,180)
        color[28,28] = (15,25,15)
        mask,report = segment(color)
        self.assertEqual(mask[2,2],255)
        self.assertEqual(mask[5,5],255)
        self.assertEqual(mask[12,12],0)
        self.assertEqual(mask[28,28],0)
        self.assertEqual(report['method'],'direct_green_subtraction')

    def test_projector_origin_must_match_even_when_dimensions_match(self):
        canvas = SimpleNamespace(winfo_rootx=lambda:2560,winfo_rooty=lambda:0,
            winfo_width=lambda:1920,winfo_height=lambda:1080)
        monitor = {'x':2560,'y':0,'width':1920,'height':1080}
        self.assertEqual(projector_canvas_geometry(canvas,monitor),monitor)
        canvas.winfo_rootx = lambda:2662
        with self.assertRaisesRegex(RuntimeError,'shifted coordinates'):
            projector_canvas_geometry(canvas,monitor)

    def test_diagnostic_regions_exclude_green_boundary_without_cutting_cyan(self):
        color = np.full((50,80,3),(30,160,30),np.uint8)
        color[10:40,20:60] = (160,160,30)
        regions = comparison_regions(color)
        self.assertTrue(regions['green_background'][5,5])
        self.assertTrue(regions['non_green_candidate'][25,40])
        self.assertFalse(regions['non_green_candidate'][10,20])
        self.assertFalse(regions['green_background'][25,40])
        with self.assertRaisesRegex(ValueError,'mask must match'):
            comparison_regions(color,np.zeros((10,10),np.uint8))

    def test_distorted_rays_recover_metric_depth_and_reject_bad_matches(self):
        c = calibration()
        points = np.array([[20.,-30.,600.],[-40.,40.,800.],[20.,10.,-500.]])
        cam = cv2.projectPoints(points,np.zeros(3),np.zeros(3),np.array(c['camera_matrix']),np.array(c['camera_distortion']))[0].reshape(-1,2)
        proj = cv2.projectPoints(points,cv2.Rodrigues(np.array(c['R_camera_to_projector']))[0],np.array(c['t_camera_to_projector_mm']),np.array(c['projector_matrix']),np.array(c['projector_distortion']))[0].reshape(-1,2)
        xyz,valid,error = triangulate(cam,proj,c)
        np.testing.assert_allclose(xyz,points,atol=.001)
        np.testing.assert_array_equal(valid,[True,True,False])
        self.assertLess(error.max(),.001)
        proj[0,0] += 25
        self.assertFalse(triangulate(cam,proj,c)[1][0])
        diagnostics = {}
        triangulate(cam,proj,c,diagnostics=diagnostics)
        self.assertEqual(diagnostics['input_points'],3)
        self.assertEqual(diagnostics['kept_points'],1)
        self.assertLess(diagnostics['reprojection_pass'],3)
        self.assertEqual(triangulate([],[],c)[0].shape,(0,3))

    def test_crop_offset_depth_file_mask_and_ply_colors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); decoded = root/'decoded'; group = decoded/'group_0001'
            group.mkdir(parents=True); (decoded/'color_images').mkdir()
            c = calibration()
            # Small crop away from the optical center; principal point must remain full-frame.
            x0,y0,w,h = 330,200,20,20
            yy,xx = np.indices((h,w))
            cam = np.column_stack((xx.ravel()+x0,yy.ravel()+y0)).astype(float)
            normalized = cv2.undistortPoints(cam[:,None,:],np.array(c['camera_matrix']),np.array(c['camera_distortion'])).reshape(-1,2)
            xyz = np.column_stack((normalized,np.ones(w*h)))*700
            proj = cv2.projectPoints(xyz,cv2.Rodrigues(np.array(c['R_camera_to_projector']))[0],np.array(c['t_camera_to_projector_mm']),np.array(c['projector_matrix']),np.array(c['projector_distortion']))[0].reshape(h,w,2)
            np.savez_compressed(group/'decoded.npz',valid=np.ones((h,w),bool),projector_x=proj[:,:,0],projector_y=proj[:,:,1])
            save_image(decoded/'color_images/group_0001.png',np.full((h,w,3),(10,20,30),np.uint8))
            save_json(decoded/'decode_summary.json',{'camera':c['camera'],'projector':c['projector'],'crop_roi_xywh':[x0,y0,w,h],'groups':[{'group':'group_0001'}]})
            save_json(root/'calibration.json',c)
            masks = root/'masks'; masks.mkdir()
            mask = np.zeros((h,w),np.uint8); mask[:10] = 255
            save_image(masks/'group_0001.png',mask)
            output = root/'output'
            process(decoded,root/'calibration.json',output,masks)
            with np.load(output/'group_0001/depth_mm.npz') as depth:
                np.testing.assert_allclose(depth['depth_mm'][:10],700,atol=.01)
                self.assertTrue(np.isnan(depth['depth_mm'][10:]).all())
            vertices,_,attrs = read_ply(output/'group_0001/cloud.ply',with_attributes=True)
            self.assertEqual(len(vertices),200)
            self.assertEqual(attrs['red'][0],30)
            self.assertTrue(json.loads((output/'status.json').read_text())['complete'])
            np.savez_compressed(group/'decoded.npz',valid=np.zeros((h,w),bool),projector_x=proj[:,:,0],projector_y=proj[:,:,1])
            failed = root/'failed'
            with self.assertRaisesRegex(ValueError,'zero valid points'):
                process(decoded,root/'calibration.json',failed,masks)
            status = json.loads((failed/'status.json').read_text())
            self.assertTrue(status['failed'])
            self.assertFalse(status['complete'])
            self.assertFalse((failed/'group_0001/cloud.ply').exists())


if __name__=='__main__':
    unittest.main()
