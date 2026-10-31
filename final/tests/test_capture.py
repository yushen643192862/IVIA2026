from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import cv2
import numpy as np

from scanner.capture import sharpness, white_balance, apply_white_balance, find_white_wall_roi
from scanner.structured_light import capture_sequence_11, capture_sequence_graycode, render_capture_pattern
from tools.capture_structured_light import CaptureApp
from tools.balance_capture_session import balance_group


class CaptureTests(unittest.TestCase):
    def test_ten_phase_frames_recover_projector_coordinate(self):
        frames = capture_sequence_11()
        self.assertEqual(len(frames),11)
        self.assertEqual(sum(f['role']=='phase' for f in frames),10)
        self.assertEqual(frames[-1]['role'],'white')
        width = 1920
        decoded = []
        shifts = 2*np.pi*np.arange(5)/5
        for group in (frames[:5],frames[5:10]):
            images = np.array([render_capture_pattern(f,width,2)[0] for f in group],float)
            phase = np.mod(np.arctan2(-np.sum(images*np.sin(shifts)[:,None],axis=0),
                                      np.sum(images*np.cos(shifts)[:,None],axis=0)),2*np.pi)
            decoded.append(phase)
        position = (decoded[0]-np.pi/2)/np.pi
        fine = decoded[1]
        unwrapped = fine+2*np.pi*np.rint((2*np.pi*16*position-fine)/(2*np.pi))
        recovered = unwrapped/(2*np.pi*16)*width-.5
        self.assertLess(float(np.max(np.abs(recovered-np.arange(width)))),.1)

    def test_white_card_balance_and_blur_check(self):
        neutral = np.full((64,64,3),(90,120,150),np.uint8)
        gains = white_balance(neutral,(0,0,64,64))
        corrected = apply_white_balance(neutral,gains)
        np.testing.assert_array_equal(corrected[0,0],[120,120,120])
        with self.assertRaisesRegex(ValueError,'dark or clipped'):
            white_balance(np.full_like(neutral,255),(0,0,64,64))
        board = np.indices((64,64)).sum(axis=0)//4%2*180+30
        sharp = np.repeat(board[:,:,None],3,axis=2).astype(np.uint8)
        blurred = cv2.GaussianBlur(sharp,(11,11),3)
        self.assertGreater(sharpness(sharp,(0,0,64,64))['laplacian_variance'],
                           sharpness(blurred,(0,0,64,64))['laplacian_variance']*5)

    def test_white_wall_search_avoids_green_screen_and_clipped_white(self):
        image = np.full((256,384,3),(25,150,25),np.uint8)
        image[:40] = (150,170,190)
        x,y,w,h = find_white_wall_roi(image)
        self.assertLessEqual(y+h,40)
        gains = white_balance(image,(x,y,w,h))
        np.testing.assert_allclose(apply_white_balance(image,gains)[y,x],[170,170,170])
        image[:40] = 255
        with self.assertRaisesRegex(ValueError,'neutral wall'):
            find_white_wall_roi(image)

    def make_app(self, root, change_settings=False):
        class Camera:
            count = 0
            def information(self):
                return {'Width':32,'Height':32,'ExposureTime':1000 if self.count<5 or not change_settings else 2000}
            def grab(self):
                self.count += 1
                image = np.full((32,32,3),100,np.uint8)
                image[:,:,0] = np.arange(32)[None,:]+80
                image[:,:,1] = np.arange(32)[:,None]+80
                return image,{'camera_timestamp':self.count}
        app = CaptureApp.__new__(CaptureApp)
        app.camera = Camera()
        app.args = SimpleNamespace(settle_ms=600,max_clipped=.05)
        app.monitor = {'width':32,'height':32}
        app.root = SimpleNamespace(after=lambda delay,callback:callback())
        app.frames = capture_sequence_graycode(32,32)
        app.roi = (4,6,20,18)
        app.object_roi = app.roi
        app.white_roi = (0,0,16,16)
        app.gains = [1.5,1.,.5]
        app.closing = False; app.busy = True; app.focus_ready = True
        app.current_records = []; app.group_index = 0
        app.group_configuration = app.camera.information()
        app.preflight_metrics = {'laplacian_variance':100}
        app.current_group = root/'group_0001'; app.current_group.mkdir()
        app.project = lambda pixels,role:None
        app.set_message = lambda text:None
        app.log = lambda event,**kwargs:None
        app.update_preview = lambda image:None
        app.update_calibration = lambda:None
        app.show_white = lambda:None
        return app

    def test_group_balances_all_exports_and_preserves_measurement_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            app = self.make_app(path)
            app.next_frame(0)
            group = path/'group_0001'
            self.assertEqual(len(list(group.glob('*.png'))),len(app.frames))
            result = json.loads((group/'group.json').read_text(encoding='utf-8'))
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['frames']),len(app.frames))
            self.assertEqual(result['expected_images'],len(app.frames))
            pattern = cv2.imread(str(group/(app.frames[0]['name']+'.png')))
            self.assertEqual(pattern.shape,(18,20,3))
            np.testing.assert_array_equal(pattern[0,0],[126,86,50])
            raw = cv2.imread(str(group/'raw'/(app.frames[0]['name']+'.png')))
            np.testing.assert_array_equal(raw[0,0],[84,86,100])
            reference = cv2.imread(str(group/(app.frames[-2]['name']+'.png')))
            np.testing.assert_array_equal(reference[0,0],[126,86,50])
            color = cv2.imread(str(group/(app.frames[-1]['name']+'.png')))
            np.testing.assert_array_equal(color[0,0],[126,86,50])
            self.assertEqual(result['crop_roi_xywh'],[4,6,20,18])
            self.assertEqual(result['sensor_image_size'],[32,32])
            self.assertEqual(result['saved_image_size'],[20,18])
            self.assertTrue(result['frames'][0]['software_white_balance_applied'])
            self.assertTrue(result['frames'][-2]['software_white_balance_applied'])
            self.assertTrue(result['frames'][-1]['software_white_balance_applied'])
            self.assertEqual(len(list((group/'raw').glob('*.png'))),len(app.frames)-1)

    def test_changed_camera_settings_abort_group_without_marking_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            app = self.make_app(path,change_settings=True)
            app.next_frame(0)
            result = json.loads((path/'group_0001/group.json').read_text(encoding='utf-8'))
            self.assertFalse(result['complete'])
            self.assertEqual(result['saved_images'],5)
            self.assertIn('settings changed',result['error'])
            self.assertEqual(app.group_index,0)

    def test_calibration_keeps_full_sensor_and_rejects_board_motion(self):
        for displacement in (0.,2.):
            with self.subTest(displacement=displacement), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                app = self.make_app(path)
                app.capture_kind = 'calibration'
                corners = np.array([[8.,8.],[16.,8.],[8.,16.],[16.,16.]])
                app.board_detection = {'quality_roi_xywh':[0,0,32,32],
                    'corners_before_xy':corners.tolist()}
                app.detect_board = lambda image: corners+displacement
                app.next_frame(0)
                group = path/'group_0001'
                result = json.loads((group/'group.json').read_text(encoding='utf-8'))
                self.assertEqual(result['complete'],displacement==0)
                self.assertEqual(result['crop_roi_xywh'],[0,0,32,32])
                self.assertEqual(result['capture_type'],'camera_projector_calibration')
                self.assertEqual(app.object_roi,(4,6,20,18))
                self.assertEqual(app.group_index,0)
                image = cv2.imread(str(group/'raw'/(app.frames[0]['name']+'.png')))
                self.assertEqual(image.shape,(32,32,3))
                if displacement:
                    self.assertIn('移动',result['error'])

    def test_detect_physical_chessboard_and_reject_blank(self):
        app = CaptureApp.__new__(CaptureApp)
        app.board = {'inner_corners':[9,6],'square_size_mm':20}
        image = np.full((430,580,3),220,np.uint8)
        for y in range(7):
            for x in range(10):
                image[40+y*50:40+(y+1)*50,40+x*50:40+(x+1)*50] = 220 if (x+y)%2 else 20
        corners = app.detect_board(image)
        self.assertEqual(corners.shape,(54,2))
        with self.assertRaisesRegex(ValueError,'未识别'):
            app.detect_board(np.full_like(image,120))

    def test_white_selection_is_not_reused_as_capture_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            app = self.make_app(Path(temp))
            app.output = Path(temp)
            app.last = (np.zeros((100,100,3),np.uint8),None,None,None)
            app.preview_transform = 1
            app.preview = SimpleNamespace(delete=lambda tag:None)
            app.busy = False; app.crop_locked = False
            app.object_roi = None; app.white_roi = None
            app.selection_stage = 'white'
            app.drag_start = (3,4)
            app.end_roi(SimpleNamespace(x=40,y=30))
            self.assertEqual(app.white_roi,(3,4,37,26))
            self.assertIsNone(app.object_roi)
            app.selection_stage = 'object'
            app.drag_start = (20,25)
            app.end_roi(SimpleNamespace(x=90,y=95))
            self.assertEqual(app.white_roi,(3,4,37,26))
            self.assertEqual(app.object_roi,(20,25,70,70))
            saved = json.loads((Path(temp)/'capture_roi.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['roi_xywh'],[20,25,70,70])
            app.crop_locked = True
            app.drag_start = (0,0)
            app.end_roi(SimpleNamespace(x=40,y=40))
            self.assertEqual(app.object_roi,(20,25,70,70))

    def test_existing_exports_migrate_once_and_preserve_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp)
            source = np.full((18,20,3),(80,100,160),np.uint8)
            frames = []
            for name,purpose in [('01_pattern','coordinate'),('02_white_reference','shadow_reference'),('03_color','color')]:
                balanced = purpose=='color'
                image = apply_white_balance(source,[1.5,1,.5]) if balanced else source
                cv2.imwrite(str(group/(name+'.png')),image)
                frames.append({'file':name+'.png','pattern':{'purpose':purpose},
                               'software_white_balance_applied':balanced})
            first_bytes = (group/'01_pattern.png').read_bytes()
            (group/'group.json').write_text(json.dumps({'complete':True,'expected_images':3,
                'frames':frames,'white_balance_bgr':[1.5,1,.5]}),encoding='utf-8')
            self.assertEqual(balance_group(group,[1,1,1]),2)
            self.assertEqual((group/'raw/01_pattern.png').read_bytes(),first_bytes)
            np.testing.assert_array_equal(cv2.imread(str(group/'01_pattern.png'))[0,0],[120,100,80])
            np.testing.assert_array_equal(cv2.imread(str(group/'03_color.png'))[0,0],[120,100,80])
            self.assertEqual(balance_group(group,[1,1,1]),0)
            np.testing.assert_array_equal(cv2.imread(str(group/'01_pattern.png'))[0,0],[120,100,80])

    def test_resume_reuses_setup_but_rejects_camera_settings_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app = self.make_app(root)
            documents = {
                'session.json':{'frames':app.frames,'projector':app.monitor},
                'white_balance.json':{'camera':app.camera.information(),'roi':app.white_roi,'gains_bgr':app.gains},
                'capture_roi.json':{'roi_xywh':app.object_roi,'sensor_image_size':[32,32]},
                'focus_check.json':{'roi':app.roi,'metrics':{'laplacian_variance':100}},
                'group_0001/group.json':{'complete':True}}
            for name,data in documents.items():
                (root/name).write_text(json.dumps(data),encoding='utf-8')
            result = CaptureApp.read_setup(root,app.camera.information(),app.monitor,app.frames)
            self.assertEqual(result['complete_count'],1)
            self.assertEqual(result['max_attempt'],1)
            changed = dict(app.camera.information(),ExposureTime=2000)
            with self.assertRaisesRegex(ValueError,'Camera settings differ'):
                CaptureApp.read_setup(root,changed,app.monitor,app.frames)


if __name__=='__main__':
    unittest.main()
