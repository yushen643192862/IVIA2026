"""Manual-turntable Basler capture: complete Gray code, references and color."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scanner.capture import BaslerCamera, sharpness, white_balance, apply_white_balance, roi_pixels
from scanner.structured_light import capture_sequence_graycode, render_capture_pattern
from tools.project_patterns import enable_dpi_awareness, list_displays, create_projector_window, projector_canvas_geometry


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def save_image(path, image):
    ok,encoded = cv2.imencode('.png',image)
    if not ok:
        raise RuntimeError(f'Could not encode {path}')
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())


class CaptureApp:
    def __init__(self, camera, monitor, args, output):
        import tkinter as tk
        from tkinter import ttk
        self.tk,self.ttk = tk,ttk
        self.camera,self.monitor,self.args,self.output = camera,monitor,args,output
        self.motion = None
        if args.motion_adapter:
            from scanner.motion import MotionController
            self.motion = MotionController(args.motion_adapter)
        self.frames = capture_sequence_graycode(monitor['width'],monitor['height'],args.projector_level,args.cell_size)
        self.board = json.loads(args.board.read_text(encoding='utf-8'))
        corners = self.board.get('inner_corners',[])
        if len(corners)!=2 or any(type(n) is not int or n<3 for n in corners) or not float(self.board.get('square_size_mm',0))>0:
            raise ValueError('Invalid calibration board metadata')
        self.capture_kind = 'subject'
        self.board_detection = None
        restored = None
        if args.resume_session:
            restored = self.read_setup(output,camera.information(),monitor,self.frames)
        self.root,self.projector,self.projector_item = create_projector_window(monitor,'Basler structured-light capture - projector')
        self.control = tk.Toplevel(self.root)
        self.control.title('Basler capture | W white balance | C focus pattern | F confirm focus | S capture group | Esc exit')
        self.control.geometry('1160x850+20+20')
        self.preview = tk.Canvas(self.control,width=820,height=610,background='#151515',highlightthickness=0)
        self.preview.grid(row=0,column=0,rowspan=5,padx=8,pady=8)
        self.preview_item = self.preview.create_image(0,0,anchor='nw')
        self.crop_label = ttk.Label(self.control)
        self.crop_label.grid(row=0,column=1,padx=8,pady=8)
        self.metric = tk.StringVar(value='Waiting for first frame')
        ttk.Label(self.control,textvariable=self.metric,wraplength=290).grid(row=1,column=1,sticky='nw')
        self.calibration = tk.StringVar()
        ttk.Label(self.control,textvariable=self.calibration,wraplength=290).grid(row=2,column=1,sticky='nw')
        buttons = ttk.Frame(self.control); buttons.grid(row=3,column=1,sticky='nw')
        for label,callback in [('W: 确认已框选的白平衡',self.calibrate_white),('B: 重选白平衡区域',self.select_white_roi),('C: 棋盘格检查对焦',self.focus_pattern),
                               ('F: 确认当前清晰度',self.confirm_focus),(f'S: 已转好，采集{len(self.frames)}张',self.start_group)]:
            ttk.Button(buttons,text=label,command=callback).pack(fill='x',pady=4)
        settings = ttk.Frame(self.control); settings.grid(row=4,column=1,sticky='nw')
        ttk.Button(buttons,text='J: 联合标定拍照（实体棋盘格）',command=self.start_calibration).pack(fill='x',pady=4)
        information = camera.information()
        exposure = information.get('ExposureTime',information.get('ExposureTimeAbs',10000))
        gain = information.get('Gain',information.get('GainRaw',0))
        self.exposure_entry = tk.StringVar(value=str(exposure))
        self.gain_entry = tk.StringVar(value=str(gain))
        ttk.Label(settings,text='Exposure (us)').grid(row=0,column=0)
        ttk.Entry(settings,textvariable=self.exposure_entry,width=12).grid(row=0,column=1)
        ttk.Label(settings,text='Gain').grid(row=1,column=0)
        ttk.Entry(settings,textvariable=self.gain_entry,width=12).grid(row=1,column=1)
        ttk.Button(settings,text='应用曝光/增益',command=self.apply_settings).grid(row=2,column=0,columnspan=2,pady=5)
        self.message = tk.StringVar(value='步骤1：在预览上拖框选择白墙，按W确认白平衡；然后再框选手办保存区域。')
        ttk.Label(self.control,textvariable=self.message,wraplength=1120).grid(row=5,column=0,columnspan=2,padx=10,pady=10,sticky='w')
        ttk.Label(self.control,text=f'{information["model"]} | SN {information["serial"]}\n{output}',wraplength=1120).grid(row=6,column=0,columnspan=2,padx=10,sticky='w')
        self.roi = None
        self.white_roi = None
        self.object_roi = None
        self.selection_stage = 'white'
        self.crop_locked = False
        self.preview_transform = None
        self.drag_start = None
        self.last = None
        self.gains = None
        self.focus_baseline = None
        self.focus_ready = False
        self.group_index = 0
        self.group_attempt = 0
        self.busy = False
        self.closing = False
        self.pressed = set()
        self.current_group = None
        self.current_records = []
        self.current_role = 'white'
        self.frame_at = time.monotonic()
        self.group_configuration = None
        self.preflight_metrics = None
        self.events = (output/'events.jsonl').open('a' if restored else 'x',encoding='utf-8',buffering=1)
        if restored:
            self.white_roi = tuple(restored['white']['roi'])
            self.gains = restored['white']['gains_bgr']
            self.object_roi = tuple(restored['crop']['roi_xywh'])
            self.roi = self.object_roi
            self.selection_stage = 'object'
            self.focus_baseline = restored['focus']['metrics']['laplacian_variance']
            self.focus_ready = True
            self.group_index = restored['complete_count']
            self.group_attempt = restored['max_attempt']
            self.crop_locked = self.group_attempt>0
        self.preview.bind('<ButtonPress-1>',self.begin_roi)
        self.preview.bind('<B1-Motion>',self.drag_roi)
        self.preview.bind('<ButtonRelease-1>',self.end_roi)
        self.root.bind_all('<KeyPress>',self.key_press)
        self.root.bind_all('<KeyRelease>',lambda event:self.pressed.discard(event.keysym.lower()))
        self.control.protocol('WM_DELETE_WINDOW',self.close)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.show_white()
        self.update_calibration()
        if not restored:
            save_json(output/'session.json',{'camera':information,'projector':monitor,'frames':self.frames,
            'settle_ms':args.settle_ms,'projector_level':args.projector_level,
            'method':'complete Gray code: horizontal then vertical, dense to sparse, each bit and inverse; raw black/white references; separate white color frame',
            'expected_images':len(self.frames),'cell_size':args.cell_size,
            'note':'Manual rotation between groups. No motion/turntable control. Software settling is not hardware exposure synchronization. '
                   'Depth needs camera/projector calibration, confidence-filtered Gray-code decoding and triangulation.'})
        else:
            self.log('session_resumed',completed_groups=self.group_index)
            self.set_message('已恢复白平衡、裁剪区域和对焦基准。所有导出照片应用白平衡，raw目录保留测深原图；停稳后按S继续下一组。')
        self.root.after(100,self.preview_tick)
        if self.motion and self.focus_ready and self.group_index < args.auto_groups:
            self.root.after(500, self.start_group)
        self.control.lift(); self.control.focus_force()

    @staticmethod
    def read_setup(output, camera, monitor, frames):
        session = json.loads((output/'session.json').read_text(encoding='utf-8'))
        if session['frames']!=frames or session['projector']!=monitor:
            raise ValueError('Session patterns/projector changed; start a new session')
        white = json.loads((output/'white_balance.json').read_text(encoding='utf-8'))
        if white['camera']!=camera:
            raise ValueError('Camera settings differ from white-balance calibration; start a new session')
        gains = np.asarray(white['gains_bgr'],float)
        if gains.shape!=(3,) or not np.isfinite(gains).all() or np.any(gains<=0):
            raise ValueError('Invalid saved white-balance gains')
        crop = json.loads((output/'capture_roi.json').read_text(encoding='utf-8'))
        focus = json.loads((output/'focus_check.json').read_text(encoding='utf-8'))
        size = [camera['Width'],camera['Height']]
        if crop['sensor_image_size']!=size or list(focus['roi'])!=list(crop['roi_xywh']):
            raise ValueError('Saved crop/focus geometry is inconsistent')
        for roi in (white['roi'],crop['roi_xywh']):
            x,y,w,h = roi
            if min(x,y)<0 or min(w,h)<16 or x+w>size[0] or y+h>size[1]:
                raise ValueError('Saved ROI is outside the camera image')
        complete,attempt = 0,0
        for group in output.glob('group_*'):
            if not group.is_dir() or not group.name[6:].isdigit():
                continue
            attempt = max(attempt,int(group.name[6:]))
            path = group/'group.json'
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                complete += bool(record.get('complete'))
        return {'white':white,'crop':crop,'focus':focus,'complete_count':complete,'max_attempt':attempt}

    def log(self, event, **extra):
        self.events.write(json.dumps({'event':event,'utc':datetime.now(timezone.utc).isoformat(),
                                      'monotonic':time.perf_counter(),**extra})+'\n')

    def set_message(self, text):
        self.message.set(text); print(text,flush=True)

    def project(self, pixels, role):
        h,w = pixels.shape
        if (w,h)!=(self.monitor['width'],self.monitor['height']):
            raise ValueError('Projection pattern size differs from monitor; rescaling is forbidden')
        data = f'P6\n{w} {h}\n255\n'.encode()+pixels[:,:,None].repeat(3,axis=2).tobytes()
        photo = self.tk.PhotoImage(data=data,format='PPM')
        self.projector.itemconfigure(self.projector_item,image=photo)
        self.projector.photo = photo
        self.root.update_idletasks()
        self.projector_geometry = projector_canvas_geometry(self.projector,self.monitor)
        self.current_role = role
        self.frame_at = time.monotonic()
        self.log('pattern_submitted',role=role,canvas_geometry=self.projector_geometry,pattern_image_size=[w,h])

    def show_white(self):
        self.project(np.full((self.monitor['height'],self.monitor['width']),self.args.projector_level,np.uint8),'white')

    def show_checkerboard(self):
        h,w = self.monitor['height'],self.monitor['width']
        yy,xx = np.indices((h,w))
        pixels = (((xx//self.args.focus_square+yy//self.args.focus_square)%2)*self.args.projector_level).astype(np.uint8)
        self.project(pixels,'focus_checkerboard')

    def focus_pattern(self):
        if self.busy:
            return
        if self.gains is None or self.object_roi is None:
            self.set_message('先框选白墙并按W，再框选手办保存区域。'); return
        self.show_checkerboard()
        self.set_message('调节镜头对焦环，观察右侧原像素局部图和清晰度；满意后按F。ROI应包含条纹和手办表面。')

    def photo(self, image):
        h,w = image.shape[:2]
        return self.tk.PhotoImage(data=f'P6\n{w} {h}\n255\n'.encode()+cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes(),format='PPM')

    def update_preview(self, image):
        if self.roi is None:
            h,w = image.shape[:2]
            self.roi = (int(w*.35),int(h*.35),int(w*.3),int(h*.3))
        x,y,w,h = self.roi
        metrics = sharpness(image,self.roi)
        self.metric.set(f'ROI Laplacian: {metrics["laplacian_variance"]:.1f}\nContrast: {metrics["contrast_5_95"]:.1f}\nClipped: {metrics["clipped_fraction"]:.1%}\n右侧为ROI中心的原像素局部图（最多280×280）。')
        # Show a real 1:1 crop, not an enlarged low-resolution preview.
        cx,cy = x+w//2,y+h//2
        display = apply_white_balance(image,self.gains) if self.gains is not None else image
        crop = display[max(y,cy-140):min(y+h,cy+140),max(x,cx-140):min(x+w,cx+140)]
        photo = self.photo(crop); self.crop_label.configure(image=photo); self.crop_label.photo = photo
        factor = min(820/image.shape[1],610/image.shape[0])
        size = (round(image.shape[1]*factor),round(image.shape[0]*factor))
        small = cv2.resize(display,size,interpolation=cv2.INTER_AREA)
        preview = self.photo(small)
        self.preview.itemconfigure(self.preview_item,image=preview); self.preview.photo = preview
        self.preview.delete('roi')
        if self.object_roi is not None:
            ox,oy,ow,oh = self.object_roi
            self.preview.create_rectangle(ox*factor,oy*factor,(ox+ow)*factor,(oy+oh)*factor,outline='yellow',width=2,tags='roi')
        if self.white_roi is not None:
            wx,wy,ww,wh = self.white_roi
            self.preview.create_rectangle(wx*factor,wy*factor,(wx+ww)*factor,(wy+wh)*factor,outline='cyan',tags='roi')
        self.preview_transform = factor
        self.last = (image,metrics,time.monotonic(),self.current_role)

    def preview_tick(self):
        if self.closing:
            return
        try:
            if not self.busy and time.monotonic()-self.frame_at>=self.args.settle_ms/1000:
                image,_ = self.camera.grab()
                self.update_preview(image)
        except Exception as exc:
            self.set_message(f'相机预览错误：{exc}')
        self.root.after(150,self.preview_tick)

    def begin_roi(self, event):
        if self.crop_locked:
            self.set_message('采集区域已固定；本会话所有角度使用同一裁剪区域。需要改变请新开会话。'); return
        if not self.busy and self.preview_transform:
            self.drag_start = (event.x,event.y)
            self.preview.focus_set()

    def drag_roi(self, event):
        if self.drag_start is None or self.busy:
            return
        self.preview.delete('selection')
        self.preview.create_rectangle(*self.drag_start,event.x,event.y,
            outline='cyan' if self.selection_stage=='white' else 'yellow',width=2,tags='selection')

    def end_roi(self, event):
        if self.drag_start is None or self.last is None or self.busy or self.crop_locked:
            return
        factor = self.preview_transform
        h,w = self.last[0].shape[:2]
        x0,y0 = self.drag_start; self.drag_start = None
        self.preview.delete('selection')
        xa,xb = sorted((max(0,min(w,int(x0/factor))),max(0,min(w,int(event.x/factor)))))
        ya,yb = sorted((max(0,min(h,int(y0/factor))),max(0,min(h,int(event.y/factor)))))
        if min(xb-xa,yb-ya)>=16:
            selected = (xa,ya,xb-xa,yb-ya)
            if self.selection_stage=='white':
                self.white_roi = selected
                self.roi = selected
                self.set_message('白墙区域已选中（青框），按W完成白平衡，然后再框选手办。')
            else:
                self.object_roi = selected
                self.roi = selected
                self.log('capture_roi_selected',roi=selected)
                save_json(self.output/'capture_roi.json',{
                    'roi_xywh':selected,'sensor_image_size':[w,h],
                    'saved_image_size':[selected[2],selected[3]],
                    'coordinate_mapping':'u_original=u_saved+x; v_original=v_saved+y',
                    'intrinsics_adjustment':'fx/fy unchanged; cx_saved=cx_original-x; cy_saved=cy_original-y',
                    'note':'Fixed software crop, not a change to the camera hardware ROI.'})
                self.set_message(f'手办保存区域已选中（黄框）。请留足旋转后尾巴/武器的范围；按C检查、F确认，S保存{len(self.frames)}张裁剪图。')
            self.focus_ready = False
            self.update_calibration()

    def update_calibration(self):
        region = f'{self.object_roi[2]}×{self.object_roi[3]}' if self.object_roi else '待框选'
        self.calibration.set(f'白平衡：{"已固定" if self.gains is not None else "待框选白墙并按W"}\n保存区域：{region}\n对焦：{"已确认" if self.focus_ready else "待确认"}\n已完成组：{self.group_index}')

    def select_white_roi(self):
        if self.busy or self.crop_locked:
            return
        self.selection_stage = 'white'
        self.white_roi = None
        self.gains = None
        self.focus_ready = False
        self.object_roi = None
        self.update_calibration()
        self.show_white()
        self.set_message('步骤1：重新拖框选择白墙，再按W。')

    def calibrate_white(self):
        if self.busy:
            return
        if self.crop_locked:
            self.set_message('已开始采集，白平衡固定；需要更改请新开会话。'); return
        if self.white_roi is None:
            self.set_message('先在预览画面拖框选择白墙，再按W。'); return
        self.busy = True
        self.show_white()
        self.set_message('正在等待白场稳定，再从你框选的白墙采样白平衡……')

        def finish():
            if self.closing:
                return
            try:
                image,_ = self.camera.grab()
                self.update_preview(image)
                self.gains = white_balance(image,self.white_roi)
                save_json(self.output/'white_balance.json',{'roi':self.white_roi,'gains_bgr':self.gains,
                    'camera':self.camera.information(),'method':'fixed software BGR gains from neutral upper wall or manually selected neutral ROI',
                    'applied_to':'all root PNG exports and previews; raw/ preserves unbalanced measurement patterns and references'})
                self.log('white_balance_calibrated',roi=self.white_roi,gains_bgr=self.gains)
                self.selection_stage = 'object'
                self.object_roi = None
                self.focus_ready = False
                self.set_message('步骤2：白平衡已完成。现在拖框选手办保存区域（留足旋转范围），然后按C/F检查对焦。')
            except Exception as exc:
                self.gains = None; self.selection_stage = 'white'; self.object_roi = None
                self.focus_ready = False; self.set_message(f'白平衡失败：{exc}')
            finally:
                self.busy = False; self.update_calibration()
        self.root.after(self.args.settle_ms,finish)

    def confirm_focus(self):
        if self.busy:
            return
        if self.gains is None or self.object_roi is None:
            self.set_message('先完成白墙白平衡，再框选手办区域。'); return
        if self.current_role!='focus_checkerboard':
            self.focus_pattern(); return
        if self.last is None or self.last[3]!='focus_checkerboard' or time.monotonic()-self.last[2]>2:
            self.set_message('等待棋盘格的新预览帧后，再按F。'); return
        metrics = sharpness(self.last[0],self.roi)
        if metrics['laplacian_variance']<self.args.min_sharpness or metrics['contrast_5_95']<20:
            self.set_message('清晰度/对比度不足：选择条纹区域并调整镜头或投影对焦。'); return
        if metrics['clipped_fraction']>self.args.max_clipped:
            self.set_message('ROI过曝，请降低曝光或增益，再校准。'); return
        self.focus_baseline = metrics['laplacian_variance']
        self.focus_ready = True
        save_json(self.output/'focus_check.json',{'roi':self.roi,'metrics':metrics,
            'relative_floor':self.args.focus_ratio,'minimum':self.args.min_sharpness,
            'note':'Operator confirms focus after viewing 1:1 crop. Sharpness is a heuristic, not proof of absolute focus.'})
        self.log('focus_confirmed',roi=self.roi,metrics=metrics)
        self.update_calibration()
        self.show_white()
        if self.motion and self.group_index < self.args.auto_groups:
            self.root.after(500, self.start_group)
        self.set_message('对焦已确认。手动旋转并等稳定后按S；采集期间不要移动物体。')

    def apply_settings(self):
        if self.busy or self.crop_locked:
            self.set_message('采集中或已保存数据时固定曝光/增益；更改请新开会话。'); return
        try:
            self.camera.set_exposure_gain(float(self.exposure_entry.get()),float(self.gain_entry.get()))
            self.gains = None; self.focus_ready = False
            self.selection_stage = 'white'; self.object_roi = None
            self.update_calibration(); self.set_message('曝光/增益已修改。先确认白墙区域并W，再框选手办、C/F检查对焦。')
        except Exception as exc:
            self.set_message(str(exc))

    def start_group(self):
        if self.busy:
            return
        if self.gains is None or self.object_roi is None or not self.focus_ready:
            self.set_message('先框选白墙并W，再框选手办并C/F检查对焦，最后按S。'); return
        self.capture_kind = 'subject'
        self.busy = True
        self.show_checkerboard()
        self.set_message('采集前复查清晰度，请保持物体静止……')
        self.root.after(self.args.settle_ms,self.preflight)

    def start_calibration(self):
        if self.busy:
            return
        if self.gains is None:
            self.set_message('请先框选白墙并按W校准白平衡，再按J拍标定板。'); return
        self.capture_kind = 'calibration'
        self.busy = True
        self.show_white()
        self.set_message('联合标定：白光检查实体棋盘格。请固定标定板，整个序列不能移动。')
        self.root.after(self.args.settle_ms,self.calibration_preflight)

    def detect_board(self, image):
        size = tuple(int(n) for n in self.board['inner_corners'])
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        found,corners = cv2.findChessboardCornersSB(gray,size,flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            raise ValueError(f'未识别到完整 {size[0]}×{size[1]} 内角点；请让实体棋盘格完整入镜、避免反光并保持清晰')
        return corners.reshape(-1,2)

    def calibration_preflight(self):
        if self.closing:
            return
        try:
            image,_ = self.camera.grab()
            self.update_preview(image)
            corners = self.detect_board(image)
            x,y,w,h = cv2.boundingRect(corners.astype(np.float32))
            metrics = sharpness(image,(x,y,w,h))
            if metrics['laplacian_variance']<self.args.min_sharpness:
                raise ValueError('实体棋盘格清晰度不足，请检查相机对焦')
            directory = self.output/'calibration'
            directory.mkdir(exist_ok=True)
            configuration = {'camera':self.camera.information(),'projector':self.monitor,
                'board':self.board,'frames':self.frames,'settle_ms':self.args.settle_ms}
            manifest = directory/'session.json'
            if manifest.exists():
                if json.loads(manifest.read_text(encoding='utf-8'))!=configuration:
                    raise ValueError('标定期间相机、投影或棋盘格配置已改变，请使用新的采集会话')
            else:
                save_json(manifest,configuration)
            attempts = [int(p.name[5:]) for p in directory.glob('pose_*') if p.is_dir() and p.name[5:].isdigit()]
            self.current_group = directory/f'pose_{max(attempts,default=0)+1:04d}'
            self.current_group.mkdir(exist_ok=False)
            self.current_records = []
            self.group_configuration = self.camera.information()
            self.preflight_metrics = metrics
            self.board_detection = {'board':self.board,'corners_before_xy':corners.tolist(),
                'quality_roi_xywh':[x,y,w,h],'coordinate_system':'full camera image; no subject crop',
                'object_points_mm':(np.pad(np.mgrid[0:self.board['inner_corners'][0],0:self.board['inner_corners'][1]].T.reshape(-1,2),((0,0),(0,1)))*float(self.board['square_size_mm'])).tolist()}
            preview = apply_white_balance(image,self.gains)
            cv2.drawChessboardCorners(preview,tuple(self.board['inner_corners']),corners.reshape(-1,1,2),True)
            save_image(self.current_group/'board_corners_before.png',preview)
            self.write_group(False)
            self.next_frame(0)
        except Exception as exc:
            if self.current_group is not None:
                self.write_group(False,str(exc))
                self.current_group = None
            self.busy = False
            self.set_message(f'联合标定未开始：{exc}。摆好标定板后再按J。')

    def is_calibration(self):
        return getattr(self,'capture_kind','subject')=='calibration'

    def saved_roi(self):
        if self.is_calibration():
            return (0,0,self.group_configuration['Width'],self.group_configuration['Height'])
        return self.object_roi

    def preflight(self):
        if self.closing:
            return
        try:
            image,_ = self.camera.grab(); self.update_preview(image)
            metrics = sharpness(image,self.roi)
            floor = max(self.args.min_sharpness,self.focus_baseline*self.args.focus_ratio)
            if metrics['laplacian_variance']<floor or metrics['contrast_5_95']<20:
                raise ValueError('当前角度清晰度低于检查基准，请C/F重新检查或选择更合适的ROI')
            if metrics['clipped_fraction']>self.args.max_clipped:
                raise ValueError('当前ROI过曝，采集未开始')
            self.group_attempt += 1
            self.current_group = self.output/f'group_{self.group_attempt:04d}'
            self.current_group.mkdir(exist_ok=False)
            self.current_records = []
            self.group_configuration = self.camera.information()
            self.crop_locked = True
            self.preflight_metrics = metrics
            self.write_group(False)
            self.next_frame(0)
        except Exception as exc:
            self.busy = False; self.focus_ready = False; self.update_calibration()
            self.set_message(f'本组未开始：{exc}')

    def next_frame(self, index):
        if self.closing:
            return
        frame = self.frames[index]
        pixels = render_capture_pattern(frame,self.monitor['width'],self.monitor['height'])
        self.project(pixels,frame['name'])
        self.set_message(f'{self.current_group.name}：等待图案稳定 {index+1}/{len(self.frames)}，保持物体静止')
        submitted = datetime.now(timezone.utc).isoformat()
        self.root.after(self.args.settle_ms,lambda:self.capture_frame(index,submitted))

    def capture_frame(self, index, submitted):
        if self.closing:
            return
        try:
            frame = self.frames[index]
            actual = self.camera.information()
            if actual!=self.group_configuration:
                raise RuntimeError('Camera settings changed during sequence')
            image,metadata = self.camera.grab()
            quality_roi = self.board_detection['quality_roi_xywh'] if self.is_calibration() else self.roi
            metrics = sharpness(image,quality_roi)
            self.update_preview(image)
            if metrics['clipped_fraction']>self.args.max_clipped:
                raise ValueError('ROI过曝，本组标记为未完成；请检查曝光')
            filename = frame['name']+'.png'
            cropped = roi_pixels(image,self.saved_roi()).copy()
            if self.is_calibration() and frame.get('purpose')=='color':
                corners = self.detect_board(image)
                before = np.asarray(self.board_detection['corners_before_xy'])
                # A detector can reverse the ordering of a symmetric chessboard.
                if np.mean(np.linalg.norm(corners[::-1]-before,axis=1))<np.mean(np.linalg.norm(corners-before,axis=1)):
                    corners = corners[::-1].copy()
                motion = float(np.max(np.linalg.norm(corners-before,axis=1)))
                self.board_detection.update(corners_after_xy=corners.tolist(),max_corner_motion_px=motion)
                if motion>1.0:
                    raise ValueError(f'标定板在采集前后移动了 {motion:.2f} 像素，本姿态无效；请固定标定板后重拍')
            raw_file = None
            if frame.get('purpose')!='color':
                raw_directory = self.current_group/'raw'
                raw_directory.mkdir(exist_ok=True)
                raw_file = 'raw/'+filename
                save_image(self.current_group/raw_file,cropped)
            saved = apply_white_balance(cropped,self.gains)
            save_image(self.current_group/filename,saved)
            self.current_records.append({'file':filename,'pattern':frame,
                'projector_canvas_geometry':getattr(self,'projector_geometry',None),
                'pattern_submitted_utc':submitted,'software_settle_ms':self.args.settle_ms,
                'software_white_balance_applied':True,'raw_file':raw_file,'quality':metrics,**metadata})
            self.write_group(False)
            self.log('image_saved',group=self.current_group.name,file=filename)
            if index+1<len(self.frames):
                self.root.after(10,lambda:self.next_frame(index+1))
            else:
                self.write_group(True)
                if not self.is_calibration():
                    self.group_index += 1
                self.log('group_completed',group=self.current_group.name,count=len(self.frames))
                self.current_group = None; self.busy = False
                self.update_calibration(); self.show_white()
                if not self.is_calibration() and self.motion and self.args.auto_groups:
                    self.root.after(100, self.advance_automatic)
                if self.is_calibration():
                    self.set_message(f'联合标定姿态已保存{len(self.frames)}张。改变标定板倾角和距离，固定后再按J；建议15～20个姿态。')
                else:
                    self.set_message(f'第{self.group_index}组{len(self.frames)}张已保存。现在可以手动旋转，停稳后再按S。')
        except Exception as exc:
            if self.current_group:
                self.write_group(False,str(exc))
                self.current_group = None
            self.busy = False; self.focus_ready = False; self.update_calibration()
            retry = '固定标定板后按J重拍。' if self.is_calibration() else '重新C/F检查后可按S另存新组。'
            self.set_message(f'本组失败，部分数据标为未完成：{exc}。{retry}')

    def advance_automatic(self):
        if self.closing or self.busy:
            return
        if self.group_index >= self.args.auto_groups:
            self.log('automatic_sweep_completed', groups=self.group_index)
            self.close()
            return
        try:
            self.motion.rotate(self.args.angle_step)
            self.log('turntable_rotated', degrees=self.args.angle_step,
                     next_group=self.group_index + 1)
        except Exception as exc:
            self.set_message(f'转台移动失败，自动采集已停止：{exc}')
            self.log('motion_failed', error=str(exc))
            return
        self.root.after(self.args.settle_ms, self.start_group)

    def write_group(self, complete, error=None):
        crop = self.saved_roi()
        save_json(self.current_group/'group.json',{'complete':complete,'expected_images':len(self.frames),
            'capture_type':'camera_projector_calibration' if self.is_calibration() else 'subject',
            'calibration_board':self.board_detection if self.is_calibration() else None,
            'measurement_image_source':'raw/; root PNGs are white-balanced viewing/color exports',
            'saved_images':len(self.current_records),'camera':self.group_configuration,
            'projector':self.monitor,'roi':self.roi,'white_balance_bgr':self.gains,
            'white_balance_roi':self.white_roi,'crop_roi_xywh':crop,
            'sensor_image_size':[self.group_configuration['Width'],self.group_configuration['Height']],
            'saved_image_size':[crop[2],crop[3]],
            'coordinate_mapping':'u_original=u_saved+crop_x; v_original=v_saved+crop_y',
            'intrinsics_adjustment':'cx_saved=cx_original-crop_x; cy_saved=cy_original-crop_y',
            'preflight':self.preflight_metrics,'frames':self.current_records,'error':error})

    def key_press(self, event):
        key = event.keysym.lower()
        if key in self.pressed:
            return
        self.pressed.add(key)
        if key=='escape':
            self.close(); return
        if event.widget.winfo_class() in ('Entry','TEntry'):
            return
        action = {'w':self.calibrate_white,'b':self.select_white_roi,'c':self.focus_pattern,'f':self.confirm_focus,'s':self.start_group,'j':self.start_calibration}
        if key in action:
            action[key]()

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.current_group is not None:
            self.write_group(False,'Operator closed the application')
        self.log('closed'); self.events.close()
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if not self.closing:
                self.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list-cameras',action='store_true')
    p.add_argument('--check',action='store_true',help='Read one camera frame without projector/GUI')
    p.add_argument('--serial')
    p.add_argument('--display',type=int,help='Windows DISPLAY number; default is the sole secondary screen')
    p.add_argument('--output',type=Path)
    p.add_argument('--resume-session',type=Path,help='Continue an existing session with identical camera/projector settings')
    p.add_argument('--board',type=Path,default=ROOT/'outputs/structured_light/calibration_board/board.json',help='Printed calibration board metadata (inner_corners, square_size_mm)')
    p.add_argument('--exposure-us',type=float)
    p.add_argument('--gain',type=float)
    p.add_argument('--settle-ms',type=int,default=600)
    p.add_argument('--projector-level',type=int,default=220)
    p.add_argument('--cell-size',type=int,default=1,help='Gray-code cell size in projector pixels; 1=full resolution')
    p.add_argument('--focus-square',type=int,default=32)
    p.add_argument('--min-sharpness',type=float,default=20)
    p.add_argument('--focus-ratio',type=float,default=.4)
    p.add_argument('--max-clipped',type=float,default=.05)
    p.add_argument('--motion-adapter',type=Path,help='Python module defining rotate(degrees), slide(millimeters), optional wait_idle()')
    p.add_argument('--auto-groups',type=int,default=0,help='Capture this many groups automatically after white balance and focus checks')
    p.add_argument('--angle-step',type=float,default=10.,help='Signed turntable rotation between automatic groups')
    a = p.parse_args()
    if a.auto_groups < 0 or (a.auto_groups and not a.motion_adapter):
        p.error('--auto-groups requires a positive count and --motion-adapter')
    if a.resume_session and (a.output or a.check):
        p.error('--resume-session cannot be combined with --output or --check')
    if a.settle_ms<100 or a.focus_square<2 or not 0<a.focus_ratio<=1 or not 0<a.max_clipped<1 or a.min_sharpness<=0:
        p.error('Invalid settling/quality parameters')
    capture_sequence_graycode(1,1,a.projector_level,a.cell_size)
    from pypylon import pylon
    found = list(pylon.TlFactory.GetInstance().EnumerateDevices())
    if a.list_cameras:
        print(json.dumps([{'model':d.GetModelName(),'serial':d.GetSerialNumber()} for d in found],indent=2)); return
    enable_dpi_awareness()
    monitor = None
    if not a.check:
        monitors = list_displays()
        secondary = [m for m in monitors if not m['primary']]
        monitor = next((m for m in monitors if m['id']==a.display),None) if a.display is not None else (secondary[0] if len(secondary)==1 else None)
        if monitor is None:
            p.error(f'Select a projector with --display. Available: {monitors}')
    camera = BaslerCamera(a.serial,a.exposure_us,a.gain)
    try:
        if a.check:
            image,metadata = camera.grab()
            result = {'camera':camera.information(),'image_shape':list(image.shape),'frame':metadata}
            if a.output:
                a.output.mkdir(parents=True,exist_ok=False)
                save_image(a.output/'camera_check.png',image)
                save_json(a.output/'camera_check.json',result)
            print(json.dumps(result,indent=2)); return
        output = (a.resume_session or a.output or ROOT/'outputs/structured_light/structured_capture'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
        if a.resume_session:
            if not output.is_dir():
                p.error('Resume session directory not found')
        else:
            output.mkdir(parents=True,exist_ok=False)
        print('Session:',output,flush=True)
        print('Camera:',camera.information(),flush=True)
        CaptureApp(camera,monitor,a,output).run()
    finally:
        camera.close()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
