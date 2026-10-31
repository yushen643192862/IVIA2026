"""Basler single-frame acquisition and observable capture quality checks."""
from datetime import datetime, timezone
import math

import cv2
import numpy as np


def roi_pixels(image, roi):
    x,y,w,h = map(int,roi)
    if min(w,h)<16 or x<0 or y<0 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('Select an ROI of at least 16x16 pixels inside the image')
    return image[y:y+h,x:x+w]


def sharpness(image, roi):
    gray = cv2.cvtColor(roi_pixels(image,roi),cv2.COLOR_BGR2GRAY)
    return {'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
            'contrast_5_95':float(np.percentile(gray,95)-np.percentile(gray,5)),
            'clipped_fraction':float(np.mean(gray>=250)),
            'dark_fraction':float(np.mean(gray<=5))}


def white_balance(image, roi):
    pixels = roi_pixels(image,roi).reshape(-1,3)
    usable = pixels[(pixels.min(axis=1)>15)&(pixels.max(axis=1)<240)]
    if len(usable)<max(100,len(pixels)*.6):
        raise ValueError('White/gray card too dark or clipped: adjust exposure and press W again')
    medians = np.median(usable,axis=0)
    gray = np.mean(usable,axis=1)
    if (np.percentile(gray,90)-np.percentile(gray,10))/max(float(np.median(gray)),1)>.45:
        raise ValueError('ROI is not uniform: fill it with a neutral white/gray card')
    gains = float(medians.mean())/medians
    if np.any((gains<.4)|(gains>2.5)):
        raise ValueError('Strong color cast: use a neutral card, not the green screen')
    return gains.tolist()


def apply_white_balance(image, gains):
    return np.rint(np.clip(image.astype(np.float32)*np.asarray(gains,np.float32),0,255)).astype(np.uint8)


def find_white_wall_roi(image):
    """Find an evenly lit neutral patch in the upper side regions, away from subject."""
    h,w = image.shape[:2]
    patch_w,patch_h = max(32,int(w*.08)),max(24,int(h*.06))
    best = None
    for cy in (.055,.12,.20,.28):
        for cx in (.10,.22,.78,.90):
            x,y = max(0,int(w*cx)-patch_w//2),max(0,int(h*cy)-patch_h//2)
            roi = (x,y,min(patch_w,w-x),min(patch_h,h-y))
            patch = roi_pixels(image,roi)
            hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            usable = (patch.min(axis=2)>35)&(patch.max(axis=2)<240)&(hsv[:,:,1]<95)
            if usable.mean()<.85:
                continue
            brightness = float(np.median(gray[usable]))
            spread = float(np.percentile(gray[usable],90)-np.percentile(gray[usable],10))
            if brightness<70 or spread>max(25,brightness*.18):
                continue
            saturation = float(np.median(hsv[:,:,1][usable]))
            score = brightness-spread*2-saturation
            if best is None or score>best[0]:
                best = (score,roi)
    if best is None:
        raise ValueError('No uniform, unclipped neutral wall patch found above. Adjust exposure or select a white-wall ROI manually.')
    return best[1]


class BaslerCamera:
    """Fresh GrabOne per exposure; preserve ROI/lens settings and avoid stale buffers."""
    def __init__(self, serial=None, exposure_us=None, gain=None):
        from pypylon import pylon,genicam
        self.pylon,self.genicam = pylon,genicam
        factory = pylon.TlFactory.GetInstance()
        devices = list(factory.EnumerateDevices())
        if serial:
            devices = [d for d in devices if d.GetSerialNumber()==serial]
        if len(devices)!=1:
            raise RuntimeError(f'Expected one selected Basler camera; found {len(devices)}. Use --list-cameras / --serial')
        self.camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        self.camera.Open()
        try:
            # Do not load a user set: doing so could change calibrated image geometry.
            for name in ('ExposureAuto','GainAuto','BalanceWhiteAuto'):
                self.set_feature(name,'Off',optional=True)
            self.set_feature('ExposureMode','Timed',optional=True)
            self.set_feature('TriggerSelector','FrameStart',optional=True)
            self.set_feature('TriggerMode','Off')
            self.set_feature('AcquisitionFrameRateEnable',False,optional=True)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            if exposure_us is not None or gain is not None:
                self.set_exposure_gain(exposure_us,gain)
        except Exception:
            self.close()
            raise

    def feature(self, name):
        return self.camera.GetNodeMap().GetNode(name)

    def set_feature(self, name, value, optional=False):
        node = self.feature(name)
        if node is None or not self.genicam.IsWritable(node):
            if optional:
                if node is not None and self.genicam.IsReadable(node) and node.GetValue()!=value:
                    raise RuntimeError(f'{name} is locked at {node.GetValue()}, expected {value}')
                return False
            raise RuntimeError(f'Camera setting is unavailable/read-only: {name}')
        node.SetValue(value)
        return True

    def first_feature(self, names):
        for name in names:
            node = self.feature(name)
            if node is not None and self.genicam.IsReadable(node):
                return name,node
        raise RuntimeError(f'Camera has none of these features: {names}')

    def set_exposure_gain(self, exposure_us=None, gain=None):
        for names,value in [(('ExposureTime','ExposureTimeAbs'),exposure_us),(('Gain','GainRaw'),gain)]:
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError('Exposure/gain must be finite')
            name,node = self.first_feature(names)
            if not self.genicam.IsWritable(node) or not node.GetMin()<=value<=node.GetMax():
                raise ValueError(f'{name} must be writable and within {node.GetMin()} .. {node.GetMax()}')
            self.set_feature(name,int(value) if name=='GainRaw' else value)

    def information(self):
        device = self.camera.GetDeviceInfo()
        info = {'model':device.GetModelName(),'serial':device.GetSerialNumber()}
        for name in ('Width','Height','OffsetX','OffsetY','PixelFormat','ExposureTime','ExposureTimeAbs',
                     'Gain','GainRaw','Gamma','ExposureAuto','GainAuto','BalanceWhiteAuto'):
            node = self.feature(name)
            if node is not None and self.genicam.IsReadable(node):
                info[name] = node.GetValue()
        return info

    def grab(self):
        _,exposure = self.first_feature(('ExposureTime','ExposureTimeAbs'))
        timeout = max(5000,math.ceil(float(exposure.GetValue())/1000)+3000)
        requested = datetime.now(timezone.utc).isoformat()
        result = self.camera.GrabOne(timeout,self.pylon.TimeoutHandling_ThrowException)
        converted = None
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(result.ErrorDescription)
            converted = self.converter.Convert(result)
            image = converted.GetArray().copy()
            metadata = {'host_request_utc':requested,
                        'host_received_utc':datetime.now(timezone.utc).isoformat(),
                        'camera_timestamp':int(result.GetTimeStamp()),'block_id':int(result.GetBlockID())}
            return image,metadata
        finally:
            if converted is not None:
                converted.Release()
            result.Release()

    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()
