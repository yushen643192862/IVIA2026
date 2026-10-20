"""Basler stereo preview. S/P saves a pair; W calibrates white; Q/Esc exits.

Connection and BGR conversion follow lab1/Lab1_Group4/Code/grab_basler1.1.py.
Uses two independent capture threads; acquisition is NOT hardware synchronized.
"""
import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np


def feature(camera, name, value, optional=False):
    from pypylon import genicam
    node = camera.GetNodeMap().GetNode(name)
    if node is None or not genicam.IsWritable(node):
        if optional:
            return False
        raise RuntimeError(f'Camera feature unavailable or read-only: {name}')
    node.SetValue(value)
    return True


def settings(camera, args):
    # Preserve resolution, ROI and current calibration-related settings.
    # Loading the default user set here could invalidate an existing calibration.
    feature(camera, 'ExposureAuto', 'Off', optional=True)
    feature(camera, 'GainAuto', 'Off', optional=True)
    feature(camera, 'BalanceWhiteAuto', 'Off', optional=True)
    feature(camera, 'ExposureMode', 'Timed', optional=True)
    feature(camera, 'TriggerSelector', 'FrameStart', optional=True)
    feature(camera, 'TriggerMode', 'Off')
    if args.exposure_us is not None:
        if not feature(camera, 'ExposureTime', args.exposure_us, optional=True):
            feature(camera, 'ExposureTimeAbs', args.exposure_us)
    if args.gain is not None:
        feature(camera, 'Gain', args.gain)


def camera_info(camera):
    from pypylon import genicam
    info = camera.GetDeviceInfo()
    result = dict(model=info.GetModelName(), serial=info.GetSerialNumber())
    for name in ('Width', 'Height', 'OffsetX', 'OffsetY', 'PixelFormat',
                 'ExposureTime', 'ExposureTimeAbs', 'Gain', 'Gamma'):
        node = camera.GetNodeMap().GetNode(name)
        if node is not None and genicam.IsReadable(node):
            result[name] = node.GetValue()
    return result


def synchronize_right(left, right):
    """Copy imaging settings in dependency order and verify actual readback.

    Do not copy commands, device identity, network/USB bandwidth settings or
    persistent user sets. Selector-dependent values must be copied separately.
    """
    from pypylon import genicam
    report = dict(copied=[], unavailable=[], mismatches=[])

    def node(camera, name):
        return camera.GetNodeMap().GetNode(name)

    def equal(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return bool(np.isclose(a, b, rtol=1e-7, atol=1e-8))
        return a == b

    def copy(name, context=''):
        source, target = node(left,name), node(right,name)
        label = context+name
        if source is None or not genicam.IsReadable(source):
            report['unavailable'].append(label)
            return
        expected = source.GetValue()
        if target is None or not genicam.IsReadable(target):
            report['mismatches'].append(dict(feature=label,left=expected,right='unavailable'))
            return
        try:
            if not equal(expected,target.GetValue()):
                if not genicam.IsWritable(target):
                    raise RuntimeError('Right feature is read-only')
                target.SetValue(expected)
            actual = target.GetValue()
            if not equal(expected,actual):
                raise RuntimeError(f'Readback differs: {actual}')
            report['copied'].append(dict(feature=label,value=actual))
        except Exception as exc:
            report['mismatches'].append(dict(feature=label,left=expected,error=str(exc)))

    def selected(selector, features):
        a, b = node(left,selector), node(right,selector)
        if a is None or not genicam.IsReadable(a):
            for name in features:
                copy(name)
            return
        if b is None or not genicam.IsWritable(b):
            report['mismatches'].append(dict(feature=selector,error='Right selector unavailable/read-only'))
            return
        original = a.GetValue()
        try:
            if genicam.IsWritable(a):
                for entry in a.GetEntries():
                    if not genicam.IsAvailable(entry):
                        continue
                    value = entry.GetSymbolic()
                    a.SetValue(value)
                    b.SetValue(value)
                    for name in features:
                        copy(name, f'{selector}={value}/')
            else:
                b.SetValue(original)
                for name in features:
                    copy(name,f'{selector}={original}/')
        finally:
            if genicam.IsWritable(a):
                a.SetValue(original)
            b.SetValue(original)

    # Free the right ROI before copying sampling modes and dimensions.
    for name in ('OffsetX','OffsetY','Width','Height'):
        n = node(right,name)
        if n is not None and genicam.IsWritable(n):
            n.SetValue(n.GetMin())
    for name in ('BinningHorizontalMode','BinningVerticalMode',
                 'BinningHorizontal','BinningVertical','DecimationHorizontal','DecimationVertical',
                 'PixelFormat','Width','Height','OffsetX','OffsetY','ReverseX','ReverseY',
                 'ExposureMode','ExposureAuto','ExposureTime','ExposureTimeAbs',
                 'GainAuto','BalanceWhiteAuto','GammaEnable','GammaSelector','Gamma',
                 'AcquisitionMode','AcquisitionFrameRateEnable','AcquisitionFrameRate',
                 'AcquisitionFrameRateAbs','LightSourcePreset','BslLightSourcePreset',
                 'ColorSpace','BslColorSpace','BslBrightness','BslContrast',
                 'BslSaturation','LUTEnable'):
        copy(name)
    for selector, features in (
        ('GainSelector',('Gain','GainRaw')),
        ('BlackLevelSelector',('BlackLevel','BlackLevelRaw')),
        ('BalanceRatioSelector',('BalanceRatio','BalanceRatioAbs')),
    ):
        selected(selector,features)
    copy('ColorTransformationSelector')
    copy('ColorTransformationEnable')
    selected('ColorTransformationValueSelector',('ColorTransformationValue',))
    selected('TriggerSelector',('TriggerMode',))
    # An enabled custom LUT cannot be reproduced by just copying its enable bit.
    lut = node(left,'LUTEnable')
    if lut is not None and genicam.IsReadable(lut) and lut.GetValue():
        report['mismatches'].append(dict(feature='LUTEnable',error='Custom LUT is enabled; table copying is not supported'))
    return report


class Stream:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            from pypylon import pylon
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.GetNodeMap().GetNode('OutputBitAlignment').SetValue('MsbAligned')
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self.stop.is_set():
                grab = self.camera.RetrieveResult(300, pylon.TimeoutHandling_Return)
                if grab is None:
                    continue
                converted = None
                try:
                    if not grab.GrabSucceeded():
                        raise RuntimeError(f'Grab failed: {grab.ErrorDescription}')
                    converted = converter.Convert(grab)
                    array = converted.GetArray().copy()
                    item = dict(image=array, received_monotonic=time.monotonic(),
                                received_at=datetime.now().astimezone().isoformat(timespec='milliseconds'),
                                camera_timestamp=int(grab.GetTimeStamp()),
                                block_id=int(grab.GetBlockID()))
                    with self.lock:
                        self.latest = item
                finally:
                    if converted is not None:
                        converted.Release()
                    grab.Release()
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

    def snapshot(self):
        with self.lock:
            return self.latest


def save_pair(session, pair, devices):
    # Use one shared index; never overwrite a file from an earlier capture.
    stamp = datetime.now().astimezone()
    used = [int(match.group(1)) for path in session.iterdir()
            if (match := re.fullmatch(r'[lr](\d+)\.bmp', path.name))]
    used += [int(path.stem.split('_')[1]) for path in session.glob('pair_*.json')
             if path.stem.split('_')[1].isdigit()]
    index = max(used, default=0)+1
    name = f'pair_{index}'
    pending = session / ('.pending_' + stamp.strftime('%Y%m%d_%H%M%S_%f'))
    pending.mkdir()
    committed = []
    try:
        for role, frame in zip(('l', 'r'), pair):
            ok, encoded = cv.imencode('.bmp', frame['image'])
            if not ok:
                raise RuntimeError(f'BMP encoding failed: {role}')
            with (pending / f'{role}{index}.bmp').open('xb') as f:
                f.write(encoded.tobytes())
        frames = [{k: v for k, v in item.items() if k not in ('image', 'received_monotonic')}
                  for item in pair]
        metadata = dict(saved_at=stamp.isoformat(), pair_index=index,
            files=[f'l{index}.bmp',f'r{index}.bmp'], cameras=devices, frames=frames,
            host_receive_difference_ms=abs(pair[0]['received_monotonic']-pair[1]['received_monotonic'])*1000,
            note='Latest displayed frames; independent free-running cameras. Host receive times and camera timestamps do not prove exposure synchronization.')
        (pending / f'{name}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        for filename in (f'l{index}.bmp',f'r{index}.bmp',f'{name}.json'):
            destination = session / filename
            if destination.exists():
                raise FileExistsError(destination)
            (pending / filename).rename(destination)
            committed.append(destination)
        pending.rmdir()
        return session / f'l{index}.bmp'
    except Exception:
        for path in committed:
            path.unlink()
        # Remove only files created for this uncommitted pair; never existing data.
        for path in pending.iterdir():
            path.unlink()
        pending.rmdir()
        raise


def white_balance_gains(pair):
    """Estimate fixed software BGR gains from the central neutral-paper patch."""
    medians = []
    for item in pair:
        image = item['image']
        h,w = image.shape[:2]
        patch = image[h*3//10:h*7//10,w*3//10:w*7//10]
        pixels = patch.reshape(-1,3)
        usable = pixels[(pixels.min(axis=1)>12)&(pixels.max(axis=1)<240)]
        if len(usable) < 100:
            raise ValueError('White patch too dark/bright. Adjust exposure and try W again.')
        medians.append(np.median(usable,axis=0))
    # Shared target makes the two neutral patches equally bright as well.
    target = float(np.mean(medians))
    gains = [target/np.maximum(values,1) for values in medians]
    if any(np.any((g < .5)|(g > 2.)) for g in gains):
        raise ValueError('Patch has strong color cast. Fill BOTH center boxes with neutral paper.')
    return [g.tolist() for g in gains]


def balanced_frame(item, gains):
    if item is None:
        return None
    adjusted = dict(item)
    adjusted['image'] = np.rint(np.clip(item['image'].astype(np.float32)*
                                np.asarray(gains,np.float32),0,255)).astype(np.uint8)
    adjusted['software_white_balance_bgr'] = list(gains)
    return adjusted


def panel(item, label, width):
    height = round(width*.75)
    canvas = np.full((height+48,width,3),25,np.uint8)
    cv.putText(canvas,label,(10,30),0,.6,(255,255,255),1)
    if item is None:
        cv.putText(canvas,'Waiting for frames...', (20,100),0,.7,(0,220,255),2)
    else:
        im = item['image']
        scale = min(width/im.shape[1],height/im.shape[0])
        small = cv.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)),interpolation=cv.INTER_AREA)
        y,x = 48+(height-small.shape[0])//2,(width-small.shape[1])//2
        canvas[y:y+small.shape[0],x:x+small.shape[1]]=small
        cv.rectangle(canvas,(x+small.shape[1]*3//10,y+small.shape[0]*3//10),
                     (x+small.shape[1]*7//10,y+small.shape[0]*7//10),(0,210,255),1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List camera serials without opening them')
    parser.add_argument('--check', action='store_true', help='Open both cameras and verify frames without a GUI')
    parser.add_argument('--left-serial')
    parser.add_argument('--right-serial')
    parser.add_argument('--swap', action='store_true', help='Swap the selected left/right roles')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'captures')
    parser.add_argument('--exposure-us', type=float)
    parser.add_argument('--gain', type=float, help='Gain value in the camera Gain feature units')
    parser.add_argument('--panel-width', type=int, default=640)
    args = parser.parse_args()
    if args.panel_width < 160:
        parser.error('--panel-width must be at least 160')
    from pypylon import pylon
    factory = pylon.TlFactory.GetInstance()
    found = sorted(factory.EnumerateDevices(),key=lambda d:d.GetSerialNumber())
    for device in found:
        print(device.GetModelName(),device.GetSerialNumber(),flush=True)
    if args.list:
        return 0
    if len(found) < 2:
        raise RuntimeError(f'Two Basler cameras are required; detected {len(found)}')
    if bool(args.left_serial) != bool(args.right_serial):
        parser.error('Specify both --left-serial and --right-serial')
    by_serial = {d.GetSerialNumber(): d for d in found}
    if args.left_serial:
        if args.left_serial == args.right_serial:
            parser.error('Left and right must be different cameras')
        if args.left_serial not in by_serial or args.right_serial not in by_serial:
            parser.error('Requested serial not found; run --list')
        chosen = [by_serial[args.left_serial],by_serial[args.right_serial]]
    else:
        chosen = found[:2]
        print('Roles assigned by serial order. Verify physical LEFT/RIGHT in preview; use --swap if needed.',flush=True)
    if args.swap:
        chosen.reverse()
    opened, streams = [], []
    window = 'Basler stereo | S/P save | W white balance | Q/Esc exit'
    try:
        for device in chosen:
            camera = pylon.InstantCamera(factory.CreateDevice(device))
            opened.append(camera)
            camera.Open()
            settings(camera,args)
        synchronization = synchronize_right(opened[0],opened[1])
        print('Right camera settings copied from LEFT:',
              json.dumps(synchronization,ensure_ascii=False,indent=2),flush=True)
        if synchronization['mismatches']:
            raise RuntimeError('Camera settings could not be matched; see mismatches above. Capture has not started.')
        info = [camera_info(c) for c in opened]
        for role in (0,1):
            info[role]['matched_imaging_settings'] = synchronization['copied']
        print('LEFT:',info[0], '\nRIGHT:',info[1],flush=True)
        for camera in opened:
            stream = Stream(camera)
            streams.append(stream)
            stream.thread.start()
        if args.check:
            deadline = time.monotonic()+15
            while time.monotonic()<deadline:
                if any(s.error for s in streams):
                    raise RuntimeError('; '.join(s.error for s in streams if s.error))
                pair = [s.snapshot() for s in streams]
                if all(item is not None for item in pair):
                    print('PASS: two cameras streaming:',[p['image'].shape for p in pair],flush=True)
                    return 0
                time.sleep(.03)
            raise RuntimeError('Timed out waiting for both cameras')
        session = args.output / datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
        session.mkdir(parents=True)
        (session/'session.json').write_text(json.dumps(dict(cameras=info,synchronization='free_running',
            settings_copy=synchronization),indent=2),encoding='utf-8')
        cv.namedWindow(window,cv.WINDOW_AUTOSIZE)
        cv.moveWindow(window,20,30)
        message = 'Click preview! S/P: save | W: white paper in BOTH center boxes | Q: quit'
        gains = [[1.,1.,1.],[1.,1.,1.]]
        last_saved = None
        print('Output:',session,flush=True)
        while True:
            if any(s.error for s in streams):
                raise RuntimeError('; '.join(s.error for s in streams if s.error))
            raw_pair = [s.snapshot() for s in streams]
            pair = [balanced_frame(item,g) for item,g in zip(raw_pair,gains)]
            canvases = [panel(item,f'{role} | SN {device["serial"]}',args.panel_width)
                        for item,role,device in zip(pair,('LEFT','RIGHT'),info)]
            display = np.hstack(canvases)
            footer = np.full((55,display.shape[1],3),25,np.uint8)
            cv.putText(footer,message,(10,32),0,.6,(0,255,220),1)
            cv.imshow(window,np.vstack((display,footer)))
            key = cv.waitKey(1)&0xff
            if key in (27,ord('q'),ord('Q')) or cv.getWindowProperty(window,cv.WND_PROP_VISIBLE)<1:
                break
            if key in (ord('w'),ord('W')):
                if any(item is None or time.monotonic()-item['received_monotonic']>1. for item in raw_pair):
                    message = 'WB not set: wait for fresh frames from BOTH cameras'
                    continue
                if last_saved is not None:
                    message = 'WB locked after first save; restart for a new pattern sequence'
                    continue
                try:
                    gains = white_balance_gains(raw_pair)
                    profile = dict(method='fixed software BGR gains from neutral center patches',
                                   serials=[c['serial'] for c in info],gains_bgr=gains,
                                   calibrated_at=datetime.now().astimezone().isoformat())
                    (session/'white_balance.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
                    message = 'WB calibrated and fixed. Remove paper; S/P saves l1.bmp + r1.bmp'
                    print('White balance:',profile,flush=True)
                except ValueError as exc:
                    message = str(exc)
                continue
            if key in (ord('p'),ord('P'),ord('s'),ord('S')):
                if any(item is None or time.monotonic()-item['received_monotonic']>1. for item in pair):
                    message = 'Not saved: waiting for fresh frames from BOTH cameras'
                    continue
                identity = tuple(item['received_monotonic'] for item in pair)
                if identity == last_saved:
                    continue
                destination = save_pair(session,pair,info)
                last_saved = identity
                message = f'SAVED {destination.name} + r{destination.stem[1:]}.bmp'
                print(message,flush=True)
        return 0
    finally:
        for stream in streams:
            stream.stop.set()
        for stream in streams:
            stream.thread.join(timeout=3)
        for camera in reversed(opened):
            try:
                if camera.IsOpen():
                    if camera.IsGrabbing():
                        camera.StopGrabbing()
                    camera.Close()
            except Exception as exc:
                print('Camera cleanup:',exc,file=sys.stderr)
        cv.destroyAllWindows()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
