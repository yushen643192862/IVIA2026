"""Scan exposure (microseconds) and gain (dB), saving one BMP per pair.

Dependencies: pip install pypylon opencv-python
Run this file to start acquisition. Press Ctrl+C to stop and keep saved images.
"""

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
from pypylon import genicam, pylon


default_cameraSettings = {
    'r_balance': 1,
    'g_balance': 1,
    'b_balance': 1,
    'PixelFormat': 'RGB8',
    'gamma': 1.0,
}

# 29 exposures x 24 gains = 696 images. Gain 0 is not included.
# Increase the step at 10,000 and 100,000 us; keep the 990,000 us endpoint.
EXPOSURE_TIMES_US = (
    10,
    *range(1000, 10001, 1000),
    *range(20000, 100001, 10000),
    *range(200000, 900001, 100000),
    990000,
)
GAIN_VALUES_DB = tuple(range(1, 25))


def OpenFirstCamera():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError('No Basler camera connected.')
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()
    return camera


def SetOptional(camera, name, value):
    """Set a model-dependent feature only when the camera provides it."""
    node = camera.GetNodeMap().GetNode(name)
    if node is not None and genicam.IsAvailable(node):
        node.SetValue(value)


def SetCamera(camera, cameraSettings):
    # Load defaults FIRST so they cannot overwrite the settings below.
    camera.UserSetSelector.SetValue('Default')
    camera.UserSetLoad.Execute()
    camera.PixelFormat.SetValue(cameraSettings['PixelFormat'])
    SetOptional(camera, 'ExposureMode', 'Timed')
    SetOptional(camera, 'AcquisitionFrameRateEnable', False)
    SetOptional(camera, 'TriggerMode', 'Off')
    camera.ExposureAuto.SetValue('Off')
    camera.GainAuto.SetValue('Off')

    camera.BalanceWhiteAuto.SetValue('Off')
    for channel, key in [('Red', 'r_balance'), ('Green', 'g_balance'),
                         ('Blue', 'b_balance')]:
        camera.BalanceRatioSelector.SetValue(channel)
        camera.BalanceRatio.SetValue(cameraSettings[key])

    SetOptional(camera, 'GammaEnable', True)
    SetOptional(camera, 'GammaSelector', 'User')
    camera.Gamma.SetValue(cameraSettings['gamma'])


def ValidateScan(camera, exposure_times, gains):
    """Reject unsupported ranges before collecting a partial dataset."""
    if not exposure_times or not gains:
        raise ValueError('The exposure and gain lists must not be empty.')
    for label, node, values in [
        ('ExposureTime (us)', camera.ExposureTime, exposure_times),
        ('Gain (dB)', camera.Gain, gains),
    ]:
        lower, upper = node.GetMin(), node.GetMax()
        print(f'{label}: camera supports {lower} .. {upper}')
        invalid = [value for value in values if not lower <= value <= upper]
        if invalid:
            raise ValueError(
                f'{label}: requested {invalid[0]} is outside [{lower}, {upper}]. '
                'Scan stopped; no values are silently clipped or skipped.'
            )


def SaveBmp(converter, result, path):
    converted = converter.Convert(result)
    try:
        ok, encoded = cv2.imencode('.bmp', converted.GetArray())
        if not ok:
            raise RuntimeError(f'BMP encoding failed: {path.name}')
        # Python file I/O supports Chinese paths and refuses to overwrite files.
        with path.open('xb') as output:
            output.write(encoded.tobytes())
    finally:
        converted.Release()


def ScanGrid(camera, save_root, exposure_times=EXPOSURE_TIMES_US,
             gains=GAIN_VALUES_DB):
    ValidateScan(camera, exposure_times, gains)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=False)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    total = len(exposure_times) * len(gains)
    exposure_hours = sum(exposure_times) * len(gains) / 1_000_000 / 3600
    print(f'Saving {total} images to: {save_root}')
    print(f'Exposure alone takes {exposure_hours:.2f} hours; '
          'transfer and disk writes take additional time.')
    print('Press Ctrl+C to stop. Completed images will be kept.')

    fields = ['filename', 'requested_exposure_us', 'requested_gain_db',
              'actual_exposure_us', 'actual_gain_db', 'gamma', 'pixel_format',
              'camera_model', 'serial_number', 'captured_at']
    info = camera.GetDeviceInfo()
    model = info.GetModelName()
    serial = info.GetSerialNumber()
    gamma = camera.Gamma.GetValue()
    pixel_format = camera.PixelFormat.GetValue()
    count = 0
    with (save_root / 'scan_parameters.csv').open(
            'x', newline='', encoding='utf-8-sig') as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        log.flush()
        for gain in gains:
            camera.Gain.SetValue(gain)
            actual_gain = camera.Gain.GetValue()
            for exposure in exposure_times:
                filename = f'e{exposure}g{gain}.bmp'
                try:
                    # GrabOne starts/stops acquisition for each pair, avoiding
                    # buffered frames from the previous exposure/gain setting.
                    camera.ExposureTime.SetValue(exposure)
                    actual_exposure = camera.ExposureTime.GetValue()
                    timeout_ms = max(5000, math.ceil(actual_exposure / 1000) + 5000)
                    result = camera.GrabOne(timeout_ms, pylon.TimeoutHandling_ThrowException)
                    try:
                        if not result.GrabSucceeded():
                            raise RuntimeError(
                                f'Grab failed ({result.ErrorCode}): {result.ErrorDescription}'
                            )
                        SaveBmp(converter, result, save_root / filename)
                    finally:
                        result.Release()

                    writer.writerow({
                        'filename': filename,
                        'requested_exposure_us': exposure,
                        'requested_gain_db': gain,
                        'actual_exposure_us': actual_exposure,
                        'actual_gain_db': actual_gain,
                        'gamma': gamma,
                        'pixel_format': pixel_format,
                        'camera_model': model,
                        'serial_number': serial,
                        'captured_at': datetime.now().isoformat(timespec='seconds'),
                    })
                    log.flush()
                    count += 1
                    print(f'[{count}/{total}] {filename} '
                          f'(readback: {actual_exposure:g} us, {actual_gain:g} dB)',
                          flush=True)
                except Exception as exc:
                    raise RuntimeError(f'Failed at {filename}: {exc}') from exc
    return count


def main():
    camera = None
    save_root = Path(__file__).resolve().parent / datetime.now().strftime(
        'scan_%Y-%m-%d_%H-%M-%S_%f')
    try:
        camera = OpenFirstCamera()
        SetCamera(camera, default_cameraSettings)
        print('Camera:', camera.GetDeviceInfo().GetModelName())
        print('Gamma:', camera.Gamma.GetValue())
        count = ScanGrid(camera, save_root)
        print(f'Finished: {count} images saved to {save_root}')
        return 0
    except KeyboardInterrupt:
        print(f'\nStopped. Completed images remain in: {save_root}')
        return 130
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        print(f'Output path (if collection started): {save_root}', file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            finally:
                camera.Close()


if __name__ == '__main__':
    sys.exit(main())
