"""Locate archived acquisitions and the compact submission dataset."""
from pathlib import Path


def scan_folders(base):
    base = Path(base).resolve()
    roots = [base / 'RawData', base.parent / 'RawData', base.parent.parent / 'RawData',
             base / 'PythonGrabBasler' / 'PythonGrabSample']
    found = {p.resolve() for root in roots for p in root.glob('scan_*') if p.is_dir()}
    return sorted(found, key=lambda p: p.name, reverse=True)


def latest_scan(base):
    for folder in scan_folders(base):
        if (folder / 'scan_parameters.csv').is_file() and any(folder.glob('*.bmp')):
            return folder
    raise FileNotFoundError('No complete acquisition found. Use --input to specify the raw image folder.')


def latest_features(base):
    base = Path(base).resolve()
    packaged = base / 'Data' / 'noise_analysis' / 'noise_features.csv'
    if packaged.is_file():
        return packaged
    for folder in scan_folders(base):
        path = folder / 'noise_analysis' / 'noise_features.csv'
        if path.is_file():
            return path
    raise FileNotFoundError('No noise_features.csv found. Use --input to specify it.')
