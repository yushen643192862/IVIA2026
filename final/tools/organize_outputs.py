"""Move output datasets into camera5, demo, and structured_light folders."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'

CAMERA5 = (
    'camera5_dense_foreground',
    'camera5_original_masks_v2',
    'camera5_refined_mask_review',
)
STRUCTURED = (
    'calibration',
    'calibration_board',
    'camera_projector_calibration_20261026_204848',
    'structured_capture',
    'structured_decode_20261026_210453',
    'structured_depth_green_only_v2',
    'structured_depth_mask_inset10_v1',
    'structured_depth_mask_inset16_v1',
    'structured_depth_mask_inset2_v1',
    'structured_depth_mask_inset6_relative_trial_v1',
    'structured_depth_mask_inset6_v1',
    'structured_fused_green_only_v3',
    'structured_groups_3_2pct_v1',
    'structured_subject_green_only_v2',
    'projector_offset_all31_audit.json',
)


def inside_outputs(path):
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(OUTPUTS.resolve()):
        raise ValueError(f'Path escapes outputs: {path}')


def mapping():
    return {name: 'camera5' for name in CAMERA5} | {name: 'structured_light' for name in STRUCTURED}


def replace_paths(content, destinations):
    changed = content
    for name, category in destinations.items():
        for separator in ('/', chr(92), chr(92) * 2):
            old = 'outputs' + separator + name
            new = 'outputs' + separator + category + separator + name
            changed = changed.replace(old, new)
    return changed


def text_files():
    for path in OUTPUTS.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.json', '.ini', '.cfg'}:
            yield path
    yield from ROOT.glob('*.md')
    yield from ROOT.glob('*.ps1')
    yield from (ROOT / 'tools').glob('*.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Perform moves and update text references')
    args = parser.parse_args()
    destinations = mapping()
    if not OUTPUTS.is_dir():
        raise FileNotFoundError(OUTPUTS)
    operations = []
    for name, category in destinations.items():
        source = OUTPUTS / name
        target = OUTPUTS / category / name
        inside_outputs(source)
        inside_outputs(target)
        if source.exists() and target.exists():
            raise FileExistsError(f'Both source and target exist: {source}, {target}')
        if not source.exists() and not target.exists():
            raise FileNotFoundError(f'Neither source nor target exists: {name}')
        operations.append((source, target, source.exists()))
    print('Directory moves:')
    for source, target, pending in operations:
        print(('MOVE ' if pending else 'DONE ') + f'{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}')
    if not args.apply:
        return
    for source, target, pending in operations:
        if pending:
            target.parent.mkdir(exist_ok=True)
            source.rename(target)
    edited = []
    for path in text_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            continue
        updated = replace_paths(content, destinations)
        if updated != content:
            path.write_text(updated, encoding='utf-8')
            edited.append(str(path.relative_to(ROOT)))
    manifest = {'categories': {
        'camera5': list(CAMERA5), 'demo': ['demo'], 'structured_light': list(STRUCTURED)},
        'updated_text_files': edited,
        'note': 'Existing outputs moved without deleting or changing binary data. JSON/code references were updated.'}
    (OUTPUTS / 'organization_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Updated text references in {len(edited)} files')


if __name__ == '__main__':
    main()
