"""Command-line green-screen reconstruction with explicit, repeatable stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['all', 'segment', 'sparse', 'dense'], default='all')
    p.add_argument('--colmap', type=Path, default=ROOT/'tools/colmap-4.2.1-cuda/bin/colmap.exe')
    p.add_argument('--window-radius', type=int, default=10)
    args = p.parse_args()
    if args.window_radius < 1:
        p.error('--window-radius must be positive')
    images, output, colmap = args.images.resolve(), args.output.resolve(), args.colmap.resolve()
    if not images.is_dir():
        p.error(f'Image directory not found: {images}')
    if not colmap.is_file():
        p.error(f'COLMAP not found: {colmap}')
    initial, masks, review = output/'masks_initial', output/'masks', output/'mask_review'
    reconstruction = output/'reconstruction'
    config = {'images': str(images), 'output': str(output), 'window_radius': args.window_radius,
              'segmentation': 'model_plus_green_refinement_v1', 'colmap': str(colmap)}
    manifest = output/'pipeline_config.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != config:
            p.error('Configuration differs from saved pipeline_config.json; use a new output directory.')
    elif args.stage in ['sparse', 'dense']:
        p.error('Run --stage segment first in this output directory.')
    if args.stage in ['all', 'segment'] and (initial.exists() or masks.exists() or reconstruction.exists()):
        p.error('Segmentation/reconstruction already exists. Use a new output or continue with --stage sparse/dense.')
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, indent=2), encoding='utf-8')

    def run(script, *arguments):
        command = [sys.executable, str(ROOT/script), *map(str, arguments)]
        print('Running:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if args.stage in ['all', 'segment']:
        run('tools/segment_subject_model.py', images, initial, '--review', review/'initial')
        run('tools/refine_mask_green.py', images, initial, masks, '--review', review)
        run('tools/export_masked_images.py', '--images', images, '--masks', masks,
            '--output', output/'masked_images')
        print('Mask preview:', review/'refined_masks_contact.jpg', flush=True)
    if args.stage in ['all', 'sparse']:
        run('scan.py', 'reconstruct', '--images', images, '--masks', masks,
            '--output', reconstruction, '--colmap', colmap, '--high-detail', '--sparse-only',
            '--patch-window-radius', args.window_radius)
        print('Sparse statistics:', reconstruction/'sparse_statistics.json', flush=True)
    if args.stage in ['all', 'dense']:
        run('scan.py', 'resume-dense', '--images', images, '--masks', masks,
            '--output', reconstruction, '--colmap', colmap, '--patch-window-radius', args.window_radius)
        run('tools/preview_cloud.py', reconstruction/'dense.ply', reconstruction/'dense_preview.png')
        print('Cloud:', reconstruction/'dense.ply', '\nMesh:', reconstruction/'mesh.ply', flush=True)
    print('Finished stage:', args.stage, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
