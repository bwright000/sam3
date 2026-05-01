"""
Smoke test for the F:/Datasets/CRCD/ build (top-N candidate snippets).

Reuses the same TUM-loader-style validation as smoke_test_tum_loader.py but
points at the new dataset root. Tolerates absent semantic_instance/ PNGs and
absent info_semantic.json (these are candidate snippets without GT yet).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path('F:/Datasets/CRCD')


def parse_index(path):
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        rows.append((float(parts[0]), parts[1]))
    return rows


def parse_groundtruth(path):
    poses = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) != 8:
            continue
        poses.append((float(parts[0]),
                      np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                      np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])))
    return poses


def parse_associations(path):
    pairs = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) == 4:
            pairs.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    return pairs


def test_snippet(snippet_dir):
    errors = []

    try:
        rgb_idx = parse_index(snippet_dir / 'rgb.txt')
        n_rgb = len(rgb_idx)
        if n_rgb == 0:
            errors.append('rgb.txt empty')
            return errors, {}
    except Exception as e:
        errors.append(f'rgb.txt: {e}')
        return errors, {}

    try:
        depth_idx = parse_index(snippet_dir / 'depth.txt')
        if len(depth_idx) != n_rgb:
            errors.append(f'depth.txt count {len(depth_idx)} != rgb {n_rgb}')
    except Exception as e:
        errors.append(f'depth.txt: {e}')

    if (snippet_dir / 'rgbright.txt').exists():
        try:
            rgbright_idx = parse_index(snippet_dir / 'rgbright.txt')
            if len(rgbright_idx) != n_rgb:
                errors.append(f'rgbright.txt count {len(rgbright_idx)} != rgb {n_rgb}')
        except Exception as e:
            errors.append(f'rgbright.txt: {e}')
    # else: mono-only snippet (e.g. D_2/D_3 episodes); no rgbright is expected.

    try:
        poses = parse_groundtruth(snippet_dir / 'groundtruth.txt')
        if len(poses) != n_rgb:
            errors.append(f'groundtruth count {len(poses)} != rgb {n_rgb}')
    except Exception as e:
        errors.append(f'groundtruth.txt: {e}')

    try:
        assoc = parse_associations(snippet_dir / 'associations.txt')
        if len(assoc) != n_rgb:
            errors.append(f'associations count {len(assoc)} != rgb {n_rgb}')
    except Exception as e:
        errors.append(f'associations.txt: {e}')

    try:
        intr = yaml.safe_load(open(snippet_dir / 'intrinsics.yaml'))
        for k in ['camera', 'depth', 'frames', 'segmentation']:
            if k not in intr:
                errors.append(f'intrinsics.yaml missing key: {k}')
        if intr.get('depth', {}).get('png_depth_scale') != 10000:
            errors.append('png_depth_scale != 10000')
        if intr.get('frames', {}).get('count') != n_rgb:
            errors.append('intrinsics frames.count mismatch')
    except Exception as e:
        errors.append(f'intrinsics.yaml: {e}')

    try:
        ts_rgb, rgb_path = rgb_idx[0]
        rgb = cv2.imread(str(snippet_dir / rgb_path), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape != (720, 1280, 3):
            errors.append(f'rgb decode failed: {rgb.shape if rgb is not None else None}')
        if abs(poses[0][0] - ts_rgb) > 1e-6:
            errors.append(f'first pose ts {poses[0][0]} != first rgb ts {ts_rgb}')
    except Exception as e:
        errors.append(f'frame load: {e}')

    # video_left.mp4
    v = snippet_dir / 'video_left.mp4'
    if v.exists():
        cap = cv2.VideoCapture(str(v))
        nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if nv != n_rgb:
            errors.append(f'video_left frames {nv} != rgb {n_rgb}')
        if (wv, hv) != (1280, 720):
            errors.append(f'video_left resolution {wv}x{hv} != 1280x720')
    else:
        errors.append('video_left.mp4 missing')

    # depth/ + semantic_instance/ should exist (.gitkeep present, no PNGs)
    for placeholder in ('depth', 'semantic_instance'):
        d = snippet_dir / placeholder
        if not d.is_dir():
            errors.append(f'{placeholder}/ missing')

    summary = {
        'n_frames': n_rgb,
        'rank': intr.get('rank') if 'intr' in dir() else None,
        'description': intr.get('description') if 'intr' in dir() else None,
        'has_cluster_meta': (snippet_dir / 'cluster_metadata.json').exists(),
    }
    return errors, summary


def main():
    if not ROOT.exists():
        print(f'ROOT does not exist: {ROOT}'); return 1
    total = 0
    failed = 0
    for ep_dir in sorted(ROOT.iterdir()):
        if not ep_dir.is_dir():
            continue
        for snip in sorted(ep_dir.iterdir()):
            if not snip.is_dir() or not snip.name.startswith('snippet_'):
                continue
            total += 1
            errors, summary = test_snippet(snip)
            label = f'{ep_dir.name}/{snip.name}'
            if errors:
                failed += 1
                print(f'FAIL {label}')
                for e in errors:
                    print(f'  - {e}')
            else:
                tag = 'cluster' if summary['has_cluster_meta'] else 'single'
                print(f'OK   {label}  rank=#{summary["rank"]}  {tag}  n={summary["n_frames"]}  "{summary["description"]}"')
    print(f'\n=== {total - failed}/{total} snippets pass ===')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
