"""
Smoke test: parse a converted snippet using a generic TUM-RGB-D-style loader.

Mimics what off-the-shelf SLAM tools do when consuming TUM data:
  1. read rgb.txt
  2. read depth.txt
  3. read groundtruth.txt
  4. read associations.txt
  5. read intrinsics.yaml
  6. read info_semantic.json (Replica-style)
  7. load one frame's RGB + semantic_instance + pose

Run on every converted snippet, report pass/fail.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_index(path):
    """Read TUM-style index file: '# header' + 'timestamp filename' lines."""
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        ts = float(parts[0])
        fname = parts[1]
        rows.append((ts, fname))
    return rows


def parse_groundtruth(path):
    """Read TUM groundtruth.txt: '# header' + 'ts tx ty tz qx qy qz qw' lines."""
    poses = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) != 8:
            continue
        ts = float(parts[0])
        t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
        poses.append((ts, t, q))
    return poses


def parse_associations(path):
    """TUM associations.txt: 'ts_rgb rgb_path ts_depth depth_path' per line."""
    pairs = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) != 4:
            continue
        pairs.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    return pairs


def test_snippet(snippet_dir):
    snippet_dir = Path(snippet_dir)
    errors = []

    # 1. rgb.txt
    try:
        rgb_idx = parse_index(snippet_dir / 'rgb.txt')
        n_rgb = len(rgb_idx)
        if n_rgb == 0:
            errors.append('rgb.txt empty')
    except Exception as e:
        errors.append(f'rgb.txt parse error: {e}')
        return errors, {}

    # 2. depth.txt
    try:
        depth_idx = parse_index(snippet_dir / 'depth.txt')
        if len(depth_idx) != n_rgb:
            errors.append(f'depth.txt count {len(depth_idx)} != rgb.txt count {n_rgb}')
    except Exception as e:
        errors.append(f'depth.txt parse error: {e}')

    # 3. rgbright.txt (deviation from TUM)
    try:
        rgbright_idx = parse_index(snippet_dir / 'rgbright.txt')
        if len(rgbright_idx) != n_rgb:
            errors.append(f'rgbright.txt count {len(rgbright_idx)} != rgb.txt count {n_rgb}')
    except Exception as e:
        errors.append(f'rgbright.txt parse error: {e}')

    # 4. groundtruth.txt
    try:
        poses = parse_groundtruth(snippet_dir / 'groundtruth.txt')
        if len(poses) != n_rgb:
            errors.append(f'groundtruth count {len(poses)} != rgb count {n_rgb}')
    except Exception as e:
        errors.append(f'groundtruth.txt parse error: {e}')

    # 5. associations.txt
    try:
        assoc = parse_associations(snippet_dir / 'associations.txt')
        if len(assoc) != n_rgb:
            errors.append(f'associations count {len(assoc)} != rgb count {n_rgb}')
        # check the file paths in the first row resolve
        if assoc:
            ts_rgb, rgb_path, ts_d, d_path = assoc[0]
            if not (snippet_dir / rgb_path).exists():
                errors.append(f'first associations rgb path missing: {rgb_path}')
    except Exception as e:
        errors.append(f'associations.txt parse error: {e}')

    # 6. intrinsics.yaml
    try:
        intr = yaml.safe_load(open(snippet_dir / 'intrinsics.yaml'))
        for k in ['camera', 'depth', 'frames']:
            if k not in intr:
                errors.append(f'intrinsics.yaml missing key: {k}')
        if intr.get('depth', {}).get('png_depth_scale') != 10000:
            errors.append(f'png_depth_scale != 10000')
        if intr.get('frames', {}).get('count') != n_rgb:
            errors.append(f'intrinsics frames.count != rgb count')
    except Exception as e:
        errors.append(f'intrinsics.yaml parse error: {e}')

    # 7. info_semantic.json
    info_path = snippet_dir / 'info_semantic.json'
    if info_path.exists():
        try:
            info = json.load(open(info_path))
            if 'classes' not in info or 'background_id' not in info:
                errors.append('info_semantic.json missing keys')
        except Exception as e:
            errors.append(f'info_semantic.json parse error: {e}')

    # 8. Load one RGB frame + matching semantic_instance + pose
    try:
        ts_rgb, rgb_path = rgb_idx[0]
        rgb = cv2.imread(str(snippet_dir / rgb_path), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape != (720, 1280, 3):
            errors.append(f'rgb decode failed or wrong shape: {rgb.shape if rgb is not None else None}')
        # Same frame's semantic_instance
        frame_n = int(Path(rgb_path).stem.replace('frame_', ''))
        sem_path = snippet_dir / 'semantic_instance' / f'frame_{frame_n:06d}.png'
        if sem_path.exists():
            sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
            if sem is None or sem.dtype != np.uint16 or sem.shape != (720, 1280):
                errors.append(f'semantic_instance decode failed or wrong dtype/shape')
        # Verify first pose timestamp matches first rgb timestamp
        if abs(poses[0][0] - ts_rgb) > 1e-6:
            errors.append(f'first pose ts {poses[0][0]} != first rgb ts {ts_rgb}')
    except Exception as e:
        errors.append(f'frame load error: {e}')

    summary = {
        'n_frames': n_rgb,
        'first_frame_path': rgb_idx[0][1] if rgb_idx else None,
        'depth_scale': intr.get('depth', {}).get('png_depth_scale') if 'intr' in dir() else None,
        'has_segmentation': info_path.exists(),
        'segmentation_state': intr.get('segmentation', {}).get('state') if 'intr' in dir() else None,
    }
    return errors, summary


def main():
    ROOT = Path('F:/Datasets/CRCD')
    total = 0
    failed = 0
    for ep in ['C_1', 'E_3', 'F_3']:
        ep_dir = ROOT / ep
        if not ep_dir.exists():
            continue
        for snip in sorted(ep_dir.iterdir()):
            if not snip.is_dir() or '.bak' in snip.name:
                continue
            total += 1
            errors, summary = test_snippet(snip)
            label = f'{ep}/{snip.name}'
            if errors:
                failed += 1
                print(f'FAIL {label}')
                for e in errors:
                    print(f'  - {e}')
            else:
                print(f'OK   {label}  ({summary["n_frames"]} frames, scale={summary["depth_scale"]}, seg={summary["segmentation_state"]})')
    print(f'\n=== {total - failed}/{total} snippets pass TUM loader smoke test ===')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
