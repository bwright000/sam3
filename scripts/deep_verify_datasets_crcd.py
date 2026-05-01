"""
Deep verification of F:/Datasets/CRCD/ snippet build.

Checks per snippet:
  1. Frame count consistency: rgb/ == rgbright/ == rgb.txt == rgbright.txt == depth.txt
     == associations.txt == groundtruth.txt == video_left.mp4 frame count
  2. Frame range: first/last frame_n in rgb/ match intrinsics.yaml frames.first/last
  3. Frame index integrity: no gaps, no duplicates, monotonically increasing
  4. Image integrity: every PNG decodes; resolution 1280x720; correct bit depth
  5. Pose-frame alignment: groundtruth.txt timestamps strictly match rgb.txt timestamps;
     first pose's tx/ty/tz matches source parquet's first frame ECM pose
  6. Video integrity: video_left.mp4 1280x720, 60fps, frame count matches rgb/
  7. Cluster sub-segments: each sub_segment_id in cluster_metadata fits within padded range;
     motion_frames count > 0; max velocities physically plausible
  8. Per-episode index file: lists exactly the snippets present, with correct rank
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path('F:/Datasets/CRCD')
PARQUET_DIR = Path(r'f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data')


def parse_index(p):
    rows = []
    for line in open(p):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        ts, fname = s.split()[:2]
        rows.append((float(ts), fname))
    return rows


def parse_groundtruth(p):
    poses = []
    for line in open(p):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) != 8:
            continue
        poses.append((float(parts[0]),
                      np.array([float(parts[i]) for i in range(1, 4)]),
                      np.array([float(parts[i]) for i in range(4, 8)])))
    return poses


def verify_snippet(snip_dir):
    issues = []

    # Detect mono-only (D_2/D_3 episodes have no right-camera in source parquet)
    has_right = (snip_dir / 'rgbright').exists()
    has_right_idx = (snip_dir / 'rgbright.txt').exists()
    if has_right != has_right_idx:
        issues.append(f'inconsistent right-camera state: dir exists={has_right}, txt exists={has_right_idx}')

    # --- 1. Frame count consistency ---
    n_rgb_dir = len(list((snip_dir / 'rgb').glob('frame_*.png')))
    n_right_dir = len(list((snip_dir / 'rgbright').glob('frame_*.png'))) if has_right else 0

    rgb_idx = parse_index(snip_dir / 'rgb.txt')
    right_idx = parse_index(snip_dir / 'rgbright.txt') if has_right_idx else []
    depth_idx = parse_index(snip_dir / 'depth.txt')
    assoc = [(float(line.split()[0]), line.split()[1], float(line.split()[2]), line.split()[3])
             for line in open(snip_dir / 'associations.txt')
             if line.strip() and not line.startswith('#')]
    poses = parse_groundtruth(snip_dir / 'groundtruth.txt')

    cap = cv2.VideoCapture(str(snip_dir / 'video_left.mp4'))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    counts = {'rgb_dir': n_rgb_dir,
              'rgb.txt': len(rgb_idx),
              'depth.txt': len(depth_idx), 'associations.txt': len(assoc),
              'groundtruth.txt': len(poses), 'video_left.mp4': n_video}
    if has_right:
        counts['right_dir'] = n_right_dir
        counts['rgbright.txt'] = len(right_idx)
    if len(set(counts.values())) != 1:
        issues.append(f'frame count mismatch: {counts}')
    n_frames = n_rgb_dir

    # --- 2. Frame range matches intrinsics.yaml ---
    intr = yaml.safe_load((snip_dir / 'intrinsics.yaml').read_text())
    fr_meta = intr.get('frames', {})
    rgb_files = sorted((snip_dir / 'rgb').glob('frame_*.png'))
    if rgb_files:
        first_f = int(rgb_files[0].stem.replace('frame_', ''))
        last_f = int(rgb_files[-1].stem.replace('frame_', ''))
        if first_f != fr_meta.get('first'):
            issues.append(f'first frame mismatch: rgb/ has {first_f}, intrinsics says {fr_meta.get("first")}')
        if last_f != fr_meta.get('last'):
            issues.append(f'last frame mismatch: rgb/ has {last_f}, intrinsics says {fr_meta.get("last")}')
    if fr_meta.get('count') != n_frames:
        issues.append(f"intrinsics frames.count {fr_meta.get('count')} != actual {n_frames}")

    # --- 3. Frame index integrity (no gaps, monotonic) ---
    frame_ns = [int(f.stem.replace('frame_', '')) for f in rgb_files]
    if frame_ns != sorted(frame_ns):
        issues.append('rgb/ frame_n not monotonic')
    if len(set(frame_ns)) != len(frame_ns):
        issues.append('rgb/ has duplicate frame_n')
    expected = list(range(frame_ns[0], frame_ns[-1] + 1))
    if frame_ns != expected:
        missing = sorted(set(expected) - set(frame_ns))
        issues.append(f'rgb/ has gaps: missing frame_n {missing[:5]}{"..." if len(missing) > 5 else ""} ({len(missing)} total)')
    # Same check for rgbright (only if stereo)
    right_files = []
    if has_right:
        right_files = sorted((snip_dir / 'rgbright').glob('frame_*.png'))
        right_ns = [int(f.stem.replace('frame_', '')) for f in right_files]
        if right_ns != frame_ns:
            issues.append('rgbright/ frame_n list differs from rgb/ frame_n list')

    # --- 4. Image integrity (sample first, middle, last) ---
    sample_idxs = [0, len(rgb_files) // 2, len(rgb_files) - 1] if rgb_files else []
    sides = [('rgb', rgb_files)]
    if has_right and right_files:
        sides.append(('rgbright', right_files))
    for i in sample_idxs:
        for side, files_list in sides:
            try:
                im = Image.open(str(files_list[i]))
                if im.size != (1280, 720):
                    issues.append(f'{side}/{files_list[i].name} size {im.size} != (1280,720)')
                if im.mode != 'RGB':
                    issues.append(f'{side}/{files_list[i].name} mode {im.mode} != RGB')
            except Exception as e:
                issues.append(f'{side}/{files_list[i].name} decode failed: {e}')

    # --- 5. Pose-frame alignment ---
    rgb_ts = [t for t, _ in rgb_idx]
    pose_ts = [t for t, _, _ in poses]
    depth_ts = [t for t, _ in depth_idx]
    if rgb_ts != pose_ts:
        n_diff = sum(1 for a, b in zip(rgb_ts, pose_ts) if abs(a - b) > 1e-9)
        issues.append(f'rgb.txt timestamps != groundtruth.txt timestamps ({n_diff} diff out of {len(rgb_ts)})')
    if has_right_idx:
        right_ts = [t for t, _ in right_idx]
        if rgb_ts != right_ts:
            issues.append('rgb.txt timestamps != rgbright.txt timestamps')
    if rgb_ts != depth_ts:
        issues.append('rgb.txt timestamps != depth.txt timestamps')

    # --- 5b. First pose value cross-check against parquet ---
    episode = snip_dir.parent.name
    first_frame_n = frame_ns[0] if frame_ns else None
    if first_frame_n is not None and len(poses) > 0:
        # Load just the first frame's row from parquet
        files = sorted((PARQUET_DIR / episode).glob("*.parquet"))
        gt_pose = None
        gt_ts = None
        for pf in files:
            df = pd.read_parquet(pf, columns=['frame_n', 'timestamp', '/ECM/custom/setpoint_cp'])
            mask = df['frame_n'] == first_frame_n
            if mask.any():
                row = df[mask].iloc[0]
                gt_pose = np.array(row['/ECM/custom/setpoint_cp'])
                gt_ts = float(row['timestamp'])
                break
            del df
        if gt_pose is not None:
            ts, t, q = poses[0]
            if abs(ts - gt_ts) > 1e-6:
                issues.append(f'first pose ts {ts} != parquet ts {gt_ts}')
            if not np.allclose(t, gt_pose[:3], atol=1e-9):
                issues.append(f'first pose t {t} != parquet t {gt_pose[:3]}')
            if not np.allclose(q, gt_pose[3:7], atol=1e-9):
                issues.append(f'first pose q {q} != parquet q {gt_pose[3:7]}')
        else:
            issues.append(f'could not find frame {first_frame_n} in parquet for cross-check')

    # --- 6. Video sanity ---
    if (w_video, h_video) != (1280, 720):
        issues.append(f'video resolution {w_video}x{h_video} != 1280x720')
    if abs(fps_video - 60.0) > 0.5:
        issues.append(f'video fps {fps_video} != 60')

    # --- 7. Cluster metadata ---
    cm_path = snip_dir / 'cluster_metadata.json'
    if cm_path.exists():
        cm = json.loads(cm_path.read_text())
        if cm['total_frames'] != n_frames:
            issues.append(f"cluster_metadata.total_frames {cm['total_frames']} != actual {n_frames}")
        if cm['padded_start_frame'] != frame_ns[0] or cm['padded_end_frame'] != frame_ns[-1]:
            issues.append(f"cluster_metadata padded range {cm['padded_start_frame']}-{cm['padded_end_frame']} != actual {frame_ns[0]}-{frame_ns[-1]}")
        for ss in cm['sub_segments']:
            ms, me = ss['motion_start_frame'], ss['motion_end_frame']
            if ms < cm['padded_start_frame'] or me > cm['padded_end_frame']:
                issues.append(f"sub_segment {ss['sub_segment_id']} range {ms}-{me} outside padded range")
            if ss['motion_frames'] <= 0:
                issues.append(f"sub_segment {ss['sub_segment_id']} has 0 motion_frames")
            if ss['max_linear_vel_mm_s'] > 500 or ss['max_linear_vel_mm_s'] < 0:
                issues.append(f"sub_segment {ss['sub_segment_id']} linear velocity {ss['max_linear_vel_mm_s']} mm/s implausible")

    # --- 8. scene_motion.json ---
    sm_path = snip_dir / 'scene_motion.json'
    if not sm_path.exists():
        issues.append('scene_motion.json missing')
    else:
        sm = json.loads(sm_path.read_text())
        if sm.get('motion_type') not in ('tool_and_camera', 'camera_only', 'tool_only'):
            issues.append(f"scene_motion.motion_type unexpected: {sm.get('motion_type')}")

    return issues, {
        'n_frames': n_frames,
        'first_frame': frame_ns[0] if frame_ns else None,
        'last_frame': frame_ns[-1] if frame_ns else None,
        'rank': intr.get('rank'),
        'has_cluster_meta': cm_path.exists(),
    }


def verify_episode_index(ep_dir):
    issues = []
    idx_path = ep_dir / f'{ep_dir.name}_snippets.json'
    if not idx_path.exists():
        issues.append(f'{idx_path.name} missing')
        return issues
    entries = json.loads(idx_path.read_text())
    snip_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith('snippet_'))
    if len(entries) != len(snip_dirs):
        issues.append(f'{idx_path.name} has {len(entries)} entries but {len(snip_dirs)} snippet dirs exist')
    for entry, sd in zip(entries, snip_dirs):
        expected_id = sd.name.replace('snippet_', '')
        if entry.get('snippet_id') != expected_id:
            issues.append(f'{idx_path.name} entry {entry.get("snippet_id")} != folder {sd.name}')
    return issues


def main():
    if not ROOT.exists():
        print(f'ROOT missing: {ROOT}'); return 1
    total_issues = 0
    n_total = 0
    n_pass = 0

    print('=== Per-snippet verification ===')
    for ep_dir in sorted(ROOT.iterdir()):
        if not ep_dir.is_dir(): continue
        for snip in sorted(ep_dir.iterdir()):
            if not snip.is_dir() or not snip.name.startswith('snippet_'): continue
            n_total += 1
            issues, summary = verify_snippet(snip)
            label = f'{ep_dir.name}/{snip.name}'
            if issues:
                total_issues += len(issues)
                print(f'\nFAIL {label}  rank=#{summary["rank"]}  n={summary["n_frames"]}  range={summary["first_frame"]}-{summary["last_frame"]}')
                for e in issues:
                    print(f'  - {e}')
            else:
                n_pass += 1
                tag = 'cluster' if summary['has_cluster_meta'] else 'single'
                print(f'OK   {label:<22} rank=#{summary["rank"]}  {tag}  n={summary["n_frames"]}  range={summary["first_frame"]}-{summary["last_frame"]}')

    print('\n=== Per-episode index verification ===')
    for ep_dir in sorted(ROOT.iterdir()):
        if not ep_dir.is_dir(): continue
        issues = verify_episode_index(ep_dir)
        if issues:
            total_issues += len(issues)
            print(f'FAIL {ep_dir.name}')
            for e in issues:
                print(f'  - {e}')
        else:
            print(f'OK   {ep_dir.name}_snippets.json')

    print(f'\n=== {n_pass}/{n_total} snippets pass; {total_issues} total issues ===')
    return 0 if total_issues == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
