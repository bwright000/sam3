#!/usr/bin/env python3
"""Find candidate snippets for SLAM-relevant SAM3 annotation across all CRCD episodes.

Reads ECM (camera) + PSM (tool) kinematics from parquet, scores candidate
windows on a multi-criteria rubric, and emits a categorised list:

  * Short   (span 10-30s)            — fast turn-around scenes for sample diversity
  * Long    (span 60-150s)           — sustained sequences for tracker stress
  * Fast    (peak ECM vel ≥ 50 mm/s) — high-velocity camera moves (parallax-rich)
  * Concurrent (camera + tool motion ≥ 30% of window)
                                     — the hardest case for visual SLAM, where
                                       moving foreground objects coincide with
                                       camera ego-motion

Each window is also annotated with average velocity (when moving) so the original
sensitive threshold (0.1 mm/s) classifies sub-segments while a stricter
threshold (1 mm/s) is used for "visually-significant" motion in the
concurrent-motion calculation.

Excludes regions overlapping the existing top-10 picks (build_top10_snippets.py
SPECS). Outputs to outputs/slam_candidates.json + prints a per-category table.

Usage:
    python scripts/find_slam_candidates.py [--out outputs/slam_candidates.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PARQ = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
MOTION_SEG_FILE = Path(r"f:/2026 vibes/MPHY Project/Detailed Analysis/all_episodes_motion_segments.json")

FPS = 60
CAM_THRESH_LO = 0.1   # mm/s — original sensitive (matches all_episodes_motion_segments.json)
CAM_THRESH_HI = 1.0   # mm/s — visually-significant camera motion
TOOL_THRESH = 1.0     # mm/s — visually-significant tool motion (PSM)

# Existing top-10 regions (build_top10_snippets.py SPECS) — exclude overlap
EXCLUDED_REGIONS = {
    'F_1': [(252, 3730), (11219, 12385)],
    'G_2': [(39191, 42053)],
    'C_2': [(30538, 31147)],
    'E_1': [(37740, 39727)],
    'G_3': [(0, 1926)],
    'C_3': [(529, 1935)],
}

MONO_ONLY = {'D_2', 'D_3'}


def overlaps_excluded(ep: str, frame_start: int, frame_end: int) -> bool:
    for ex_s, ex_e in EXCLUDED_REGIONS.get(ep, []):
        if not (frame_end < ex_s or frame_start > ex_e):
            return True
    return False


def load_episode_kinematics(ep: str) -> dict | None:
    """Load all parquets for an episode and compute per-frame velocities."""
    files = sorted((PARQ / ep).glob("*.parquet"))
    if not files:
        return None
    cols = ['frame_n', 'timestamp',
            '/ECM/custom/setpoint_cp',
            '/PSM1/custom/setpoint_cp',
            '/PSM2/custom/setpoint_cp']
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f, columns=cols))
        except Exception:
            return None
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).sort_values('frame_n').reset_index(drop=True)
    n = len(df)
    if n < 5:
        return None

    ts = df['timestamp'].values.astype(np.float64)
    dt = np.diff(ts)
    dt = np.maximum(dt, 1e-6)

    # ECM (camera) ---------------------------------------------------------
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist(), dtype=np.float64)
    ecm_pos = ecm[:, :3]                        # m
    ecm_quat = ecm[:, 3:7]                      # xyzw
    ecm_lin = np.linalg.norm(np.diff(ecm_pos, axis=0), axis=1) / dt * 1000  # mm/s
    # Angular velocity from quaternion delta
    dot = np.clip(np.abs(np.einsum('ij,ij->i', ecm_quat[:-1], ecm_quat[1:])), 0.0, 1.0)
    ecm_ang = np.degrees(2 * np.arccos(dot) / dt)

    # PSMs (tools) ---------------------------------------------------------
    psm_lins = []
    for col in ['/PSM1/custom/setpoint_cp', '/PSM2/custom/setpoint_cp']:
        if col not in df.columns or df[col].iloc[0] is None:
            psm_lins.append(np.zeros(n - 1))
            continue
        try:
            psm = np.array(df[col].tolist(), dtype=np.float64)
            psm_pos = psm[:, :3]
            psm_lin = np.linalg.norm(np.diff(psm_pos, axis=0), axis=1) / dt * 1000
            psm_lins.append(psm_lin)
        except Exception:
            psm_lins.append(np.zeros(n - 1))

    return {
        'frame_n': df['frame_n'].values.astype(np.int64),
        'ts': ts,
        'ecm_pos': ecm_pos,
        'ecm_lin': ecm_lin,
        'ecm_ang': ecm_ang,
        'psm1_lin': psm_lins[0],
        'psm2_lin': psm_lins[1],
        'n_frames': n,
    }


def score_window(kin: dict, i: int, j: int) -> dict | None:
    """Score frames [i..j) of the episode. j is exclusive on raw frames;
    velocities/diffs cover up to j-1."""
    if j <= i + 2 or j > kin['n_frames']:
        return None
    # Velocity arrays are length n-1; constrain to i..j-1
    ecm_lin = kin['ecm_lin'][i:j-1]
    ecm_ang = kin['ecm_ang'][i:j-1]
    psm1_lin = kin['psm1_lin'][i:j-1]
    psm2_lin = kin['psm2_lin'][i:j-1]
    pos = kin['ecm_pos'][i:j]
    n_v = len(ecm_lin)
    if n_v < 2:
        return None

    # Per-frame motion flags at two thresholds
    cam_mov_lo = (ecm_lin > CAM_THRESH_LO) | (ecm_ang > CAM_THRESH_LO)   # sensitive
    cam_mov_hi = ecm_lin > CAM_THRESH_HI                                  # visually-significant
    tool_mov = (psm1_lin > TOOL_THRESH) | (psm2_lin > TOOL_THRESH)
    concurrent = cam_mov_hi & tool_mov

    span_frames = j - i
    span_s = span_frames / FPS

    # Path + bbox (3D coverage)
    diffs = np.diff(pos, axis=0)
    path_mm = float(np.linalg.norm(diffs, axis=1).sum() * 1000)
    bbox = (pos.max(axis=0) - pos.min(axis=0)) * 1000   # mm
    bbox_vol_mm3 = float(max(bbox.prod(), 1e-6))

    return {
        'frame_start': int(kin['frame_n'][i]),
        'frame_end': int(kin['frame_n'][j-1]),
        'span_frames': span_frames,
        'span_s': round(span_s, 1),
        'motion_s_threshold_low': round(cam_mov_lo.sum() / FPS, 1),
        'motion_s_threshold_hi': round(cam_mov_hi.sum() / FPS, 1),
        'concurrent_motion_s': round(concurrent.sum() / FPS, 1),
        'concurrent_motion_ratio': round(concurrent.sum() / max(n_v, 1), 3),
        'avg_cam_vel_mm_s': round(float(ecm_lin.mean()), 2),
        'avg_cam_vel_when_moving_mm_s': round(float(ecm_lin[cam_mov_lo].mean()) if cam_mov_lo.any() else 0.0, 2),
        'peak_cam_vel_mm_s': round(float(ecm_lin.max()), 1),
        'avg_ang_vel_deg_s': round(float(ecm_ang.mean()), 2),
        'peak_ang_vel_deg_s': round(float(ecm_ang.max()), 1),
        'avg_tool_vel_when_moving_mm_s': round(float((psm1_lin + psm2_lin)[tool_mov].mean()) if tool_mov.any() else 0.0, 2),
        'peak_tool_vel_mm_s': round(float(np.maximum(psm1_lin, psm2_lin).max()), 1),
        'path_length_mm': round(path_mm, 1),
        'bbox_volume_mm3': round(bbox_vol_mm3, 1),
        'bbox_extent_mm': [round(float(x), 1) for x in bbox.tolist()],
    }


def find_candidate_windows(kin: dict, ep: str, motion_segs_for_ep: list) -> list[dict]:
    """Generate candidate windows for an episode, score each, return list."""
    n = kin['n_frames']
    frame_n = kin['frame_n']
    # Map frame_n -> index for fast lookup
    fn_to_idx = {int(f): i for i, f in enumerate(frame_n)}

    candidates = []

    # 1) Per-segment windows: each motion segment + 60-frame padding
    for seg in motion_segs_for_ep:
        s = seg['motion_start_frame']
        e = seg['motion_end_frame']
        i = fn_to_idx.get(max(s - 60, int(frame_n[0])))
        j = fn_to_idx.get(min(e + 60, int(frame_n[-1])))
        if i is None or j is None:
            continue
        sc = score_window(kin, i, j + 1)
        if sc:
            sc['source'] = f'seg{seg["segment_index"]}'
            candidates.append(sc)

    # 2) Multi-segment clusters at increasing widths (gap tolerance up to 30s)
    sorted_segs = sorted(motion_segs_for_ep, key=lambda s: s['motion_start_frame'])
    for gap_tol_s in (10, 20, 30, 60):
        gap_tol_f = int(gap_tol_s * FPS)
        i_seg = 0
        while i_seg < len(sorted_segs):
            cluster = [sorted_segs[i_seg]]
            j_seg = i_seg + 1
            while j_seg < len(sorted_segs):
                gap = sorted_segs[j_seg]['motion_start_frame'] - cluster[-1]['motion_end_frame']
                if gap > gap_tol_f:
                    break
                cluster.append(sorted_segs[j_seg])
                j_seg += 1
            if len(cluster) >= 2:
                s = cluster[0]['motion_start_frame']
                e = cluster[-1]['motion_end_frame']
                i = fn_to_idx.get(max(s - 60, int(frame_n[0])))
                j = fn_to_idx.get(min(e + 60, int(frame_n[-1])))
                if i is not None and j is not None:
                    sc = score_window(kin, i, j + 1)
                    if sc:
                        sc['source'] = f'cluster_segs_{cluster[0]["segment_index"]}-{cluster[-1]["segment_index"]}_gap{gap_tol_s}s'
                        candidates.append(sc)
            i_seg = j_seg if j_seg > i_seg else i_seg + 1

    # 3) Adaptive: for each segment, grow forward/backward until reaching motion_s targets
    for target_motion_s in (15, 30, 60):
        target_f = int(target_motion_s * FPS)
        for seg in motion_segs_for_ep:
            s = seg['motion_start_frame']
            e = seg['motion_end_frame']
            # walk forward until enough total motion
            i = fn_to_idx.get(s)
            if i is None:
                continue
            j = fn_to_idx.get(e, n - 1)
            # extend forward
            while j < n - 1:
                cur = score_window(kin, i, j + 1)
                if cur and cur['motion_s_threshold_low'] * FPS >= target_f:
                    break
                step = min(int(5 * FPS), n - 1 - j)   # 5s steps
                if step <= 0:
                    break
                j += step
            sc = score_window(kin, i, j + 1)
            if sc:
                sc['source'] = f'adaptive_target{target_motion_s}s_seg{seg["segment_index"]}'
                # cap extreme spans
                if sc['span_s'] <= 200:
                    candidates.append(sc)

    # Deduplicate by (frame_start, frame_end)
    seen = set()
    deduped = []
    for c in candidates:
        key = (c['frame_start'], c['frame_end'])
        if key in seen:
            continue
        seen.add(key)
        # Skip overlap with existing top-10
        if overlaps_excluded(ep, c['frame_start'], c['frame_end']):
            c['overlaps_existing'] = True
        else:
            c['overlaps_existing'] = False
        c['episode'] = ep
        c['mono_only'] = ep in MONO_ONLY
        deduped.append(c)
    return deduped


def categorise(candidates: list[dict]) -> dict:
    """Bucket windows by SLAM-data-diversity category."""
    free = [c for c in candidates if not c['overlaps_existing']]

    buckets = {
        'short':       [],   # span 10-30s
        'long':        [],   # span 60-150s
        'fast':        [],   # peak cam vel >=50 mm/s
        'concurrent':  [],   # >=30% concurrent (cam+tool) motion
    }
    for c in free:
        if 10 <= c['span_s'] <= 30:
            buckets['short'].append(c)
        if 60 <= c['span_s'] <= 150:
            buckets['long'].append(c)
        if c['peak_cam_vel_mm_s'] >= 50:
            buckets['fast'].append(c)
        if c['concurrent_motion_ratio'] >= 0.3:
            buckets['concurrent'].append(c)

    # Sort each bucket by relevant primary metric
    buckets['short'].sort(key=lambda c: -c['avg_cam_vel_when_moving_mm_s'])
    buckets['long'].sort(key=lambda c: -c['motion_s_threshold_low'])
    buckets['fast'].sort(key=lambda c: -c['peak_cam_vel_mm_s'])
    buckets['concurrent'].sort(key=lambda c: -c['concurrent_motion_ratio'])

    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='outputs/slam_candidates.json')
    ap.add_argument('--top-per-category', type=int, default=5,
                    help='top N candidates per category in printed report')
    ap.add_argument('--include-mono', action='store_true',
                    help='include D_2/D_3 (mono-only) episodes')
    args = ap.parse_args()

    # Load existing motion segment metadata as the seed for candidate windows
    motion_data = json.load(open(MOTION_SEG_FILE))
    segs_by_ep = defaultdict(list)
    for s in motion_data['segments']:
        segs_by_ep[s['episode']].append(s)

    all_candidates = []
    print('Episode kinematics scan ...')
    for ep in sorted(segs_by_ep.keys()):
        if not args.include_mono and ep in MONO_ONLY:
            print(f'  skip {ep} (mono-only)')
            continue
        t0 = time.time()
        kin = load_episode_kinematics(ep)
        if kin is None:
            print(f'  fail {ep}: parquet load')
            continue
        cands = find_candidate_windows(kin, ep, segs_by_ep[ep])
        print(f'  {ep}: n_frames={kin["n_frames"]:>6}  segments={len(segs_by_ep[ep]):>3}  candidates={len(cands):>4}  ({time.time()-t0:.1f}s)')
        all_candidates.extend(cands)

    print(f'\\ntotal candidates (all episodes): {len(all_candidates)}')
    free = [c for c in all_candidates if not c['overlaps_existing']]
    print(f'free (non-overlapping with existing top-10): {len(free)}')

    buckets = categorise(all_candidates)
    print(f'\\nCategorised:')
    for name, cs in buckets.items():
        print(f'  {name:<12}: {len(cs)} candidates')

    # Print top N per category
    print('\\n' + '='*120)
    fmt = '{ep:<5} {span_s:>7}  {motion_s:>9}  {avg_v:>10}  {peak_v:>10}  {tool_avg:>10}  {conc_pct:>9}  {peak_ang:>9}  {path:>9}  {bbox:>9}  {fr_range:<14}  {src}'
    header = fmt.format(
        ep='ep', span_s='span_s', motion_s='motion_s', avg_v='avg_v',
        peak_v='peak_v', tool_avg='tool_avg', conc_pct='conc_pct',
        peak_ang='peak_ang', path='path_mm', bbox='bbox_mm3',
        fr_range='frame_range', src='source')
    for cat in ('fast', 'concurrent', 'long', 'short'):
        print(f'\\n=== {cat.upper()} (top {args.top_per_category}) ===')
        print(header)
        for c in buckets[cat][:args.top_per_category]:
            mono = ' (mono)' if c.get('mono_only') else ''
            print(fmt.format(
                ep=c['episode'] + mono,
                span_s=c['span_s'],
                motion_s=c['motion_s_threshold_low'],
                avg_v=c['avg_cam_vel_when_moving_mm_s'],
                peak_v=c['peak_cam_vel_mm_s'],
                tool_avg=c['avg_tool_vel_when_moving_mm_s'],
                conc_pct=f'{int(c["concurrent_motion_ratio"]*100)}%',
                peak_ang=c['peak_ang_vel_deg_s'],
                path=c['path_length_mm'],
                bbox=int(c['bbox_volume_mm3']),
                fr_range=f'{c["frame_start"]}..{c["frame_end"]}',
                src=c['source']))

    # Write JSON output (top 20 per category for downstream processing)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'parameters': {
                'cam_thresh_lo_mm_s': CAM_THRESH_LO,
                'cam_thresh_hi_mm_s': CAM_THRESH_HI,
                'tool_thresh_mm_s': TOOL_THRESH,
                'fps': FPS,
                'excluded_regions': {ep: rgs for ep, rgs in EXCLUDED_REGIONS.items()},
            },
            'total_candidates': len(all_candidates),
            'free_candidates': len(free),
            'categories': {
                cat: cs[:20] for cat, cs in buckets.items()
            },
            'all_free_candidates': free,
        }, f, indent=2)
    print(f'\\nwrote {out_path}')


if __name__ == '__main__':
    main()
