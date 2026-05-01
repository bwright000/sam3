"""
Verify candidate clusters by loading raw ECM pose data and computing per-frame velocities.
Reports: motion-region stats, gap-region stats, max velocity, any glitches.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")

d = json.load(open('f:/2026 vibes/MPHY Project/Detailed Analysis/all_episodes_motion_segments.json'))
by_ep = defaultdict(list)
for s in d['segments']:
    by_ep[s['episode']].append(s)
FPS = 60.0


def cluster(segs, max_gap_s):
    if not segs:
        return []
    clusters = [[segs[0]]]
    for s in segs[1:]:
        prev = clusters[-1][-1]
        gap = (s['motion_start_frame'] - prev['motion_end_frame']) / FPS
        if gap <= max_gap_s:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return clusters


def load_episode_lightweight(episode):
    parquet_files = sorted((PARQUET_DIR / episode).glob("*.parquet"))
    dfs = [pd.read_parquet(pf, columns=['frame_n', 'timestamp', '/ECM/custom/setpoint_cp']) for pf in parquet_files]
    df = pd.concat(dfs, ignore_index=True).sort_values('frame_n').reset_index(drop=True)
    return df


def compute_velocities(timestamps, positions, quaternions):
    dt = np.maximum(np.diff(timestamps), 1e-6)
    pos_delta = np.diff(positions, axis=0)
    linear_vel = np.linalg.norm(pos_delta, axis=1) / dt * 1000
    angular_changes = []
    for i in range(len(quaternions) - 1):
        dot = np.clip(np.abs(np.dot(quaternions[i], quaternions[i + 1])), -1.0, 1.0)
        angular_changes.append(2 * np.arccos(dot))
    angular_vel = np.degrees(np.array(angular_changes) / dt)
    return linear_vel, angular_vel


candidates = []
for ep in sorted(by_ep.keys()):
    segs = sorted(by_ep[ep], key=lambda x: x['motion_start_frame'])
    for c in cluster(segs, 15):
        span = (c[-1]['motion_end_frame'] - c[0]['motion_start_frame']) / FPS
        motion = sum(s['duration_s'] for s in c)
        ratio = motion / span if span > 0 else 0
        if 20 <= span <= 80 and ratio >= 0.15 and len(c) >= 2:
            candidates.append({
                'episode': ep, 'type': 'cluster',
                'segs': [s['segment_index'] for s in c],
                'start': c[0]['motion_start_frame'], 'end': c[-1]['motion_end_frame'],
                'sub_segs': [(s['motion_start_frame'], s['motion_end_frame']) for s in c],
                'span_s': span, 'motion_s': motion, 'ratio': ratio,
                'reported_max_lin': max(s['max_linear_vel_mm_s'] for s in c),
                'reported_max_ang': max(s['max_angular_vel_deg_s'] for s in c),
                'path_mm': sum(s['path_length_mm'] for s in c),
                'rot_deg': sum(s['total_rotation_deg'] for s in c),
            })
    for s in segs:
        if s['duration_s'] >= 10:
            candidates.append({
                'episode': ep, 'type': 'single',
                'segs': [s['segment_index']],
                'start': s['motion_start_frame'], 'end': s['motion_end_frame'],
                'sub_segs': [(s['motion_start_frame'], s['motion_end_frame'])],
                'span_s': s['duration_s'], 'motion_s': s['duration_s'], 'ratio': 1.0,
                'reported_max_lin': s['max_linear_vel_mm_s'],
                'reported_max_ang': s['max_angular_vel_deg_s'],
                'path_mm': s['path_length_mm'],
                'rot_deg': s['total_rotation_deg'],
            })

print(f"Total candidates: {len(candidates)}")
print(f"Episodes needed: {sorted(set(c['episode'] for c in candidates))}")

verified = []
for ep in sorted(set(c['episode'] for c in candidates)):
    print(f"\nLoading {ep}...", flush=True)
    df = load_episode_lightweight(ep)
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quaternions = ecm[:, :3], ecm[:, 3:7]
    lin_vel, ang_vel = compute_velocities(timestamps, positions, quaternions)

    for cand in [c for c in candidates if c['episode'] == ep]:
        s, e = cand['start'], cand['end']
        is_motion = np.zeros(e - s + 1, dtype=bool)
        for ms, me in cand['sub_segs']:
            is_motion[max(0, ms - s):min(e - s + 1, me - s + 1)] = True
        v_start = s
        v_end = min(e, len(lin_vel))
        seg_lin = lin_vel[v_start:v_end]
        seg_ang = ang_vel[v_start:v_end]
        is_motion_v = is_motion[:len(seg_lin)]
        motion_lin = seg_lin[is_motion_v] if is_motion_v.any() else np.array([0])
        gap_lin = seg_lin[~is_motion_v] if (~is_motion_v).any() else np.array([0])
        motion_ang = seg_ang[is_motion_v] if is_motion_v.any() else np.array([0])
        gap_ang = seg_ang[~is_motion_v] if (~is_motion_v).any() else np.array([0])
        glitch_count = int(np.sum(seg_lin > 200))
        ts_deltas = np.diff(timestamps[s:e + 1])
        ts_anomalies = int(np.sum(ts_deltas > 0.05))

        cand['actual_max_lin'] = float(seg_lin.max()) if len(seg_lin) > 0 else 0
        cand['actual_max_ang'] = float(seg_ang.max()) if len(seg_ang) > 0 else 0
        cand['motion_mean_lin'] = float(motion_lin.mean())
        cand['gap_mean_lin'] = float(gap_lin.mean())
        cand['gap_max_lin'] = float(gap_lin.max())
        cand['motion_mean_ang'] = float(motion_ang.mean())
        cand['gap_mean_ang'] = float(gap_ang.mean())
        cand['glitch_count_lin_gt200'] = glitch_count
        cand['ts_anomalies_gt50ms'] = ts_anomalies
        verified.append(cand)
    del df

print('\n' + '=' * 160)
print('VERIFICATION REPORT')
print('=' * 160)
print(f"{'Ep':<5} {'Type':<6} {'Segs':<14} {'Frames':<16} {'Span':<6} {'Mot(s)':<7} {'Ratio':<6} {'RepLin':<8} {'ActLin':<8} {'GapMeanL':<10} {'GapMaxL':<9} {'Glitch':<7} {'TSan':<5}")
print('-' * 160)
for c in verified:
    seg_str = '-'.join(map(str, [c['segs'][0], c['segs'][-1]])) if len(c['segs']) > 1 else str(c['segs'][0])
    fr = f"{c['start']}-{c['end']}"
    print(f"{c['episode']:<5} {c['type']:<6} {seg_str:<14} {fr:<16} {c['span_s']:<5.1f} {c['motion_s']:<6.2f} {c['ratio']:<5.2f} {c['reported_max_lin']:<7.2f} {c['actual_max_lin']:<7.2f} {c['gap_mean_lin']:<9.4f} {c['gap_max_lin']:<8.4f} {c['glitch_count_lin_gt200']:<6} {c['ts_anomalies_gt50ms']:<4}")

with open('c:/Users/benli/sam3facebook/scripts/verified_candidates.json', 'w') as f:
    json.dump(verified, f, indent=2)
print(f'\nVerified {len(verified)} candidates. Saved JSON.')
