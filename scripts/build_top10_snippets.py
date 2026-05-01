"""
Build 8 unique top-ranked motion snippets (de-duplicated from top-10) directly from CRCD
parquets, into the TUM-RGB-D + Replica-style layout at F:/Datasets/CRCD/.

For each spec:
  - Load parquet rows in [padded_start, padded_end] inclusive (frames + ECM + PSM)
  - Write rgb/, rgbright/ (PNG), depth/ + semantic_instance/ (.gitkeep placeholders)
  - Write rgb.txt, rgbright.txt, depth.txt, associations.txt
  - Write groundtruth.txt (TUM quaternion poses)
  - Write intrinsics.yaml (depth_scale=10000, frame counts)
  - Write scene_motion.json (tool/camera classification from PSM)
  - Build video_left.mp4 (60fps, libx264)
  - For clusters: write cluster_metadata.json with sub-segment kinematics

Layout per snippet (matches data/Segments/ canonical layout):
  F:/Datasets/CRCD/{Episode}/snippet_NNN/
    ├── rgb/frame_NNNNNN.png
    ├── rgbright/frame_NNNNNN.png
    ├── depth/.gitkeep                       (empty placeholder)
    ├── semantic_instance/.gitkeep           (empty placeholder)
    ├── rgb.txt / rgbright.txt / depth.txt
    ├── associations.txt
    ├── groundtruth.txt
    ├── intrinsics.yaml
    ├── scene_motion.json
    ├── video_left.mp4
    └── cluster_metadata.json                (cluster snippets only)
"""
import io
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
OUT_ROOT = Path(r"F:/Datasets/CRCD")
FPS = 60
PAD_FRAMES = 60
PNG_DEPTH_SCALE = 10000
TOOL_THRESH_MM_S = 1.0

# 8 unique top-ranked specs (after de-duplicating G_2 #16 inside G_2 cluster, E_1 #13 inside E_1 cluster)
# Per-episode snippet numbering. Each spec produces F:/Datasets/CRCD/{episode}/snippet_NNN/
SPECS = [
    # F_1: 2 snippets
    {'episode': 'F_1', 'snippet_id': '001', 'rank': 1,
     'description': 'cluster of motion segments 1-5',
     'motion_start': 252, 'motion_end': 3730,
     'sub_segments': [
         {'id': 1, 'motion_start': 252,  'motion_end': 694},
         {'id': 2, 'motion_start': 1233, 'motion_end': 1691},
         {'id': 3, 'motion_start': 2363, 'motion_end': 2482},
         {'id': 4, 'motion_start': 3082, 'motion_end': 3175},
         {'id': 5, 'motion_start': 3500, 'motion_end': 3730},
     ],
     'source_segment_ids': [1, 2, 3, 4, 5]},

    {'episode': 'F_1', 'snippet_id': '002', 'rank': 2,
     'description': 'single motion segment #6',
     'motion_start': 11219, 'motion_end': 12385,
     'sub_segments': None,
     'source_segment_ids': [6]},

    # G_2: 1 snippet
    {'episode': 'G_2', 'snippet_id': '001', 'rank': 5,
     'description': 'cluster of motion segments 13-17',
     'motion_start': 39191, 'motion_end': 42053,
     'sub_segments': [
         {'id': 13, 'motion_start': 39191, 'motion_end': 39268},
         {'id': 14, 'motion_start': 39593, 'motion_end': 39687},
         {'id': 15, 'motion_start': 40462, 'motion_end': 40546},
         {'id': 16, 'motion_start': 40902, 'motion_end': 41648},
         {'id': 17, 'motion_start': 41954, 'motion_end': 42053},
     ],
     'source_segment_ids': [13, 14, 15, 16, 17]},

    # C_2: 1 snippet
    {'episode': 'C_2', 'snippet_id': '001', 'rank': 6,
     'description': 'single motion segment #4',
     'motion_start': 30538, 'motion_end': 31147,
     'sub_segments': None,
     'source_segment_ids': [4]},

    # E_1: 1 snippet
    {'episode': 'E_1', 'snippet_id': '001', 'rank': 7,
     'description': 'cluster of motion segments 11-13',
     'motion_start': 37740, 'motion_end': 39727,
     'sub_segments': [
         {'id': 11, 'motion_start': 37740, 'motion_end': 37799},
         {'id': 12, 'motion_start': 38568, 'motion_end': 38640},
         {'id': 13, 'motion_start': 39138, 'motion_end': 39727},
     ],
     'source_segment_ids': [11, 12, 13]},

    # G_3: 1 snippet
    {'episode': 'G_3', 'snippet_id': '001', 'rank': 8,
     'description': 'cluster of motion segments 1-4',
     'motion_start': 0, 'motion_end': 1926,
     'sub_segments': [
         {'id': 1, 'motion_start': 0,    'motion_end': 13},
         {'id': 2, 'motion_start': 359,  'motion_end': 464},
         {'id': 3, 'motion_start': 880,  'motion_end': 929},
         {'id': 4, 'motion_start': 1633, 'motion_end': 1926},
     ],
     'source_segment_ids': [1, 2, 3, 4]},

    # C_3: 1 snippet
    {'episode': 'C_3', 'snippet_id': '001', 'rank': 9,
     'description': 'cluster of motion segments 1-3',
     'motion_start': 529, 'motion_end': 1935,
     'sub_segments': [
         {'id': 1, 'motion_start': 529,  'motion_end': 644},
         {'id': 2, 'motion_start': 1110, 'motion_end': 1391},
         {'id': 3, 'motion_start': 1822, 'motion_end': 1935},
     ],
     'source_segment_ids': [1, 2, 3]},

    # D_3 (rank 10) DROPPED on 2026-04-29 — D_3 episode is mono-only in CRCD source
    # parquet (frame_right is null for all rows). User chose to remove rather than keep
    # as mono. To re-add, uncomment below and the build script will produce a
    # mono-only snippet with no rgbright/ artefacts.
    # {'episode': 'D_3', 'snippet_id': '001', 'rank': 10,
    #  'description': 'cluster of motion segments 1-3',
    #  'motion_start': 2239, 'motion_end': 3638,
    #  'sub_segments': [
    #      {'id': 1, 'motion_start': 2239, 'motion_end': 2399},
    #      {'id': 2, 'motion_start': 2725, 'motion_end': 2785},
    #      {'id': 3, 'motion_start': 3554, 'motion_end': 3638},
    #  ],
    #  'source_segment_ids': [1, 2, 3]},
]


# ----------- core helpers -----------

def load_parquet_range(episode, start, end, with_frames=True):
    """Load parquet rows in [start, end] inclusive (frame_n)."""
    files = sorted((PARQUET_DIR / episode).glob("*.parquet"))
    cols = None if with_frames else ['frame_n', 'timestamp',
                                     '/ECM/custom/setpoint_cp',
                                     '/PSM1/custom/setpoint_cp',
                                     '/PSM2/custom/setpoint_cp']
    rows = []
    for pf in files:
        df = pd.read_parquet(pf, columns=cols)
        df = df[(df['frame_n'] >= start) & (df['frame_n'] <= end)]
        if len(df):
            rows.append(df)
        del df
    if not rows:
        raise RuntimeError(f"No frames in [{start},{end}] for {episode}")
    out = pd.concat(rows, ignore_index=True).sort_values('frame_n').reset_index(drop=True)
    return out


def compute_velocities(timestamps, positions, quaternions):
    dt = np.maximum(np.diff(timestamps), 1e-6)
    pos_delta = np.diff(positions, axis=0)
    linear_vel = np.linalg.norm(pos_delta, axis=1) / dt * 1000
    ang_changes = []
    for i in range(len(quaternions) - 1):
        dot = np.clip(np.abs(np.dot(quaternions[i], quaternions[i + 1])), -1.0, 1.0)
        ang_changes.append(2 * np.arccos(dot))
    angular_vel = np.degrees(np.array(ang_changes) / dt)
    return linear_vel, angular_vel


# ----------- output writers -----------

def write_rgb_frames(df, snip_dir):
    """Write rgb/ and rgbright/ as full-HD PNG.

    Returns dict with 'left_count' and 'right_count'. If right-camera frames are
    all null (e.g. D_2/D_3 episodes), rgbright/ is removed afterwards.
    """
    counts = {'left': 0, 'right': 0}
    for side, col, dir_name in [('left', 'frame_left', 'rgb'),
                                ('right', 'frame_right', 'rgbright')]:
        out_dir = snip_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            fr = int(row['frame_n'])
            fdata = row[col]
            if fdata is None or 'bytes' not in fdata:
                continue
            img = Image.open(io.BytesIO(fdata['bytes'])).convert('RGB')
            img.save(str(out_dir / f'frame_{fr:06d}.png'), format='PNG', compress_level=3)
            counts[side] += 1
    # If right is empty, remove the empty folder so the snippet is cleanly mono
    if counts['right'] == 0:
        import shutil
        shutil.rmtree(snip_dir / 'rgbright', ignore_errors=True)
    return counts


def write_groundtruth(df, snip_dir, spec):
    """TUM-format pose file from ECM kinematics."""
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    out = snip_dir / 'groundtruth.txt'
    sub_str = ', '.join(f"({s['motion_start']},{s['motion_end']})" for s in (spec.get('sub_segments') or []))
    with open(out, 'w') as f:
        f.write('# ground truth trajectory\n')
        f.write('# TUM format: timestamp tx ty tz qx qy qz qw\n')
        f.write(f"# {spec['episode']} {spec['description']} | padded "
                f"{spec['_padded_start']}-{spec['_padded_end']} | motion "
                f"{spec['motion_start']}-{spec['motion_end']}\n")
        if sub_str:
            f.write(f'# Sub-segments: [{sub_str}]\n')
        for ts, pose in zip(timestamps, ecm):
            x, y, z = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            f.write(f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def write_index_files(df, snip_dir, has_right=True):
    """Write rgb.txt, rgbright.txt (if has_right), depth.txt, associations.txt."""
    timestamps = df['timestamp'].values
    frame_ns = df['frame_n'].values
    snip_name = snip_dir.name

    plan = [
        ('rgb.txt', 'rgb', f'RGB images for snippet {snip_name}'),
        ('depth.txt', 'depth', f'depth images for snippet {snip_name}'),
    ]
    if has_right:
        plan.append(('rgbright.txt', 'rgbright', f'right-camera RGB images for snippet {snip_name}'))

    for fname, folder, header in plan:
        with open(snip_dir / fname, 'w') as f:
            f.write(f'# {header}\n')
            f.write('# timestamp filename\n')
            for ts, fr in zip(timestamps, frame_ns):
                f.write(f'{ts:.9f} {folder}/frame_{int(fr):06d}.png\n')

    with open(snip_dir / 'associations.txt', 'w') as f:
        f.write('# RGB-depth associations (1:1 since synthesised)\n')
        f.write('# ts_rgb rgb_path ts_depth depth_path\n')
        for ts, fr in zip(timestamps, frame_ns):
            f.write(f'{ts:.9f} rgb/frame_{int(fr):06d}.png {ts:.9f} depth/frame_{int(fr):06d}.png\n')


def write_intrinsics(snip_dir, spec, n_frames, has_right=True):
    if has_right:
        stereo_block = """stereo:
  available: true
  baseline_m: TBD     # da Vinci stereo baseline TBD
  right_camera_relative_pose: TBD"""
        right_dev = '  - "rgbright/ folder added for stereo right camera (TUM is monocular)."'
    else:
        stereo_block = f"""stereo:
  available: false        # MONO ONLY for {spec['episode']} episode
  reason: "{spec['episode']} episode missing right-camera data in CRCD source parquet."
  baseline_m: null
  right_camera_relative_pose: null"""
        right_dev = '  - "MONO ONLY: rgbright/ and rgbright.txt absent because CRCD source episode has no right-camera frames."'

    yaml_str = f"""camera:
  width: 1280
  height: 720
  fps: {FPS}
  fx: TBD             # CRCD da Vinci calibration not yet sourced
  fy: TBD
  cx: TBD
  cy: TBD
  distortion: [TBD, TBD, TBD, TBD, TBD]   # k1 k2 p1 p2 k3
depth:
  png_depth_scale: {PNG_DEPTH_SCALE}   # custom: 0.1 mm precision, max 6.55 m
  units: metres
  # NOTE: differs from TUM default (5000) and Replica default (6553.5).
  # Standard TUM/Replica loaders must be configured to read this value, not assume the default.
{stereo_block}
deviations_from_tum:
  - "Frame names use sequential frame_NNNNNN.png instead of TUM Unix-timestamp filenames."
{right_dev}
  - "depth/ scale = {PNG_DEPTH_SCALE} instead of TUM default 5000."
deviations_from_replica:
  - "Pose format is TUM quaternion (groundtruth.txt), not Replica 4x4 matrix (traj.txt)."
  - "Depth scale {PNG_DEPTH_SCALE} instead of Replica's 6553.5."
segmentation:
  state: absent       # candidate snippet — no CRCD GT polygons yet
  # info_semantic.json intentionally omitted (no class taxonomy until annotated).
frames:
  count: {n_frames}
  first: {spec['_padded_start']}
  last: {spec['_padded_end']}
  motion_start: {spec['motion_start']}
  motion_end: {spec['motion_end']}
rank: {spec['rank']}
description: "{spec['description']}"
source_segment_ids: {spec['source_segment_ids']}
"""
    (snip_dir / 'intrinsics.yaml').write_text(yaml_str)


def write_placeholders(snip_dir):
    """depth/ and semantic_instance/ folders with .gitkeep, no PNGs."""
    for name in ('depth', 'semantic_instance'):
        d = snip_dir / name
        d.mkdir(exist_ok=True)
        (d / '.gitkeep').touch()


def write_scene_motion(df, snip_dir, spec):
    """Classify camera + tool motion across the snippet, plus per-sub-segment if cluster."""
    timestamps = df['timestamp'].values
    frame_ns = df['frame_n'].values
    psm1 = np.array(df['/PSM1/custom/setpoint_cp'].tolist())
    psm2 = np.array(df['/PSM2/custom/setpoint_cp'].tolist())

    def _stats(ms, me):
        ms_i = int(np.searchsorted(frame_ns, ms))
        me_i = min(int(np.searchsorted(frame_ns, me)), len(timestamps) - 1)
        if me_i <= ms_i + 1:
            return {'tool_motion': False, 'psm1_motion': False, 'psm2_motion': False,
                    'psm1_max_vel_mm_s': 0.0, 'psm2_max_vel_mm_s': 0.0}
        ts = timestamps[ms_i:me_i + 1]
        dt = np.maximum(np.diff(ts), 1e-6)
        v1 = np.linalg.norm(np.diff(psm1[ms_i:me_i + 1, :3], axis=0), axis=1) / dt * 1000
        v2 = np.linalg.norm(np.diff(psm2[ms_i:me_i + 1, :3], axis=0), axis=1) / dt * 1000
        return {
            'tool_motion': bool((v1 > TOOL_THRESH_MM_S).any() or (v2 > TOOL_THRESH_MM_S).any()),
            'psm1_motion': bool((v1 > TOOL_THRESH_MM_S).any()),
            'psm2_motion': bool((v2 > TOOL_THRESH_MM_S).any()),
            'psm1_max_vel_mm_s': float(v1.max()) if len(v1) else 0.0,
            'psm2_max_vel_mm_s': float(v2.max()) if len(v2) else 0.0,
        }

    overall = _stats(spec['motion_start'], spec['motion_end'])
    out = {
        'motion_type': 'tool_and_camera' if overall['tool_motion'] else 'camera_only',
        'camera_motion': True,
        'tool_motion': overall['tool_motion'],
        'psm1_motion': overall['psm1_motion'],
        'psm2_motion': overall['psm2_motion'],
        'psm1_max_vel_mm_s': overall['psm1_max_vel_mm_s'],
        'psm2_max_vel_mm_s': overall['psm2_max_vel_mm_s'],
    }
    if spec.get('sub_segments'):
        per_sub = []
        for ss in spec['sub_segments']:
            m = _stats(ss['motion_start'], ss['motion_end'])
            m.update({'sub_segment_id': ss['id'],
                      'motion_start_frame': ss['motion_start'],
                      'motion_end_frame': ss['motion_end'],
                      'motion_type': 'tool_and_camera' if m['tool_motion'] else 'camera_only'})
            per_sub.append(m)
        out['sub_segments'] = per_sub

    (snip_dir / 'scene_motion.json').write_text(json.dumps(out, indent=2))


def write_cluster_metadata(df, snip_dir, spec):
    """Per-sub-segment kinematics for cluster snippets."""
    if not spec.get('sub_segments'):
        return
    timestamps = df['timestamp'].values
    frame_ns = df['frame_n'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quats = ecm[:, :3], ecm[:, 3:7]
    lin_v, ang_v = compute_velocities(timestamps, positions, quats)
    duration = float(timestamps[-1] - timestamps[0])

    sub_meta = []
    for ss in spec['sub_segments']:
        ms_i = int(np.searchsorted(frame_ns, ss['motion_start']))
        me_i = min(int(np.searchsorted(frame_ns, ss['motion_end'])), len(timestamps) - 1)
        v_lin = lin_v[ms_i:me_i] if me_i > ms_i else np.array([0.0])
        v_ang = ang_v[ms_i:me_i] if me_i > ms_i else np.array([0.0])
        pos_mm = positions[ms_i:me_i + 1] * 1000
        sub_meta.append({
            'sub_segment_id': ss['id'],
            'motion_start_frame': ss['motion_start'],
            'motion_end_frame': ss['motion_end'],
            'duration_s': float(timestamps[me_i] - timestamps[ms_i]),
            'motion_frames': int(me_i - ms_i + 1),
            'max_linear_vel_mm_s': float(v_lin.max()) if len(v_lin) else 0.0,
            'mean_linear_vel_mm_s': float(v_lin.mean()) if len(v_lin) else 0.0,
            'max_angular_vel_deg_s': float(v_ang.max()) if len(v_ang) else 0.0,
            'mean_angular_vel_deg_s': float(v_ang.mean()) if len(v_ang) else 0.0,
            'path_length_mm': float(np.sum(np.linalg.norm(np.diff(pos_mm, axis=0), axis=1))) if len(pos_mm) > 1 else 0.0,
            'displacement_mm': float(np.linalg.norm(pos_mm[-1] - pos_mm[0])) if len(pos_mm) > 1 else 0.0,
        })

    metadata = {
        'episode': spec['episode'],
        'snippet_id': spec['snippet_id'],
        'snippet_type': 'multi_segment_cluster',
        'rank': spec['rank'],
        'description': spec['description'],
        'padded_start_frame': spec['_padded_start'],
        'padded_end_frame': spec['_padded_end'],
        'motion_start_frame': spec['motion_start'],
        'motion_end_frame': spec['motion_end'],
        'total_frames': int(len(df)),
        'duration_s': duration,
        'total_motion_seconds': float(sum(s['duration_s'] for s in sub_meta)),
        'motion_ratio': float(sum(s['duration_s'] for s in sub_meta) / duration) if duration > 0 else 0,
        'fps': FPS,
        'overall_max_linear_vel_mm_s': float(lin_v.max()) if len(lin_v) else 0.0,
        'overall_max_angular_vel_deg_s': float(ang_v.max()) if len(ang_v) else 0.0,
        'overall_path_length_mm': float(np.sum(np.linalg.norm(np.diff(positions * 1000, axis=0), axis=1))),
        'sub_segments': sub_meta,
        'source_segment_ids_in_motion_analysis': spec['source_segment_ids'],
    }
    (snip_dir / 'cluster_metadata.json').write_text(json.dumps(metadata, indent=2))


def build_video_left(snip_dir):
    files = sorted((snip_dir / 'rgb').glob('frame_*.png'))
    if not files:
        return 0
    frames = []
    for f in files:
        img = np.array(Image.open(str(f)).convert('RGB'))
        frames.append(img)
    imageio.mimwrite(str(snip_dir / 'video_left.mp4'), frames,
                     fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    return len(frames)


# ----------- per-episode index file -----------

def write_episode_index(episode):
    """Write F:/Datasets/CRCD/{Episode}/{Episode}_snippets.json with all snippets in that episode."""
    ep_dir = OUT_ROOT / episode
    if not ep_dir.exists():
        return
    entries = []
    for snip_dir in sorted(ep_dir.iterdir()):
        if not snip_dir.is_dir() or snip_dir.name.startswith('.'):
            continue
        # Try cluster_metadata.json first, fall back to intrinsics.yaml
        meta_path = snip_dir / 'cluster_metadata.json'
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            entries.append({
                'snippet_id': m['snippet_id'],
                'snippet_type': m['snippet_type'],
                'rank': m['rank'],
                'description': m['description'],
                'padded_start_frame': m['padded_start_frame'],
                'padded_end_frame': m['padded_end_frame'],
                'motion_start_frame': m['motion_start_frame'],
                'motion_end_frame': m['motion_end_frame'],
                'total_frames': m['total_frames'],
                'duration_s': m['duration_s'],
                'motion_ratio': m['motion_ratio'],
                'source_segment_ids': m['source_segment_ids_in_motion_analysis'],
            })
        else:
            # Single-segment: read intrinsics.yaml lightly
            import yaml
            intr = yaml.safe_load((snip_dir / 'intrinsics.yaml').read_text())
            entries.append({
                'snippet_id': snip_dir.name.replace('snippet_', ''),
                'snippet_type': 'single_segment',
                'rank': intr.get('rank'),
                'description': intr.get('description'),
                'padded_start_frame': intr['frames']['first'],
                'padded_end_frame': intr['frames']['last'],
                'motion_start_frame': intr['frames']['motion_start'],
                'motion_end_frame': intr['frames']['motion_end'],
                'total_frames': intr['frames']['count'],
                'source_segment_ids': intr.get('source_segment_ids', []),
            })
    entries.sort(key=lambda x: x['snippet_id'])
    (ep_dir / f'{episode}_snippets.json').write_text(json.dumps(entries, indent=2))


# ----------- main -----------

def build_one(spec):
    spec['_padded_start'] = max(0, spec['motion_start'] - PAD_FRAMES)
    spec['_padded_end'] = spec['motion_end'] + PAD_FRAMES
    snip_dir = OUT_ROOT / spec['episode'] / f"snippet_{spec['snippet_id']}"
    snip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {spec['episode']}/snippet_{spec['snippet_id']} (rank #{spec['rank']}) ===")
    print(f"  {spec['description']}")
    print(f"  padded {spec['_padded_start']}-{spec['_padded_end']}, motion {spec['motion_start']}-{spec['motion_end']}")

    print(f"  loading parquet ...", flush=True)
    df = load_parquet_range(spec['episode'], spec['_padded_start'], spec['_padded_end'], with_frames=True)
    print(f"    {len(df)} frames")

    print(f"  writing rgb/ + rgbright/ PNGs ...", flush=True)
    counts = write_rgb_frames(df, snip_dir)
    has_right = counts['right'] > 0
    if not has_right:
        print(f"    WARNING: {spec['episode']} has no right-camera frames; building mono-only snippet")

    print(f"  writing TUM index files + groundtruth ...")
    write_index_files(df, snip_dir, has_right=has_right)
    write_groundtruth(df, snip_dir, spec)

    print(f"  writing intrinsics.yaml + scene_motion.json + placeholders ...")
    write_intrinsics(snip_dir, spec, len(df), has_right=has_right)
    write_scene_motion(df, snip_dir, spec)
    write_placeholders(snip_dir)

    if spec.get('sub_segments'):
        print(f"  writing cluster_metadata.json ...")
        write_cluster_metadata(df, snip_dir, spec)

    print(f"  encoding video_left.mp4 ...", flush=True)
    n = build_video_left(snip_dir)
    print(f"    {n} frames at libx264")

    del df
    print(f"  done: {snip_dir}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        build_one(spec)

    print('\n=== writing per-episode index files ===')
    for ep in sorted(set(s['episode'] for s in SPECS)):
        write_episode_index(ep)
        print(f'  {ep}_snippets.json')

    print('\nAll done.')


if __name__ == '__main__':
    main()
