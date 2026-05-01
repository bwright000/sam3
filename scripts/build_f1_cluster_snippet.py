"""
Build a multi-segment cluster snippet for F_1 segs #1-5.

Cluster spec:
  motion frames     252-3730 (58.0s span, 22.4s motion, ratio 0.39)
  padded frames     192-3790 (with +/-60 frame pad)
  sub-segments      (252,694), (1233,1691), (2363,2482), (3082,3175), (3500,3730)

Output structure mirrors extract_motion_snippets.py outputs but adds cluster_metadata.json
describing the 5 sub-segments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import imageio
import base64

PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
SNIPPETS_DIR = Path(r"f:/2026 vibes/MPHY Project/Detailed Analysis/snippets/F_1")

EPISODE = "F_1"
CLUSTER = {
    "name": "snippet_cluster_segs1to5",
    "motion_start": 252,
    "motion_end": 3730,
    "padded_start": 192,
    "padded_end": 3790,
    "sub_segments": [
        {"id": 1, "motion_start": 252,  "motion_end": 694},
        {"id": 2, "motion_start": 1233, "motion_end": 1691},
        {"id": 3, "motion_start": 2363, "motion_end": 2482},
        {"id": 4, "motion_start": 3082, "motion_end": 3175},
        {"id": 5, "motion_start": 3500, "motion_end": 3730},
    ],
}

FPS = 60
MOTION_THRESH_LIN = 0.1
MOTION_THRESH_ANG = 0.1
TOOL_THRESH = 1.0  # mm/s for PSM tool motion classification


def load_full_episode():
    """Load all parquet files (with frames) for the cluster's padded range only.
    Memory: filter per-file, only keep cluster-range rows."""
    files = sorted((PARQUET_DIR / EPISODE).glob("*.parquet"))
    s, e = CLUSTER['padded_start'], CLUSTER['padded_end']
    rows = []
    for pf in files:
        print(f"  reading {pf.name}...", flush=True)
        df = pd.read_parquet(pf)
        df = df[(df['frame_n'] >= s) & (df['frame_n'] <= e)]
        if len(df) > 0:
            rows.append(df)
        del df
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


def export_tum_poses(df, out_path):
    """TUM format: timestamp tx ty tz qx qy qz qw"""
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    frame_ns = df['frame_n'].values
    with open(out_path, 'w') as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write(f"# F_1 cluster snippet, padded frames {CLUSTER['padded_start']}-{CLUSTER['padded_end']}, motion frames {CLUSTER['motion_start']}-{CLUSTER['motion_end']}\n")
        f.write(f"# Sub-segments: {[(s['motion_start'], s['motion_end']) for s in CLUSTER['sub_segments']]}\n")
        for ts, fr, pose in zip(timestamps, frame_ns, ecm):
            x, y, z = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            f.write(f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def make_velocity_plot(df, out_path):
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quats = ecm[:, :3], ecm[:, 3:7]
    lin_v, ang_v = compute_velocities(timestamps, positions, quats)
    frame_ns = df['frame_n'].values

    t_rel = timestamps - timestamps[0]
    t_v = t_rel[:-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(t_rel[0], t_rel[-1], alpha=0.15, color='lightgray', label='Padding/gap')
    legend_added = False
    for ss in CLUSTER['sub_segments']:
        ms_idx = np.searchsorted(frame_ns, ss['motion_start'])
        me_idx = np.searchsorted(frame_ns, ss['motion_end'])
        if ms_idx < len(t_rel) and me_idx < len(t_rel):
            for ax in (ax1, ax2):
                lbl = 'Motion sub-segment' if not legend_added else None
                ax.axvspan(t_rel[ms_idx], t_rel[me_idx], alpha=0.30, color='gold', label=lbl)
            legend_added = True

    ax1.plot(t_v, lin_v, 'b-', lw=0.7)
    ax1.axhline(y=MOTION_THRESH_LIN, color='r', ls='--', alpha=0.5, label=f'Threshold ({MOTION_THRESH_LIN} mm/s)')
    ax1.set_ylabel('Linear velocity (mm/s)')
    ax1.set_title(f"F_1 Cluster #1-5 | padded frames {CLUSTER['padded_start']}-{CLUSTER['padded_end']} | motion {CLUSTER['motion_start']}-{CLUSTER['motion_end']} | 5 sub-segments")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_v, ang_v, 'g-', lw=0.7)
    ax2.axhline(y=MOTION_THRESH_ANG, color='r', ls='--', alpha=0.5, label=f'Threshold ({MOTION_THRESH_ANG} deg/s)')
    ax2.set_ylabel('Angular velocity (deg/s)')
    ax2.set_xlabel('Time within snippet (s)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def classify_scene_motion(df):
    """Compute motion type both overall (for whole padded span) and per sub-segment.
    Uses PSM kinematics to detect tool motion."""
    timestamps = df['timestamp'].values
    frame_ns = df['frame_n'].values
    psm1 = np.array(df['/PSM1/custom/setpoint_cp'].tolist())
    psm2 = np.array(df['/PSM2/custom/setpoint_cp'].tolist())

    def motion_in_range(ms, me):
        ms_i = np.searchsorted(frame_ns, ms)
        me_i = min(np.searchsorted(frame_ns, me), len(timestamps) - 1)
        if me_i <= ms_i + 1:
            return {'tool_motion': False, 'psm1_motion': False, 'psm2_motion': False,
                    'psm1_max_vel_mm_s': 0.0, 'psm2_max_vel_mm_s': 0.0}
        ts = timestamps[ms_i:me_i + 1]
        dt = np.maximum(np.diff(ts), 1e-6)
        v1 = np.linalg.norm(np.diff(psm1[ms_i:me_i + 1, :3], axis=0), axis=1) / dt * 1000
        v2 = np.linalg.norm(np.diff(psm2[ms_i:me_i + 1, :3], axis=0), axis=1) / dt * 1000
        return {
            'tool_motion': bool((v1 > TOOL_THRESH).any() or (v2 > TOOL_THRESH).any()),
            'psm1_motion': bool((v1 > TOOL_THRESH).any()),
            'psm2_motion': bool((v2 > TOOL_THRESH).any()),
            'psm1_max_vel_mm_s': float(v1.max()) if len(v1) else 0.0,
            'psm2_max_vel_mm_s': float(v2.max()) if len(v2) else 0.0,
        }

    overall = motion_in_range(CLUSTER['motion_start'], CLUSTER['motion_end'])
    per_sub = []
    for ss in CLUSTER['sub_segments']:
        m = motion_in_range(ss['motion_start'], ss['motion_end'])
        m['sub_segment_id'] = ss['id']
        m['motion_start_frame'] = ss['motion_start']
        m['motion_end_frame'] = ss['motion_end']
        m['motion_type'] = 'tool_and_camera' if m['tool_motion'] else 'camera_only'
        per_sub.append(m)

    overall_type = 'tool_and_camera' if overall['tool_motion'] else 'camera_only'

    return {
        'motion_type': overall_type,
        'camera_motion': True,
        'tool_motion': overall['tool_motion'],
        'psm1_motion': overall['psm1_motion'],
        'psm2_motion': overall['psm2_motion'],
        'psm1_max_vel_mm_s': overall['psm1_max_vel_mm_s'],
        'psm2_max_vel_mm_s': overall['psm2_max_vel_mm_s'],
        'sub_segments': per_sub,
    }


def export_frames(df, out_dir):
    left_dir = out_dir / 'frames_left'
    right_dir = out_dir / 'frames_right'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        fr = int(row['frame_n'])
        if row['frame_left'] is not None and 'bytes' in row['frame_left']:
            with open(left_dir / f'frame_{fr:06d}.webp', 'wb') as f:
                f.write(row['frame_left']['bytes'])
        if row['frame_right'] is not None and 'bytes' in row['frame_right']:
            with open(right_dir / f'frame_{fr:06d}.webp', 'wb') as f:
                f.write(row['frame_right']['bytes'])


def export_video_left(df, out_path):
    frames = []
    for _, row in df.iterrows():
        fdata = row['frame_left']
        if fdata is not None and 'bytes' in fdata:
            img = Image.open(io.BytesIO(fdata['bytes'])).convert('RGB')
            frames.append(np.array(img))
    if frames:
        imageio.mimwrite(out_path, frames, fps=FPS, codec='libx264', quality=8)
    return len(frames)


def export_video_stereo(df, out_path):
    frames = []
    for _, row in df.iterrows():
        ld, rd = row['frame_left'], row['frame_right']
        if ld is None or rd is None:
            continue
        L = Image.open(io.BytesIO(ld['bytes'])).convert('RGB')
        R = Image.open(io.BytesIO(rd['bytes'])).convert('RGB')
        w, h = L.size
        Ls = L.resize((w // 2, h // 2))
        Rs = R.resize((w // 2, h // 2))
        combined = Image.new('RGB', (w, h // 2))
        combined.paste(Ls, (0, 0))
        combined.paste(Rs, (w // 2, 0))
        frames.append(np.array(combined))
    if frames:
        imageio.mimwrite(out_path, frames, fps=FPS, codec='libx264', quality=8)
    return len(frames)


def make_visualization_html(df, out_path):
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions = ecm[:, :3] * 1000
    lin_v, _ = compute_velocities(timestamps, positions / 1000, ecm[:, 3:7])
    vel = np.append(lin_v, lin_v[-1] if len(lin_v) else 0)
    frame_ns = df['frame_n'].values

    # Mark sub-segment frames
    in_motion = np.zeros(len(positions), dtype=bool)
    for ss in CLUSTER['sub_segments']:
        m = (frame_ns >= ss['motion_start']) & (frame_ns <= ss['motion_end'])
        in_motion |= m

    vel_max = max(0.5, float(np.percentile(vel, 99)))
    vel_norm = np.clip(vel / vel_max, 0, 1)
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in cm.viridis(vel_norm)]

    fig = go.Figure()
    for i in range(len(positions) - 1):
        if in_motion[i]:
            fig.add_trace(go.Scatter3d(
                x=[positions[i, 0], positions[i + 1, 0]],
                y=[positions[i, 1], positions[i + 1, 1]],
                z=[positions[i, 2], positions[i + 1, 2]],
                mode='lines', line=dict(color=colors[i], width=5),
                showlegend=False, hoverinfo='skip'))
        else:
            fig.add_trace(go.Scatter3d(
                x=[positions[i, 0], positions[i + 1, 0]],
                y=[positions[i, 1], positions[i + 1, 1]],
                z=[positions[i, 2], positions[i + 1, 2]],
                mode='markers', marker=dict(size=2, color='lightgray', opacity=0.5),
                showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        marker=dict(size=0.1, color=[0, vel_max], colorscale='Viridis', cmin=0, cmax=vel_max,
                    colorbar=dict(title='Vel (mm/s)', thickness=15, len=0.7)),
        showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                               mode='markers', marker=dict(size=8, color='lime'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                               mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='End'))

    fig.update_layout(
        title=f'F_1 Cluster #1-5 (frames {CLUSTER["padded_start"]}-{CLUSTER["padded_end"]})',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        height=600, margin=dict(l=0, r=0, t=50, b=0))

    # Sample frames for thumbnail strip
    n_thumbs = 12
    thumb_idx = np.linspace(0, len(df) - 1, n_thumbs, dtype=int)
    frames_info = []
    for idx in thumb_idx:
        row = df.iloc[idx]
        if row['frame_left'] is None:
            continue
        img = Image.open(io.BytesIO(row['frame_left']['bytes']))
        img.thumbnail((300, 170))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        fr = int(row['frame_n'])
        is_m = any(ss['motion_start'] <= fr <= ss['motion_end'] for ss in CLUSTER['sub_segments'])
        t = (timestamps[idx] - timestamps[0])
        frames_info.append({'frame': fr, 't': t, 'b64': b64, 'motion': is_m})

    duration = timestamps[-1] - timestamps[0]
    motion_total = sum((ss['motion_end'] - ss['motion_start'] + 1) / FPS for ss in CLUSTER['sub_segments'])
    html = f"""<!DOCTYPE html><html><head>
<title>F_1 Cluster #1-5 Visualization</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font-family:sans-serif;margin:0;padding:20px;background:#f5f5f5}}
.hdr{{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
.hdr h1{{margin:0 0 10px 0;color:#333}} .stats{{display:flex;gap:25px;color:#666;font-size:14px}}
.stats strong{{color:#333}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.panel{{background:#fff;border-radius:8px;padding:20px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
.panel h2{{margin:0 0 15px 0;color:#333;font-size:18px;border-bottom:2px solid #f4b400;padding-bottom:10px}}
#plot{{width:100%;height:600px}}
.thumbs{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;max-height:600px;overflow-y:auto}}
.thumb{{background:#fafafa;border-radius:5px;padding:6px;border:2px solid #ddd}}
.thumb.motion{{border-color:#f4b400}} .thumb img{{width:100%;border-radius:3px}}
.thumb p{{margin:6px 0 0 0;font-size:11px;color:#555;text-align:center}}
.video{{margin-top:12px;padding:10px;background:#e8f5e9;border-radius:5px;text-align:center}}
.video a{{color:#2e7d32;text-decoration:none;font-weight:500;margin:0 8px}}
</style></head><body>
<div class="hdr"><h1>F_1 Cluster #1-5 (multi-segment)</h1>
<div class="stats">
<span><strong>Padded frames:</strong> {CLUSTER['padded_start']}-{CLUSTER['padded_end']}</span>
<span><strong>Motion frames:</strong> {CLUSTER['motion_start']}-{CLUSTER['motion_end']}</span>
<span><strong>Span:</strong> {duration:.2f}s</span>
<span><strong>Motion total:</strong> {motion_total:.1f}s ({motion_total/duration*100:.0f}%)</span>
<span><strong>Sub-segments:</strong> {len(CLUSTER['sub_segments'])}</span>
<span><strong>Max vel:</strong> {vel_max:.1f} mm/s</span>
</div></div>
<div class="grid">
<div class="panel"><h2>3D Camera Trajectory</h2><div id="plot"></div>
<p style="font-size:12px;color:#888;margin-top:10px">Coloured = motion frames (viridis by velocity); grey markers = stationary gap frames</p></div>
<div class="panel"><h2>Sample Frames (Left Camera)</h2><div class="thumbs">"""
    for f in frames_info:
        cls = 'motion' if f['motion'] else ''
        html += f'<div class="thumb {cls}"><img src="data:image/jpeg;base64,{f["b64"]}"><p>f{f["frame"]} | t={f["t"]:.1f}s | {"MOTION" if f["motion"] else "gap"}</p></div>'
    html += f"""</div><div class="video">
<a href="video_left.mp4" target="_blank">Open Left Video</a> |
<a href="video_stereo.mp4" target="_blank">Open Stereo Video</a>
</div></div></div>
<script>var pd={fig.to_json()};Plotly.newPlot('plot',pd.data,pd.layout,{{responsive:true}});</script>
</body></html>"""
    with open(out_path, 'w') as f:
        f.write(html)


def main():
    out_dir = SNIPPETS_DIR / CLUSTER['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    print("Loading parquet data for cluster range...")
    df = load_full_episode()
    print(f"  loaded {len(df)} rows (frames {df['frame_n'].min()} to {df['frame_n'].max()})")

    timestamps = df['timestamp'].values
    duration = timestamps[-1] - timestamps[0]
    print(f"  duration {duration:.2f}s")

    print("Exporting TUM poses...")
    export_tum_poses(df, out_dir / 'poses.txt')

    print("Building velocity plot...")
    make_velocity_plot(df, out_dir / 'velocity.png')

    print("Classifying scene motion (overall + per sub-segment)...")
    scene = classify_scene_motion(df)
    with open(out_dir / 'scene_motion.json', 'w') as f:
        json.dump(scene, f, indent=2)
    print(f"  overall motion_type: {scene['motion_type']}")

    print("Writing cluster_metadata.json...")
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quats = ecm[:, :3], ecm[:, 3:7]
    lin_v, ang_v = compute_velocities(timestamps, positions, quats)
    frame_ns = df['frame_n'].values

    sub_meta = []
    for ss in CLUSTER['sub_segments']:
        ms_i = np.searchsorted(frame_ns, ss['motion_start'])
        me_i = min(np.searchsorted(frame_ns, ss['motion_end']), len(timestamps) - 1)
        v_lin = lin_v[ms_i:me_i] if me_i > ms_i else np.array([0])
        v_ang = ang_v[ms_i:me_i] if me_i > ms_i else np.array([0])
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
        'episode': EPISODE,
        'snippet_id': 'cluster_segs1to5',
        'snippet_type': 'multi_segment_cluster',
        'padded_start_frame': CLUSTER['padded_start'],
        'padded_end_frame': CLUSTER['padded_end'],
        'motion_start_frame': CLUSTER['motion_start'],
        'motion_end_frame': CLUSTER['motion_end'],
        'total_frames': int(len(df)),
        'duration_s': float(duration),
        'total_motion_seconds': float(sum(s['duration_s'] for s in sub_meta)),
        'motion_ratio': float(sum(s['duration_s'] for s in sub_meta) / duration),
        'fps': FPS,
        'overall_max_linear_vel_mm_s': float(lin_v.max()) if len(lin_v) else 0.0,
        'overall_max_angular_vel_deg_s': float(ang_v.max()) if len(ang_v) else 0.0,
        'overall_path_length_mm': float(np.sum(np.linalg.norm(np.diff(positions * 1000, axis=0), axis=1))),
        'sub_segments': sub_meta,
        'source_segment_ids_in_F1_snippets_json': [1, 2, 3, 4, 5],
    }
    with open(out_dir / 'cluster_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Exporting frames (left + right webp)...")
    export_frames(df, out_dir)

    print("Exporting left video...")
    n_left = export_video_left(df, out_dir / 'video_left.mp4')
    print(f"  {n_left} frames")

    print("Exporting stereo video...")
    n_st = export_video_stereo(df, out_dir / 'video_stereo.mp4')
    print(f"  {n_st} frames")

    print("Building visualization HTML...")
    make_visualization_html(df, out_dir / 'visualization.html')

    print(f"\nDone. Snippet built at: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        sz = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        print(f"  {p.name}  {sz/1024:.1f} KB")


if __name__ == '__main__':
    main()
