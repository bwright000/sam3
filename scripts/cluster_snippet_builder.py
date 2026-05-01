"""
Consolidated cluster snippet builder.

Replaces (and supersedes) build_f1_cluster_snippet.py + finish_f1_cluster_snippet.py.

Produces multi-segment cluster snippets matching the structure of
extract_motion_snippets.py output, plus a cluster_metadata.json describing
the constituent sub-segments. Stereo video is exported at FULL resolution
(2*W x H), not half.

Functions:
    build_cluster(spec)        — full build from parquet
    extend_cluster(spec, n)    — add N frames to padded_end of an existing snippet
    rebuild_videos(snip_dir)   — re-encode video_left.mp4 + video_stereo.mp4 from
                                 existing webp frames at FULL resolution

Cluster spec format:
    {
        'episode': 'F_1',
        'name':    'snippet_cluster_segs1to5',
        'motion_start': 252,       # first frame of first sub-seg
        'motion_end':   3730,      # last frame of last sub-seg
        'pad': 60,                 # frames each side
        'sub_segments': [{'id':1,'motion_start':252,'motion_end':694}, ...],
        'source_segment_ids': [1,2,3,4,5],   # for cross-ref to {EP}_snippets.json
    }
"""
import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
SNIPPETS_ROOT = Path(r"f:/2026 vibes/MPHY Project/Detailed Analysis/snippets")
FPS = 60
MOTION_THRESH_LIN = 0.1
MOTION_THRESH_ANG = 0.1
TOOL_THRESH_MM_S = 1.0


# ============================== I/O helpers ==============================

def load_episode_range(episode, start, end, with_frames=True):
    """Load all parquet rows in [start, end] inclusive."""
    files = sorted((PARQUET_DIR / episode).glob("*.parquet"))
    cols = None if with_frames else ['frame_n', 'timestamp', '/ECM/custom/setpoint_cp',
                                     '/PSM1/custom/setpoint_cp', '/PSM2/custom/setpoint_cp']
    rows = []
    for pf in files:
        df = pd.read_parquet(pf, columns=cols)
        df = df[(df['frame_n'] >= start) & (df['frame_n'] <= end)]
        if len(df) > 0:
            rows.append(df)
        del df
    if not rows:
        raise RuntimeError(f"No frames in range [{start},{end}] for {episode}")
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


# ============================== Output writers ==============================

def export_tum_poses(df, out_path, header_lines):
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    with open(out_path, 'w') as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        for line in header_lines:
            f.write(f"# {line}\n")
        for ts, pose in zip(timestamps, ecm):
            x, y, z = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            f.write(f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def export_frames(df, out_dir):
    """Write left+right webp frames named frame_NNNNNN.webp."""
    left_dir = out_dir / 'frames_left'
    right_dir = out_dir / 'frames_right'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        fr = int(row['frame_n'])
        ld, rd = row['frame_left'], row['frame_right']
        if ld is not None and 'bytes' in ld:
            (left_dir / f'frame_{fr:06d}.webp').write_bytes(ld['bytes'])
        if rd is not None and 'bytes' in rd:
            (right_dir / f'frame_{fr:06d}.webp').write_bytes(rd['bytes'])


def _webp_to_rgb(path):
    arr = np.array(Image.open(str(path)).convert('RGB'))
    return arr  # H x W x 3


def write_video_libx264(rgb_frames, out_path, fps=FPS, quality=8):
    """Write MP4 via imageio + ffmpeg with libx264 codec."""
    imageio.mimwrite(str(out_path), rgb_frames, fps=fps, codec='libx264', quality=quality, macro_block_size=1)


def build_video_left_full_res(snip_dir):
    """Build video_left.mp4 at full webp resolution."""
    files = sorted((snip_dir / 'frames_left').glob('frame_*.webp'))
    if not files:
        return 0
    frames = [_webp_to_rgb(f) for f in files]
    write_video_libx264(frames, snip_dir / 'video_left.mp4')
    h, w = frames[0].shape[:2]
    return len(frames), (w, h)


def build_video_stereo_full_res(snip_dir):
    """Build video_stereo.mp4 at FULL stereo resolution: width = 2*W, height = H.
    No resizing — concatenates left+right at native resolution."""
    left_dir = snip_dir / 'frames_left'
    right_dir = snip_dir / 'frames_right'
    left_files = sorted(left_dir.glob('frame_*.webp'))
    right_files = sorted(right_dir.glob('frame_*.webp'))
    common = sorted(set(p.name for p in left_files) & set(p.name for p in right_files))
    if not common:
        return 0, None
    frames = []
    out_size = None
    for name in common:
        L = _webp_to_rgb(left_dir / name)
        R = _webp_to_rgb(right_dir / name)
        if L.shape != R.shape:
            R = cv2.resize(R, (L.shape[1], L.shape[0]))
        combined = np.concatenate([L, R], axis=1)  # full res side-by-side
        frames.append(combined)
        if out_size is None:
            out_size = (combined.shape[1], combined.shape[0])
    write_video_libx264(frames, snip_dir / 'video_stereo.mp4')
    return len(frames), out_size


def make_velocity_plot(df, sub_segments, out_path, title):
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
    for ss in sub_segments:
        ms_idx = int(np.searchsorted(frame_ns, ss['motion_start']))
        me_idx = int(np.searchsorted(frame_ns, ss['motion_end']))
        if ms_idx < len(t_rel) and me_idx < len(t_rel):
            for ax in (ax1, ax2):
                lbl = 'Motion sub-segment' if not legend_added else None
                ax.axvspan(t_rel[ms_idx], t_rel[me_idx], alpha=0.30, color='gold', label=lbl)
            legend_added = True

    ax1.plot(t_v, lin_v, 'b-', lw=0.7)
    ax1.axhline(y=MOTION_THRESH_LIN, color='r', ls='--', alpha=0.5, label=f'Threshold ({MOTION_THRESH_LIN} mm/s)')
    ax1.set_ylabel('Linear velocity (mm/s)')
    ax1.set_title(title)
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


def classify_scene_motion(df, sub_segments, motion_start, motion_end):
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

    overall = _stats(motion_start, motion_end)
    per_sub = []
    for ss in sub_segments:
        m = _stats(ss['motion_start'], ss['motion_end'])
        m.update({'sub_segment_id': ss['id'], 'motion_start_frame': ss['motion_start'],
                  'motion_end_frame': ss['motion_end'],
                  'motion_type': 'tool_and_camera' if m['tool_motion'] else 'camera_only'})
        per_sub.append(m)
    return {
        'motion_type': 'tool_and_camera' if overall['tool_motion'] else 'camera_only',
        'camera_motion': True,
        'tool_motion': overall['tool_motion'],
        'psm1_motion': overall['psm1_motion'],
        'psm2_motion': overall['psm2_motion'],
        'psm1_max_vel_mm_s': overall['psm1_max_vel_mm_s'],
        'psm2_max_vel_mm_s': overall['psm2_max_vel_mm_s'],
        'sub_segments': per_sub,
    }


def make_cluster_metadata(df, spec):
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quats = ecm[:, :3], ecm[:, 3:7]
    lin_v, ang_v = compute_velocities(timestamps, positions, quats)
    frame_ns = df['frame_n'].values
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

    return {
        'episode': spec['episode'],
        'snippet_id': spec.get('snippet_id', spec['name'].replace('snippet_', '')),
        'snippet_type': 'multi_segment_cluster',
        'padded_start_frame': spec['padded_start'],
        'padded_end_frame': spec['padded_end'],
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
        'source_segment_ids_in_episode_snippets_json': spec.get('source_segment_ids', []),
    }


def make_visualization_html(df, meta, snip_dir):
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions = ecm[:, :3] * 1000
    lin_v, _ = compute_velocities(timestamps, ecm[:, :3], ecm[:, 3:7])
    vel = np.append(lin_v, lin_v[-1] if len(lin_v) else 0)
    frame_ns = df['frame_n'].values

    in_motion = np.zeros(len(positions), dtype=bool)
    for ss in meta['sub_segments']:
        m = (frame_ns >= ss['motion_start_frame']) & (frame_ns <= ss['motion_end_frame'])
        in_motion |= m

    vel_max = max(0.5, float(np.percentile(vel, 99)))
    vel_norm = np.clip(vel / vel_max, 0, 1)
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in cm.viridis(vel_norm)]

    fig = go.Figure()
    for i in range(len(positions) - 1):
        if in_motion[i]:
            fig.add_trace(go.Scatter3d(x=[positions[i, 0], positions[i + 1, 0]],
                                       y=[positions[i, 1], positions[i + 1, 1]],
                                       z=[positions[i, 2], positions[i + 1, 2]],
                                       mode='lines', line=dict(color=colors[i], width=5),
                                       showlegend=False, hoverinfo='skip'))
        else:
            fig.add_trace(go.Scatter3d(x=[positions[i, 0], positions[i + 1, 0]],
                                       y=[positions[i, 1], positions[i + 1, 1]],
                                       z=[positions[i, 2], positions[i + 1, 2]],
                                       mode='markers', marker=dict(size=2, color='lightgray', opacity=0.5),
                                       showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                               marker=dict(size=0.1, color=[0, vel_max], colorscale='Viridis',
                                           cmin=0, cmax=vel_max,
                                           colorbar=dict(title='Vel (mm/s)', thickness=15, len=0.7)),
                               showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                               mode='markers', marker=dict(size=8, color='lime'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
                               mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='End'))
    fig.update_layout(
        title=f'{meta["episode"]} {meta["snippet_id"]} (frames {meta["padded_start_frame"]}-{meta["padded_end_frame"]})',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        height=600, margin=dict(l=0, r=0, t=50, b=0))

    left_files = sorted((snip_dir / 'frames_left').glob('frame_*.webp'))
    n_thumbs = 12
    idxs = np.linspace(0, len(left_files) - 1, n_thumbs, dtype=int)
    items = []
    for idx in idxs:
        path = left_files[idx]
        img = Image.open(str(path))
        img.thumbnail((300, 170))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        fr = int(path.stem.replace('frame_', ''))
        is_m = any(ss['motion_start_frame'] <= fr <= ss['motion_end_frame'] for ss in meta['sub_segments'])
        idx_in_df = min(int(np.searchsorted(frame_ns, fr)), len(timestamps) - 1)
        t = float(timestamps[idx_in_df] - timestamps[0])
        items.append({'frame': fr, 't': t, 'b64': b64, 'motion': is_m})

    duration = meta['duration_s']
    motion_total = meta['total_motion_seconds']
    rows = ''
    for ss in meta['sub_segments']:
        rows += f"<tr><td>#{ss['sub_segment_id']}</td><td>{ss['motion_start_frame']}</td><td>{ss['motion_end_frame']}</td><td>{ss['motion_frames']}</td><td>{ss['duration_s']:.2f}</td><td>{ss['max_linear_vel_mm_s']:.2f}</td><td>{ss['max_angular_vel_deg_s']:.2f}</td><td>{ss['path_length_mm']:.2f}</td><td>{ss['displacement_mm']:.2f}</td></tr>"
    thumbs = ''
    for f in items:
        cls = 'motion' if f['motion'] else ''
        thumbs += f'<div class="thumb {cls}"><img src="data:image/jpeg;base64,{f["b64"]}"><p>f{f["frame"]} | t={f["t"]:.1f}s | {"MOTION" if f["motion"] else "gap"}</p></div>'

    html = f"""<!DOCTYPE html><html><head>
<title>{meta['episode']} {meta['snippet_id']}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font-family:sans-serif;margin:0;padding:20px;background:#f5f5f5}}
.hdr{{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
.hdr h1{{margin:0 0 10px 0;color:#333}} .stats{{display:flex;gap:25px;color:#666;font-size:14px;flex-wrap:wrap}}
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
.subtbl{{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}}
.subtbl th,.subtbl td{{border:1px solid #ddd;padding:5px;text-align:right}}
.subtbl th{{background:#f9f9f9}}
</style></head><body>
<div class="hdr"><h1>{meta['episode']} {meta['snippet_id']}</h1>
<div class="stats">
<span><strong>Padded frames:</strong> {meta['padded_start_frame']}-{meta['padded_end_frame']}</span>
<span><strong>Motion frames:</strong> {meta['motion_start_frame']}-{meta['motion_end_frame']}</span>
<span><strong>Total frames:</strong> {meta['total_frames']}</span>
<span><strong>Span:</strong> {duration:.2f}s</span>
<span><strong>Motion total:</strong> {motion_total:.1f}s ({motion_total/duration*100:.0f}%)</span>
<span><strong>Sub-segments:</strong> {len(meta['sub_segments'])}</span>
<span><strong>Max vel:</strong> {meta['overall_max_linear_vel_mm_s']:.1f} mm/s</span>
</div>
<table class="subtbl">
<tr><th>Sub</th><th>Start</th><th>End</th><th>MotF</th><th>Dur(s)</th><th>MaxLin</th><th>MaxAng</th><th>Path(mm)</th><th>Disp(mm)</th></tr>
{rows}
</table></div>
<div class="grid">
<div class="panel"><h2>3D Camera Trajectory</h2><div id="plot"></div>
<p style="font-size:12px;color:#888;margin-top:10px">Coloured = motion frames (viridis by velocity); grey markers = stationary gap frames</p></div>
<div class="panel"><h2>Sample Frames (Left Camera)</h2><div class="thumbs">{thumbs}</div>
<div class="video"><a href="video_left.mp4" target="_blank">Open Left Video</a> | <a href="video_stereo.mp4" target="_blank">Open Stereo Video</a></div></div></div>
<script>var pd={fig.to_json()};Plotly.newPlot("plot",pd.data,pd.layout,{{responsive:true}});</script>
</body></html>"""
    with open(snip_dir / 'visualization.html', 'w') as f:
        f.write(html)


# ============================== Top-level ops ==============================

def build_cluster(spec):
    """Full build of a multi-segment cluster snippet."""
    spec['padded_start'] = spec['motion_start'] - spec.get('pad', 60)
    spec['padded_end'] = spec['motion_end'] + spec.get('pad', 60)
    spec['snippet_id'] = spec.get('snippet_id', spec['name'].replace('snippet_', ''))
    out_dir = SNIPPETS_ROOT / spec['episode'] / spec['name']
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_cluster] {spec['episode']}/{spec['name']}  frames {spec['padded_start']}-{spec['padded_end']}")
    print("  loading parquet range with frames...")
    df = load_episode_range(spec['episode'], spec['padded_start'], spec['padded_end'], with_frames=True)
    print(f"    got {len(df)} frames")

    print("  exporting webp frames...")
    export_frames(df, out_dir)

    print("  exporting TUM poses...")
    sub_str = ', '.join(f"({s['motion_start']},{s['motion_end']})" for s in spec['sub_segments'])
    export_tum_poses(df, out_dir / 'poses.txt',
                     header_lines=[f"{spec['episode']} cluster snippet, padded frames {spec['padded_start']}-{spec['padded_end']}, motion frames {spec['motion_start']}-{spec['motion_end']}",
                                   f"Sub-segments: [{sub_str}]"])

    print("  velocity plot...")
    title = f"{spec['episode']} {spec['snippet_id']} | padded {spec['padded_start']}-{spec['padded_end']} | motion {spec['motion_start']}-{spec['motion_end']} | {len(spec['sub_segments'])} sub-segs"
    make_velocity_plot(df, [{'motion_start': s['motion_start'], 'motion_end': s['motion_end']} for s in spec['sub_segments']], out_dir / 'velocity.png', title)

    print("  scene_motion classification...")
    scene = classify_scene_motion(df, [{'id': s['id'], 'motion_start': s['motion_start'], 'motion_end': s['motion_end']} for s in spec['sub_segments']],
                                  spec['motion_start'], spec['motion_end'])
    with open(out_dir / 'scene_motion.json', 'w') as f:
        json.dump(scene, f, indent=2)

    print("  cluster_metadata.json...")
    meta = make_cluster_metadata(df, spec)
    with open(out_dir / 'cluster_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("  building left video at full resolution...")
    n_l, lsize = build_video_left_full_res(out_dir)
    print(f"    {n_l} frames at {lsize}")

    print("  building stereo video at full resolution...")
    n_s, ssize = build_video_stereo_full_res(out_dir)
    print(f"    {n_s} frames at {ssize}")

    print("  visualization HTML...")
    make_visualization_html(df, meta, out_dir)

    print(f"[done] {out_dir}")
    return meta


def extend_cluster(snip_dir, n_frames=1):
    """Add n frames to padded_end of an existing cluster snippet."""
    snip_dir = Path(snip_dir)
    meta_path = snip_dir / 'cluster_metadata.json'
    meta = json.load(open(meta_path))
    episode = meta['episode']
    old_end = meta['padded_end_frame']
    new_end = old_end + n_frames
    print(f"[extend_cluster] {snip_dir.name}: padded_end {old_end} -> {new_end} (+{n_frames})")

    print("  loading new frames from parquet...")
    df_new = load_episode_range(episode, old_end + 1, new_end, with_frames=True)
    if len(df_new) != n_frames:
        raise RuntimeError(f"Expected {n_frames} new frames, got {len(df_new)}")
    print(f"    fetched {len(df_new)} frames ({old_end+1}-{new_end})")

    print("  saving new webp frames...")
    export_frames(df_new, snip_dir)

    print("  appending poses.txt...")
    timestamps = df_new['timestamp'].values
    ecm = np.array(df_new['/ECM/custom/setpoint_cp'].tolist())
    with open(snip_dir / 'poses.txt', 'a') as f:
        for ts, pose in zip(timestamps, ecm):
            x, y, z = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            f.write(f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

    print("  reloading full pose+psm range for metadata recompute...")
    df_full = load_episode_range(episode, meta['padded_start_frame'], new_end, with_frames=False)

    print("  rebuilding velocity.png...")
    sub_segs_simple = [{'motion_start': s['motion_start_frame'], 'motion_end': s['motion_end_frame']} for s in meta['sub_segments']]
    title = f"{episode} {meta['snippet_id']} | padded {meta['padded_start_frame']}-{new_end} | motion {meta['motion_start_frame']}-{meta['motion_end_frame']} | {len(meta['sub_segments'])} sub-segs"
    make_velocity_plot(df_full, sub_segs_simple, snip_dir / 'velocity.png', title)

    print("  rebuilding cluster_metadata.json...")
    spec_for_meta = {
        'episode': episode, 'name': snip_dir.name,
        'snippet_id': meta['snippet_id'],
        'padded_start': meta['padded_start_frame'], 'padded_end': new_end,
        'motion_start': meta['motion_start_frame'], 'motion_end': meta['motion_end_frame'],
        'sub_segments': [{'id': s['sub_segment_id'], 'motion_start': s['motion_start_frame'], 'motion_end': s['motion_end_frame']} for s in meta['sub_segments']],
        'source_segment_ids': meta.get('source_segment_ids_in_episode_snippets_json', []),
    }
    new_meta = make_cluster_metadata(df_full, spec_for_meta)
    with open(meta_path, 'w') as f:
        json.dump(new_meta, f, indent=2)

    print("  rebuilding scene_motion.json...")
    scene = classify_scene_motion(df_full, [{'id': s['id'], 'motion_start': s['motion_start'], 'motion_end': s['motion_end']} for s in spec_for_meta['sub_segments']],
                                  meta['motion_start_frame'], meta['motion_end_frame'])
    with open(snip_dir / 'scene_motion.json', 'w') as f:
        json.dump(scene, f, indent=2)

    print("  rebuilding videos at full resolution...")
    n_l, lsize = build_video_left_full_res(snip_dir)
    n_s, ssize = build_video_stereo_full_res(snip_dir)
    print(f"    left {n_l}@{lsize}, stereo {n_s}@{ssize}")

    print("  rebuilding visualization.html...")
    make_visualization_html(df_full, new_meta, snip_dir)
    print(f"[done] now {n_l} frames")


def rebuild_videos(snip_dir):
    """Rebuild video_left.mp4 + video_stereo.mp4 at FULL resolution from existing webp frames.
    For retrofitting existing snippets that were built with half-res stereo."""
    snip_dir = Path(snip_dir)
    print(f"[rebuild_videos] {snip_dir}")
    n_l, lsize = build_video_left_full_res(snip_dir)
    n_s, ssize = build_video_stereo_full_res(snip_dir)
    print(f"  left {n_l}@{lsize}, stereo {n_s}@{ssize}")
    return (n_l, lsize, n_s, ssize)


# ============================== CLI ==============================

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    p_build = sub.add_parser('build', help='Build cluster from spec JSON')
    p_build.add_argument('spec_json', help='Path to spec JSON file')

    p_ext = sub.add_parser('extend', help='Extend existing cluster by N frames')
    p_ext.add_argument('snippet_dir')
    p_ext.add_argument('--n', type=int, default=1)

    p_reb = sub.add_parser('rebuild_videos', help='Rebuild videos at full resolution from webp frames')
    p_reb.add_argument('snippet_dir')

    args = p.parse_args()
    if args.cmd == 'build':
        spec = json.load(open(args.spec_json))
        build_cluster(spec)
    elif args.cmd == 'extend':
        extend_cluster(args.snippet_dir, n_frames=args.n)
    elif args.cmd == 'rebuild_videos':
        rebuild_videos(args.snippet_dir)
