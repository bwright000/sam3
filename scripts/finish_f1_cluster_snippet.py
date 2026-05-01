"""
Resume F_1 cluster snippet build:
  - Build video_left.mp4 + video_stereo.mp4 from already-exported webp frames using OpenCV
  - Build visualization.html from existing pose data and a sample of frames
"""
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import io
import base64
import pandas as pd
import matplotlib.cm as cm
import plotly.graph_objects as go

SNIP_DIR = Path(r"f:/2026 vibes/MPHY Project/Detailed Analysis/snippets/F_1/snippet_cluster_segs1to5")
PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
EPISODE = "F_1"
FPS = 60

with open(SNIP_DIR / 'cluster_metadata.json') as f:
    META = json.load(f)


def webp_to_bgr(path):
    img = Image.open(str(path)).convert('RGB')
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), arr.shape


def build_video_left():
    left_dir = SNIP_DIR / 'frames_left'
    files = sorted(left_dir.glob('frame_*.webp'))
    if not files:
        print("  no left frames found"); return 0
    bgr, (h, w, _) = webp_to_bgr(files[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = str(SNIP_DIR / 'video_left.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
    n = 0
    for f in files:
        bgr, _ = webp_to_bgr(f)
        writer.write(bgr); n += 1
    writer.release()
    print(f"  wrote {n} frames to {out_path}")
    return n


def build_video_stereo():
    left_dir = SNIP_DIR / 'frames_left'
    right_dir = SNIP_DIR / 'frames_right'
    left_files = sorted(left_dir.glob('frame_*.webp'))
    right_files = sorted(right_dir.glob('frame_*.webp'))
    common = sorted(set(p.name for p in left_files) & set(p.name for p in right_files))
    if not common:
        print("  no paired frames"); return 0
    L0, (h, w, _) = webp_to_bgr(left_dir / common[0])
    out_w, out_h = w, h // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = str(SNIP_DIR / 'video_stereo.mp4')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (out_w, out_h))
    n = 0
    for name in common:
        L, _ = webp_to_bgr(left_dir / name)
        R, _ = webp_to_bgr(right_dir / name)
        Ls = cv2.resize(L, (w // 2, h // 2))
        Rs = cv2.resize(R, (w // 2, h // 2))
        combined = np.concatenate([Ls, Rs], axis=1)
        writer.write(combined); n += 1
    writer.release()
    print(f"  wrote {n} frames to {out_path}")
    return n


def build_visualization_html():
    files = sorted((PARQUET_DIR / EPISODE).glob("*.parquet"))
    s, e = META['padded_start_frame'], META['padded_end_frame']
    rows = []
    for pf in files:
        df = pd.read_parquet(pf, columns=['frame_n', 'timestamp', '/ECM/custom/setpoint_cp'])
        df = df[(df['frame_n'] >= s) & (df['frame_n'] <= e)]
        if len(df) > 0:
            rows.append(df)
    df = pd.concat(rows, ignore_index=True).sort_values('frame_n').reset_index(drop=True)
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions = ecm[:, :3] * 1000
    quats = ecm[:, 3:7]
    dt = np.maximum(np.diff(timestamps), 1e-6)
    pos_delta = np.diff(positions / 1000, axis=0)
    lin_v = np.linalg.norm(pos_delta, axis=1) / dt * 1000
    vel = np.append(lin_v, lin_v[-1] if len(lin_v) else 0)
    frame_ns = df['frame_n'].values

    in_motion = np.zeros(len(positions), dtype=bool)
    for ss in META['sub_segments']:
        m = (frame_ns >= ss['motion_start_frame']) & (frame_ns <= ss['motion_end_frame'])
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
        title=f'F_1 Cluster #1-5 (frames {s}-{e})',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        height=600, margin=dict(l=0, r=0, t=50, b=0))

    left_dir = SNIP_DIR / 'frames_left'
    left_files = sorted(left_dir.glob('frame_*.webp'))
    n_thumbs = 12
    thumb_idx = np.linspace(0, len(left_files) - 1, n_thumbs, dtype=int)
    frames_info = []
    for idx in thumb_idx:
        path = left_files[idx]
        img = Image.open(str(path))
        img.thumbnail((300, 170))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        fr = int(path.stem.replace('frame_', ''))
        is_m = any(ss['motion_start_frame'] <= fr <= ss['motion_end_frame'] for ss in META['sub_segments'])
        idx_in_df = np.searchsorted(frame_ns, fr)
        idx_in_df = min(idx_in_df, len(timestamps) - 1)
        t = float(timestamps[idx_in_df] - timestamps[0])
        frames_info.append({'frame': fr, 't': t, 'b64': b64, 'motion': is_m})

    duration = META['duration_s']
    motion_total = META['total_motion_seconds']
    html = f"""<!DOCTYPE html><html><head>
<title>F_1 Cluster #1-5 Visualization</title>
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
<div class="hdr"><h1>F_1 Cluster #1-5 (multi-segment)</h1>
<div class="stats">
<span><strong>Padded frames:</strong> {s}-{e}</span>
<span><strong>Motion frames:</strong> {META['motion_start_frame']}-{META['motion_end_frame']}</span>
<span><strong>Span:</strong> {duration:.2f}s</span>
<span><strong>Motion total:</strong> {motion_total:.1f}s ({motion_total/duration*100:.0f}%)</span>
<span><strong>Sub-segments:</strong> {len(META['sub_segments'])}</span>
<span><strong>Max vel:</strong> {META['overall_max_linear_vel_mm_s']:.1f} mm/s</span>
</div>
<table class="subtbl">
<tr><th>Sub</th><th>Start</th><th>End</th><th>MotF</th><th>Dur(s)</th><th>MaxLin</th><th>MaxAng</th><th>Path(mm)</th><th>Disp(mm)</th></tr>"""
    for ss in META['sub_segments']:
        html += f"<tr><td>#{ss['sub_segment_id']}</td><td>{ss['motion_start_frame']}</td><td>{ss['motion_end_frame']}</td><td>{ss['motion_frames']}</td><td>{ss['duration_s']:.2f}</td><td>{ss['max_linear_vel_mm_s']:.2f}</td><td>{ss['max_angular_vel_deg_s']:.2f}</td><td>{ss['path_length_mm']:.2f}</td><td>{ss['displacement_mm']:.2f}</td></tr>"
    html += "</table></div>"
    html += '<div class="grid"><div class="panel"><h2>3D Camera Trajectory</h2><div id="plot"></div>'
    html += '<p style="font-size:12px;color:#888;margin-top:10px">Coloured = motion frames (viridis by velocity); grey markers = stationary gap frames</p></div>'
    html += '<div class="panel"><h2>Sample Frames (Left Camera)</h2><div class="thumbs">'
    for fi in frames_info:
        cls = 'motion' if fi['motion'] else ''
        html += f'<div class="thumb {cls}"><img src="data:image/jpeg;base64,{fi["b64"]}"><p>f{fi["frame"]} | t={fi["t"]:.1f}s | {"MOTION" if fi["motion"] else "gap"}</p></div>'
    html += '</div><div class="video">'
    html += '<a href="video_left.mp4" target="_blank">Open Left Video</a> | '
    html += '<a href="video_stereo.mp4" target="_blank">Open Stereo Video</a></div></div></div>'
    html += f'<script>var pd={fig.to_json()};Plotly.newPlot("plot",pd.data,pd.layout,{{responsive:true}});</script></body></html>'

    with open(SNIP_DIR / 'visualization.html', 'w') as f:
        f.write(html)
    print(f"  wrote {SNIP_DIR / 'visualization.html'}")


def main():
    # Remove broken video stub
    bad = SNIP_DIR / 'video_left.mp4'
    if bad.exists() and bad.stat().st_size < 1000:
        bad.unlink()

    print("Building left video...")
    build_video_left()
    print("Building stereo video...")
    build_video_stereo()
    print("Building visualization HTML...")
    build_visualization_html()

    print("\nFinal snippet contents:")
    for p in sorted(SNIP_DIR.iterdir()):
        sz = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        print(f"  {p.name:40s} {sz/1024/1024:.1f} MB")


if __name__ == '__main__':
    main()
