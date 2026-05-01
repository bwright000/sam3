"""
Visual confirmation for F_1 snippets:
  1. cluster #1-5  -> frames 252-3730  (58.0s, 22.4s motion)
  2. single  #6    -> frames 11219-12385 (19.4s, 19.4s motion)

Outputs:
  - velocity plot (PNG): linear+angular vel with motion/gap regions shaded
  - 3D trajectory (HTML): interactive Plotly camera path coloured by velocity
  - contact sheet (HTML): sampled left-camera frames so user can eyeball scene continuity
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from PIL import Image
import io
import base64

PARQUET_DIR = Path(r"f:/2026 vibes/MPHY Project/CRCD_manual/hub/datasets--SITL-Eng--CRCD/snapshots/f597d230356f4e6d46516b83c2baa4f52c923358/data")
OUT_DIR = Path(r"c:/Users/benli/sam3facebook/outputs/F_1_visual_verify")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE = "F_1"
SNIPPETS = [
    {
        "name": "01_F1_cluster_segs1to5",
        "start": 252,
        "end": 3730,
        "title": "F_1 cluster #1-5  (frames 252-3730, 58.0s span, 22.4s motion, ratio 0.39)",
        "motion_subsegs": [(252, 694), (1233, 1691), (2363, 2482), (3082, 3175), (3500, 3730)],
    },
    {
        "name": "02_F1_single_seg6",
        "start": 11219,
        "end": 12385,
        "title": "F_1 single #6  (frames 11219-12385, 19.4s span, 100% motion)",
        "motion_subsegs": [(11219, 12385)],
    },
]


def load_pose_only():
    files = sorted((PARQUET_DIR / EPISODE).glob("*.parquet"))
    dfs = [pd.read_parquet(pf, columns=['frame_n', 'timestamp', '/ECM/custom/setpoint_cp']) for pf in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('frame_n').reset_index(drop=True)
    return df


def load_frames_in_range(start, end):
    """Load only frame_n + frame_left for a frame range (one parquet at a time)."""
    files = sorted((PARQUET_DIR / EPISODE).glob("*.parquet"))
    rows = []
    for pf in files:
        df = pd.read_parquet(pf, columns=['frame_n', 'frame_left'])
        df = df[(df['frame_n'] >= start) & (df['frame_n'] <= end)]
        if len(df) > 0:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values('frame_n').reset_index(drop=True)


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


def make_velocity_plot(snippet, timestamps, lin_vel, ang_vel, out_path):
    s, e = snippet['start'], snippet['end']
    t_rel = timestamps[s:e + 1] - timestamps[s]
    lin = lin_vel[s:e]
    ang = ang_vel[s:e]
    t_v = t_rel[:len(lin)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(t_rel[0], t_rel[-1], alpha=0.15, color='lightgray', label='Gap (camera stationary)')
    for sub_s, sub_e in snippet['motion_subsegs']:
        sub_s_rel = timestamps[sub_s] - timestamps[s]
        sub_e_rel = timestamps[sub_e] - timestamps[s]
        for ax in (ax1, ax2):
            ax.axvspan(sub_s_rel, sub_e_rel, alpha=0.30, color='gold', label='Motion sub-segment' if sub_s == snippet['motion_subsegs'][0][0] else None)

    ax1.plot(t_v, lin, 'b-', lw=0.7)
    ax1.axhline(y=0.1, color='r', ls='--', alpha=0.5, label='Threshold (0.1 mm/s)')
    ax1.set_ylabel('Linear velocity (mm/s)')
    ax1.set_title(snippet['title'])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_v, ang, 'g-', lw=0.7)
    ax2.axhline(y=0.1, color='r', ls='--', alpha=0.5, label='Threshold (0.1 deg/s)')
    ax2.set_ylabel('Angular velocity (deg/s)')
    ax2.set_xlabel('Time within snippet (s)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def make_3d_trajectory(snippet, positions, lin_vel, out_path):
    s, e = snippet['start'], snippet['end']
    pos_mm = positions[s:e + 1] * 1000
    vel = np.append(lin_vel[s:e], lin_vel[e - 1] if e > 0 else 0)

    # mark which frames are in motion sub-segments vs in gap
    in_motion = np.zeros(len(pos_mm), dtype=bool)
    for sub_s, sub_e in snippet['motion_subsegs']:
        in_motion[max(0, sub_s - s):min(len(pos_mm), sub_e - s + 1)] = True

    vel_max = max(0.5, np.percentile(vel, 99))
    vel_norm = np.clip(vel / vel_max, 0, 1)
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in cm.viridis(vel_norm)]

    fig = go.Figure()
    for i in range(len(pos_mm) - 1):
        if in_motion[i]:
            fig.add_trace(go.Scatter3d(
                x=[pos_mm[i, 0], pos_mm[i + 1, 0]],
                y=[pos_mm[i, 1], pos_mm[i + 1, 1]],
                z=[pos_mm[i, 2], pos_mm[i + 1, 2]],
                mode='lines',
                line=dict(color=colors[i], width=5),
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[pos_mm[i, 0], pos_mm[i + 1, 0]],
                y=[pos_mm[i, 1], pos_mm[i + 1, 1]],
                z=[pos_mm[i, 2], pos_mm[i + 1, 2]],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=0.1, color=[0, vel_max], colorscale='Viridis', cmin=0, cmax=vel_max,
                    colorbar=dict(title='Vel (mm/s)', thickness=15, len=0.7)),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(x=[pos_mm[0, 0]], y=[pos_mm[0, 1]], z=[pos_mm[0, 2]],
                               mode='markers', marker=dict(size=8, color='lime'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[pos_mm[-1, 0]], y=[pos_mm[-1, 1]], z=[pos_mm[-1, 2]],
                               mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='End'))

    fig.update_layout(
        title=snippet['title'],
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        height=700, margin=dict(l=0, r=0, t=50, b=0)
    )
    fig.write_html(str(out_path), include_plotlyjs='cdn')


def make_contact_sheet(snippet, frames_df, out_path, n_samples=24):
    s, e = snippet['start'], snippet['end']
    if len(frames_df) == 0:
        print(f"  [warn] no frames available for contact sheet")
        return
    indices = np.linspace(0, len(frames_df) - 1, n_samples, dtype=int)
    in_motion_for_sample = []
    sample_rows = frames_df.iloc[indices]
    for _, row in sample_rows.iterrows():
        fr = row['frame_n']
        is_m = any(sub_s <= fr <= sub_e for sub_s, sub_e in snippet['motion_subsegs'])
        in_motion_for_sample.append(is_m)

    items = []
    for (_, row), is_m in zip(sample_rows.iterrows(), in_motion_for_sample):
        fdata = row['frame_left']
        if fdata is None or 'bytes' not in fdata:
            continue
        img = Image.open(io.BytesIO(fdata['bytes']))
        img.thumbnail((320, 180))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        t_rel = (row['frame_n'] - s) / 60.0
        items.append({'frame': int(row['frame_n']), 't': t_rel, 'b64': b64, 'motion': is_m})

    html = f"""<!DOCTYPE html><html><head><title>{snippet['title']}</title>
<style>
body{{font-family:sans-serif;background:#f5f5f5;margin:0;padding:20px}}
.hdr{{background:#fff;padding:15px;border-radius:6px;margin-bottom:15px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.cell{{background:#fff;padding:6px;border-radius:5px;border:2px solid #ddd}}
.cell.motion{{border-color:#f4b400}}
.cell.gap{{border-color:#bbb}}
.cell img{{width:100%;display:block;border-radius:3px}}
.cell p{{margin:6px 0 0 0;font-size:11px;color:#666;text-align:center}}
.legend{{display:flex;gap:20px;margin-top:8px;font-size:13px;color:#555}}
.swatch{{display:inline-block;width:14px;height:14px;border:2px solid #f4b400;margin-right:5px;vertical-align:middle}}
.swatch.gap{{border-color:#bbb}}
</style></head><body>
<div class="hdr"><h2>{snippet['title']}</h2>
<div class="legend"><span><span class="swatch"></span>Motion sub-segment</span>
<span><span class="swatch gap"></span>Gap (camera stationary)</span></div></div>
<div class="grid">"""
    for it in items:
        cls = 'motion' if it['motion'] else 'gap'
        html += f'<div class="cell {cls}"><img src="data:image/jpeg;base64,{it["b64"]}"><p>frame {it["frame"]} | t={it["t"]:.2f}s | {"MOTION" if it["motion"] else "gap"}</p></div>\n'
    html += "</div></body></html>"
    with open(out_path, 'w') as f:
        f.write(html)


def main():
    print(f"Loading {EPISODE} pose data...")
    df = load_pose_only()
    timestamps = df['timestamp'].values
    ecm = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions, quaternions = ecm[:, :3], ecm[:, 3:7]
    lin_vel, ang_vel = compute_velocities(timestamps, positions, quaternions)
    print(f"  loaded {len(df)} frames")

    for snip in SNIPPETS:
        print(f"\n{snip['name']}: frames {snip['start']}-{snip['end']}")
        snip_dir = OUT_DIR / snip['name']
        snip_dir.mkdir(exist_ok=True)

        print("  velocity plot...", flush=True)
        make_velocity_plot(snip, timestamps, lin_vel, ang_vel, snip_dir / 'velocity.png')

        print("  3D trajectory...", flush=True)
        make_3d_trajectory(snip, positions, lin_vel, snip_dir / 'trajectory_3d.html')

        print("  loading frames for contact sheet...", flush=True)
        frames_df = load_frames_in_range(snip['start'], snip['end'])
        print(f"    got {len(frames_df)} frames")
        n_samples = 24 if snip['end'] - snip['start'] > 1500 else 16
        make_contact_sheet(snip, frames_df, snip_dir / 'contact_sheet.html', n_samples=n_samples)
        del frames_df

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == '__main__':
    main()
