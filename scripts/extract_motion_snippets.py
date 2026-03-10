"""
Motion Snippet Extraction and Analysis

Extracts video snippets where camera movement occurs, generates reports,
exports poses in TUM format, and creates side-by-side visualizations.

https://github.com/MichaelGrupp/evo
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import imageio

# Configuration
BASE_DIR = Path(r"f:\2026 vibes\MPHY Project")
PARQUET_DIR = BASE_DIR / "CRCD_manual" / "hub" / "datasets--SITL-Eng--CRCD" / "snapshots" / "f597d230356f4e6d46516b83c2baa4f52c923358" / "data"
OUTPUT_DIR = BASE_DIR / "Detailed Analysis" / "snippets"

# User-specified parameters
PADDING_FRAMES = 60          # ~1 second at 60Hz
MERGE_GAP_SECONDS = 5.0      # Merge segments with gaps < 5s
MIN_MOTION_FRAMES = 5        # Ignore very short bursts
MOTION_THRESH_LINEAR = 0.1   # mm/s
MOTION_THRESH_ANGULAR = 0.1  # deg/s

# Episodes to process
EPISODES = ["C_1", "F_3", "E_3",]


def load_episode_data(episode):
    """Load all parquet data for an episode."""
    episode_dir = PARQUET_DIR / episode
    parquet_files = sorted(episode_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('frame_n').reset_index(drop=True)
    return df


def compute_velocities(timestamps, positions, quaternions):
    """Compute linear and angular velocities."""
    dt = np.diff(timestamps)
    dt = np.maximum(dt, 1e-6)

    # Linear velocity (mm/s)
    pos_delta = np.diff(positions, axis=0)
    linear_vel = np.linalg.norm(pos_delta, axis=1) / dt * 1000

    # Angular velocity (deg/s)
    angular_changes = []
    for i in range(len(quaternions) - 1):
        q1, q2 = quaternions[i], quaternions[i+1]
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        angular_changes.append(angle)
    angular_vel = np.degrees(np.array(angular_changes) / dt)

    return linear_vel, angular_vel, dt


def classify_motion(linear_vel, angular_vel):
    """Classify each frame as motion or stationary."""
    is_motion = (linear_vel > MOTION_THRESH_LINEAR) | (angular_vel > MOTION_THRESH_ANGULAR)
    return is_motion


def find_motion_segments(is_motion, timestamps):
    """Find continuous motion segments and merge nearby ones."""
    segments = []
    in_segment = False
    start_idx = 0

    for i, motion in enumerate(is_motion):
        if motion and not in_segment:
            start_idx = i
            in_segment = True
        elif not motion and in_segment:
            if i - start_idx >= MIN_MOTION_FRAMES:
                segments.append((start_idx, i - 1))
            in_segment = False

    # Handle segment at end
    if in_segment and len(is_motion) - start_idx >= MIN_MOTION_FRAMES:
        segments.append((start_idx, len(is_motion) - 1))

    if not segments:
        return []

    # Merge segments with gaps < MERGE_GAP_SECONDS
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_end = merged[-1][1]
        gap_time = timestamps[start] - timestamps[prev_end] if start < len(timestamps) and prev_end < len(timestamps) else float('inf')

        if gap_time < MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def add_padding(segments, total_frames):
    """Add padding frames before and after each segment."""
    padded = []
    for start, end in segments:
        padded_start = max(0, start - PADDING_FRAMES)
        padded_end = min(total_frames - 1, end + PADDING_FRAMES)
        padded.append({
            'start': padded_start,
            'end': padded_end,
            'motion_start': start,
            'motion_end': end,
        })
    return padded


def classify_scene_motion(df, snippet, timestamps):
    """Classify what type of motion is in the scene (tool, camera, or both)."""
    start, end = snippet['start'], snippet['end']

    # Get PSM poses
    psm1_data = np.array(df['/PSM1/custom/setpoint_cp'].iloc[start:end+1].tolist())
    psm2_data = np.array(df['/PSM2/custom/setpoint_cp'].iloc[start:end+1].tolist())

    ts = timestamps[start:end+1]

    # Compute PSM velocities
    if len(ts) > 1:
        dt = np.diff(ts)
        dt = np.maximum(dt, 1e-6)

        psm1_vel = np.linalg.norm(np.diff(psm1_data[:, :3], axis=0), axis=1) / dt * 1000
        psm2_vel = np.linalg.norm(np.diff(psm2_data[:, :3], axis=0), axis=1) / dt * 1000

        psm1_motion = np.any(psm1_vel > 1.0)  # 1 mm/s threshold for tools
        psm2_motion = np.any(psm2_vel > 1.0)
        tool_motion = psm1_motion or psm2_motion

        psm1_max = psm1_vel.max() if len(psm1_vel) > 0 else 0
        psm2_max = psm2_vel.max() if len(psm2_vel) > 0 else 0
    else:
        tool_motion = False
        psm1_max = psm2_max = 0

    # ECM (camera) is moving by definition in these snippets
    camera_motion = True

    if tool_motion and camera_motion:
        motion_type = "tool_and_camera"
    elif tool_motion:
        motion_type = "tool_only"
    else:
        motion_type = "camera_only"

    return {
        'motion_type': motion_type,
        'camera_motion': bool(camera_motion),
        'tool_motion': bool(tool_motion),
        'psm1_motion': bool(psm1_motion) if len(ts) > 1 else False,
        'psm2_motion': bool(psm2_motion) if len(ts) > 1 else False,
        'psm1_max_vel_mm_s': float(psm1_max),
        'psm2_max_vel_mm_s': float(psm2_max),
    }


def export_tum_poses(df, snippet, output_path):
    """Export poses in TUM format: timestamp tx ty tz qx qy qz qw"""
    start, end = snippet['start'], snippet['end']

    timestamps = df['timestamp'].iloc[start:end+1].values
    ecm_data = np.array(df['/ECM/custom/setpoint_cp'].iloc[start:end+1].tolist())

    with open(output_path, 'w') as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write(f"# Episode snippet from frame {start} to {end}\n")
        for i, (ts, pose) in enumerate(zip(timestamps, ecm_data)):
            x, y, z = pose[:3]
            qx, qy, qz, qw = pose[3:7]
            f.write(f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def create_velocity_plot(timestamps, linear_vel, angular_vel, snippet, output_path):
    """Create velocity plot highlighting motion vs padding regions."""
    start, end = snippet['start'], snippet['end']
    motion_start, motion_end = snippet['motion_start'], snippet['motion_end']

    # Time array (relative to start)
    t = timestamps[start:end] - timestamps[start]
    lin_v = linear_vel[start:end-1] if end > start else []
    ang_v = angular_vel[start:end-1] if end > start else []

    # Adjust for velocity being one element shorter
    t_vel = t[:-1] if len(t) > 1 else t

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Motion region highlighting
    motion_t_start = timestamps[motion_start] - timestamps[start]
    motion_t_end = timestamps[motion_end] - timestamps[start]

    for ax in [ax1, ax2]:
        ax.axvspan(0, motion_t_start, alpha=0.2, color='gray', label='Padding')
        ax.axvspan(motion_t_start, motion_t_end, alpha=0.2, color='yellow', label='Motion')
        ax.axvspan(motion_t_end, t[-1] if len(t) > 0 else 0, alpha=0.2, color='gray')

    # Plot velocities
    if len(t_vel) > 0 and len(lin_v) > 0:
        ax1.plot(t_vel, lin_v, 'b-', linewidth=0.8)
        ax1.axhline(y=MOTION_THRESH_LINEAR, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({MOTION_THRESH_LINEAR} mm/s)')
    ax1.set_ylabel('Linear Velocity (mm/s)')
    ax1.set_title(f'Snippet: Frames {start}-{end} ({len(t)} frames, {t[-1]:.1f}s)' if len(t) > 0 else 'Snippet')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    if len(t_vel) > 0 and len(ang_v) > 0:
        ax2.plot(t_vel, ang_v, 'g-', linewidth=0.8)
        ax2.axhline(y=MOTION_THRESH_ANGULAR, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({MOTION_THRESH_ANGULAR} deg/s)')
    ax2.set_ylabel('Angular Velocity (deg/s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_frames(df, snippet, output_dir):
    """Extract stereo video frames for the snippet."""
    start, end = snippet['start'], snippet['end']

    left_dir = output_dir / 'frames_left'
    right_dir = output_dir / 'frames_right'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # Sample frames (every 10th frame to keep size manageable)
    sample_indices = list(range(start, end + 1, 1))
    if end not in sample_indices:
        sample_indices.append(end)

    for idx in sample_indices:
        frame_n = df['frame_n'].iloc[idx]

        # Left frame
        left_data = df['frame_left'].iloc[idx]
        if left_data is not None and 'bytes' in left_data:
            with open(left_dir / f'frame_{frame_n:06d}.webp', 'wb') as f:
                f.write(left_data['bytes'])

        # Right frame
        right_data = df['frame_right'].iloc[idx]
        if right_data is not None and 'bytes' in right_data:
            with open(right_dir / f'frame_{frame_n:06d}.webp', 'wb') as f:
                f.write(right_data['bytes'])

    return sample_indices


def export_video(df, snippet, output_path, fps=30, camera='left'):
    """Export video snippet as MP4."""
    start, end = snippet['start'], snippet['end']

    frames = []
    frame_col = 'frame_left' if camera == 'left' else 'frame_right'

    # Get all frames (not just sampled)
    for idx in range(start, end + 1):
        frame_data = df[frame_col].iloc[idx]
        if frame_data is not None and 'bytes' in frame_data:
            img = Image.open(io.BytesIO(frame_data['bytes']))
            # Convert to RGB numpy array
            frames.append(np.array(img.convert('RGB')))

    if frames:
        # Write video
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        return len(frames)
    return 0


def export_stereo_video(df, snippet, output_path, fps=30):
    """Export side-by-side stereo video as MP4."""
    start, end = snippet['start'], snippet['end']

    frames = []

    for idx in range(start, end + 1):
        left_data = df['frame_left'].iloc[idx]
        right_data = df['frame_right'].iloc[idx]

        if left_data is not None and right_data is not None:
            left_img = Image.open(io.BytesIO(left_data['bytes'])).convert('RGB')
            right_img = Image.open(io.BytesIO(right_data['bytes'])).convert('RGB')

            # Resize for side-by-side (half width each)
            w, h = left_img.size
            left_small = left_img.resize((w // 2, h // 2))
            right_small = right_img.resize((w // 2, h // 2))

            # Combine side by side
            combined = Image.new('RGB', (w, h // 2))
            combined.paste(left_small, (0, 0))
            combined.paste(right_small, (w // 2, 0))

            frames.append(np.array(combined))

    if frames:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        return len(frames)
    return 0


def create_visualization_html(df, snippet, timestamps, positions, linear_vel, sample_indices, output_path):
    """Create interactive HTML with video frames and 3D trajectory side by side."""
    start, end = snippet['start'], snippet['end']

    # Get positions for this snippet
    pos_mm = positions[start:end+1] * 1000
    vel = linear_vel[start:end] if end > start else np.array([0])
    vel_extended = np.append(vel, vel[-1]) if len(vel) > 0 else np.array([0])

    # Create standalone 3D figure (not using subplots for better interactivity)
    fig = go.Figure()

    # 3D trajectory colored by velocity using SEGMENTED LINES
    # (Plotly 3D doesn't support continuous color gradients on lines via to_json())
    vel_max_display = max(0.5, vel_extended.max())

    # Create color array using viridis colormap
    import matplotlib.cm as cm
    vel_normalized = np.clip(vel_extended / vel_max_display, 0, 1)
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
              for r, g, b, _ in cm.viridis(vel_normalized)]

    # Draw trajectory as individual colored segments
    for i in range(len(pos_mm) - 1):
        fig.add_trace(go.Scatter3d(
            x=[pos_mm[i, 0], pos_mm[i+1, 0]],
            y=[pos_mm[i, 1], pos_mm[i+1, 1]],
            z=[pos_mm[i, 2], pos_mm[i+1, 2]],
            mode='lines',
            line=dict(color=colors[i], width=6),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add invisible trace for colorbar
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            size=0.1,
            color=[0, vel_max_display],
            colorscale='Viridis',
            cmin=0, cmax=vel_max_display,
            colorbar=dict(title=f'Velocity<br>(mm/s)', thickness=15, len=0.7)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[pos_mm[0, 0]], y=[pos_mm[0, 1]], z=[pos_mm[0, 2]],
        mode='markers', marker=dict(size=6, color='lime', symbol='circle'),
        name='Start', showlegend=True
    ))

    # End marker
    fig.add_trace(go.Scatter3d(
        x=[pos_mm[-1, 0]], y=[pos_mm[-1, 1]], z=[pos_mm[-1, 2]],
        mode='markers', marker=dict(size=6, color='red', symbol='diamond'),
        name='End', showlegend=True
    ))

    # Update layout to match evo trajectory style
    fig.update_layout(
        title=dict(
            text=f'Snippet {snippet["motion_start"]}-{snippet["motion_end"]} - Interactive 3D<br><sup>Velocity scale: 0 to {vel_max_display:.1f} mm/s</sup>',
            font=dict(size=16),
            x=0.02,
            xanchor='left'
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgba(230,236,245,0.9)'
        ),
        height=550,
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor='white'
    )

    # Load and embed sample frames as thumbnails
    frames_info = []
    for i, idx in enumerate(sample_indices[:8]):  # Show up to 8 frames
        left_data = df['frame_left'].iloc[idx]
        if left_data is not None and 'bytes' in left_data:
            img = Image.open(io.BytesIO(left_data['bytes']))
            img.thumbnail((400, 225))  # Larger thumbnails
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            frames_info.append({
                'frame': df['frame_n'].iloc[idx],
                'time': timestamps[idx] - timestamps[start],
                'b64': b64
            })

    # Create HTML with side-by-side layout
    duration = timestamps[end] - timestamps[start]
    motion_frames = snippet['motion_end'] - snippet['motion_start'] + 1

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Motion Snippet Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .header {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header h1 {{ margin: 0 0 10px 0; color: #333; }}
        .stats {{ display: flex; gap: 30px; color: #666; }}
        .stats span {{ font-size: 14px; }}
        .stats strong {{ color: #333; }}
        .main-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .panel {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .panel h2 {{ margin: 0 0 15px 0; color: #333; font-size: 18px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        #plot {{ width: 100%; height: 550px; }}
        .frame-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; max-height: 550px; overflow-y: auto; }}
        .frame-item {{ text-align: center; background: #f9f9f9; border-radius: 6px; padding: 8px; }}
        .frame-item img {{ width: 100%; border-radius: 4px; border: 1px solid #ddd; }}
        .frame-item p {{ margin: 8px 0 0 0; font-size: 12px; color: #666; }}
        .video-link {{ margin-top: 15px; padding: 10px; background: #e8f5e9; border-radius: 6px; text-align: center; }}
        .video-link a {{ color: #2e7d32; text-decoration: none; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Motion Snippet: Frames {start} - {end}</h1>
        <div class="stats">
            <span><strong>Duration:</strong> {duration:.2f}s</span>
            <span><strong>Total Frames:</strong> {end - start + 1}</span>
            <span><strong>Motion Frames:</strong> {motion_frames}</span>
            <span><strong>Max Velocity:</strong> {vel_extended.max():.2f} mm/s</span>
        </div>
    </div>

    <div class="main-container">
        <div class="panel">
            <h2>3D Camera Pose Trajectory</h2>
            <div id="plot"></div>
            <p style="font-size:12px; color:#888; margin-top:10px;">Drag to rotate, scroll to zoom, double-click to reset view</p>
        </div>
        <div class="panel">
            <h2>Video Frames (Left Camera)</h2>
            <div class="frame-grid">
"""

    for info in frames_info:
        html_content += f"""                <div class="frame-item">
                    <img src="data:image/jpeg;base64,{info['b64']}" alt="Frame {info['frame']}">
                    <p>Frame {info['frame']} | t={info['time']:.2f}s</p>
                </div>
"""

    html_content += f"""            </div>
            <div class="video-link">
                <a href="video_left.mp4" target="_blank">Open Full Video (Left Camera)</a> |
                <a href="video_stereo.mp4" target="_blank">Open Stereo Video</a>
            </div>
        </div>
    </div>

    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('plot', plotData.data, plotData.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_content)


def process_episode(episode):
    """Process a single episode and extract all motion snippets."""
    print(f"\n{'='*60}")
    print(f"Processing {episode}...")
    print('='*60)

    # Load data
    df = load_episode_data(episode)
    timestamps = df['timestamp'].values
    ecm_data = np.array(df['/ECM/custom/setpoint_cp'].tolist())
    positions = ecm_data[:, :3]
    quaternions = ecm_data[:, 3:7]

    print(f"  Loaded {len(df)} frames")

    # Compute velocities
    linear_vel, angular_vel, dt = compute_velocities(timestamps, positions, quaternions)

    # Classify motion
    is_motion = classify_motion(linear_vel, angular_vel)
    motion_pct = np.mean(is_motion) * 100
    print(f"  Motion frames: {np.sum(is_motion)} ({motion_pct:.1f}%)")

    # Find and merge segments
    segments = find_motion_segments(is_motion, timestamps)
    print(f"  Found {len(segments)} motion segments (after merging)")

    if not segments:
        print("  No motion segments found!")
        return []

    # Add padding
    snippets = add_padding(segments, len(df))

    # Create output directory
    episode_dir = OUTPUT_DIR / episode
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Process each snippet
    snippet_results = []
    for i, snippet in enumerate(snippets):
        snippet_id = f"{i+1:03d}"
        snippet_dir = episode_dir / f"snippet_{snippet_id}"
        snippet_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Snippet {snippet_id}: frames {snippet['start']}-{snippet['end']}")

        # Calculate statistics
        start, end = snippet['start'], snippet['end']
        motion_start, motion_end = snippet['motion_start'], snippet['motion_end']
        num_frames = end - start + 1
        motion_frames = motion_end - motion_start + 1
        duration = timestamps[end] - timestamps[start]

        # Get velocities for this snippet
        snip_lin_vel = linear_vel[start:end] if end > start else []
        snip_ang_vel = angular_vel[start:end] if end > start else []

        # Scene motion classification
        scene_motion = classify_scene_motion(df, snippet, timestamps)
        print(f"    Motion type: {scene_motion['motion_type']}")

        # Save scene motion JSON
        with open(snippet_dir / 'scene_motion.json', 'w') as f:
            json.dump(scene_motion, f, indent=2)

        # Export TUM poses
        export_tum_poses(df, snippet, snippet_dir / 'poses.txt')
        print(f"    Exported TUM poses")

        # Create velocity plot
        create_velocity_plot(timestamps, linear_vel, angular_vel, snippet, snippet_dir / 'velocity.png')
        print(f"    Created velocity plot")

        # Extract video frames (sampled for thumbnails)
        sample_indices = extract_frames(df, snippet, snippet_dir)
        print(f"    Extracted {len(sample_indices)} sample frames")

        # Export video snippets
        video_frames = export_video(df, snippet, snippet_dir / 'video_left.mp4', fps=60, camera='left')
        print(f"    Exported left camera video ({video_frames} frames)")

        stereo_frames = export_stereo_video(df, snippet, snippet_dir / 'video_stereo.mp4', fps=60)
        print(f"    Exported stereo video ({stereo_frames} frames)")

        # Create visualization HTML
        create_visualization_html(df, snippet, timestamps, positions, linear_vel, sample_indices, snippet_dir / 'visualization.html')
        print(f"    Created visualization HTML")

        # Compile result
        result = {
            'snippet_id': snippet_id,
            'episode': episode,
            'start_frame': int(start),
            'end_frame': int(end),
            'motion_start_frame': int(motion_start),
            'motion_end_frame': int(motion_end),
            'num_frames': int(num_frames),
            'motion_frames': int(motion_frames),
            'padding_frames': int(num_frames - motion_frames),
            'duration_s': float(duration),
            'max_linear_vel': float(max(snip_lin_vel)) if len(snip_lin_vel) > 0 else 0,
            'max_angular_vel': float(max(snip_ang_vel)) if len(snip_ang_vel) > 0 else 0,
            'scene_motion': scene_motion,
            'tissue_motion': "",
        }
        snippet_results.append(result)

    # Save episode summary
    with open(episode_dir / f'{episode}_snippets.json', 'w') as f:
        json.dump(snippet_results, f, indent=2)

    return snippet_results


def main():
    import sys

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if specific episode requested
    episodes_to_process = EPISODES
    if len(sys.argv) > 1:
        ep = sys.argv[1]
        if ep in EPISODES:
            episodes_to_process = [ep]
        else:
            print(f"Unknown episode: {ep}")
            print(f"Available: {EPISODES}")
            return

    print("="*60)
    print("MOTION SNIPPET EXTRACTION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Padding: {PADDING_FRAMES} frames (~{PADDING_FRAMES/60:.1f}s)")
    print(f"  Merge gap: {MERGE_GAP_SECONDS}s")
    print(f"  Min motion: {MIN_MOTION_FRAMES} frames")
    print(f"  Thresholds: {MOTION_THRESH_LINEAR} mm/s, {MOTION_THRESH_ANGULAR} deg/s")

    all_results = []
    for episode in episodes_to_process:
        results = process_episode(episode)
        all_results.extend(results)

    # Save global summary
    with open(OUTPUT_DIR / 'snippets_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Episode':<8} {'Snippet':<10} {'Frames':<10} {'Duration':<10} {'Motion Type':<15}")
    print("-"*60)
    for r in all_results:
        print(f"{r['episode']:<8} {r['snippet_id']:<10} {r['num_frames']:<10} {r['duration_s']:.1f}s{'':<5} {r['scene_motion']['motion_type']:<15}")

    print(f"\nTotal snippets: {len(all_results)}")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
