"""
Render poster assets:
  1. 8 segmentation overlay frames from C_1 snippet_001
  2. 3D camera trajectory PNGs for C_1, E_3, F_3
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root for shared_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scripts.shared_config import (
    CATEGORY_COLORS_BGR,
    render_mask_overlay,
)

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS = ROOT / "data" / "Segments"
OUT = ROOT / "for poster"
OUT.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def coco_poly_to_mask(segmentation, h, w):
    """Decode COCO polygon segmentation to binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_poses(poses_path):
    """Load TUM-format poses: timestamp tx ty tz qx qy qz qw"""
    data = []
    with open(poses_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                data.append([float(x) for x in parts[:4]])  # ts, tx, ty, tz
    return np.array(data)


# ── Part 1: Segmentation frames ─────────────────────────────────────────────

def render_segmentation_frames():
    snippet_dir = SEGMENTS / "C_1" / "snippet_001"
    frames_dir = snippet_dir / "frames_left"
    ann_path = snippet_dir / "snippet_annotations.json"

    with open(ann_path) as f:
        coco = json.load(f)

    # Build category id -> name map
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Build image_id -> annotations
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Get sorted frame files
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    n = len(frame_files)
    print(f"Found {n} frames in C_1/snippet_001")

    # Pick 8 evenly spaced frames
    indices = [int(i * (n - 1) / 7) for i in range(8)]

    # Map frame number -> image_id (frame_001561.webp -> image_id 1561)
    for idx in indices:
        fpath = frame_files[idx]
        frame_num = int(fpath.stem.split("_")[1])

        # Load frame
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  SKIP: cannot read {fpath.name}")
            continue

        h, w = frame.shape[:2]

        # Find annotations for this frame
        anns = ann_by_img.get(frame_num, [])
        overlay_count = 0

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list):
                continue
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            color = CATEGORY_COLORS_BGR.get(cat_name, (0, 255, 255))
            mask = coco_poly_to_mask(seg, h, w)
            frame = render_mask_overlay(frame, mask, color, alpha=0.30,
                                        contour_thickness=2, contour_outline=True)
            overlay_count += 1

        # Draw legend
        y_pos = 30
        cats_present = set()
        for ann in anns:
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            if cat_name not in cats_present:
                cats_present.add(cat_name)
                color = CATEGORY_COLORS_BGR.get(cat_name, (0, 255, 255))
                cv2.putText(frame, cat_name, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, cat_name, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 28

        # Frame label
        label = f"Frame {frame_num}"
        cv2.putText(frame, label, (w - 200, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = OUT / f"C1_seg_frame_{frame_num:06d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"  Saved {out_path.name} ({overlay_count} masks)")


# ── Part 2: 3D Trajectories ─────────────────────────────────────────────────

def render_trajectory(episode):
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ep_dir = SEGMENTS / episode
    snippets = sorted(ep_dir.glob("snippet_*"))

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    # Clean white background for poster
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.xaxis.pane.set_facecolor("#fafafa")
    ax.yaxis.pane.set_facecolor("#f5f5f5")
    ax.zaxis.pane.set_facecolor("#f0f0f0")
    ax.xaxis.pane.set_edgecolor("#cccccc")
    ax.yaxis.pane.set_edgecolor("#cccccc")
    ax.zaxis.pane.set_edgecolor("#cccccc")
    ax.grid(True, alpha=0.3, color="#cccccc")

    # Collect all velocities across snippets for global normalization
    all_data = []
    for sdir in snippets:
        poses_path = sdir / "poses.txt"
        if not poses_path.exists():
            continue
        data = load_poses(poses_path)
        if len(data) > 0:
            all_data.append(data)

    # Global velocity range for consistent coloring
    all_vels = []
    for data in all_data:
        tx, ty, tz = data[:, 1], data[:, 2], data[:, 3]
        diffs = np.sqrt(np.diff(tx)**2 + np.diff(ty)**2 + np.diff(tz)**2)
        all_vels.extend(diffs.tolist())
    global_vmax = np.percentile(all_vels, 98) if all_vels else 1.0

    total_points = 0
    for i, data in enumerate(all_data):
        tx, ty, tz = data[:, 1], data[:, 2], data[:, 3]
        total_points += len(data)

        # Compute velocity for color gradient
        diffs = np.sqrt(np.diff(tx)**2 + np.diff(ty)**2 + np.diff(tz)**2)
        vel = np.concatenate([[0], diffs])
        vel_norm = np.clip(vel / (global_vmax + 1e-8), 0, 1)

        # Build colored line segments
        points = np.array([tx, ty, tz]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment by its velocity (average of endpoints)
        seg_vel = (vel_norm[:-1] + vel_norm[1:]) / 2
        seg_colors = plt.cm.plasma(seg_vel)

        lc = Line3DCollection(segments, colors=seg_colors, linewidths=2.2, alpha=0.9)
        ax.add_collection3d(lc)

        # Mark start and end
        ax.scatter([tx[0]], [ty[0]], [tz[0]], c="#2ecc71", s=50,
                   marker="o", edgecolors="#333333", linewidths=0.8, zorder=5)
        ax.scatter([tx[-1]], [ty[-1]], [tz[-1]], c="#e74c3c", s=50,
                   marker="^", edgecolors="#333333", linewidths=0.8, zorder=5)

        # Label snippet
        mid = len(tx) // 2
        ax.text(tx[mid], ty[mid], tz[mid], f"  S{i+1}", fontsize=8,
                color="#333333", fontweight="bold", alpha=0.85)

    # Auto-scale axes (Line3DCollection doesn't auto-scale)
    if all_data:
        all_pts = np.vstack([d[:, 1:4] for d in all_data])
        ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
        ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
        ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    ax.set_title(f"{episode} — Camera Trajectory ({total_points} poses)",
                 color="#222222", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("X (m)", color="#555555", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (m)", color="#555555", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (m)", color="#555555", fontsize=10, labelpad=8)
    ax.tick_params(colors="#555555", labelsize=7)

    # Colorbar for velocity
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(0, global_vmax * 1000))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label("Speed (mm/frame)", color="#555555", fontsize=9)
    cbar.ax.tick_params(colors="#555555", labelsize=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="#333", markersize=8, label="Start", linestyle="None"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="#333", markersize=8, label="End", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              facecolor="white", edgecolor="#cccccc", labelcolor="#333333")

    plt.tight_layout()
    out_path = OUT / f"{episode}_3D_trajectory.png"
    fig.savefig(str(out_path), facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved {out_path.name} ({len(all_data)} snippets, {total_points} poses)")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Rendering segmentation frames ===")
    render_segmentation_frames()

    print("\n=== Rendering 3D trajectories ===")
    for ep in ["C_1", "E_3", "F_3"]:
        render_trajectory(ep)

    print(f"\nAll assets saved to: {OUT}")
