"""
Verify GT annotation-to-frame mapping by overlaying GT masks on snippet frames.

Uses per-snippet snippet_annotations.json via COCOAnnotationLoader to render
GT masks at keyframes (offset 0 of each split). Saves overlays to outputs/gt_verify/.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_annotations import COCOAnnotationLoader

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "gt_verify"

from scripts.shared_config import CATEGORY_COLORS_BGR

# Episodes to check with their snippet directories
EPISODES = {
    "C_1": BASE_DIR / "data" / "Segments" / "C_1",
    "E_3": BASE_DIR / "data" / "Segments" / "E_3",
    "F_3": BASE_DIR / "data" / "Segments" / "F_3",
}


def render_overlay(image_bgr, masks_dict, label_text=""):
    """Render category masks on a frame image."""
    frame = image_bgr.copy()
    h, w = frame.shape[:2]

    for cat_name, mask in masks_dict.items():
        color = CATEGORY_COLORS_BGR.get(cat_name, (0, 255, 255))
        color_np = np.array(color, dtype=np.uint8)

        mask_bool = mask > 0
        colored = np.where(mask_bool[..., None], color_np, frame)
        frame = cv2.addWeighted(frame, 0.75, colored, 0.25, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 5)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 3)
        cv2.drawContours(frame, contours, -1, color, 2)

    # Legend
    y = 30
    for cat_name, mask in masks_dict.items():
        color = CATEGORY_COLORS_BGR.get(cat_name, (0, 255, 255))
        area_pct = 100 * (mask > 0).sum() / max(mask.size, 1)
        label = f"{cat_name}: {area_pct:.1f}%"
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30

    if label_text:
        cv2.putText(frame, label_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, label_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def process_snippet(episode, snippet_dir, split_size):
    """Process one snippet: find GT keyframes, overlay masks, save results."""
    snippet_name = snippet_dir.name
    ann_path = snippet_dir / "snippet_annotations.json"
    frames_dir = snippet_dir / "frames_left"

    if not ann_path.exists():
        print(f"  {snippet_name}: no snippet_annotations.json, skipping")
        return 0

    if not frames_dir.exists():
        print(f"  {snippet_name}: no frames_left/, skipping")
        return 0

    # Load annotations via COCOAnnotationLoader with explicit split_size
    loader = COCOAnnotationLoader(str(ann_path), str(snippet_dir), split_size=split_size)
    loader.load()

    # Find available frames
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print(f"  {snippet_name}: no frames found")
        return 0

    frame_map = {}
    for fp in frame_files:
        vf = int(fp.stem.split("_")[1])
        frame_map[vf] = fp

    first_frame = min(frame_map.keys())
    last_frame = max(frame_map.keys())

    # Find GT keyframes (offset 0 = multiples of split_size)
    first_gt = ((first_frame + split_size - 1) // split_size) * split_size
    gt_frames = list(range(first_gt, last_frame + 1, split_size))

    rendered_count = 0
    for vf in gt_frames:
        split_num = vf // split_size

        if vf not in frame_map:
            print(f"    frame {vf} (split {split_num}): frame file missing")
            continue

        # Get masks from loader
        masks = loader.get_frame_masks_by_frame_num(vf)
        if masks is None or len(masks) == 0:
            print(f"    frame {vf} (split {split_num}): no annotations")
            continue

        # Load frame image
        img_bgr = cv2.imread(str(frame_map[vf]))
        if img_bgr is None:
            print(f"    frame {vf} (split {split_num}): cannot read image")
            continue

        label = f"{episode} {snippet_name} | frame {vf} | split {split_num} (GT keyframe)"
        overlay = render_overlay(img_bgr, masks, label)

        out_name = f"{episode}_{snippet_name}_frame{vf:06d}_split{split_num}.jpg"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cat_names = sorted(masks.keys())
        print(f"    frame {vf} (split {split_num}): {', '.join(cat_names)} -> {out_name}")
        rendered_count += 1

    return rendered_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for episode, ep_dir in EPISODES.items():
        print(f"\n{'=' * 60}")
        print(f"  Episode: {episode}")
        print(f"{'=' * 60}")

        # Load snippets metadata to get split_size
        snippets_json = ep_dir / f"{episode}_snippets.json"
        if not snippets_json.exists():
            print(f"  No snippets JSON found")
            continue

        with open(snippets_json) as f:
            snippets = json.load(f)

        # Get split_size from first snippet that has it
        split_size = None
        for s in snippets:
            if "split_size" in s:
                split_size = s["split_size"]
                break

        if split_size is None:
            print(f"  No split_size in snippets metadata, skipping")
            continue

        print(f"  Split size: {split_size}")

        # Process each snippet
        snippet_dirs = sorted(ep_dir.glob("snippet_*"))
        for snippet_dir in snippet_dirs:
            if not snippet_dir.is_dir():
                continue
            count = process_snippet(episode, snippet_dir, split_size)
            total += count

    print(f"\n{'=' * 60}")
    print(f"  Total overlays rendered: {total}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
