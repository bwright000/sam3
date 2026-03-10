"""
Extract separate layer images for a single frame:
  1. Original image (no overlays)
  2. Tool masks only
  3. GT masks only

Usage:
    python scripts/extract_frame_layers.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# -- Add project root so we can import check_annotations --
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_annotations import COCOAnnotationLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODE = "C_1"
SNIPPET = "snippet_001"
FRAME = "frame_001582"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEGMENTS_DIR = DATA_DIR / "Segments" / EPISODE / SNIPPET
RESULTS_JSON = ROOT / "outputs" / "segments" / EPISODE / SNIPPET / f"{SNIPPET}_results.json"
FRAME_PATH = SEGMENTS_DIR / "frames_left" / f"{FRAME}.webp"
OUTPUT_DIR = ROOT / "outputs" / "segments" / EPISODE / SNIPPET / "layer_extracts"

from scripts.shared_config import (
    CATEGORY_COLORS_BGR_LOWER as CATEGORY_COLORS,
    GT_CAT_MAP,
)


def draw_mask_layer(img_bgr, mask_uint8, color):
    """Draw a single mask with alpha blend + triple contours."""
    overlay = img_bgr.copy()
    mask_bool = mask_uint8 > 0
    color_np = np.array(color, dtype=np.uint8)
    color_layer = overlay.copy()
    color_layer[mask_bool] = color_np
    overlay = cv2.addWeighted(overlay, 0.75, color_layer, 0.25, 0)

    contours, _ = cv2.findContours(
        (mask_uint8 * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 7)
    cv2.drawContours(overlay, contours, -1, (0, 0, 0), 5)
    cv2.drawContours(overlay, contours, -1, color, 3)
    return overlay


def draw_legend(img, labels):
    """Draw a simple legend in top-left corner."""
    y = 25
    for label in labels:
        color = CATEGORY_COLORS.get(label, (0, 255, 255))
        cv2.putText(img, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load original frame
    print(f"Loading frame: {FRAME_PATH}")
    from PIL import Image
    pil_image = Image.open(FRAME_PATH).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # --- 1. Save original image as-is ---
    out_original = OUTPUT_DIR / f"{FRAME}_original.jpg"
    cv2.imwrite(str(out_original), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out_original}")

    # --- 2. Tool masks only ---
    print("Loading results JSON...")
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)

    frame_result = None
    for entry in all_results:
        if entry["frame"] == FRAME:
            frame_result = entry
            break
    if frame_result is None:
        print(f"ERROR: {FRAME} not found in results JSON")
        return

    tool_overlay = img_bgr.copy()
    tool_masks = frame_result["masks"].get("tool", [])
    print(f"  Tool masks: {len(tool_masks)}")
    for mask_data in tool_masks:
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in mask_data["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        tool_overlay = draw_mask_layer(tool_overlay, mask, CATEGORY_COLORS["tool"])

    draw_legend(tool_overlay, [f"tool: {len(tool_masks)}"])
    out_tool = OUTPUT_DIR / f"{FRAME}_tool_masks.jpg"
    cv2.imwrite(str(out_tool), tool_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out_tool}")

    # --- 3. GT masks only ---
    print("Loading COCO annotations (this may take a moment)...")
    loader = COCOAnnotationLoader(str(DATA_DIR / "train.json"), str(DATA_DIR))
    loader.load()
    # Merge test.json
    test_path = DATA_DIR / "test.json"
    if test_path.exists():
        print("  Merging test.json...")
        with open(test_path) as f:
            test_data = json.load(f)
        added = 0
        for img in test_data.get("images", []):
            if img["id"] not in loader.images:
                loader.images[img["id"]] = img
                loader.file_to_id[img["file_name"]] = img["id"]
                added += 1
        for ann in test_data.get("annotations", []):
            loader.annotations[ann["image_id"]].append(ann)
        # Also register categories
        for cat in test_data.get("categories", []):
            if cat["id"] not in loader.categories:
                loader.categories[cat["id"]] = cat["name"]
        print(f"    +{added} images from test.json")
        loader.build_frame_mapping()

    frame_num = int(FRAME.split("_")[1])
    gt_masks = loader.get_frame_masks_by_frame_num(frame_num)

    gt_overlay = img_bgr.copy()
    gt_labels = []
    if gt_masks:
        for gt_cat, color_key in GT_CAT_MAP.items():
            if gt_cat not in gt_masks:
                continue
            mask = gt_masks[gt_cat].astype(np.uint8)
            gt_overlay = draw_mask_layer(gt_overlay, mask, CATEGORY_COLORS[color_key])
            gt_labels.append(f"{color_key}: GT")
        print(f"  GT categories found: {list(gt_masks.keys())}")
    else:
        print(f"  WARNING: No GT masks found for frame {frame_num}")

    draw_legend(gt_overlay, gt_labels)
    out_gt = OUTPUT_DIR / f"{FRAME}_gt_masks.jpg"
    cv2.imwrite(str(out_gt), gt_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {out_gt}")

    print(f"\nDone! All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
