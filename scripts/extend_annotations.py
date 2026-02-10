#!/usr/bin/env python3
"""
Extend tissue annotations using SAM3 video mask propagation.

The CRCD annotation pipeline (Track Anything) produces 120-frame clips per split,
but F_3's exported annotations only contain offsets 0-99, leaving a systematic
16.7% gap at offsets 100-119. This script fills those gaps by using SAM3's video
predictor to propagate GT masks from the last annotated frame forward through the
actual video frames.

Approach:
  For each snippet, start ONE SAM3 video session with all snippet frames.
  Add GT Liver/Gallbladder masks as mask prompts at every offset-99 frame
  (the last annotated frame per split). SAM3 propagates these through the
  video, filling offsets 100-119 with temporally-coherent masks.

Requires:
  - CUDA GPU for SAM3 inference
  - Source frame images (from snippets directory)
  - GT annotation JSON files (tissue_segmentation)

Output: A gap-fill-only JSON file (F_3_fill.json) containing ONLY the newly
generated frames. Integrates with _load_episode_annotations() which merges
all JSON files in the episode directory.

Usage:
    # Dry-run: report gaps and which snippets can fill them
    python scripts/extend_annotations.py \\
        --tissue-seg-dir "path/to/tissue_segmentation" \\
        --snippets-dir "path/to/snippets" \\
        --episode F_3 --dry-run

    # Generate gap-fill annotations using SAM3
    python scripts/extend_annotations.py \\
        --tissue-seg-dir "path/to/tissue_segmentation" \\
        --snippets-dir "path/to/snippets" \\
        --episode F_3
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Annotation loading and analysis (no SAM3 needed)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extend tissue annotations using SAM3 mask propagation",
    )
    parser.add_argument(
        "--tissue-seg-dir", required=True,
        help="Path to tissue_segmentation directory",
    )
    parser.add_argument(
        "--snippets-dir", required=True,
        help="Path to snippets directory (contains episode/snippet_NNN/frames_left/)",
    )
    parser.add_argument(
        "--episode", default="F_3",
        help="Episode to extend (default: F_3)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: {episode}_fill.json in episode dir)",
    )
    parser.add_argument(
        "--frames-per-split", type=int, default=120,
        help="Expected frames per split (default: 120)",
    )
    parser.add_argument(
        "--min-area", type=int, default=500,
        help="Minimum mask area in pixels to keep (default: 500)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report gaps without running SAM3 inference",
    )
    return parser.parse_args()


def load_and_merge(ep_dir):
    """Load and merge all annotation JSON files for an episode."""
    all_images = {}
    all_annotations = []
    categories = []

    for name in sorted(ep_dir.glob("*.json")):
        if "_fill" in name.stem or "_extended" in name.stem:
            continue
        print(f"  Loading {name.name}...")
        with open(name) as f:
            data = json.load(f)

        if not categories and "categories" in data:
            categories = data["categories"]

        for img in data.get("images", []):
            if img["id"] not in all_images:
                all_images[img["id"]] = img

        all_annotations.extend(data.get("annotations", []))

    return all_images, all_annotations, categories


def analyze_splits(images, frames_per_split):
    """Analyze which splits exist and what offsets they have."""
    splits = defaultdict(set)
    split_images = defaultdict(dict)

    for img_id, img in images.items():
        fname = img["file_name"]
        m = re.search(r"split_(\d+)", fname)
        if not m:
            continue
        split_num = int(m.group(1))
        offset = int(Path(fname).stem)
        splits[split_num].add(offset)
        split_images[split_num][offset] = img

    return splits, split_images


def build_annotation_index(annotations):
    """Build image_id -> [annotations] lookup."""
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)
    return ann_by_image


def reconstruct_mask(anns, category_name, categories, h, w):
    """Reconstruct a binary mask from COCO polygon annotations for a category."""
    cat_id = None
    for cat in categories:
        if cat["name"] == category_name:
            cat_id = cat["id"]
            break
    if cat_id is None:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        if ann["category_id"] != cat_id:
            continue
        for poly in ann.get("segmentation", []):
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    if mask.sum() == 0:
        return None
    return mask


def mask_to_coco_polygons(binary_mask):
    """Convert binary mask to COCO polygon format."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = contour.flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


# ---------------------------------------------------------------------------
# Snippet-to-split mapping
# ---------------------------------------------------------------------------

def build_snippet_frame_map(snippet_dir):
    """
    Build a mapping from video frame number to (snippet_frame_index, frame_path)
    for all frames in a snippet's frames_left/ directory.
    """
    frames_dir = snippet_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    frame_map = {}
    for idx, fpath in enumerate(frame_files):
        # frame_010898.webp -> video frame 10898
        video_frame = int(fpath.stem.split("_")[1])
        frame_map[video_frame] = (idx, fpath)

    return frame_files, frame_map


def find_splits_needing_fill(frame_map, splits, frames_per_split):
    """
    For a snippet's frame range, find splits that:
    1. Have an annotated frame at their max offset (anchor for propagation)
    2. Are missing offsets up to frames_per_split
    3. Have the missing offset frames available in the snippet

    Returns list of (split_num, anchor_video_frame, anchor_snippet_idx,
                      [(missing_video_frame, missing_snippet_idx, offset), ...])
    """
    fillable = []

    for video_frame in sorted(frame_map.keys()):
        split_num = video_frame // frames_per_split
        offset = video_frame % frames_per_split

        # We only care about the anchor frame (last annotated offset)
        if split_num not in splits:
            continue
        max_annotated = max(splits[split_num])
        if offset != max_annotated:
            continue

        # Check which missing offsets have frames in this snippet
        missing_with_frames = []
        for off in range(max_annotated + 1, frames_per_split):
            target_vf = split_num * frames_per_split + off
            if target_vf in frame_map:
                missing_with_frames.append(
                    (target_vf, frame_map[target_vf][0], off)
                )

        if missing_with_frames:
            anchor_idx = frame_map[video_frame][0]
            fillable.append(
                (split_num, video_frame, anchor_idx, missing_with_frames)
            )

    return fillable


# ---------------------------------------------------------------------------
# SAM3 mask propagation
# ---------------------------------------------------------------------------

def propagate_snippet_masks(
    predictor, snippet_dir, frame_files, frame_map,
    fillable_splits, ann_by_image, split_images, categories,
    min_area, img_h, img_w,
):
    """
    Run SAM3 video predictor on a snippet to propagate GT masks into gap frames.

    For each split needing fill:
      - Reconstructs GT Liver/Gallbladder masks at the anchor frame (offset 99)
      - Adds them as mask prompts to the SAM3 session
    Then propagates through ALL snippet frames.
    Collects output only at the gap frames (offsets 100-119).

    Returns list of (video_frame, offset, split_num, category_masks_dict)
    where category_masks_dict = {cat_name: binary_mask_np}
    """
    frames_dir = snippet_dir / "frames_left"

    # Category name -> object ID mapping for SAM3 tracking
    tissue_cats = ["Liver", "Gallbladder"]
    cat_to_obj_id = {cat: i + 1 for i, cat in enumerate(tissue_cats)}

    # Collect all gap frame indices we need output for
    gap_frame_indices = set()
    gap_frame_info = {}  # snippet_idx -> (video_frame, offset, split_num)
    for split_num, anchor_vf, anchor_idx, missing_frames in fillable_splits:
        for vf, snippet_idx, offset in missing_frames:
            gap_frame_indices.add(snippet_idx)
            gap_frame_info[snippet_idx] = (vf, offset, split_num)

    if not gap_frame_indices:
        return []

    # Start SAM3 video session
    print(f"    Starting SAM3 session ({len(frame_files)} frames)...")
    session = predictor.start_session(resource_path=str(frames_dir))
    sid = session["session_id"]
    inference_state = session.get("inference_state")

    # Add mask prompts at each anchor frame
    prompts_added = 0
    for split_num, anchor_vf, anchor_idx, _ in fillable_splits:
        # Get GT annotations for the anchor frame
        source_img = split_images[split_num].get(anchor_vf % 120)
        if source_img is None:
            continue
        source_anns = ann_by_image.get(source_img["id"], [])
        if not source_anns:
            continue

        for cat_name in tissue_cats:
            mask_np = reconstruct_mask(
                source_anns, cat_name, categories, img_h, img_w
            )
            if mask_np is None:
                continue

            obj_id = cat_to_obj_id[cat_name]
            mask_tensor = torch.from_numpy(mask_np).float() / 255.0

            # Add mask prompt via tracker's add_new_mask
            try:
                if inference_state is not None:
                    predictor.tracker.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=anchor_idx,
                        obj_id=obj_id,
                        mask=mask_tensor,
                    )
                else:
                    # Try high-level API
                    predictor.add_new_mask(
                        session_id=sid,
                        frame_idx=anchor_idx,
                        obj_id=obj_id,
                        mask=mask_tensor,
                    )
                prompts_added += 1
            except Exception as e:
                print(f"    WARNING: Failed to add mask for {cat_name} "
                      f"at frame {anchor_idx}: {e}")

    if prompts_added == 0:
        print("    No mask prompts added — skipping propagation")
        predictor.close_session(session_id=sid)
        return []

    print(f"    Added {prompts_added} mask prompts across "
          f"{len(fillable_splits)} anchor frames")

    # Propagate through all frames
    results = []
    t0 = time.time()
    frames_processed = 0

    print(f"    Propagating masks...")
    for response in predictor.propagate_in_video(
        session_id=sid,
        propagation_direction="both",
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]
        frames_processed += 1

        # Only collect results for gap frames
        if frame_idx not in gap_frame_indices:
            continue

        vf, offset, split_num = gap_frame_info[frame_idx]

        # Extract per-category masks from SAM3 output
        cat_masks = {}
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks", np.empty((0,)))

        for i, oid in enumerate(obj_ids):
            oid_int = int(oid)
            # Reverse-lookup category name from obj_id
            for cat_name, mapped_oid in cat_to_obj_id.items():
                if mapped_oid == oid_int:
                    mask_np = binary_masks[i]
                    if isinstance(mask_np, torch.Tensor):
                        mask_np = mask_np.cpu().numpy()
                    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                    area = int(mask_uint8.sum() // 255)
                    if area >= min_area:
                        cat_masks[cat_name] = mask_uint8
                    break

        if cat_masks:
            results.append((vf, offset, split_num, cat_masks))

    elapsed = time.time() - t0
    predictor.close_session(session_id=sid)
    print(f"    Propagation done: {frames_processed} frames in {elapsed:.1f}s, "
          f"{len(results)} gap frames filled")

    return results


# ---------------------------------------------------------------------------
# COCO output generation
# ---------------------------------------------------------------------------

def results_to_coco(results, categories, img_w, img_h, start_img_id, start_ann_id):
    """Convert propagated mask results to COCO image + annotation entries."""
    new_images = []
    new_annotations = []
    img_id = start_img_id
    ann_id = start_ann_id

    # Build category name -> id mapping
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    for vf, offset, split_num, cat_masks in results:
        new_fname = f"./split_imgs/split_{split_num}/{offset:05d}.jpg"
        new_img = {
            "id": img_id,
            "file_name": new_fname,
            "width": img_w,
            "height": img_h,
        }
        new_images.append(new_img)

        for cat_name, mask_uint8 in cat_masks.items():
            cat_id = cat_name_to_id.get(cat_name)
            if cat_id is None:
                continue

            polygons = mask_to_coco_polygons(mask_uint8)
            if not polygons:
                continue

            area = float(mask_uint8.sum() // 255)
            ys, xs = np.where(mask_uint8 > 0)
            bbox = [
                float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min()),
            ]

            new_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": polygons,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return new_images, new_annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tissue_dir = Path(args.tissue_seg_dir)
    snippets_dir = Path(args.snippets_dir)
    ep_dir = tissue_dir / args.episode
    expected = args.frames_per_split

    if not ep_dir.exists():
        print(f"ERROR: Episode directory not found: {ep_dir}")
        sys.exit(1)

    snippet_ep_dir = snippets_dir / args.episode
    if not snippet_ep_dir.exists():
        print(f"ERROR: Snippet episode directory not found: {snippet_ep_dir}")
        sys.exit(1)

    # --- Load annotations ---
    print(f"Loading {args.episode} annotations...")
    images, annotations, categories = load_and_merge(ep_dir)
    print(f"  {len(images)} images, {len(annotations)} annotations")

    ann_by_image = build_annotation_index(annotations)
    splits, split_images = analyze_splits(images, expected)

    # --- Classify splits ---
    full_splits = []
    partial_splits = []
    for split_num in sorted(splits.keys()):
        if len(splits[split_num]) >= expected:
            full_splits.append(split_num)
        else:
            partial_splits.append(
                (split_num, len(splits[split_num]),
                 min(splits[split_num]), max(splits[split_num]))
            )

    all_split_nums = sorted(splits.keys())
    expected_range = range(all_split_nums[0], all_split_nums[-1] + 1)
    missing_splits = [s for s in expected_range if s not in splits]

    print(f"\nSplit analysis for {args.episode}:")
    print(f"  Split range: {all_split_nums[0]}-{all_split_nums[-1]}")
    print(f"  Total splits present: {len(splits)}")
    print(f"  Full ({expected} frames): {len(full_splits)}")
    print(f"  Partial (<{expected} frames): {len(partial_splits)}")
    print(f"  Missing entirely: {len(missing_splits)}")

    if partial_splits:
        sizes = [s[1] for s in partial_splits]
        print(f"  Partial frame counts: min={min(sizes)}, max={max(sizes)}, "
              f"most common={max(set(sizes), key=sizes.count)}")

    if not partial_splits:
        print("\nNo partial splits found — nothing to extend.")
        return

    # --- Discover snippets and map to fillable splits ---
    snippet_dirs = sorted(
        [d for d in snippet_ep_dir.glob("snippet_*") if d.is_dir()]
    )
    print(f"\nDiscovered {len(snippet_dirs)} snippets in {snippet_ep_dir}")

    # Determine image dimensions from first frame
    first_snippet_frames = sorted(
        (snippet_dirs[0] / "frames_left").glob("frame_*.webp")
    )
    if not first_snippet_frames:
        print("ERROR: No frames found in first snippet")
        sys.exit(1)
    sample_img = cv2.imread(str(first_snippet_frames[0]))
    img_h, img_w = sample_img.shape[:2]
    print(f"  Frame dimensions: {img_w}x{img_h}")

    # Build per-snippet fill plans
    total_fillable = 0
    snippet_plans = []
    for snippet_dir in snippet_dirs:
        frame_files, frame_map = build_snippet_frame_map(snippet_dir)
        if not frame_files:
            continue

        fillable = find_splits_needing_fill(frame_map, splits, expected)
        if fillable:
            n_gap_frames = sum(len(mf) for _, _, _, mf in fillable)
            snippet_plans.append((snippet_dir, frame_files, frame_map, fillable))
            total_fillable += n_gap_frames
            print(f"  {snippet_dir.name}: {len(fillable)} splits, "
                  f"{n_gap_frames} gap frames")

    print(f"\nTotal fillable gap frames: {total_fillable} "
          f"(across {sum(len(fp[3]) for fp in snippet_plans)} splits)")

    # --- Dry run: show coverage improvement ---
    if args.dry_run:
        current_frames = set()
        for img in images.values():
            m = re.search(r"split_(\d+)", img["file_name"])
            if m:
                sn = int(m.group(1))
                off = int(Path(img["file_name"]).stem)
                current_frames.add(sn * 120 + off)

        extended_frames = set(current_frames)
        for _, _, _, fillable in snippet_plans:
            for split_num, _, _, missing_frames in fillable:
                for vf, _, _ in missing_frames:
                    extended_frames.add(vf)

        # Load snippet metadata for coverage report
        snippet_json = snippet_ep_dir / f"{args.episode}_snippets.json"
        if snippet_json.exists():
            print(f"\nSnippet coverage improvement:")
            with open(snippet_json) as f:
                snippet_meta = json.load(f)
            for snip in snippet_meta:
                sid = snip["snippet_id"]
                start, end = snip["start_frame"], snip["end_frame"]
                total = end - start + 1
                cur = sum(1 for f in range(start, end + 1) if f in current_frames)
                ext = sum(1 for f in range(start, end + 1) if f in extended_frames)
                print(f"  snippet_{sid}: {cur}/{total} "
                      f"({100*cur/total:.1f}%) -> "
                      f"{ext}/{total} ({100*ext/total:.1f}%)")

        print("\n[Dry run complete — no SAM3 inference performed]")
        return

    # --- SAM3 propagation ---
    print("\n" + "=" * 60)
    print("Loading SAM3 video predictor...")
    print("=" * 60)

    from sam3 import Sam3VideoPredictor

    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    device = next(predictor.model.parameters()).device
    print(f"Model device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_mem / (1024**3):.1f} GB)")

    # Process each snippet
    all_results = []
    t_total = time.time()

    for snippet_dir, frame_files, frame_map, fillable in snippet_plans:
        print(f"\n{'=' * 60}")
        print(f"Processing {snippet_dir.name} "
              f"({len(fillable)} splits, {len(frame_files)} frames)")
        print(f"{'=' * 60}")

        results = propagate_snippet_masks(
            predictor=predictor,
            snippet_dir=snippet_dir,
            frame_files=frame_files,
            frame_map=frame_map,
            fillable_splits=fillable,
            ann_by_image=ann_by_image,
            split_images=split_images,
            categories=categories,
            min_area=args.min_area,
            img_h=img_h,
            img_w=img_w,
        )
        all_results.extend(results)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"SAM3 propagation complete: {len(all_results)} gap frames filled "
          f"in {total_time:.1f}s")
    print(f"{'=' * 60}")

    if not all_results:
        print("No gap frames filled — nothing to write.")
        return

    # --- Convert to COCO and save ---
    max_img_id = max(images.keys())
    max_ann_id = max(a["id"] for a in annotations) if annotations else 0

    new_images, new_annotations = results_to_coco(
        all_results, categories, img_w, img_h,
        start_img_id=max_img_id + 1,
        start_ann_id=max_ann_id + 1,
    )

    output_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ep_dir / f"{args.episode}_fill.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done! {file_size:.1f} MB")
    print(f"  {len(new_images)} images, {len(new_annotations)} annotations")

    total_before = len(images)
    total_after = total_before + len(new_images)
    print(f"\nCoverage: {total_before} -> {total_after} frames (+{len(new_images)})")
    print(f"Remaining gaps: {len(missing_splits)} missing splits")
    print(f"\nThe fill file integrates automatically with "
          f"_load_episode_annotations()")


if __name__ == "__main__":
    main()
