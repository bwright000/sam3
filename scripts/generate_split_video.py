"""
Split-aware segmentation pipeline for E_3 and F_3.

Generates fully-segmented videos with 3 masks: Tool, Liver, Gallbladder.

The CRCD dataset provides human-annotated GT only for frame 0 of each split
(every 100 frames for F_3, every 120 for E_3). This script:
  1. Uses GT tissue masks (Liver, Gallbladder) at split keyframes
  2. Propagates tissue forward/backward with SAM3 tracker
  3. Monitors tissue area; backpropagates from next GT if degradation detected
  4. Detects tools via text prompt "tool" per split (re-prompts until found)
  5. Renders overlays and stitches final video

Usage:
    # Single snippet
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 --snippet 1 \\
        --tissue-seg-dir "F:\\2026 vibes\\MPHY Project\\annotated_dataset\\tissue_segmentation"

    # All snippets for an episode
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 \\
        --tissue-seg-dir "F:\\2026 vibes\\MPHY Project\\annotated_dataset\\tissue_segmentation"

    # Test mode (first N frames)
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 --snippet 1 --test 100 \\
        --tissue-seg-dir "F:\\2026 vibes\\MPHY Project\\annotated_dataset\\tissue_segmentation"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_tool_masks import (
    mask_to_coco_polygons,
    _load_episode_annotations,
    _render_overlay_from_results,
    _stitch_snippet_video,
    CATEGORY_COLORS,
    DEFAULT_COLOR,
)
from scripts.generate_tool_masks_video import (
    load_video_model,
    _convert_video_output,
    _load_frames_for_tracker,
)

from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# Tissue categories
TISSUE_CATEGORIES = ["liver", "gallbladder"]
GT_CAT_MAP = {"Liver": "liver", "Gallbladder": "gallbladder"}
TISSUE_OBJ_IDS = {"Liver": 1, "Gallbladder": 2}
OBJ_ID_TO_KEY = {1: "liver", 2: "gallbladder"}


# ---------------------------------------------------------------------------
# Split detection and GT keyframe identification
# ---------------------------------------------------------------------------

def detect_split_size(annotation_loader):
    """Get the auto-detected split size from the annotation loader."""
    return getattr(annotation_loader, '_split_size', 120)


def find_gt_keyframes(frame_files, split_size):
    """
    Identify GT keyframes within a snippet's frame list.

    GT keyframes are frame 0 of each split, i.e. where
    video_frame_num % split_size == 0.

    Returns:
        List of (local_idx, video_frame_num) tuples, sorted by local_idx.
    """
    keyframes = []
    for i, fpath in enumerate(frame_files):
        frame_num = int(fpath.stem.split("_")[1])
        if frame_num % split_size == 0:
            keyframes.append((i, frame_num))
    return keyframes


# ---------------------------------------------------------------------------
# Tissue segmentation: GT + tracker propagation
# ---------------------------------------------------------------------------

def process_tissue(
    tracker,
    frame_files,
    effective_frames,
    annotation_loader,
    gt_keyframes,
    min_area=500,
):
    """
    Process tissue (Liver, Gallbladder) using GT at split keyframes + propagation.

    1. Creates a tracker state for the snippet
    2. Adds GT masks at each GT keyframe (frame 0 of each split)
    3. Propagates bidirectionally through all frames
    4. Returns per-frame tissue results and area timeseries

    Args:
        tracker: Sam3TrackingPredictor instance
        frame_files: sorted frame file paths
        effective_frames: number of frames to process
        annotation_loader: COCOAnnotationLoader with GT masks
        gt_keyframes: list of (local_idx, video_frame_num) tuples
        min_area: minimum mask area to keep

    Returns:
        tissue_results: dict {frame_idx: {cat_key: mask_data_dict}}
        areas: dict {cat_key: list of (frame_idx, area) tuples}
    """
    if not gt_keyframes:
        print("    No GT keyframes found — skipping tissue processing")
        return {}, {}

    print(f"\n  Tissue processing: {len(gt_keyframes)} GT keyframes, "
          f"{effective_frames} total frames")

    # Load frames for tracker
    print(f"    Loading {effective_frames} frames for tracker...")
    images, video_height, video_width = _load_frames_for_tracker(
        frame_files, effective_frames, tracker.image_size,
    )

    # Create tracker state (no video_path — we inject images manually)
    tracker_state = tracker.init_state(
        video_height=video_height,
        video_width=video_width,
        num_frames=effective_frames,
    )
    tracker_state["images"] = images

    # Add GT masks at each keyframe
    prompts_added = 0
    for local_idx, frame_num in gt_keyframes:
        if local_idx >= effective_frames:
            continue
        gt_masks = annotation_loader.get_frame_masks_by_frame_num(frame_num)
        if gt_masks is None:
            print(f"    WARNING: No GT at frame {frame_num} (idx {local_idx})")
            continue

        for gt_cat, obj_id in TISSUE_OBJ_IDS.items():
            if gt_cat not in gt_masks:
                continue
            mask_np = gt_masks[gt_cat].astype(np.uint8)
            if mask_np.sum() == 0:
                continue
            mask_tensor = torch.from_numpy(mask_np).float()
            try:
                tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=local_idx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
                prompts_added += 1
            except Exception as e:
                print(f"    WARNING: Failed to add tissue mask at frame {local_idx}: {e}")

    if prompts_added == 0:
        print("    No tissue mask prompts added — skipping propagation")
        del tracker_state, images
        return {}, {}

    print(f"    Added {prompts_added} mask prompts across {len(gt_keyframes)} GT keyframes")

    # Consolidate and propagate
    tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

    tissue_results = {}  # frame_idx -> {cat_key: mask_data}
    areas = {"liver": [], "gallbladder": []}
    gt_indices = set(idx for idx, _ in gt_keyframes if idx < effective_frames)

    t0 = time.time()
    for reverse in [False, True]:
        for frame_idx, obj_ids, _low_res, video_res_masks, _scores in tracker.propagate_in_video(
            tracker_state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=reverse,
        ):
            if frame_idx >= effective_frames:
                continue

            if frame_idx not in tissue_results:
                tissue_results[frame_idx] = {}

            for i, oid in enumerate(obj_ids):
                cat_key = OBJ_ID_TO_KEY.get(int(oid))
                if cat_key is None:
                    continue

                mask_logits = video_res_masks[i]
                if isinstance(mask_logits, torch.Tensor):
                    mask_np = mask_logits.squeeze(0).cpu().numpy()
                else:
                    mask_np = mask_logits
                mask_uint8 = (mask_np > 0.0).astype(np.uint8)
                area = float(mask_uint8.sum())

                if area < min_area:
                    continue

                polygons = mask_to_coco_polygons(mask_uint8 * 255)
                if not polygons:
                    continue

                ys, xs = np.where(mask_uint8 > 0)
                bbox = [float(xs.min()), float(ys.min()),
                        float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                source = "ground_truth" if frame_idx in gt_indices else "sam3_propagated"
                tissue_results[frame_idx][cat_key] = {
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "source": source,
                }

    # Build area timeseries (sorted by frame index)
    for fidx in sorted(tissue_results.keys()):
        for cat_key in ["liver", "gallbladder"]:
            if cat_key in tissue_results[fidx]:
                areas[cat_key].append((fidx, tissue_results[fidx][cat_key]["area"]))

    elapsed = time.time() - t0
    filled = len(tissue_results)
    print(f"    Tissue propagation done: {filled}/{effective_frames} frames "
          f"filled in {elapsed:.1f}s")

    del tracker_state, images
    return tissue_results, areas


# ---------------------------------------------------------------------------
# Area degradation detection
# ---------------------------------------------------------------------------

def detect_area_degradation(areas_list, window=60):
    """
    Detect frames where tissue area drops anomalously.

    Uses exponential moving average (EMA) rolling statistics: if area drops
    below rolling_mean - 2*rolling_std, mark as degraded.

    Args:
        areas_list: list of (frame_idx, area) tuples, sorted by frame_idx
        window: smoothing window size

    Returns:
        List of (start_frame_idx, end_frame_idx) degraded regions.
    """
    if len(areas_list) < window:
        return []

    degraded_frames = []
    initial_area = np.mean([a for _, a in areas_list[:min(10, len(areas_list))]])
    rolling_mean = initial_area
    rolling_sq = initial_area ** 2
    alpha = 2.0 / (window + 1)

    for frame_idx, area in areas_list:
        rolling_mean = alpha * area + (1 - alpha) * rolling_mean
        rolling_sq = alpha * (area ** 2) + (1 - alpha) * rolling_sq
        rolling_std = max((rolling_sq - rolling_mean ** 2) ** 0.5, 1.0)
        if area < rolling_mean - 2 * rolling_std:
            degraded_frames.append(frame_idx)

    if not degraded_frames:
        return []

    # Group consecutive degraded frames into regions
    regions = []
    start = degraded_frames[0]
    prev = degraded_frames[0]
    for fidx in degraded_frames[1:]:
        if fidx - prev > 5:  # gap > 5 frames = new region
            regions.append((start, prev))
            start = fidx
        prev = fidx
    regions.append((start, prev))

    return regions


# ---------------------------------------------------------------------------
# Tissue backpropagation for degraded regions
# ---------------------------------------------------------------------------

def backpropagate_tissue(
    tracker,
    frame_files,
    effective_frames,
    annotation_loader,
    gt_keyframes,
    degraded_regions,
    tissue_results,
    min_area=500,
):
    """
    Backpropagate tissue masks from the next GT keyframe for degraded regions.

    For each degraded region, finds the next GT keyframe after the region,
    creates a new tracker state with just that GT, and propagates backward
    to fill the degraded frames.

    Args:
        degraded_regions: dict {cat_key: [(start_idx, end_idx), ...]}
        tissue_results: existing results (modified in-place)

    Returns:
        Number of frames recovered.
    """
    total_recovered = 0

    for cat_key, regions in degraded_regions.items():
        if not regions:
            continue
        gt_cat = "Liver" if cat_key == "liver" else "Gallbladder"
        obj_id = TISSUE_OBJ_IDS[gt_cat]

        for region_start, region_end in regions:
            # Find next GT keyframe after the degraded region
            next_gt = None
            for local_idx, frame_num in gt_keyframes:
                if local_idx > region_end and local_idx < effective_frames:
                    next_gt = (local_idx, frame_num)
                    break

            if next_gt is None:
                print(f"    No GT keyframe after degraded region "
                      f"[{region_start}-{region_end}] for {cat_key}")
                continue

            next_idx, next_frame_num = next_gt
            gt_masks = annotation_loader.get_frame_masks_by_frame_num(next_frame_num)
            if gt_masks is None or gt_cat not in gt_masks:
                continue

            print(f"    Backpropagating {cat_key} from GT frame {next_frame_num} "
                  f"(idx {next_idx}) to cover [{region_start}-{region_end}]")

            # Create a small tracker state for just this backpropagation
            num_frames_needed = next_idx - region_start + 1
            sub_frame_files = frame_files[region_start:next_idx + 1]

            images, vh, vw = _load_frames_for_tracker(
                sub_frame_files, num_frames_needed, tracker.image_size,
            )
            sub_state = tracker.init_state(
                video_height=vh, video_width=vw, num_frames=num_frames_needed,
            )
            sub_state["images"] = images

            # Add GT mask at the last frame of this sub-range
            mask_np = gt_masks[gt_cat].astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_np).float()
            try:
                tracker.add_new_mask(
                    inference_state=sub_state,
                    frame_idx=num_frames_needed - 1,  # GT is at end of sub-range
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
            except Exception as e:
                print(f"    WARNING: Failed to add backprop mask: {e}")
                del sub_state, images
                continue

            tracker.propagate_in_video_preflight(sub_state, run_mem_encoder=True)

            # Propagate backward (reverse=True)
            recovered = 0
            for frame_idx, obj_ids, _low, video_res_masks, _scores in tracker.propagate_in_video(
                sub_state,
                start_frame_idx=None,
                max_frame_num_to_track=None,
                reverse=True,
            ):
                actual_idx = region_start + frame_idx
                if actual_idx >= effective_frames:
                    continue

                for i, oid in enumerate(obj_ids):
                    if int(oid) != obj_id:
                        continue
                    mask_logits = video_res_masks[i]
                    if isinstance(mask_logits, torch.Tensor):
                        mask_np_out = mask_logits.squeeze(0).cpu().numpy()
                    else:
                        mask_np_out = mask_logits
                    mask_uint8 = (mask_np_out > 0.0).astype(np.uint8)
                    area = float(mask_uint8.sum())
                    if area < min_area:
                        continue

                    polygons = mask_to_coco_polygons(mask_uint8 * 255)
                    if not polygons:
                        continue

                    ys, xs = np.where(mask_uint8 > 0)
                    bbox = [float(xs.min()), float(ys.min()),
                            float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                    # Only overwrite if this frame was degraded
                    if actual_idx not in tissue_results:
                        tissue_results[actual_idx] = {}
                    tissue_results[actual_idx][cat_key] = {
                        "segmentation": polygons,
                        "area": area,
                        "bbox": bbox,
                        "source": "sam3_backpropagated",
                    }
                    recovered += 1

            del sub_state, images
            total_recovered += recovered
            print(f"    Recovered {recovered} frames via backpropagation")

    return total_recovered


# ---------------------------------------------------------------------------
# Tool detection: text prompt per split
# ---------------------------------------------------------------------------

def process_tools(
    predictor,
    frames_dir,
    frame_files,
    gt_keyframes,
    split_size,
    effective_frames,
    min_area=5000,
    reprompt_interval=20,
):
    """
    Detect tools using text prompt "tool" per split segment.

    For each split:
    1. Try text prompt "tool" at the split's start frame
    2. Check if 1-2 tool masks detected
    3. If not, try at +20, +40, etc. within the split
    4. Once tools found, propagate forward through the split
    5. Collect tool masks for all frames in the split

    Args:
        predictor: Sam3VideoPredictor instance
        frames_dir: Path to frames_left/ directory
        frame_files: sorted frame file paths
        gt_keyframes: list of (local_idx, video_frame_num) tuples
        split_size: frames per split
        effective_frames: number of frames to process
        min_area: minimum mask area in pixels
        reprompt_interval: frames between re-prompt attempts

    Returns:
        tool_results: dict {frame_idx: mask_data_list}
    """
    print(f"\n  Tool detection: {len(gt_keyframes)} splits, "
          f"reprompt_interval={reprompt_interval}")

    tool_results = {}  # frame_idx -> [mask_data_dict, ...]

    # Start one session for the whole snippet (loads frames once)
    session = predictor.start_session(resource_path=str(frames_dir))
    sid = session["session_id"]

    # Build split boundaries from GT keyframes
    split_boundaries = []
    for i, (local_idx, frame_num) in enumerate(gt_keyframes):
        if local_idx >= effective_frames:
            continue
        # End of this split = start of next split or end of snippet
        if i + 1 < len(gt_keyframes):
            end_idx = min(gt_keyframes[i + 1][0], effective_frames)
        else:
            end_idx = effective_frames
        split_boundaries.append((local_idx, end_idx))

    # Also handle frames before the first GT keyframe
    if gt_keyframes and gt_keyframes[0][0] > 0:
        split_boundaries.insert(0, (0, gt_keyframes[0][0]))

    for split_start, split_end in split_boundaries:
        split_len = split_end - split_start
        if split_len <= 0:
            continue

        print(f"\n    Split [{split_start}-{split_end}) ({split_len} frames)")

        # Try prompting at different offsets until tools found
        tools_found = False
        for offset in range(0, split_len, reprompt_interval):
            prompt_idx = split_start + offset
            if prompt_idx >= effective_frames:
                break

            # add_prompt resets state internally, so each attempt is fresh
            response = predictor.add_prompt(
                session_id=sid, frame_idx=prompt_idx, text="tool"
            )
            outputs = response["outputs"]

            # Check how many tool masks were detected
            binary_masks = outputs.get("out_binary_masks", [])
            n_tools = 0
            if len(binary_masks) > 0:
                for mask in binary_masks:
                    if isinstance(mask, np.ndarray):
                        area = float(mask.sum())
                    else:
                        area = float(mask.sum().item())
                    if area >= min_area:
                        n_tools += 1

            print(f"      Prompt at frame {prompt_idx}: {n_tools} tools detected")

            if n_tools >= 1:
                # Tools found! Propagate forward through this split
                frames_to_track = split_end - prompt_idx
                for response in predictor.propagate_in_video(
                    session_id=sid,
                    propagation_direction="forward",
                    start_frame_idx=prompt_idx,
                    max_frame_num_to_track=frames_to_track,
                ):
                    fidx = response["frame_index"]
                    if fidx >= effective_frames:
                        continue
                    result = _convert_video_output(
                        response["outputs"],
                        frame_files[fidx],
                        "tool",
                        min_area,
                    )
                    tool_masks = result["masks"].get("tool", [])
                    if tool_masks:
                        tool_results[fidx] = tool_masks

                # Also propagate backward to cover frames before prompt_idx in this split
                if prompt_idx > split_start:
                    # Re-prompt at same frame (add_prompt resets state)
                    predictor.add_prompt(
                        session_id=sid, frame_idx=prompt_idx, text="tool"
                    )
                    frames_to_backtrack = prompt_idx - split_start + 1
                    for response in predictor.propagate_in_video(
                        session_id=sid,
                        propagation_direction="backward",
                        start_frame_idx=prompt_idx,
                        max_frame_num_to_track=frames_to_backtrack,
                    ):
                        fidx = response["frame_index"]
                        if fidx < split_start or fidx >= effective_frames:
                            continue
                        if fidx in tool_results:
                            continue  # don't overwrite forward results
                        result = _convert_video_output(
                            response["outputs"],
                            frame_files[fidx],
                            "tool",
                            min_area,
                        )
                        tool_masks = result["masks"].get("tool", [])
                        if tool_masks:
                            tool_results[fidx] = tool_masks

                tools_found = True
                frames_with = sum(1 for fidx in range(split_start, split_end) if fidx in tool_results)
                print(f"      Tools tracked for {frames_with}/{split_len} frames in split")
                break

        if not tools_found:
            print(f"      WARNING: No tools found in split [{split_start}-{split_end})")

    predictor.close_session(session_id=sid)

    total_frames_with_tools = len(tool_results)
    print(f"\n  Tool detection done: tools on {total_frames_with_tools}/{effective_frames} frames")

    return tool_results


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def render_overlays(
    all_results,
    frame_files,
    output_dir,
    effective_frames,
):
    """Render overlay images for all frames."""
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    all_prompts = ["tool", "liver", "gallbladder"]
    print(f"\n  Rendering overlays ({effective_frames} frames)...")

    for i in range(effective_frames):
        fpath = frame_files[i]
        result = all_results[i]
        _, overlay, _ = _render_overlay_from_results(
            fpath, result, all_prompts, annotation_loader=None
        )
        out_path = overlays_dir / f"{fpath.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)

    return overlays_dir


# ---------------------------------------------------------------------------
# GT tissue subtraction (remove tool pixels on tissue)
# ---------------------------------------------------------------------------

def subtract_tissue_from_tools(all_results, effective_frames, erode_px=3):
    """
    Remove tool mask pixels that overlap with tissue masks.

    For each frame, if both tool and tissue masks exist, subtract tissue
    from tools (tissue takes priority).
    """
    cleaned = 0
    for i in range(effective_frames):
        result = all_results[i]
        tool_masks = result["masks"].get("tool", [])
        if not tool_masks:
            continue

        h, w = result["height"], result["width"]
        if h == 0 or w == 0:
            continue

        # Build combined tissue mask
        tissue = np.zeros((h, w), dtype=np.uint8)
        for cat_key in ["liver", "gallbladder"]:
            for md in result["masks"].get(cat_key, []):
                for poly in md["segmentation"]:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(tissue, [pts], 1)

        if tissue.sum() == 0:
            continue

        # Optional erosion
        if erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
            )
            tissue = cv2.erode(tissue, kernel, iterations=1)

        # Subtract tissue from each tool mask
        new_tools = []
        modified = False
        for md in tool_masks:
            tool = np.zeros((h, w), dtype=np.uint8)
            for poly in md["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(tool, [pts], 1)

            cleaned_mask = tool & (~tissue)
            new_area = float(np.sum(cleaned_mask))

            if new_area > 0:
                old_area = md["area"]
                md["area"] = new_area
                md["segmentation"] = mask_to_coco_polygons(cleaned_mask * 255)
                ys, xs = np.where(cleaned_mask > 0)
                md["bbox"] = [float(xs.min()), float(ys.min()),
                              float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                new_tools.append(md)
                if new_area != old_area:
                    modified = True
            else:
                modified = True

        result["masks"]["tool"] = new_tools
        if modified:
            cleaned += 1

    return cleaned


# ---------------------------------------------------------------------------
# Main per-snippet processing
# ---------------------------------------------------------------------------

def process_snippet(
    predictor,
    snippet_dir,
    episode,
    annotation_loader,
    split_size,
    output_dir,
    min_area=5000,
    tissue_min_area=500,
    test_frames=None,
    reprompt_interval=20,
    area_window=60,
    gt_erode_px=3,
):
    """
    Process a single snippet with split-aware tissue + tool segmentation.

    Steps:
        1. Identify GT keyframes (frame 0 of each split)
        2. Tissue: GT at keyframes → tracker propagation
        3. Area monitoring → backpropagation for degraded regions
        4. Tools: text prompt per split (re-prompt until found)
        5. Tissue subtraction from tools
        6. Overlay rendering + video stitching
    """
    frames_dir = snippet_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print(f"  WARNING: No frames found in {frames_dir}")
        return []

    total_frames = len(frame_files)
    effective_frames = min(test_frames, total_frames) if test_frames else total_frames
    frame_files_eff = frame_files[:effective_frames]

    snippet_id = snippet_dir.name
    print(f"\n{'=' * 60}")
    print(f"  {episode} / {snippet_id}: {effective_frames} frames "
          f"(of {total_frames}), split_size={split_size}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # --- 1. Identify GT keyframes ---
    gt_keyframes = find_gt_keyframes(frame_files_eff, split_size)
    print(f"  GT keyframes: {len(gt_keyframes)}")
    for idx, fnum in gt_keyframes[:5]:
        print(f"    idx={idx}, frame={fnum}")
    if len(gt_keyframes) > 5:
        print(f"    ... and {len(gt_keyframes) - 5} more")

    # --- 2. Tissue: GT + tracker propagation ---
    tracker = predictor.model.tracker
    tissue_results, areas = process_tissue(
        tracker, frame_files_eff, effective_frames,
        annotation_loader, gt_keyframes, min_area=tissue_min_area,
    )

    # --- 3. Area monitoring + backpropagation ---
    degraded_regions = {}
    for cat_key in ["liver", "gallbladder"]:
        regions = detect_area_degradation(areas.get(cat_key, []), window=area_window)
        if regions:
            print(f"  Area degradation detected for {cat_key}: {len(regions)} region(s)")
            for s, e in regions:
                print(f"    [{s}-{e}]")
            degraded_regions[cat_key] = regions

    if degraded_regions:
        recovered = backpropagate_tissue(
            tracker, frame_files_eff, effective_frames,
            annotation_loader, gt_keyframes, degraded_regions,
            tissue_results, min_area=tissue_min_area,
        )
        print(f"  Backpropagation recovered {recovered} frames total")

    # --- 4. Tools: text prompt per split ---
    tool_results = process_tools(
        predictor, frames_dir, frame_files_eff, gt_keyframes,
        split_size, effective_frames, min_area=min_area,
        reprompt_interval=reprompt_interval,
    )

    # --- 5. Assemble per-frame results ---
    all_results = []
    for i in range(effective_frames):
        fpath = frame_files_eff[i]
        # Get frame dimensions from the first available mask or read image
        h, w = 0, 0
        if i in tissue_results:
            for cat_key, md in tissue_results[i].items():
                # Get dimensions from polygons (estimate from bbox)
                # Actually need image dimensions — read from first frame
                pass
        if i in tool_results:
            for md in tool_results[i]:
                if md.get("segmentation"):
                    pass

        # Read actual dimensions if not known yet
        if h == 0 or w == 0:
            img = cv2.imread(str(fpath))
            if img is not None:
                h, w = img.shape[:2]

        masks = {}

        # Tool masks
        if i in tool_results:
            masks["tool"] = tool_results[i]
        else:
            masks["tool"] = []

        # Tissue masks
        for cat_key in ["liver", "gallbladder"]:
            if i in tissue_results and cat_key in tissue_results[i]:
                masks[cat_key] = [tissue_results[i][cat_key]]
            else:
                masks[cat_key] = []

        all_results.append({
            "frame": fpath.stem,
            "height": h,
            "width": w,
            "masks": masks,
        })

    # --- 6. Tissue subtraction from tools ---
    cleaned = subtract_tissue_from_tools(all_results, effective_frames, erode_px=gt_erode_px)
    if cleaned > 0:
        print(f"\n  Tissue subtraction cleaned {cleaned} tool frames")

    # --- 7. Render overlays ---
    overlays_dir = render_overlays(all_results, frame_files_eff, output_dir, effective_frames)

    # --- 8. Stitch video ---
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{snippet_id}_overlay.mp4"
    _stitch_snippet_video(overlays_dir, video_path, fps=60)

    # --- 9. Save results JSON ---
    results_path = output_dir / f"{snippet_id}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "episode": episode,
            "snippet_id": snippet_id,
            "split_size": split_size,
            "gt_keyframes": gt_keyframes,
            "mode": "split_video",
            "num_frames": len(all_results),
            "frames": all_results,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n  {snippet_id} done: {effective_frames} frames in {total_time:.1f}s "
          f"({total_time/max(effective_frames,1):.1f}s/frame)")

    for cat in ["tool", "liver", "gallbladder"]:
        total_masks = sum(len(r["masks"].get(cat, [])) for r in all_results)
        frames_with = sum(1 for r in all_results if len(r["masks"].get(cat, [])) > 0)
        if total_masks == 0:
            continue
        gt_count = sum(1 for r in all_results
                       for m in r["masks"].get(cat, [])
                       if m.get("source") == "ground_truth")
        prop_count = sum(1 for r in all_results
                         for m in r["masks"].get(cat, [])
                         if m.get("source") in ("sam3_propagated", "sam3_backpropagated"))
        source_info = ""
        if gt_count or prop_count:
            source_info = f" (GT={gt_count}, propagated={prop_count})"
        color_name = {(255, 128, 0): "blue", (0, 0, 255): "red",
                      (0, 255, 0): "green"}.get(
            CATEGORY_COLORS.get(cat, DEFAULT_COLOR), "?"
        )
        print(f"    {cat} ({color_name}): {total_masks} masks across "
              f"{frames_with}/{effective_frames} frames{source_info}")

    return all_results


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split-aware segmentation for E_3 and F_3"
    )
    parser.add_argument(
        "--segments-dir", required=True,
        help="Path to Segments directory",
    )
    parser.add_argument(
        "--episode", required=True,
        help="Episode name (E_3 or F_3)",
    )
    parser.add_argument(
        "--snippet", type=int, default=None,
        help="Specific snippet number (default: all)",
    )
    parser.add_argument(
        "--tissue-seg-dir", required=True,
        help="Path to tissue_segmentation directory with GT annotations",
    )
    parser.add_argument(
        "--output-dir", default="outputs/split_video",
        help="Output directory (default: outputs/split_video)",
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Process first N frames only (test mode)",
    )
    parser.add_argument(
        "--min-area", type=int, default=5000,
        help="Minimum tool mask area in pixels (default: 5000)",
    )
    parser.add_argument(
        "--tissue-min-area", type=int, default=500,
        help="Minimum tissue mask area in pixels (default: 500)",
    )
    parser.add_argument(
        "--tool-reprompt-interval", type=int, default=20,
        help="Frames between tool re-prompt attempts (default: 20)",
    )
    parser.add_argument(
        "--area-window", type=int, default=60,
        help="Rolling window for area degradation detection (default: 60)",
    )
    parser.add_argument(
        "--gt-erode-px", type=int, default=3,
        help="Erosion buffer for tissue subtraction (default: 3, 0=disable)",
    )
    parser.add_argument(
        "--lora-checkpoint", default=None,
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip snippets that already have results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    tissue_seg_dir = Path(args.tissue_seg_dir)

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)
    if not tissue_seg_dir.exists():
        print(f"ERROR: Tissue segmentation directory not found: {tissue_seg_dir}")
        sys.exit(1)

    # Collect snippets
    ep_dir = segments_dir / args.episode
    if not ep_dir.exists():
        print(f"ERROR: Episode not found: {ep_dir}")
        sys.exit(1)

    snippet_list = []
    if args.snippet is not None:
        snip = ep_dir / f"snippet_{args.snippet:03d}"
        if snip.exists():
            snippet_list.append(snip)
        else:
            print(f"ERROR: Snippet not found: {snip}")
            sys.exit(1)
    else:
        snippet_list = sorted(
            [s for s in ep_dir.glob("snippet_*") if s.is_dir()]
        )

    if not snippet_list:
        print("ERROR: No snippets found")
        sys.exit(1)

    # Count frames
    total_frames = 0
    for snip_dir in snippet_list:
        frames_dir = snip_dir / "frames_left"
        if frames_dir.exists():
            n = len(list(frames_dir.glob("frame_*.webp")))
            if not n:
                n = len(list(frames_dir.glob("frame_*.png")))
            if not n:
                n = len(list(frames_dir.glob("frame_*.jpg")))
            total_frames += n

    # Header
    print("=" * 60)
    print("Split-Aware Segmentation Pipeline")
    print("=" * 60)
    print(f"Episode: {args.episode}")
    print(f"Snippets: {len(snippet_list)}")
    print(f"Total frames: {total_frames}")
    print(f"Output: {output_dir}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")

    # Load GT annotations
    print(f"\nLoading GT annotations from {tissue_seg_dir}...")
    annotation_loader = _load_episode_annotations(tissue_seg_dir, args.episode)
    if annotation_loader is None:
        print(f"ERROR: No annotations found for {args.episode}")
        sys.exit(1)

    split_size = detect_split_size(annotation_loader)
    print(f"Detected split size: {split_size}")

    # Load video model
    print()
    predictor = load_video_model(lora_checkpoint=args.lora_checkpoint)

    # Process snippets
    t_total = time.time()
    for snip_dir in snippet_list:
        snippet_id = snip_dir.name
        snip_output = output_dir / args.episode / snippet_id

        # Resume: skip if results exist
        results_path = snip_output / f"{snippet_id}_results.json"
        if args.resume and results_path.exists():
            print(f"\n  Skipping {snippet_id} (results exist, --resume)")
            continue

        process_snippet(
            predictor=predictor,
            snippet_dir=snip_dir,
            episode=args.episode,
            annotation_loader=annotation_loader,
            split_size=split_size,
            output_dir=snip_output,
            min_area=args.min_area,
            tissue_min_area=args.tissue_min_area,
            test_frames=args.test,
            reprompt_interval=args.tool_reprompt_interval,
            area_window=args.area_window,
            gt_erode_px=args.gt_erode_px,
        )

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done! {len(snippet_list)} snippets in {total_time:.1f}s")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
