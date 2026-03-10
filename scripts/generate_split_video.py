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
    # Using per-snippet annotations (from update_snippets.py, preferred)
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 --snippet 1

    # Using episode-level annotations (fallback)
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 \\
        --tissue-seg-dir "path/to/tissue_segmentation"

    # Test mode (first N frames)
    python scripts/generate_split_video.py \\
        --segments-dir data/Segments --episode F_3 --snippet 1 --test 100
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
from scripts.check_annotations import COCOAnnotationLoader
from scripts.generate_tool_masks_video import (
    load_video_model,
    _convert_video_output,
    _load_frames_for_tracker,
)

from sam3.model.sam3_video_predictor import Sam3VideoPredictor



# ---------------------------------------------------------------------------
# Per-snippet annotation loading
# ---------------------------------------------------------------------------

def load_snippet_annotations(snippet_dir, split_size=None):
    """
    Load annotations from snippet_annotations.json (created by update_snippets.py).

    Returns a COCOAnnotationLoader or None if the file doesn't exist.
    Much more memory-efficient than loading full episode annotations,
    since each snippet only contains its own frame range.

    split_size: if provided, used instead of auto-detection (critical for C_1
    where broken filenames cause incorrect auto-detection).
    """
    ann_path = snippet_dir / "snippet_annotations.json"
    if not ann_path.exists():
        return None

    loader = COCOAnnotationLoader(str(ann_path), str(snippet_dir), split_size=split_size)
    loader.load()
    return loader


# Tissue categories (from shared config)
from scripts.shared_config import GT_CAT_MAP, TISSUE_OBJ_IDS, OBJ_ID_TO_KEY
TISSUE_CATEGORIES = list(GT_CAT_MAP.values())

# Tool detection: text prompts tried in order of specificity.
# SAM3's text-guided detection responds better to descriptive terms
# than abstract ones. We cascade until detections are found.
TOOL_PROMPTS = ["surgical instrument", "grasper", "tool"]


# ---------------------------------------------------------------------------
# Tool gap detection and recovery
# ---------------------------------------------------------------------------

def find_tool_gaps(tool_results, split_start, split_end, min_gap_length=3):
    """
    Find contiguous frame regions within a split where tools are absent
    but should be present (bounded by frames that DO have tools).

    Only returns "internal" gaps — regions with tool-containing frames both
    before and after. This avoids flagging the start/end of a split where
    tools may genuinely be off-screen.

    Args:
        tool_results: dict {frame_idx: [mask_data_dict, ...]}
        split_start: first frame index of the split (inclusive)
        split_end: last frame index of the split (exclusive)
        min_gap_length: ignore gaps shorter than this (transient losses)

    Returns:
        List of (gap_start, gap_end) tuples (inclusive on both ends).
    """
    # Build a boolean array: True = has tool masks
    has_tools = []
    for fidx in range(split_start, split_end):
        has_tools.append(fidx in tool_results and len(tool_results[fidx]) > 0)

    if not any(has_tools):
        return []  # no tools at all in split — nothing to recover

    # Find contiguous empty regions
    gaps = []
    gap_start = None
    for i, has in enumerate(has_tools):
        fidx = split_start + i
        if not has:
            if gap_start is None:
                gap_start = fidx
        else:
            if gap_start is not None:
                gap_end = fidx - 1  # inclusive
                gaps.append((gap_start, gap_end))
                gap_start = None
    # Don't close a gap that extends to the end of the split (trailing gap)

    # Filter: only keep internal gaps (tools present both before AND after)
    # Trailing and leading gaps are excluded by construction:
    # - leading gaps: first frame has no tools → gap_start=split_start, but no
    #   tool frame before it → excluded
    # - trailing gaps: not added (loop above doesn't close them)
    # But also explicitly filter by min_gap_length
    filtered = []
    for gs, ge in gaps:
        if (ge - gs + 1) >= min_gap_length:
            filtered.append((gs, ge))

    return filtered


def recover_tool_gaps(
    predictor,
    sid,
    frame_files,
    tool_results,
    gaps,
    split_start,
    split_end,
    effective_frames,
    min_area,
    reprompt_interval=10,
):
    """
    Re-prompt the detector within each gap to recover lost tools.

    For each gap:
      1. Reset session (clean tracker state, keep loaded frames)
      2. Try text prompts at intervals within the gap
      3. Propagate from the first successful prompt
      4. Merge: only fill frames that don't already have tool masks

    Args:
        predictor: Sam3VideoPredictor instance
        sid: session ID (session already started, frames loaded)
        frame_files: sorted frame file paths
        tool_results: dict {frame_idx: [mask_dicts]} — modified in-place
        gaps: list of (gap_start, gap_end) from find_tool_gaps
        split_start, split_end: split boundaries
        effective_frames: total frames being processed
        min_area: minimum mask area in pixels
        reprompt_interval: frames between re-prompt attempts within the gap

    Returns:
        Number of frames recovered.
    """
    total_recovered = 0

    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start + 1

        # Reset session for a clean recovery pass
        predictor.reset_session(session_id=sid)

        # Try prompting at intervals within the gap
        recovery_found = False
        for offset in range(0, gap_len, reprompt_interval):
            recovery_frame = gap_start + offset
            if recovery_frame >= effective_frames:
                break

            # Multi-prompt cascade (same order as initial detection)
            best_prompt = None
            for prompt_text in TOOL_PROMPTS:
                response = predictor.add_prompt(
                    session_id=sid, frame_idx=recovery_frame, text=prompt_text
                )
                outputs = response["outputs"]

                binary_masks = outputs.get("out_binary_masks", [])
                n_valid = 0
                n_raw = len(binary_masks) if hasattr(binary_masks, '__len__') else 0
                if n_raw > 0:
                    for mask in binary_masks:
                        if isinstance(mask, np.ndarray):
                            area = float(mask.sum())
                        else:
                            area = float(mask.sum().item())
                        if area >= min_area:
                            n_valid += 1

                if n_valid >= 1:
                    best_prompt = prompt_text
                    break

            if best_prompt is None:
                continue  # tool genuinely absent at this frame, try next offset

            # Tools found — propagate through the gap (with some buffer)
            buffer = 5
            max_track = max(recovery_frame - gap_start, gap_end - recovery_frame) + buffer
            recovered_in_gap = 0

            for prop_response in predictor.propagate_in_video(
                session_id=sid,
                propagation_direction="both",
                start_frame_idx=recovery_frame,
                max_frame_num_to_track=max_track,
            ):
                fidx = prop_response["frame_index"]
                # Only fill within the gap boundaries (with buffer for propagation)
                if fidx < split_start or fidx >= split_end or fidx >= effective_frames:
                    continue

                # Only fill frames that don't already have tool masks
                if fidx in tool_results and len(tool_results[fidx]) > 0:
                    continue

                result = _convert_video_output(
                    prop_response["outputs"],
                    frame_files[fidx],
                    "tool",
                    min_area,
                )
                tool_masks = result["masks"].get("tool", [])
                if tool_masks:
                    # Mark as recovered
                    for md in tool_masks:
                        md["source"] = "gap_recovery"
                    tool_results[fidx] = tool_masks
                    recovered_in_gap += 1

            total_recovered += recovered_in_gap
            print(f"        Gap [{gap_start}-{gap_end}]: re-prompted at frame "
                  f"{recovery_frame} (\"{best_prompt}\"), recovered {recovered_in_gap} frames")
            recovery_found = True
            break  # done with this gap

        if not recovery_found:
            print(f"        Gap [{gap_start}-{gap_end}]: no tools found during recovery")

    return total_recovered


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

def _gt_proximity_distance(frame_idx, gt_keyframes, reverse):
    """
    Compute distance from a frame to its source GT keyframe.

    For the forward pass: source is the nearest preceding GT (frame <= frame_idx).
    For the backward pass: source is the nearest following GT (frame >= frame_idx).
    Returns the absolute distance in frames.
    """
    if reverse:
        # Backward pass originates from the nearest following GT
        for local_idx, _ in gt_keyframes:
            if local_idx >= frame_idx:
                return abs(frame_idx - local_idx)
    else:
        # Forward pass originates from the nearest preceding GT
        for local_idx, _ in reversed(gt_keyframes):
            if local_idx <= frame_idx:
                return abs(frame_idx - local_idx)
    return 9999  # no GT found in the expected direction


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
    4. Merges forward/backward passes by keeping the higher-confidence mask
    5. Returns per-frame tissue results and forward-pass confidence scores

    Args:
        tracker: Sam3TrackingPredictor instance
        frame_files: sorted frame file paths
        effective_frames: number of frames to process
        annotation_loader: COCOAnnotationLoader with GT masks
        gt_keyframes: list of (local_idx, video_frame_num) tuples
        min_area: minimum mask area to keep

    Returns:
        tissue_results: dict {frame_idx: {cat_key: mask_data_dict}}
        fwd_scores: dict {cat_key: list of (frame_idx, obj_score, mean_logit) tuples}
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

    gt_indices = set(idx for idx, _ in gt_keyframes if idx < effective_frames)

    # Collect forward and backward results separately with confidence scores
    fwd_results = {}  # frame_idx -> {cat_key: mask_data_dict (includes obj_score, mean_logit)}
    bwd_results = {}
    fwd_scores = {"liver": [], "gallbladder": []}  # for degradation monitoring

    t0 = time.time()
    for reverse in [False, True]:
        target = bwd_results if reverse else fwd_results
        for frame_idx, obj_ids, _low_res, video_res_masks, obj_scores in tracker.propagate_in_video(
            tracker_state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=reverse,
        ):
            if frame_idx >= effective_frames:
                continue

            if frame_idx not in target:
                target[frame_idx] = {}

            for i, oid in enumerate(obj_ids):
                cat_key = OBJ_ID_TO_KEY.get(int(oid))
                if cat_key is None:
                    continue

                # Extract obj_score (model confidence for this object)
                score_val = obj_scores[i]
                if isinstance(score_val, torch.Tensor):
                    obj_score = float(score_val.squeeze().cpu().item())
                else:
                    obj_score = float(score_val)

                mask_logits = video_res_masks[i]
                if isinstance(mask_logits, torch.Tensor):
                    mask_np = mask_logits.squeeze(0).cpu().numpy()
                else:
                    mask_np = mask_logits

                # Compute mean logit over positive region (pixel-level confidence)
                positive_mask = mask_np > 0.0
                mean_logit = float(mask_np[positive_mask].mean()) if positive_mask.any() else 0.0

                mask_uint8 = positive_mask.astype(np.uint8)
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
                target[frame_idx][cat_key] = {
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "source": source,
                    "obj_score": obj_score,
                    "mean_logit": mean_logit,
                }

                # Collect forward-pass confidence for degradation monitoring
                if not reverse:
                    fwd_scores[cat_key].append((frame_idx, obj_score, mean_logit))

    # Merge forward and backward: proximity-weighted obj_score with mean_logit tiebreaker.
    # Closer to the source GT keyframe = less accumulated drift = more geometrically accurate.
    tissue_results = {}
    all_frame_idxs = set(fwd_results) | set(bwd_results)
    for fidx in all_frame_idxs:
        tissue_results[fidx] = {}
        for cat_key in ["liver", "gallbladder"]:
            fwd = fwd_results.get(fidx, {}).get(cat_key)
            bwd = bwd_results.get(fidx, {}).get(cat_key)
            if fwd and bwd:
                d_fwd = _gt_proximity_distance(fidx, gt_keyframes, reverse=False)
                d_bwd = _gt_proximity_distance(fidx, gt_keyframes, reverse=True)
                fwd_score = fwd["obj_score"] * (1.0 / (1.0 + d_fwd))
                bwd_score = bwd["obj_score"] * (1.0 / (1.0 + d_bwd))
                if abs(fwd_score - bwd_score) < 1.0:
                    # Scores are close — tiebreak with mean_logit (pixel confidence)
                    pick = fwd if fwd.get("mean_logit", 0) >= bwd.get("mean_logit", 0) else bwd
                else:
                    pick = fwd if fwd_score >= bwd_score else bwd
                tissue_results[fidx][cat_key] = pick
            elif fwd:
                tissue_results[fidx][cat_key] = fwd
            elif bwd:
                tissue_results[fidx][cat_key] = bwd

    elapsed = time.time() - t0
    filled = len(tissue_results)
    print(f"    Tissue propagation done: {filled}/{effective_frames} frames "
          f"filled in {elapsed:.1f}s")

    del tracker_state, images
    return tissue_results, fwd_scores


# ---------------------------------------------------------------------------
# Degradation detection (model confidence-based)
# ---------------------------------------------------------------------------

def detect_degradation(scores_list, obj_score_threshold=0.0, mean_logit_threshold=1.0):
    """
    Detect frames where tracker confidence drops, indicating degraded masks.

    Uses the model's own confidence signals (camera-pose-independent):
    - obj_score: tracker's "is this object appearing?" logit. < 0 means lost.
    - mean_logit: average pixel confidence over the mask. Low = fuzzy/uncertain.

    Args:
        scores_list: list of (frame_idx, obj_score, mean_logit) tuples,
                     sorted by frame_idx (from forward pass)
        obj_score_threshold: flag if obj_score below this (default: 0.0)
        mean_logit_threshold: flag if mean_logit below this (default: 1.0)

    Returns:
        List of (start_frame_idx, end_frame_idx) degraded regions.
    """
    if not scores_list:
        return []

    scores_list = sorted(scores_list, key=lambda x: x[0])

    degraded_frames = []
    for frame_idx, obj_score, mean_logit in scores_list:
        # Model thinks object has disappeared
        if obj_score < obj_score_threshold:
            degraded_frames.append(frame_idx)
        # Mask exists but model is very uncertain about it
        elif mean_logit < mean_logit_threshold:
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
            for frame_idx, obj_ids, _low, video_res_masks, obj_scores in tracker.propagate_in_video(
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

                    # Extract obj_score
                    score_val = obj_scores[i]
                    if isinstance(score_val, torch.Tensor):
                        obj_score = float(score_val.squeeze().cpu().item())
                    else:
                        obj_score = float(score_val)

                    mask_logits = video_res_masks[i]
                    if isinstance(mask_logits, torch.Tensor):
                        mask_np_out = mask_logits.squeeze(0).cpu().numpy()
                    else:
                        mask_np_out = mask_logits

                    # Compute mean logit confidence
                    positive_mask = mask_np_out > 0.0
                    mean_logit = float(mask_np_out[positive_mask].mean()) if positive_mask.any() else 0.0

                    mask_uint8 = positive_mask.astype(np.uint8)
                    area = float(mask_uint8.sum())
                    if area < min_area:
                        continue

                    polygons = mask_to_coco_polygons(mask_uint8 * 255)
                    if not polygons:
                        continue

                    ys, xs = np.where(mask_uint8 > 0)
                    bbox = [float(xs.min()), float(ys.min()),
                            float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                    if actual_idx not in tissue_results:
                        tissue_results[actual_idx] = {}
                    tissue_results[actual_idx][cat_key] = {
                        "segmentation": polygons,
                        "area": area,
                        "bbox": bbox,
                        "source": "sam3_backpropagated",
                        "obj_score": obj_score,
                        "mean_logit": mean_logit,
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
    min_area=2000,
    reprompt_interval=20,
):
    """
    Detect tools using multi-prompt cascade per split segment, with gap recovery.

    For each split:
    1. Try text prompts in order of specificity (TOOL_PROMPTS) at the split's start
    2. Accept the first prompt that produces valid masks above min_area
    3. If no prompt works, re-prompt at +20, +40, etc. within the split
    4. Once tools found, propagate bidirectionally through the split
    5. Detect gaps where tools were lost mid-split
    6. Re-prompt within gaps to recover lost tools

    Args:
        predictor: Sam3VideoPredictor instance
        frames_dir: Path to frames_left/ directory
        frame_files: sorted frame file paths
        gt_keyframes: list of (local_idx, video_frame_num) tuples
        split_size: frames per split
        effective_frames: number of frames to process
        min_area: minimum mask area in pixels (default 2000)
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

            # Multi-prompt cascade: try each text prompt in order of specificity
            n_tools = 0
            best_prompt = None
            for prompt_text in TOOL_PROMPTS:
                # add_prompt resets state internally, so each attempt is fresh
                response = predictor.add_prompt(
                    session_id=sid, frame_idx=prompt_idx, text=prompt_text
                )
                outputs = response["outputs"]

                # Check how many tool masks were detected above min_area
                binary_masks = outputs.get("out_binary_masks", [])
                n_valid = 0
                n_raw = len(binary_masks) if hasattr(binary_masks, '__len__') else 0
                if n_raw > 0:
                    for mask in binary_masks:
                        if isinstance(mask, np.ndarray):
                            area = float(mask.sum())
                        else:
                            area = float(mask.sum().item())
                        if area >= min_area:
                            n_valid += 1

                if n_valid >= 1:
                    n_tools = n_valid
                    best_prompt = prompt_text
                    print(f"      Prompt at frame {prompt_idx} (\"{prompt_text}\"): "
                          f"{n_raw} raw, {n_tools} kept (min_area={min_area})")
                    break  # use first successful prompt

            if best_prompt is None:
                print(f"      Prompt at frame {prompt_idx}: 0 tools detected")

            if n_tools >= 1:
                # Tools found! Propagate bidirectionally through this split.
                # Using "both" keeps the tracker's memory bank from the forward pass
                # when it runs backward, producing better backward masks.
                # Tight bound: covers both directions without overshooting into adjacent splits.
                max_track = max(prompt_idx - split_start, split_end - prompt_idx)
                for response in predictor.propagate_in_video(
                    session_id=sid,
                    propagation_direction="both",
                    start_frame_idx=prompt_idx,
                    max_frame_num_to_track=max_track,
                ):
                    fidx = response["frame_index"]
                    if fidx < split_start or fidx >= split_end or fidx >= effective_frames:
                        continue
                    result = _convert_video_output(
                        response["outputs"],
                        frame_files[fidx],
                        "tool",
                        min_area,
                    )
                    tool_masks = result["masks"].get("tool", [])
                    if tool_masks:
                        for md in tool_masks:
                            md["source"] = "initial_detection"
                        tool_results[fidx] = tool_masks

                tools_found = True
                frames_with = sum(1 for fidx in range(split_start, split_end) if fidx in tool_results)
                print(f"      Tools tracked for {frames_with}/{split_len} frames in split")

                # --- Gap detection and recovery ---
                gaps = find_tool_gaps(tool_results, split_start, split_end)
                if gaps:
                    print(f"      Tool gaps detected: {len(gaps)} region(s)")
                    for gs, ge in gaps:
                        print(f"        [{gs}-{ge}] ({ge - gs + 1} frames)")

                    recovered = recover_tool_gaps(
                        predictor, sid, frame_files, tool_results,
                        gaps, split_start, split_end, effective_frames,
                        min_area, reprompt_interval=min(reprompt_interval, 10),
                    )
                    if recovered:
                        frames_with = sum(
                            1 for f in range(split_start, split_end) if f in tool_results
                        )
                        print(f"      After recovery: {frames_with}/{split_len} frames "
                              f"({recovered} recovered)")

                break

        if not tools_found:
            print(f"      WARNING: No tools found in split [{split_start}-{split_end})")

    predictor.close_session(session_id=sid)

    total_frames_with_tools = len(tool_results)
    initial_count = sum(
        1 for masks in tool_results.values()
        for m in masks if m.get("source") == "initial_detection"
    )
    recovery_count = sum(
        1 for masks in tool_results.values()
        for m in masks if m.get("source") == "gap_recovery"
    )
    print(f"\n  Tool detection done: tools on {total_frames_with_tools}/{effective_frames} frames")
    if recovery_count:
        print(f"    Initial detections: {initial_count} masks, "
              f"gap recoveries: {recovery_count} masks")

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
    # Share the detector's backbone so the tracker can compute image features
    # on demand. The tracker is built with backbone=None (model_builder.py:448)
    # and relies on the VG pipeline to provide cached features. When used
    # standalone, we need it to compute features itself via forward_image().
    tracker.backbone = predictor.model.detector.backbone
    tissue_results, fwd_scores = process_tissue(
        tracker, frame_files_eff, effective_frames,
        annotation_loader, gt_keyframes, min_area=tissue_min_area,
    )

    # --- 3. Confidence monitoring + backpropagation ---
    degraded_regions = {}
    for cat_key in ["liver", "gallbladder"]:
        regions = detect_degradation(fwd_scores.get(cat_key, []))
        if regions:
            print(f"  Tracking degradation detected for {cat_key}: {len(regions)} region(s)")
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
    # Read frame dimensions once (all frames in a snippet share the same resolution)
    first_img = cv2.imread(str(frame_files_eff[0]))
    snippet_h, snippet_w = first_img.shape[:2]
    del first_img

    all_results = []
    for i in range(effective_frames):
        fpath = frame_files_eff[i]
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
            "height": snippet_h,
            "width": snippet_w,
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
        "--tissue-seg-dir", default=None,
        help="Path to tissue_segmentation directory with GT annotations. "
             "Optional if snippets have snippet_annotations.json (from update_snippets.py)",
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
        "--min-area", type=int, default=2000,
        help="Minimum tool mask area in pixels (default: 2000)",
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

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)

    tissue_seg_dir = Path(args.tissue_seg_dir) if args.tissue_seg_dir else None
    if tissue_seg_dir and not tissue_seg_dir.exists():
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

    # Load snippet metadata for split_size (auto-detection is unreliable for C_1)
    snippets_json_path = ep_dir / f"{args.episode}_snippets.json"
    metadata_split_size = None
    if snippets_json_path.exists():
        with open(snippets_json_path) as f:
            snippets_meta = json.load(f)
        for s in snippets_meta:
            if "split_size" in s:
                metadata_split_size = s["split_size"]
                break
        if metadata_split_size:
            print(f"Split size from metadata: {metadata_split_size}")

    # Check annotation strategy: per-snippet vs episode-level
    has_snippet_anns = any(
        (s / "snippet_annotations.json").exists() for s in snippet_list
    )
    use_snippet_anns = has_snippet_anns  # prefer per-snippet when available

    if use_snippet_anns:
        print(f"\nUsing per-snippet annotations (snippet_annotations.json)")
    elif tissue_seg_dir:
        print(f"\nUsing episode-level annotations from {tissue_seg_dir}")
    else:
        print(f"ERROR: No snippet_annotations.json found and --tissue-seg-dir not provided")
        sys.exit(1)

    # Load episode-level annotations as fallback (only if needed)
    episode_loader = None
    episode_split_size = None
    if tissue_seg_dir:
        print(f"Loading episode GT annotations from {tissue_seg_dir}...")
        episode_loader = _load_episode_annotations(tissue_seg_dir, args.episode)
        if episode_loader:
            episode_split_size = detect_split_size(episode_loader)
            print(f"Detected split size: {episode_split_size}")

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

        # Load annotations for this snippet
        annotation_loader = None
        split_size = None

        if use_snippet_anns:
            annotation_loader = load_snippet_annotations(snip_dir, split_size=metadata_split_size)
            if annotation_loader:
                split_size = metadata_split_size or detect_split_size(annotation_loader)
                print(f"\n  Loaded snippet annotations: "
                      f"{len(annotation_loader.images)} images, "
                      f"split_size={split_size}")

        # Fall back to episode-level if snippet annotations unavailable
        if annotation_loader is None:
            if episode_loader is None and tissue_seg_dir:
                print(f"\n  Loading episode annotations (fallback)...")
                episode_loader = _load_episode_annotations(tissue_seg_dir, args.episode)
                if episode_loader:
                    episode_split_size = detect_split_size(episode_loader)
            annotation_loader = episode_loader
            split_size = episode_split_size

        if annotation_loader is None:
            print(f"\n  ERROR: No annotations available for {snippet_id}, skipping")
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
            gt_erode_px=args.gt_erode_px,
        )

        # Free snippet-level loader after processing (memory efficiency)
        if use_snippet_anns and annotation_loader is not episode_loader:
            del annotation_loader

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done! {len(snippet_list)} snippets in {total_time:.1f}s")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
