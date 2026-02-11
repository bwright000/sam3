"""
Generate tool segmentation masks using SAM3 Video Mode.

Runs separate video sessions per prompt (default: "tool") so each
category gets its own tracked masklets and distinct overlay color:
  - tool  → blue
  - liver (GT) → red
  - gallbladder (GT) → green

Pipeline per prompt:
  1. Text prompt on keyframe → initial masklets
  2. Video propagation tracks masklets through all frames
  3. GT tissue subtraction (liver/gallbladder) removes overlap
  4. Combined overlay rendering + video stitching

This script is GPU-optimized (A100 recommended). The existing image-mode
pipeline (generate_tool_masks.py) is unchanged and still usable.

Usage:
    # Process a single snippet (default prompt: tool)
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 --snippet 1

    # Custom prompts
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 --prompts tool gauze

    # With GT tissue subtraction
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 --snippet 1 \\
        --tissue-seg-dir "F:\\2026 vibes\\MPHY Project\\annotated_dataset\\tissue_segmentation"

    # With LoRA fine-tuned weights (MedSAM3 or custom Cholec80)
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 \\
        --lora-checkpoint path/to/lora_weights.pt

    # Test mode (first N frames only)
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 --snippet 1 --test 10
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse functions from existing image-mode pipeline
from scripts.generate_tool_masks import (
    mask_to_coco_polygons,
    _load_episode_annotations,
    _subtract_gt_tissue,
    _render_overlay_from_results,
    _draw_legend,
    _stitch_snippet_video,
    CATEGORY_COLORS,
    DEFAULT_COLOR,
)

# SAM3 video predictor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# For loading frames (supports .webp, .png, .jpg etc. via PIL)
from sam3.model.io_utils import _load_img_as_tensor

# Color palette: "tool" already blue from generate_tool_masks.py

# Tissue categories for GT and propagation
TISSUE_CATEGORIES = ["liver", "gallbladder"]
GT_CAT_MAP = {"Liver": "liver", "Gallbladder": "gallbladder"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_video_model(lora_checkpoint=None):
    """
    Build SAM3 video predictor with optional LoRA weights.

    The predictor uses build_sam3_video_model() internally, which creates:
      - detector (Sam3ImageOnVideoMultiGPU): text-guided per-frame detection
      - tracker (Sam3TrackerPredictor): temporal memory tracking
    Wrapped in Sam3VideoInferenceWithInstanceInteractivity for temporal disambiguation.

    Args:
        lora_checkpoint: Optional path to LoRA weights (MedSAM3 or custom Cholec80)
    """
    print("Loading SAM3 video model...")
    t0 = time.time()

    predictor = Sam3VideoPredictor(
        apply_temporal_disambiguation=True,
    )

    if lora_checkpoint:
        _apply_lora_weights(predictor.model, lora_checkpoint)

    dt = time.time() - t0
    print(f"Model loaded in {dt:.1f}s")

    # Print device info
    device = next(predictor.model.parameters()).device
    print(f"Model device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3)
        print(f"GPU: {props.name} ({mem_gb:.1f} GB)")

    return predictor


def _apply_lora_weights(model, checkpoint_path):
    """
    Apply LoRA weights to the video model's detector backbone.

    Supports:
      - MedSAM3 LoRA weights (from lal-Joey/MedSAM3_v1)
      - Custom Cholec80 LoRA weights (from SAM3_LoRA training)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"WARNING: LoRA checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading LoRA weights from {checkpoint_path}...")
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if "lora_state_dict" in state_dict:
        lora_weights = state_dict["lora_state_dict"]
    elif "model" in state_dict:
        lora_weights = state_dict["model"]
    else:
        lora_weights = state_dict

    # Try to load LoRA weights
    try:
        # Try SAM3_LoRA style: inject LoRA layers then load weights
        from sam3_lora.lora.lora_utils import inject_lora_into_model, load_lora_state_dict
        inject_lora_into_model(model)
        load_lora_state_dict(model, lora_weights)
        print(f"  LoRA weights applied (SAM3_LoRA format)")
    except ImportError:
        # Fallback: direct state dict loading
        missing, unexpected = model.load_state_dict(lora_weights, strict=False)
        loaded = len(lora_weights) - len(unexpected)
        print(f"  LoRA weights loaded directly: {loaded} tensors")
        if missing:
            print(f"  Missing keys: {len(missing)}")


# ---------------------------------------------------------------------------
# Output format conversion
# ---------------------------------------------------------------------------

def _convert_video_output(outputs, frame_path, prompt, min_area):
    """
    Convert SAM3 video predictor output to our standard per-frame format.

    Video predictor yields per-frame:
      - out_obj_ids: (N,) int64, persistent object IDs across frames
      - out_probs: (N,) float, detection confidence scores
      - out_binary_masks: (N, H, W) bool, at original resolution
      - out_boxes_xywh: (N, 4) float, normalized bounding boxes

    We convert to the same format used by the image pipeline:
      {"frame": str, "height": int, "width": int,
       "masks": {prompt: [{"segmentation": [...], "area": float, "score": float, ...}]}}
    """
    binary_masks = outputs["out_binary_masks"]

    if len(binary_masks) == 0:
        return {
            "frame": frame_path.stem,
            "height": 0,
            "width": 0,
            "masks": {prompt: []},
        }

    h, w = binary_masks.shape[1:]
    masks_list = []

    for i, obj_id in enumerate(outputs["out_obj_ids"]):
        binary_mask = binary_masks[i]  # (H, W) bool numpy
        area = float(binary_mask.sum())
        if area < min_area:
            continue

        score = float(outputs["out_probs"][i])
        polygons = mask_to_coco_polygons(binary_mask.astype(np.uint8) * 255)
        if not polygons:
            continue

        bbox = outputs["out_boxes_xywh"][i].tolist()

        masks_list.append({
            "segmentation": polygons,
            "area": area,
            "score": score,
            "obj_id": int(obj_id),
            "bbox": bbox,
        })

    return {
        "frame": frame_path.stem,
        "height": h,
        "width": w,
        "masks": {prompt: masks_list},
    }


# ---------------------------------------------------------------------------
# Tissue segmentation: GT loading + SAM3 propagation
# ---------------------------------------------------------------------------

def _add_gt_tissue_to_results(all_results, frame_files, annotation_loader):
    """
    Add GT tissue masks (liver, gallbladder) to per-frame results.

    For each frame, checks if GT annotations exist. If so, converts them to
    COCO polygon format and adds to the frame's masks dict with
    source="ground_truth". Returns set of frame indices that have GT.

    Args:
        all_results: list of per-frame result dicts (modified in-place)
        frame_files: sorted list of frame file paths
        annotation_loader: COCOAnnotationLoader with GT tissue masks

    Returns:
        gt_frame_indices: set of frame indices (into all_results) that have GT
    """
    gt_frame_indices = set()

    for i, (result, fpath) in enumerate(zip(all_results, frame_files)):
        frame_num = int(fpath.stem.split("_")[1])
        gt_masks = annotation_loader.get_frame_masks_by_frame_num(frame_num)

        if gt_masks is None:
            continue

        h, w = result["height"], result["width"]
        if h == 0 or w == 0:
            # Frame dimensions not yet set (no tool masks found) — read from image
            img = cv2.imread(str(fpath))
            if img is not None:
                h, w = img.shape[:2]
                result["height"] = h
                result["width"] = w

        has_tissue = False
        for gt_cat, color_key in GT_CAT_MAP.items():
            if gt_cat not in gt_masks:
                continue
            mask = gt_masks[gt_cat].astype(np.uint8)
            area = float(mask.sum())
            if area == 0:
                continue

            polygons = mask_to_coco_polygons(mask * 255)
            if not polygons:
                continue

            ys, xs = np.where(mask > 0)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())]

            result["masks"][color_key] = [{
                "segmentation": polygons,
                "area": area,
                "bbox": bbox,
                "source": "ground_truth",
            }]
            has_tissue = True

        if has_tissue:
            gt_frame_indices.add(i)

    return gt_frame_indices


def _load_frames_for_tracker(frame_files, num_frames, image_size):
    """
    Load video frames from the script's frame_files list for the tracker.

    Uses the same file list as the rest of the pipeline to guarantee that
    frame index N in the tracker corresponds to frame_files[N]. Supports
    all image formats (.webp, .png, .jpg, etc.) via PIL.

    Normalizes with mean=0.5, std=0.5 to match the tracker backbone's
    expected normalization (sam2_utils.py defaults).

    Returns (images_tensor, video_height, video_width).
    """
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    video_height = video_width = None
    for n, fpath in enumerate(frame_files[:num_frames]):
        images[n], video_height, video_width = _load_img_as_tensor(
            str(fpath), image_size
        )
    # Move to GPU if available
    if torch.cuda.is_available():
        images = images.cuda()
    # Normalize (mean=0.5, std=0.5 — matching tracker backbone expectations)
    img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
    img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
    if torch.cuda.is_available():
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def propagate_tissue_masks(
    predictor,
    frames_dir,
    frame_files,
    effective_frames,
    annotation_loader,
    gt_frame_indices,
    all_results,
    min_area=500,
):
    """
    Use SAM3 tracker propagation to fill tissue masks for frames without GT.

    Creates a standalone tracker state, loads frames from the script's own
    frame_files list (guaranteeing index alignment), adds mask prompts at
    all GT frames, then propagates bidirectionally.

    Note: Uses the tracker (SAM2-style) directly rather than the full
    Sam3VideoInference VG pipeline for two reasons:
      1. The VG-level inference state does not expose the tracker's
         obj_id_to_idx mapping needed by add_new_mask.
      2. The tracker's built-in frame loader (sam2_utils.py) only supports
         .jpg/.jpeg, but our frames may be .webp or .png.

    Args:
        predictor: Sam3VideoPredictor instance
        frames_dir: Path to frames_left/ directory
        frame_files: sorted list of frame file paths
        effective_frames: number of frames to process
        annotation_loader: COCOAnnotationLoader with GT tissue masks
        gt_frame_indices: set of frame indices that already have GT
        all_results: list of per-frame result dicts (modified in-place)
        min_area: minimum mask area to keep

    Returns:
        Number of frames filled via propagation.
    """
    # Identify frames needing propagation
    need_propagation = set(range(effective_frames)) - gt_frame_indices
    if not need_propagation:
        return 0

    print(f"\n  Tissue propagation: {len(need_propagation)} frames need SAM3 fill "
          f"({len(gt_frame_indices)} have GT)")

    # Category name -> SAM3 object ID mapping
    tissue_cats = {"Liver": 1, "Gallbladder": 2}

    # Use the tracker directly with its own inference state.
    # The VG-level inference state (from Sam3VideoInference.init_state) does not
    # contain 'obj_id_to_idx'; only tracker-level states do.
    tracker = predictor.model.tracker

    # Load frames from the SAME frame_files list the script uses, so that
    # frame_idx=N in the tracker is guaranteed to match frame_files[N].
    # We bypass tracker.init_state(video_path=...) because its loader
    # (sam2_utils.py) only supports .jpg/.jpeg — our frames may be .webp.
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

    # Add mask prompts at every GT frame
    prompts_added = 0
    for fidx in sorted(gt_frame_indices):
        fpath = frame_files[fidx]
        frame_num = int(fpath.stem.split("_")[1])
        gt_masks = annotation_loader.get_frame_masks_by_frame_num(frame_num)
        if gt_masks is None:
            continue

        for gt_cat, obj_id in tissue_cats.items():
            if gt_cat not in gt_masks:
                continue
            mask_np = gt_masks[gt_cat].astype(np.uint8)
            if mask_np.sum() == 0:
                continue

            mask_tensor = torch.from_numpy(mask_np).float()

            try:
                tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=fidx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
                prompts_added += 1
            except Exception as e:
                print(f"    WARNING: Failed to add tissue mask at frame {fidx}: {e}")

    if prompts_added == 0:
        print("    No tissue mask prompts added — skipping propagation")
        del tracker_state, images
        return 0

    print(f"    Added {prompts_added} mask prompts across "
          f"{len(gt_frame_indices)} GT frames")

    # Reverse lookup: obj_id -> lowercase category key
    obj_id_to_key = {1: "liver", 2: "gallbladder"}

    # Consolidate mask inputs before propagation
    tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

    # Propagate through all frames (forward then backward)
    filled = 0
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
            # Only collect output for frames WITHOUT GT
            if frame_idx not in need_propagation:
                continue

            frame_filled = False
            for i, oid in enumerate(obj_ids):
                cat_key = obj_id_to_key.get(int(oid))
                if cat_key is None:
                    continue

                # video_res_masks shape: (num_objects, 1, H, W) — logits
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

                all_results[frame_idx]["masks"][cat_key] = [{
                    "segmentation": polygons,
                    "area": area,
                    "bbox": bbox,
                    "source": "sam3_propagated",
                }]
                frame_filled = True

            if frame_filled:
                filled += 1

    elapsed = time.time() - t0
    del tracker_state, images
    print(f"    Tissue propagation done: {filled}/{len(need_propagation)} "
          f"frames filled in {elapsed:.1f}s")

    return filled


# ---------------------------------------------------------------------------
# Keyframe selection and session helpers
# ---------------------------------------------------------------------------

def _select_keyframes(effective_frames, num_keyframes):
    """Select evenly-spaced keyframe indices across the video."""
    if num_keyframes <= 1:
        return [0]
    # Cap keyframes to not exceed frame count
    num_keyframes = min(num_keyframes, effective_frames)
    step = effective_frames / num_keyframes
    return [int(i * step) for i in range(num_keyframes)]


def _run_prompt_session(
    predictor, frames_dir, frame_files, effective_frames,
    prompt, keyframe_idx, direction, min_area, test_frames,
):
    """
    Run one video session: text prompt on keyframe_idx, propagate, collect results.

    Returns:
        results: dict {frame_idx: result_dict} with per-frame mask data
        confidences: dict {frame_idx: float} with avg confidence per frame
    """
    session = predictor.start_session(resource_path=str(frames_dir))
    sid = session["session_id"]

    predictor.add_prompt(session_id=sid, frame_idx=keyframe_idx, text=prompt)

    results = {}
    confidences = {}
    pass_label = ""
    frames_seen = 0

    t0 = time.time()
    for response in predictor.propagate_in_video(
        session_id=sid,
        propagation_direction=direction,
        start_frame_idx=None,
        max_frame_num_to_track=test_frames,
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        if frame_idx >= effective_frames:
            continue

        result = _convert_video_output(
            outputs, frame_files[frame_idx], prompt, min_area
        )
        results[frame_idx] = result

        # Compute avg confidence from out_probs
        probs = outputs.get("out_probs")
        if probs is not None and len(probs) > 0:
            avg_conf = float(probs.mean())
        else:
            avg_conf = 0.0
        confidences[frame_idx] = avg_conf

        frames_seen += 1
        # Detect which pass we're on (forward: ascending, backward: descending)
        if frames_seen == 1:
            pass_label = "fwd"
        elif frame_idx < keyframe_idx or (frames_seen > effective_frames):
            pass_label = "bwd"

        n_masks = len(result["masks"].get(prompt, []))
        elapsed = time.time() - t0
        print(f"    [{pass_label}] frame {frame_idx:4d} | "
              f"{prompt}={n_masks} conf={avg_conf:.2f} | "
              f"{elapsed:.1f}s")

    predictor.close_session(session_id=sid)
    elapsed = time.time() - t0
    print(f"  Session done (keyframe={keyframe_idx}, {len(results)} frames, {elapsed:.1f}s)")
    return results, confidences


def _detect_and_recover_dropouts(
    predictor, frames_dir, frame_files, effective_frames,
    prompt, all_results, confidences, direction, min_area, test_frames,
    conf_threshold, dropout_window,
):
    """
    Scan for confidence dropouts and re-prompt to recover lost tracking.

    Finds stretches of dropout_window+ consecutive frames where avg confidence
    < conf_threshold, then re-prompts on the first frame after the dropout.

    Returns number of recovered dropout windows.
    """
    if conf_threshold <= 0 or not confidences:
        return 0

    # Build sorted frame confidence list
    sorted_frames = sorted(confidences.keys())
    if not sorted_frames:
        return 0

    # Detect dropout windows
    dropout_regions = []
    current_dropout_start = None
    consecutive_low = 0

    for fidx in sorted_frames:
        conf = confidences.get(fidx, 0.0)
        if conf < conf_threshold:
            if current_dropout_start is None:
                current_dropout_start = fidx
            consecutive_low += 1
        else:
            if consecutive_low >= dropout_window:
                dropout_regions.append((current_dropout_start, fidx - 1))
            current_dropout_start = None
            consecutive_low = 0

    # Handle dropout at end of video
    if consecutive_low >= dropout_window and current_dropout_start is not None:
        dropout_regions.append((current_dropout_start, sorted_frames[-1]))

    if not dropout_regions:
        return 0

    print(f"\n  Detected {len(dropout_regions)} dropout region(s) for '{prompt}':")
    recovered = 0

    for start_frame, end_frame in dropout_regions:
        # Find recovery frame: first frame AFTER dropout with masks
        recovery_frame = None
        for fidx in range(end_frame + 1, effective_frames):
            if confidences.get(fidx, 0) >= conf_threshold:
                recovery_frame = fidx
                break

        # Fallback: use midpoint of dropout as recovery frame
        if recovery_frame is None:
            recovery_frame = (start_frame + end_frame) // 2

        print(f"    Dropout frames {start_frame}-{end_frame} "
              f"(conf<{conf_threshold}), re-prompting at frame {recovery_frame}")

        # Run recovery session
        rec_results, rec_confidences = _run_prompt_session(
            predictor, frames_dir, frame_files, effective_frames,
            prompt, recovery_frame, direction, min_area, test_frames,
        )

        # Merge recovered results only for frames in/near the dropout region
        merge_start = max(0, start_frame - dropout_window)
        merge_end = min(effective_frames, end_frame + dropout_window + 1)
        merged = 0
        for fidx in range(merge_start, merge_end):
            rec_result = rec_results.get(fidx)
            if rec_result is None:
                continue
            rec_conf = rec_confidences.get(fidx, 0)
            old_conf = confidences.get(fidx, 0)
            # Keep recovery result if it has better confidence
            if rec_conf > old_conf:
                all_results[fidx]["masks"][prompt] = rec_result["masks"].get(prompt, [])
                if rec_result["height"]:
                    all_results[fidx]["height"] = rec_result["height"]
                    all_results[fidx]["width"] = rec_result["width"]
                confidences[fidx] = rec_conf
                merged += 1

        print(f"    Recovered {merged} frames from dropout")
        recovered += 1

    return recovered


# ---------------------------------------------------------------------------
# Per-snippet video processing
# ---------------------------------------------------------------------------

def process_snippet_video(
    predictor,
    snippet_dir,
    episode,
    snippet_id,
    prompts,
    output_dir,
    annotation_loader=None,
    min_area=5000,
    test_frames=None,
    num_keyframes=1,
    forward_only=False,
    confidence_threshold=0.3,
    dropout_window=5,
    gt_erode_px=3,
):
    """
    Process a snippet using SAM3 video mode with separate prompts.

    Runs a separate video session per prompt (e.g. "tool") so each
    category gets its own tracked masklets and distinct overlay color.

    Supports multi-keyframe prompting, bidirectional propagation, and
    confidence-triggered re-prompting for tracking dropout recovery.

    Then: GT tissue subtraction + overlay rendering + video stitching.
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
    if test_frames:
        effective_frames = min(test_frames, total_frames)
    else:
        effective_frames = total_frames

    direction = "forward" if forward_only else "both"
    keyframes = _select_keyframes(effective_frames, num_keyframes)

    print(f"\n  Processing {snippet_id}: {effective_frames} frames"
          f" (of {total_frames}), prompts={prompts}")
    print(f"  Direction: {direction}, keyframes: {keyframes}")
    if confidence_threshold > 0:
        print(f"  Confidence recovery: threshold={confidence_threshold}, "
              f"window={dropout_window}")

    # Create output directories
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Pre-initialise per-frame results with empty masks and confidence tracking
    all_results = []
    for fpath in frame_files[:effective_frames]:
        all_results.append({
            "frame": fpath.stem,
            "height": 0,
            "width": 0,
            "masks": {p: [] for p in prompts},
            "avg_confidence": {p: 0.0 for p in prompts},
        })

    # --- Run a separate video session per prompt ---
    for prompt in prompts:
        print(f"\n  --- Prompt: '{prompt}' ---")

        # --- Multi-keyframe: run one session per keyframe, collect all results ---
        all_kf_results = {}  # keyframe_idx -> {frame_idx: result}
        all_kf_confidences = {}  # keyframe_idx -> {frame_idx: float}

        for kf_idx in keyframes:
            print(f"\n  Keyframe {kf_idx} (of {keyframes}):")
            kf_results, kf_confidences = _run_prompt_session(
                predictor, frames_dir, frame_files, effective_frames,
                prompt, kf_idx, direction, min_area, test_frames,
            )
            all_kf_results[kf_idx] = kf_results
            all_kf_confidences[kf_idx] = kf_confidences

        # --- Merge keyframe results: keep highest-confidence per frame ---
        merged_confidences = {}  # frame_idx -> best avg confidence
        for fidx in range(effective_frames):
            best_conf = -1.0
            best_kf = keyframes[0]
            for kf_idx in keyframes:
                conf = all_kf_confidences.get(kf_idx, {}).get(fidx, 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_kf = kf_idx

            best_result = all_kf_results.get(best_kf, {}).get(fidx)
            if best_result is not None:
                all_results[fidx]["masks"][prompt] = best_result["masks"].get(prompt, [])
                if best_result["height"]:
                    all_results[fidx]["height"] = best_result["height"]
                    all_results[fidx]["width"] = best_result["width"]
            all_results[fidx]["avg_confidence"][prompt] = max(best_conf, 0.0)
            merged_confidences[fidx] = max(best_conf, 0.0)

        if len(keyframes) > 1:
            avg_merged = np.mean(list(merged_confidences.values())) if merged_confidences else 0
            print(f"\n  Merged {len(keyframes)} keyframe passes "
                  f"(avg confidence: {avg_merged:.3f})")

        # --- Confidence-triggered re-prompting for dropout recovery ---
        if confidence_threshold > 0:
            recovered = _detect_and_recover_dropouts(
                predictor, frames_dir, frame_files, effective_frames,
                prompt, all_results, merged_confidences, direction,
                min_area, test_frames, confidence_threshold, dropout_window,
            )
            if recovered > 0:
                # Update avg_confidence in results from merged_confidences
                for fidx in range(effective_frames):
                    all_results[fidx]["avg_confidence"][prompt] = merged_confidences.get(fidx, 0.0)

        prompt_time = time.time() - t0
        print(f"\n  '{prompt}' done ({prompt_time:.1f}s)")

    inference_time = time.time() - t0
    print(f"\n  Video inference done: {len(all_results)} frames, "
          f"{len(prompts)} prompts in {inference_time:.1f}s")

    # Collect frame files for the frames we actually processed
    processed_frame_files = frame_files[:len(all_results)]

    # --- GT tissue subtraction ---
    if annotation_loader:
        print(f"\n  GT tissue subtraction...")
        cleaned = _subtract_gt_tissue(
            all_results, processed_frame_files, annotation_loader, prompts,
            erode_px=gt_erode_px,
        )
        print(f"  GT subtraction cleaned {cleaned} frames")

    # --- Tissue segmentation: GT + SAM3 propagation ---
    if annotation_loader:
        print(f"\n  Adding tissue segmentation (GT + propagation)...")
        gt_indices = _add_gt_tissue_to_results(
            all_results, processed_frame_files, annotation_loader
        )
        print(f"  GT tissue found for {len(gt_indices)}/{len(all_results)} frames")

        # Propagate tissue through frames without GT
        if len(gt_indices) < len(all_results) and len(gt_indices) > 0:
            propagate_tissue_masks(
                predictor, frames_dir, frame_files, effective_frames,
                annotation_loader, gt_indices, all_results,
                min_area=500,
            )
        elif len(gt_indices) == 0:
            print("  WARNING: No GT tissue annotations found — tissue masks unavailable")

    # --- Render overlays ---
    # Include tissue categories in overlay rendering
    all_prompts = prompts + [tc for tc in TISSUE_CATEGORIES
                             if any(tc in r["masks"] for r in all_results)]
    print(f"  Rendering overlays (categories: {all_prompts})...")
    for i, (result, fpath) in enumerate(zip(all_results, processed_frame_files)):
        _, overlay, _ = _render_overlay_from_results(
            fpath, result, all_prompts, annotation_loader=None
        )
        out_path = overlays_dir / f"{fpath.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)

    # --- Stitch video ---
    video_path = output_dir / f"{snippet_id}_overlay.mp4"
    _stitch_snippet_video(overlays_dir, video_path, fps=60)

    # --- Save results JSON ---
    results_path = output_dir / f"{snippet_id}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "episode": episode,
            "snippet_id": snippet_id,
            "prompts": prompts,
            "tissue_categories": TISSUE_CATEGORIES,
            "mode": "video",
            "num_frames": len(all_results),
            "inference_time_s": round(inference_time, 1),
            "frames": all_results,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n  {snippet_id} done: {len(all_results)} frames in {total_time:.1f}s "
          f"({total_time/max(len(all_results),1):.1f}s/frame)")
    for cat in prompts + TISSUE_CATEGORIES:
        total_masks = sum(len(r["masks"].get(cat, [])) for r in all_results)
        frames_with = sum(1 for r in all_results if len(r["masks"].get(cat, [])) > 0)
        if total_masks == 0:
            continue
        # Count GT vs propagated for tissue
        gt_count = sum(1 for r in all_results
                       for m in r["masks"].get(cat, [])
                       if m.get("source") == "ground_truth")
        prop_count = sum(1 for r in all_results
                         for m in r["masks"].get(cat, [])
                         if m.get("source") == "sam3_propagated")
        source_info = ""
        if gt_count or prop_count:
            source_info = f" (GT={gt_count}, propagated={prop_count})"
        color_name = {(255, 128, 0): "blue", (0, 255, 255): "yellow",
                      (0, 0, 255): "red", (0, 255, 0): "green"}.get(
            CATEGORY_COLORS.get(cat, DEFAULT_COLOR), "?"
        )
        print(f"    {cat} ({color_name}): {total_masks} masks across "
              f"{frames_with}/{len(all_results)} frames{source_info}")

    return all_results


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tool masks using SAM3 video mode"
    )

    parser.add_argument(
        "--segments-dir", required=True,
        help="Path to Segments directory with episode/snippet structure",
    )
    parser.add_argument(
        "--episode", required=True,
        help="Episode name (e.g., C_1, E_3, F_3)",
    )
    parser.add_argument(
        "--snippet", type=int, default=None,
        help="Specific snippet number (default: all snippets)",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=["tool"],
        help="Text prompts for detection (default: tool)",
    )
    parser.add_argument(
        "--tissue-seg-dir", default=None,
        help="Path to tissue_segmentation directory with GT annotations",
    )
    parser.add_argument(
        "--min-area", type=int, default=5000,
        help="Minimum mask area in pixels (default: 5000)",
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Process first N frames only (test mode)",
    )
    parser.add_argument(
        "--lora-checkpoint", default=None,
        help="Path to LoRA weights (MedSAM3 or custom Cholec80)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/segments_video",
        help="Output directory (default: outputs/segments_video)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip snippets that already have results JSON (resume interrupted runs)",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Force device (default: auto-detect CUDA > CPU)",
    )

    # --- Propagation & recovery options ---
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Forward-only propagation (default: bidirectional forward+backward)",
    )
    parser.add_argument(
        "--num-keyframes", type=int, default=1,
        help="Number of evenly-spaced keyframes to prompt (default: 1 = frame 0 only)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.3,
        help="Min avg confidence before triggering dropout recovery (default: 0.3, 0=disable)",
    )
    parser.add_argument(
        "--dropout-window", type=int, default=5,
        help="Consecutive low-confidence frames before re-prompting (default: 5)",
    )
    parser.add_argument(
        "--gt-erode-px", type=int, default=3,
        help="Erosion buffer (pixels) for GT tissue mask before subtraction "
             "(default: 3, 0=disable)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)

    # --- Collect snippets ---
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

    # Count frames (check webp, then png, then jpg)
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

    # --- Header ---
    print("=" * 60)
    print("SAM3 Video Segmentation Pipeline")
    print("=" * 60)
    print(f"Episode: {args.episode}")
    print(f"Snippets: {len(snippet_list)}")
    print(f"Total frames: {total_frames}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {output_dir}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")
    if args.resume:
        print(f"Resume: enabled (skipping completed snippets)")
    if args.device:
        print(f"Device: {args.device} (forced)")
    if args.lora_checkpoint:
        print(f"LoRA checkpoint: {args.lora_checkpoint}")
    direction_label = "forward-only" if args.forward_only else "bidirectional"
    print(f"Propagation: {direction_label}")
    if args.num_keyframes > 1:
        print(f"Keyframes: {args.num_keyframes}")
    if args.confidence_threshold > 0:
        print(f"Confidence recovery: threshold={args.confidence_threshold}, "
              f"window={args.dropout_window}")
    if args.gt_erode_px > 0:
        print(f"GT erosion buffer: {args.gt_erode_px}px")

    # --- Load GT annotations ---
    ann_loader = None
    if args.tissue_seg_dir:
        tissue_seg_dir = Path(args.tissue_seg_dir)
        if tissue_seg_dir.exists():
            print(f"\nLoading GT annotations from {tissue_seg_dir}...")
            ann_loader = _load_episode_annotations(tissue_seg_dir, args.episode)
        else:
            print(f"WARNING: tissue-seg-dir not found: {tissue_seg_dir}")

    # --- Load video model ---
    print()
    predictor = load_video_model(lora_checkpoint=args.lora_checkpoint)

    # --- Force device if requested ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available, using CPU")
    elif args.device == "cpu" and torch.cuda.is_available():
        print("Forcing CPU mode (--device cpu)")
        predictor.model.cpu()

    # --- Process snippets ---
    t_total = time.time()
    for snip_dir in snippet_list:
        snippet_id = snip_dir.name

        snip_output = output_dir / args.episode / snippet_id

        # Resume: skip if results already exist
        results_path = snip_output / f"{snippet_id}_results.json"
        if args.resume and results_path.exists():
            print(f"\n  Skipping {snippet_id} (results exist, --resume)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Episode: {args.episode} / {snippet_id}")
        print(f"{'=' * 60}")

        process_snippet_video(
            predictor=predictor,
            snippet_dir=snip_dir,
            episode=args.episode,
            snippet_id=snippet_id,
            prompts=args.prompts,
            output_dir=snip_output,
            annotation_loader=ann_loader,
            min_area=args.min_area,
            test_frames=args.test,
            num_keyframes=args.num_keyframes,
            forward_only=args.forward_only,
            confidence_threshold=args.confidence_threshold,
            dropout_window=args.dropout_window,
            gt_erode_px=args.gt_erode_px,
        )

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done! {len(snippet_list)} snippets in {total_time:.1f}s")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
