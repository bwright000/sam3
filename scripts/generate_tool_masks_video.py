"""
Generate tool/cloth segmentation masks using SAM3 Video Mode.

Uses SAM3's video predictor API with temporal memory tracking:
  1. Text prompt ("surgical tool and cloth") on keyframe → initial masklets
  2. Video propagation tracks masklets through all frames
  3. GT tissue subtraction (liver/gallbladder) removes overlap
  4. Combined overlay rendering + video stitching

This script is GPU-optimized (A100 recommended). The existing image-mode
pipeline (generate_tool_masks.py) is unchanged and still usable.

Usage:
    # Process a single snippet
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1 --snippet 1

    # Process all snippets for an episode
    python scripts/generate_tool_masks_video.py \\
        --segments-dir data/Segments --episode C_1

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
        mem_gb = props.total_mem / (1024 ** 3)
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
# Per-snippet video processing
# ---------------------------------------------------------------------------

def process_snippet_video(
    predictor,
    snippet_dir,
    episode,
    snippet_id,
    prompt,
    output_dir,
    annotation_loader=None,
    min_area=5000,
    test_frames=None,
):
    """
    Process a snippet using SAM3 video mode.

    Two-stage approach:
      Stage 1: Text prompt on keyframe → masklets for tools/cloth only
      Stage 2: Video propagation → track masklets through all frames

    Then: GT tissue subtraction + overlay rendering + video stitching.
    """
    frames_dir = snippet_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("*.webp"))
    if not frame_files:
        print(f"  WARNING: No frames found in {frames_dir}")
        return []

    total_frames = len(frame_files)
    if test_frames:
        effective_frames = min(test_frames, total_frames)
    else:
        effective_frames = total_frames

    print(f"\n  Processing {snippet_id}: {effective_frames} frames"
          f" (of {total_frames}), prompt='{prompt}'")

    # Create output directories
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # --- Stage 1: Start session + text prompt → masklets ---
    print(f"  Starting video session ({total_frames} frames)...")
    session = predictor.start_session(resource_path=str(frames_dir))
    sid = session["session_id"]

    # Add text prompt on first frame — applies globally to all frames
    # This constrains detection to only find tools/cloth
    predictor.add_prompt(session_id=sid, frame_idx=0, text=prompt)
    print(f"  Text prompt applied: '{prompt}'")

    # --- Stage 2: Propagate masklets through video ---
    print(f"  Propagating...")
    all_results = []
    frame_times = []

    for frame_idx, outputs in predictor.propagate_in_video(
        session_id=sid,
        propagation_direction="forward",
        start_frame_idx=None,
        max_frame_num_to_track=test_frames,
    ):
        ft0 = time.time()

        # Map frame_idx to our frame file
        if frame_idx >= len(frame_files):
            break

        result = _convert_video_output(
            outputs, frame_files[frame_idx], prompt, min_area
        )
        all_results.append(result)

        n_masks = len(result["masks"].get(prompt, []))
        ft = time.time() - ft0
        frame_times.append(ft)

        # Compact progress logging
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            elapsed = time.time() - t0
            print(f"    [{frame_idx+1}/{effective_frames}] "
                  f"{frame_files[frame_idx].stem} | masks={n_masks} | "
                  f"{elapsed:.0f}s elapsed")

    predictor.close_session(session_id=sid)
    inference_time = time.time() - t0
    print(f"  Video inference done: {len(all_results)} frames in {inference_time:.1f}s "
          f"({inference_time/max(len(all_results),1):.1f}s/frame)")

    # Collect frame files for the frames we actually processed
    processed_frame_files = frame_files[:len(all_results)]

    # --- Pass 3: GT tissue subtraction ---
    if annotation_loader:
        print(f"\n  GT tissue subtraction...")
        cleaned = _subtract_gt_tissue(
            all_results, processed_frame_files, annotation_loader, [prompt]
        )
        print(f"  GT subtraction cleaned {cleaned} frames")

    # --- Render overlays ---
    print(f"  Rendering overlays...")
    for i, (result, fpath) in enumerate(zip(all_results, processed_frame_files)):
        _, overlay, _ = _render_overlay_from_results(
            fpath, result, [prompt], annotation_loader
        )
        out_path = overlays_dir / f"{fpath.stem}.jpg"
        cv2.imwrite(str(out_path), overlay)

    # --- Stitch video ---
    video_path = output_dir / f"{snippet_id}_overlay.mp4"
    _stitch_snippet_video(overlays_dir, video_path, fps=6)

    # --- Save results JSON ---
    results_path = output_dir / f"{snippet_id}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "episode": episode,
            "snippet_id": snippet_id,
            "prompt": prompt,
            "mode": "video",
            "num_frames": len(all_results),
            "inference_time_s": round(inference_time, 1),
            "frames": all_results,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")

    # --- Summary ---
    total_masks = sum(
        len(r["masks"].get(prompt, [])) for r in all_results
    )
    frames_with_masks = sum(
        1 for r in all_results if len(r["masks"].get(prompt, [])) > 0
    )
    total_time = time.time() - t0
    print(f"\n  {snippet_id} done: {len(all_results)} frames in {total_time:.1f}s "
          f"({total_time/max(len(all_results),1):.1f}s/frame)")
    print(f"    {prompt}: {total_masks} masks across "
          f"{frames_with_masks}/{len(all_results)} frames")

    return all_results


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tool/cloth masks using SAM3 video mode"
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
        "--prompt", default="surgical tool and cloth",
        help="Text prompt for detection (default: 'surgical tool and cloth')",
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

    # Count frames
    total_frames = 0
    for snip_dir in snippet_list:
        frames_dir = snip_dir / "frames_left"
        if frames_dir.exists():
            total_frames += len(list(frames_dir.glob("*.webp")))

    # --- Header ---
    print("=" * 60)
    print("SAM3 Video Segmentation Pipeline")
    print("=" * 60)
    print(f"Episode: {args.episode}")
    print(f"Snippets: {len(snippet_list)}")
    print(f"Total frames: {total_frames}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output: {output_dir}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")
    if args.lora_checkpoint:
        print(f"LoRA checkpoint: {args.lora_checkpoint}")

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

    # --- Process snippets ---
    t_total = time.time()
    for snip_dir in snippet_list:
        snippet_id = snip_dir.name
        print(f"\n{'=' * 60}")
        print(f"Episode: {args.episode} / {snippet_id}")
        print(f"{'=' * 60}")

        snip_output = output_dir / args.episode / snippet_id
        process_snippet_video(
            predictor=predictor,
            snippet_dir=snip_dir,
            episode=args.episode,
            snippet_id=snippet_id,
            prompt=args.prompt,
            output_dir=snip_output,
            annotation_loader=ann_loader,
            min_area=args.min_area,
            test_frames=args.test,
        )

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done! {len(snippet_list)} snippets in {total_time:.1f}s")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
