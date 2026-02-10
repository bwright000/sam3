"""
End-to-end annotated dataset pipeline for C_1, E_3, and F_3.

Produces a complete stereo dataset where every snippet frame has:
  - Tool segmentations (SAM3 video model, prompts: tool + cloth)
  - Tissue segmentations (GT where available, SAM3-propagated where not)
  - Both left and right camera outputs (stereo projection via disparity)

Pipeline per snippet:
  1. SAM3 video inference for tool/cloth masks
  2. GT tissue loading + SAM3 propagation for tissue gaps
  3. GT tissue subtraction from tool masks (tissue takes priority)
  4. Stereo projection of all masks to right camera
  5. Overlay rendering + video stitching for both cameras

Usage:
    # Full pipeline (all 3 episodes)
    python scripts/run_dataset_pipeline.py \\
        --segments-dir data/Segments \\
        --tissue-seg-dir "F:/2026 vibes/MPHY Project/annotated_dataset/tissue_segmentation" \\
        --calibration stereo_calib.pkl

    # Single episode
    python scripts/run_dataset_pipeline.py \\
        --segments-dir data/Segments \\
        --tissue-seg-dir "F:/2026 vibes/MPHY Project/annotated_dataset/tissue_segmentation" \\
        --calibration stereo_calib.pkl \\
        --episodes C_1

    # Test mode (first 10 frames per snippet, skip stereo)
    python scripts/run_dataset_pipeline.py \\
        --segments-dir data/Segments \\
        --tissue-seg-dir "F:/2026 vibes/MPHY Project/annotated_dataset/tissue_segmentation" \\
        --episodes C_1 --test 10 --skip-stereo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_tool_masks import _load_episode_annotations
from scripts.generate_tool_masks_video import (
    load_video_model,
    process_snippet_video,
    TISSUE_CATEGORIES,
)
from scripts.project_masks_stereo import (
    load_stereo_calibration,
    process_snippet_stereo,
)

ALL_EPISODES = ["C_1", "E_3", "F_3"]

# Valid snippet counts per episode (from motion analysis)
EPISODE_SNIPPETS = {
    "C_1": 3,
    "E_3": 5,
    "F_3": 7,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end annotated dataset pipeline for surgical video",
    )

    # Required paths
    parser.add_argument(
        "--segments-dir", required=True,
        help="Path to Segments directory with episode/snippet structure",
    )
    parser.add_argument(
        "--tissue-seg-dir", required=True,
        help="Path to tissue_segmentation directory with GT annotations",
    )

    # Optional paths
    parser.add_argument(
        "--calibration", default=None,
        help="Path to stereo calibration file (.pkl, .yaml, or .json). "
             "Required for stereo projection; omit to skip stereo step.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/annotated_datasets",
        help="Output directory (default: outputs/annotated_datasets)",
    )

    # Episode selection
    parser.add_argument(
        "--episodes", nargs="+", default=ALL_EPISODES,
        help=f"Episodes to process (default: {ALL_EPISODES})",
    )
    parser.add_argument(
        "--snippet", type=int, default=None,
        help="Process only this snippet number (default: all valid snippets)",
    )

    # SAM3 video options
    parser.add_argument(
        "--prompts", nargs="+", default=["tool"],
        help="Text prompts for tool detection (default: tool)",
    )
    parser.add_argument(
        "--min-area", type=int, default=5000,
        help="Minimum tool mask area in pixels (default: 5000)",
    )
    parser.add_argument(
        "--num-keyframes", type=int, default=1,
        help="Number of keyframes for video prompting (default: 1)",
    )
    parser.add_argument(
        "--forward-only", action="store_true",
        help="Forward-only propagation (default: bidirectional)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.3,
        help="Dropout recovery confidence threshold (default: 0.3, 0=disable)",
    )
    parser.add_argument(
        "--dropout-window", type=int, default=5,
        help="Consecutive low-confidence frames for dropout detection (default: 5)",
    )
    parser.add_argument(
        "--gt-erode-px", type=int, default=3,
        help="GT erosion buffer pixels for tissue subtraction (default: 3)",
    )

    # Stereo options
    parser.add_argument(
        "--assume-rectified", action="store_true", default=True,
        help="Assume stereo images are already rectified (default: True)",
    )
    parser.add_argument(
        "--no-assume-rectified", dest="assume_rectified", action="store_false",
    )
    parser.add_argument(
        "--num-disparities", type=int, default=128,
        help="SGBM max disparity range (default: 128)",
    )
    parser.add_argument(
        "--save-disparity", action="store_true",
        help="Save computed disparity maps as .npy files",
    )

    # Pipeline control
    parser.add_argument(
        "--test", type=int, default=None,
        help="Process first N frames per snippet (test mode)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip snippets that already have results",
    )
    parser.add_argument(
        "--skip-stereo", action="store_true",
        help="Skip stereo projection step",
    )
    parser.add_argument(
        "--lora-checkpoint", default=None,
        help="Path to LoRA weights for SAM3",
    )

    return parser.parse_args()


def collect_snippets(segments_dir, episode, snippet_num=None, max_snippets=None):
    """Collect valid snippet directories for an episode."""
    ep_dir = segments_dir / episode
    if not ep_dir.exists():
        print(f"  WARNING: Episode directory not found: {ep_dir}")
        return []

    if snippet_num is not None:
        snip = ep_dir / f"snippet_{snippet_num:03d}"
        if snip.exists():
            return [snip]
        print(f"  WARNING: Snippet not found: {snip}")
        return []

    snippet_list = sorted(
        [s for s in ep_dir.glob("snippet_*") if s.is_dir()]
    )

    # Filter to valid snippet count (ignore stale extra folders)
    if max_snippets and len(snippet_list) > max_snippets:
        snippet_list = snippet_list[:max_snippets]

    return snippet_list


def main():
    args = parse_args()
    segments_dir = Path(args.segments_dir)
    tissue_seg_dir = Path(args.tissue_seg_dir)
    output_dir = Path(args.output_dir)

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)
    if not tissue_seg_dir.exists():
        print(f"ERROR: Tissue segmentation directory not found: {tissue_seg_dir}")
        sys.exit(1)

    # --- Load stereo calibration (if provided) ---
    calib = None
    do_stereo = not args.skip_stereo and args.calibration is not None
    if do_stereo:
        calib_path = Path(args.calibration)
        if not calib_path.exists():
            print(f"WARNING: Calibration file not found: {calib_path}")
            print("  Stereo projection will be skipped.")
            do_stereo = False
        else:
            print("Loading stereo calibration...")
            calib = load_stereo_calibration(calib_path, args.assume_rectified)

    # --- Load SAM3 video model ---
    print()
    predictor = load_video_model(lora_checkpoint=args.lora_checkpoint)

    # --- Header ---
    print("\n" + "=" * 60)
    print("Annotated Dataset Pipeline")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Prompts: {args.prompts}")
    print(f"Tissue categories: {TISSUE_CATEGORIES}")
    print(f"Stereo projection: {'enabled' if do_stereo else 'disabled'}")
    print(f"Output: {output_dir}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")
    if args.resume:
        print(f"Resume: enabled")
    print("=" * 60)

    t_total = time.time()
    total_snippets = 0
    total_frames = 0

    for episode in args.episodes:
        print(f"\n{'#' * 60}")
        print(f"# Episode: {episode}")
        print(f"{'#' * 60}")

        # --- Load GT annotations for this episode ---
        print(f"\nLoading GT annotations for {episode}...")
        ann_loader = _load_episode_annotations(tissue_seg_dir, episode)
        if ann_loader is None:
            print(f"  WARNING: No GT annotations found for {episode}")

        # --- Collect snippets ---
        max_snippets = EPISODE_SNIPPETS.get(episode)
        snippet_list = collect_snippets(
            segments_dir, episode, args.snippet, max_snippets
        )
        if not snippet_list:
            print(f"  No snippets found for {episode} — skipping")
            continue

        print(f"  Processing {len(snippet_list)} snippets")

        for snip_dir in snippet_list:
            snippet_id = snip_dir.name
            snip_output = output_dir / episode / snippet_id

            # --- Resume check ---
            left_results_path = snip_output / f"{snippet_id}_results.json"
            right_results_path = snip_output / f"{snippet_id}_right_results.json"
            if args.resume:
                if left_results_path.exists() and (
                    not do_stereo or right_results_path.exists()
                ):
                    print(f"\n  Skipping {snippet_id} (results exist, --resume)")
                    continue

            print(f"\n{'=' * 60}")
            print(f"Episode: {episode} / {snippet_id}")
            print(f"{'=' * 60}")

            # --- Step 1+2+3: Tool segmentation + tissue GT/propagation ---
            all_results = process_snippet_video(
                predictor=predictor,
                snippet_dir=snip_dir,
                episode=episode,
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

            if not all_results:
                print(f"  WARNING: No results for {snippet_id}")
                continue

            total_snippets += 1
            total_frames += len(all_results)

            # --- Step 4: Stereo projection ---
            if do_stereo and calib is not None:
                print(f"\n  --- Stereo Projection ---")

                # Determine all prompts/categories present in results
                all_mask_keys = set()
                for fr in all_results:
                    all_mask_keys.update(fr.get("masks", {}).keys())
                all_prompts = list(all_mask_keys)

                process_snippet_stereo(
                    snippet_dir=snip_dir,
                    left_results=all_results,
                    calib=calib,
                    output_dir=snip_output,
                    prompts=all_prompts,
                    num_disparities=args.num_disparities,
                    min_area=100,
                    max_frames=args.test,
                    render_overlays=True,
                    save_disparity=args.save_disparity,
                )

    # --- Final summary ---
    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Episodes: {args.episodes}")
    print(f"  Snippets processed: {total_snippets}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
