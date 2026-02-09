"""
Combined overlay: tissue GT annotations + SAM3 tool segmentation.

Renders liver/gallbladder ground truth masks from COCO annotations alongside
SAM3 text-prompted tool segmentation on the same frames.

Usage:
    # GT-only overlay (instant, no SAM3)
    python scripts/overlay_combined.py --dataset-dir "path/to/C_1" --gt-only --num-samples 10

    # Combined overlay: GT tissue + SAM3 tools
    python scripts/overlay_combined.py --dataset-dir "path/to/C_1" --split-idx 0

    # Quick test (first N frames)
    python scripts/overlay_combined.py --dataset-dir "path/to/C_1" --split-idx 0 --test 5

    # Full split video
    python scripts/overlay_combined.py --dataset-dir "path/to/C_1" --split-idx 0 --fps 30
"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.check_annotations import COCOAnnotationLoader


# ---------------------------------------------------------------------------
# Colors (BGR for OpenCV)
# ---------------------------------------------------------------------------

COLORS = {
    "Liver": (0, 0, 255),         # Red
    "Gallbladder": (0, 255, 0),   # Green
    "tool": (255, 0, 0),          # Blue
    "Meat": (0, 200, 200),        # Yellow-ish
    "Skin": (180, 130, 70),       # Steel blue
}
DEFAULT_COLOR = (0, 255, 255)  # Yellow


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_combined_frame(
    img_bgr: np.ndarray,
    gt_masks: dict,
    sam3_masks: dict = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Render GT tissue masks and optional SAM3 tool masks on a frame.

    Args:
        img_bgr: Original BGR image
        gt_masks: {category_name: binary_mask} from COCOAnnotationLoader
        sam3_masks: {prompt: {"masks": tensor, "scores": tensor}} from SAM3 (optional)
        alpha: Overlay transparency (0.25 = SAM3 demo style)

    Returns:
        Rendered BGR frame
    """
    frame = img_bgr.copy()

    # 1. Draw GT masks (liver, gallbladder, etc.)
    for cat_name, mask in gt_masks.items():
        color = COLORS.get(cat_name, DEFAULT_COLOR)
        color_np = np.array(color, dtype=np.uint8)
        mask_bool = mask > 0

        # Alpha blend
        overlay = frame.copy()
        overlay[mask_bool] = color_np
        frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

        # Triple-layer contours
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 7)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(frame, contours, -1, color, 3)

    # 2. Draw SAM3 tool masks (if provided)
    if sam3_masks:
        import torch
        for prompt, mask_data in sam3_masks.items():
            masks_tensor = mask_data.get("masks")
            if masks_tensor is None or len(masks_tensor) == 0:
                continue

            color = COLORS.get("tool", DEFAULT_COLOR)
            color_np = np.array(color, dtype=np.uint8)

            for m in masks_tensor:
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                if m.ndim == 3:
                    m = m.squeeze()
                # Resize if needed
                if m.shape != (img_bgr.shape[0], img_bgr.shape[1]):
                    m = cv2.resize(
                        m.astype(np.float32),
                        (img_bgr.shape[1], img_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                mask_bool = (m > 0.5).astype(np.uint8)

                # Alpha blend
                overlay = frame.copy()
                overlay[mask_bool > 0] = color_np
                frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

                # Triple-layer contours
                contours, _ = cv2.findContours(
                    mask_bool * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(frame, contours, -1, (255, 255, 255), 7)
                cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
                cv2.drawContours(frame, contours, -1, color, 3)

    # 3. Legend
    _draw_legend(frame, gt_masks, sam3_masks)

    return frame


def _draw_legend(frame, gt_masks, sam3_masks):
    """Draw a small legend in the top-left corner."""
    y = 25
    entries = []
    for cat_name in gt_masks:
        color = COLORS.get(cat_name, DEFAULT_COLOR)
        entries.append((f"GT: {cat_name}", color))
    if sam3_masks:
        for prompt in sam3_masks:
            if sam3_masks[prompt].get("masks") is not None and len(sam3_masks[prompt]["masks"]) > 0:
                entries.append((f"SAM3: {prompt}", COLORS.get("tool", DEFAULT_COLOR)))

    for label, color in entries:
        cv2.rectangle(frame, (10, y - 15), (30, y + 3), color, -1)
        cv2.putText(frame, label, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, label, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 25


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_split_video(
    loader: COCOAnnotationLoader,
    split_name: str,
    image_ids: list,
    output_dir: Path,
    wrapper=None,
    tool_prompt: str = "surgical tool",
    fps: int = 30,
    max_frames: int = None,
):
    """Process a split and write an overlay video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_frames is not None:
        image_ids = image_ids[:max_frames]

    total = len(image_ids)
    print(f"\n  Processing {split_name}: {total} frames")

    # Get dimensions from first frame
    first_path = loader.resolve_frame_path(image_ids[0])
    if first_path is None:
        print(f"  ERROR: Cannot resolve path for first frame")
        return
    first_img = cv2.imread(str(first_path))
    h, w = first_img.shape[:2]

    # Create video writer
    video_name = f"{loader.dataset_dir.name}_{split_name}_combined.mp4"
    video_path = output_dir / video_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"  ERROR: Could not create video: {video_path}")
        return

    start_time = time.time()

    for idx, img_id in enumerate(image_ids):
        frame_path = loader.resolve_frame_path(img_id)
        if frame_path is None:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue

        # Get GT masks
        gt_masks = loader.get_frame_masks(img_id)

        # Get SAM3 tool masks (if wrapper provided)
        sam3_masks = None
        if wrapper is not None:
            from PIL import Image
            pil_image = Image.open(frame_path).convert("RGB")
            sam3_masks = wrapper.segment(pil_image, [tool_prompt])

        # Render combined frame
        rendered = render_combined_frame(img_bgr, gt_masks, sam3_masks)

        # Frame counter
        cv2.putText(
            rendered,
            f"{split_name} | {idx + 1}/{total}",
            (w - 300, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out.write(rendered)

        # Progress
        if (idx + 1) % 10 == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            mode = "GT+SAM3" if wrapper else "GT only"
            print(f"  [{idx + 1}/{total}] {rate:.1f} fps | ETA: {eta:.0f}s | {mode}")

    out.release()
    elapsed = time.time() - start_time
    print(f"  Video saved: {video_path}")
    print(f"  Duration: {total / fps:.1f}s at {fps} FPS | Processed in {elapsed:.1f}s")


def render_samples(
    loader: COCOAnnotationLoader,
    image_ids: list,
    output_dir: Path,
    wrapper=None,
    tool_prompt: str = "surgical tool",
    num_samples: int = 10,
):
    """Render sample frames as individual images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample evenly
    step = max(1, len(image_ids) // num_samples)
    sample_ids = image_ids[::step][:num_samples]

    print(f"\n  Rendering {len(sample_ids)} sample frames...")

    for idx, img_id in enumerate(sample_ids):
        frame_path = loader.resolve_frame_path(img_id)
        if frame_path is None:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue

        gt_masks = loader.get_frame_masks(img_id)

        sam3_masks = None
        if wrapper is not None:
            img_start = time.time()
            from PIL import Image
            pil_image = Image.open(frame_path).convert("RGB")
            sam3_masks = wrapper.segment(pil_image, [tool_prompt])
            elapsed = time.time() - img_start
            print(f"  [{idx + 1}/{len(sample_ids)}] SAM3: {elapsed:.1f}s | {frame_path.name}")
        else:
            print(f"  [{idx + 1}/{len(sample_ids)}] GT only | {frame_path.name}")

        rendered = render_combined_frame(img_bgr, gt_masks, sam3_masks)

        out_path = output_dir / f"sample_{idx:03d}_{frame_path.stem}.jpg"
        cv2.imwrite(str(out_path), rendered)

    print(f"  Saved {len(sample_ids)} samples to {output_dir}")


# ---------------------------------------------------------------------------
# Auto-detect annotation file
# ---------------------------------------------------------------------------

def find_annotation_file(dataset_dir: Path) -> Path:
    """Auto-detect annotation JSON (tries annotations.json then train.json)."""
    for name in ["annotations.json", "train.json"]:
        p = dataset_dir / name
        if p.exists():
            return p
    print(f"ERROR: No annotation file found in {dataset_dir}")
    print("  Tried: annotations.json, train.json")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined overlay: tissue GT + SAM3 tool segmentation"
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Path to tissue dataset (C_1, E_3, or F_3)",
    )
    parser.add_argument(
        "--split-idx",
        type=int,
        default=0,
        help="Which split to process (default: 0)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./outputs/combined",
        help="Output directory",
    )
    parser.add_argument(
        "--tool-prompt",
        default="tool",
        help='SAM3 text prompt for tools (default: "tool")',
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="SAM3 confidence threshold (default: 0.3, lower = more permissive)",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Only render GT annotations (skip SAM3, instant)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Render N sample frames instead of full video",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Quick test: process only N frames",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS (default: 30)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Load annotations
    print("=" * 60)
    print("Combined Overlay: Tissue GT + SAM3 Tools")
    print("=" * 60)

    ann_file = find_annotation_file(dataset_dir)
    loader = COCOAnnotationLoader(str(ann_file), str(dataset_dir))
    loader.load()

    # Get splits
    splits = loader.get_split_image_ids()
    split_names = sorted(splits.keys(), key=lambda s: int(re.search(r"\d+", s).group()))

    if not split_names:
        print("ERROR: No splits found in annotations")
        sys.exit(1)

    print(f"  Found {len(split_names)} splits")
    print(f"  Selected: split_{args.split_idx}")

    target_split = f"split_{args.split_idx}"
    if target_split not in splits:
        print(f"ERROR: {target_split} not found. Available: {split_names[:10]}...")
        sys.exit(1)

    image_ids = splits[target_split]
    print(f"  Frames in {target_split}: {len(image_ids)}")

    # Load SAM3 if not GT-only
    wrapper = None
    if not args.gt_only:
        print(f"\nLoading SAM3 model for tool segmentation...")
        print(f"  Tool prompt: \"{args.tool_prompt}\"")
        from scripts.model_loader import SAM3InferenceWrapper
        wrapper = SAM3InferenceWrapper(device=args.device, verbose=True)
        wrapper.processor.set_confidence_threshold(args.confidence)
        print(f"  Confidence threshold: {args.confidence}")
    else:
        print("\n  GT-only mode (no SAM3)")

    # Process
    if args.num_samples is not None:
        # Sample mode: render individual images
        render_samples(
            loader=loader,
            image_ids=image_ids,
            output_dir=output_dir / f"{dataset_dir.name}_{target_split}_samples",
            wrapper=wrapper,
            tool_prompt=args.tool_prompt,
            num_samples=args.num_samples,
        )
    else:
        # Video mode
        process_split_video(
            loader=loader,
            split_name=target_split,
            image_ids=image_ids,
            output_dir=output_dir,
            wrapper=wrapper,
            tool_prompt=args.tool_prompt,
            fps=args.fps,
            max_frames=args.test,
        )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
