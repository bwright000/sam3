#!/usr/bin/env python3
"""
CRCD Dataset Test Script for SAM3.

Tests SAM3 segmentation with custom prompts for medical imaging:
- liver segmentation
- gallbladder segmentation
- surgical tool segmentation

Usage:
    # CPU mode (recommended for limited hardware)
    python scripts/test_crcd_prompts.py --device cpu --data-dir ./data/crcd

    # Single image test
    python scripts/test_crcd_prompts.py --device cpu --image ./data/crcd/sample.jpg

    # Custom prompts
    python scripts/test_crcd_prompts.py --device cpu --prompts "liver,gallbladder,tool"

    # GPU mode (if available)
    python scripts/test_crcd_prompts.py --device cuda --data-dir ./data/crcd
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default prompts for CRCD medical imaging
DEFAULT_PROMPTS = ["liver", "gallbladder", "tool"]

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test SAM3 segmentation on CRCD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing CRCD images",
    )
    input_group.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="One or more image files to process",
    )
    input_group.add_argument(
        "--video",
        type=str,
        help="Video file to process (outputs segmented video)",
    )
    input_group.add_argument(
        "--frame-dir",
        type=str,
        help="Directory containing video frames (00000.jpg, 00001.jpg, etc.)",
    )

    # Model options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (downloads from HF if not provided)",
    )

    # Prompt options
    parser.add_argument(
        "--prompts",
        type=str,
        default=",".join(DEFAULT_PROMPTS),
        help=f"Comma-separated list of prompts (default: {','.join(DEFAULT_PROMPTS)})",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save results (default: ./outputs)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save individual mask images",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save overlay visualization (default: True)",
    )

    # Performance options
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for output video (default: 30)",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage during inference",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    return parser.parse_args()


def find_images(data_dir: str, max_images: Optional[int] = None) -> List[Path]:
    """Find all images in directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(data_path.glob(f"*{ext}"))
        images.extend(data_path.glob(f"*{ext.upper()}"))

    images = sorted(images)

    if max_images is not None:
        images = images[:max_images]

    return images


def load_image(image_path: str) -> Image.Image:
    """Load and prepare image for inference."""
    img = Image.open(image_path).convert("RGB")
    return img


def create_overlay(image: Image.Image, masks: dict, alpha: float = 0.25) -> Image.Image:
    """
    Create visualization overlay with colored masks and smooth contours.

    Uses SAM3 demo-style rendering with multi-layer contours for smooth boundaries:
    - Alpha blending for semi-transparent mask overlay
    - Triple-layer contours: white (7px) -> black (5px) -> color (3px)

    Args:
        image: Original PIL image
        masks: Dictionary of prompt -> mask tensor
        alpha: Overlay transparency (default 0.25 to match SAM3 demo)

    Returns:
        PIL Image with colored mask overlay and smooth contours
    """
    import cv2

    # Color palette for different prompts (BGR for OpenCV)
    colors = {
        "reddish-brown organ": (0, 0, 255),      # Red (BGR)
        "greenish-grey organ": (0, 255, 0),      # Green (BGR)
        "surgical tool": (255, 0, 0),            # Blue (BGR)
        "instrument": (255, 0, 0),               # Blue (alias)
        "cloth": (0, 165, 255),                  # Orange (BGR)
        "liver": (0, 0, 255),                    # Red
        "gallbladder": (0, 255, 0),              # Green
        "tool": (255, 0, 0),                     # Blue
        "default": (0, 255, 255),                # Yellow (BGR)
    }

    # Convert to numpy (BGR for OpenCV)
    img_np = np.array(image).copy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    masked_frame = img_bgr.copy()

    for prompt, mask_data in masks.items():
        mask_tensor = mask_data.get("masks")
        if mask_tensor is None or len(mask_tensor) == 0:
            continue

        # Get color for this prompt (BGR)
        color = colors.get(prompt.lower(), colors["default"])
        color_np = np.array(color, dtype=np.uint8)

        # Process each mask
        for i, mask in enumerate(mask_tensor):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = mask.squeeze()

            # Resize mask to image size if needed
            if mask.shape != (img_np.shape[0], img_np.shape[1]):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Binary mask
            mask_bool = (mask > 0.5).astype(np.uint8)

            # Alpha blending (SAM3 style: 75% original, 25% mask color)
            curr_masked_frame = np.where(
                mask_bool[..., None].astype(bool),
                color_np,
                masked_frame
            )
            masked_frame = cv2.addWeighted(
                masked_frame, 1.0 - alpha,
                curr_masked_frame, alpha,
                0
            )

            # Draw multi-layer contours for smooth boundaries
            contours, _ = cv2.findContours(
                mask_bool,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )

            # Triple-layer contour rendering (SAM3 demo style)
            cv2.drawContours(masked_frame, contours, -1, (255, 255, 255), 7)  # White outer
            cv2.drawContours(masked_frame, contours, -1, (0, 0, 0), 5)        # Black middle
            cv2.drawContours(masked_frame, contours, -1, color, 3)            # Color inner

    # Convert back to RGB for PIL
    result_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def save_results(
    image_path: Path,
    results: dict,
    output_dir: Path,
    original_image: Image.Image,
    save_masks: bool = False,
    save_overlay: bool = True,
):
    """Save segmentation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Include parent folder name to avoid overwriting when multiple images have same name
    # e.g., split_0/00000.jpg -> split_0_00000
    parent_name = image_path.parent.name
    stem = f"{parent_name}_{image_path.stem}" if parent_name else image_path.stem

    # Save overlay visualization
    if save_overlay:
        overlay = create_overlay(original_image, results)
        overlay_path = output_dir / f"{stem}_overlay.png"
        overlay.save(overlay_path)

    # Save individual masks
    if save_masks:
        for prompt, mask_data in results.items():
            masks = mask_data.get("masks")
            if masks is None:
                continue

            for i, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                mask_img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
                mask_path = output_dir / f"{stem}_{prompt}_{i}.png"
                mask_img.save(mask_path)

    # Save metadata
    metadata = {
        "image": str(image_path),
        "prompts": list(results.keys()),
        "results": {
            prompt: {
                "num_masks": len(data["masks"]) if data["masks"] is not None else 0,
                "scores": data["scores"].tolist() if data["scores"] is not None else [],
            }
            for prompt, data in results.items()
        },
    }

    meta_path = output_dir / f"{stem}_results.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_video(
    wrapper,
    video_path: str,
    prompts: List[str],
    output_dir: Path,
    fps: int = 30,
    max_frames: Optional[int] = None,
    verbose: bool = True,
):
    """
    Process a video file and create segmented output video.

    Args:
        wrapper: SAM3InferenceWrapper instance
        video_path: Path to input video
        prompts: List of text prompts
        output_dir: Output directory
        fps: Output video FPS
        max_frames: Maximum frames to process (None for all)
        verbose: Print progress
    """
    import cv2

    video_path = Path(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Input FPS: {input_fps:.1f}, Output FPS: {fps}")
    print(f"  Frames to process: {total_frames}")

    # Create output video writer
    output_path = output_dir / f"{video_path.stem}_segmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")

    start_time = time.time()
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Run inference
        results = wrapper.segment(image, prompts)

        # Create overlay with smooth contours
        overlay = create_overlay(image, results)

        # Convert back to BGR for video writer
        overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
        out.write(overlay_bgr)

        frame_idx += 1

        if verbose and (frame_idx % 10 == 0 or frame_idx == total_frames):
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} ({fps_actual:.2f} fps, ETA: {eta:.0f}s)")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\nVideo saved to: {output_path}")
    print(f"Total time: {total_time:.1f}s ({total_frames / total_time:.2f} fps)")


def process_frame_directory(
    wrapper,
    frame_dir: str,
    prompts: List[str],
    output_dir: Path,
    fps: int = 30,
    max_frames: Optional[int] = None,
    verbose: bool = True,
):
    """
    Process a directory of video frames and create segmented output video.

    Args:
        wrapper: SAM3InferenceWrapper instance
        frame_dir: Directory containing frames (00000.jpg, 00001.jpg, etc.)
        prompts: List of text prompts
        output_dir: Output directory
        fps: Output video FPS
        max_frames: Maximum frames to process (None for all)
        verbose: Print progress
    """
    import cv2

    frame_path = Path(frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all frames
    frames = sorted(frame_path.glob("*.jpg"))
    if not frames:
        frames = sorted(frame_path.glob("*.png"))
    if not frames:
        raise ValueError(f"No frames found in {frame_dir}")

    if max_frames:
        frames = frames[:max_frames]

    # Get dimensions from first frame
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]

    print(f"Frame directory: {frame_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Output FPS: {fps}")
    print(f"  Frames to process: {len(frames)}")

    # Create output video writer
    output_path = output_dir / f"{frame_path.name}_segmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")

    start_time = time.time()

    for frame_idx, frame_file in enumerate(frames):
        # Load frame
        image = Image.open(frame_file).convert("RGB")

        # Run inference
        results = wrapper.segment(image, prompts)

        # Create overlay with smooth contours
        overlay = create_overlay(image, results)

        # Convert to BGR for video writer
        overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
        out.write(overlay_bgr)

        if verbose and ((frame_idx + 1) % 10 == 0 or frame_idx == len(frames) - 1):
            elapsed = time.time() - start_time
            fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(frames) - frame_idx - 1) / fps_actual if fps_actual > 0 else 0
            print(f"  Frame {frame_idx + 1}/{len(frames)} ({fps_actual:.2f} fps, ETA: {eta:.0f}s)")

    out.release()

    total_time = time.time() - start_time
    print(f"\nVideo saved to: {output_path}")
    print(f"Total time: {total_time:.1f}s ({len(frames) / total_time:.2f} fps)")


def profile_memory():
    """Print memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"CPU Memory - RSS: {mem_info.rss / 1024**3:.2f} GB")
        except ImportError:
            print("Install psutil for CPU memory profiling: pip install psutil")


def run_inference(
    wrapper,
    image_path: Path,
    prompts: List[str],
    verbose: bool = True,
    profile: bool = False,
) -> dict:
    """
    Run SAM3 inference on a single image.

    Args:
        wrapper: SAM3InferenceWrapper instance
        image_path: Path to image
        prompts: List of text prompts
        verbose: Print progress
        profile: Profile memory usage

    Returns:
        Dictionary of results per prompt
    """
    if verbose:
        print(f"\nProcessing: {image_path.name}")

    start_time = time.time()

    # Load image
    image = load_image(str(image_path))

    if profile:
        print("Before inference:")
        profile_memory()

    # Run segmentation
    results = wrapper.segment(image, prompts)

    if profile:
        print("After inference:")
        profile_memory()

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Total time: {elapsed:.1f}s")
        for prompt, data in results.items():
            num_masks = len(data["masks"]) if data["masks"] is not None else 0
            print(f"  {prompt}: {num_masks} masks found")

    return results, image


def main():
    """Main entry point."""
    args = parse_args()

    # Parse prompts
    prompts = [p.strip() for p in args.prompts.split(",")]

    print("=" * 60)
    print("SAM3 CRCD Test Script")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Prompts: {prompts}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading SAM3 model...")
    from scripts.model_loader import SAM3InferenceWrapper

    wrapper = SAM3InferenceWrapper(
        device=args.device,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
    )

    output_dir = Path(args.output_dir)

    # Handle video input
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)

        print(f"\nProcessing video: {video_path.name}")
        process_video(
            wrapper,
            args.video,
            prompts,
            output_dir,
            fps=args.fps,
            max_frames=args.max_images,
            verbose=args.verbose,
        )
        return

    # Handle frame directory input
    if args.frame_dir:
        frame_path = Path(args.frame_dir)
        if not frame_path.exists():
            print(f"Error: Frame directory not found: {args.frame_dir}")
            sys.exit(1)

        print(f"\nProcessing frame directory: {frame_path.name}")
        process_frame_directory(
            wrapper,
            args.frame_dir,
            prompts,
            output_dir,
            fps=args.fps,
            max_frames=args.max_images,
            verbose=args.verbose,
        )
        return

    # Handle image inputs
    if args.image:
        image_paths = [Path(img) for img in args.image]
        # Verify all images exist
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            print(f"Error: Image(s) not found: {', '.join(str(p) for p in missing)}")
            sys.exit(1)
    else:
        image_paths = find_images(args.data_dir, args.max_images)
        if not image_paths:
            print(f"Error: No images found in {args.data_dir}")
            sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process")

    # Process images
    success_count = 0

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] ", end="")

        try:
            results, original_image = run_inference(
                wrapper,
                image_path,
                prompts,
                verbose=args.verbose,
                profile=args.profile_memory,
            )

            # Save results
            save_results(
                image_path,
                results,
                output_dir,
                original_image,
                save_masks=args.save_masks,
                save_overlay=args.save_overlay,
            )

            success_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Processed: {success_count}/{len(image_paths)} images")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
