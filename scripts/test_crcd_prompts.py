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
        help="Single image file to process",
    )

    # Model options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
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


def create_overlay(image: Image.Image, masks: dict, alpha: float = 0.5) -> Image.Image:
    """
    Create visualization overlay with colored masks.

    Args:
        image: Original PIL image
        masks: Dictionary of prompt -> mask tensor
        alpha: Overlay transparency

    Returns:
        PIL Image with colored mask overlay
    """
    # Color palette for different prompts
    colors = {
        "liver": (255, 0, 0),      # Red
        "gallbladder": (0, 255, 0), # Green
        "tool": (0, 0, 255),       # Blue
        "instrument": (0, 0, 255), # Blue (alias)
        "default": (255, 255, 0),  # Yellow for unknown
    }

    # Convert to numpy
    img_np = np.array(image).copy()
    overlay = img_np.copy()

    for prompt, mask_data in masks.items():
        mask_tensor = mask_data.get("masks")
        if mask_tensor is None or len(mask_tensor) == 0:
            continue

        # Get color for this prompt
        color = colors.get(prompt.lower(), colors["default"])

        # Process each mask
        for i, mask in enumerate(mask_tensor):
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = mask.squeeze()

            # Resize mask to image size if needed
            if mask.shape != (img_np.shape[0], img_np.shape[1]):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]))
                mask = np.array(mask_pil) > 127

            # Apply color to mask region
            mask_bool = mask > 0.5
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_bool,
                    (1 - alpha) * img_np[:, :, c] + alpha * color[c],
                    overlay[:, :, c],
                )

    return Image.fromarray(overlay.astype(np.uint8))


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
    stem = image_path.stem

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

    # Get image list
    if args.image:
        image_paths = [Path(args.image)]
        if not image_paths[0].exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
    else:
        image_paths = find_images(args.data_dir, args.max_images)
        if not image_paths:
            print(f"Error: No images found in {args.data_dir}")
            sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process")

    # Load model
    print("\nLoading SAM3 model...")
    from scripts.model_loader import SAM3InferenceWrapper

    wrapper = SAM3InferenceWrapper(
        device=args.device,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
    )

    # Process images
    output_dir = Path(args.output_dir)
    total_time = 0
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
