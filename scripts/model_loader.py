"""
Memory-optimized SAM3 model loader.

Provides a wrapper around Facebook SAM3's build_sam3_image_model() with:
- Explicit CPU device support
- Memory optimization settings
- Progress feedback during loading
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.device_config import DeviceConfig, configure_device


def load_sam3_model(
    config: Optional[DeviceConfig] = None,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    load_from_hf: bool = True,
    enable_inst_interactivity: bool = False,
    verbose: bool = True,
):
    """
    Load SAM3 image model with memory optimizations.

    Args:
        config: DeviceConfig object (if None, will auto-detect)
        device: Override device ('cpu' or 'cuda')
        checkpoint_path: Path to local checkpoint (if not using HuggingFace)
        load_from_hf: Whether to download from HuggingFace if no checkpoint
        enable_inst_interactivity: Enable SAM1-style interactive mode
        verbose: Print loading progress

    Returns:
        Loaded SAM3 model ready for inference
    """
    # Get or create device config
    if config is None:
        config = configure_device(device=device, verbose=verbose)
    elif device is not None:
        # Override config device
        config = configure_device(device=device, verbose=verbose)

    if verbose:
        print(f"\nLoading SAM3 model on {config.device}...")
        print("This may take 30-60 seconds...")
        start_time = time.time()

    # Import here to avoid slow startup
    from sam3.model_builder import build_sam3_image_model

    # Build model with appropriate settings
    model = build_sam3_image_model(
        device=config.device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        enable_segmentation=True,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=config.use_compile,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"Model loaded in {elapsed:.1f} seconds")
        print_model_info(model)

    return model


def print_model_info(model):
    """Print model information."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("SAM3 Model Info")
    print("=" * 50)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"Model in eval mode: {not model.training}")

    # Check device of first parameter
    first_param = next(model.parameters())
    print(f"Model device: {first_param.device}")
    print(f"Model dtype: {first_param.dtype}")
    print("=" * 50)


def get_processor(model, device: str = "cpu"):
    """
    Get SAM3 processor for inference.

    Args:
        model: Loaded SAM3 model
        device: Device string

    Returns:
        Sam3Processor ready for inference
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    return Sam3Processor(model, device=device)


def run_text_prompt_inference(
    processor,
    image,
    text_prompts: list,
    verbose: bool = True,
):
    """
    Run inference with text prompts.

    Args:
        processor: Sam3Processor instance
        image: PIL Image or tensor
        text_prompts: List of text prompts (e.g., ["liver", "gallbladder"])
        verbose: Print timing info

    Returns:
        Dictionary with masks and scores for each prompt
    """
    import time

    results = {}

    # First, set the image (this computes backbone features once)
    if verbose:
        print("Computing image features...")
        img_start = time.time()

    state = processor.set_image(image)

    if verbose:
        print(f"  Image features computed in {time.time() - img_start:.1f}s")

    # Then run each text prompt
    for prompt in text_prompts:
        if verbose:
            print(f"Processing prompt: '{prompt}'...")
            start_time = time.time()

        # Run text prompt (pass prompt first, then state)
        state = processor.set_text_prompt(prompt, state)

        # Get results
        masks = state.get("masks", None)
        scores = state.get("scores", None)

        results[prompt] = {
            "masks": masks.clone() if masks is not None else None,
            "scores": scores.clone() if scores is not None else None,
        }

        if verbose:
            elapsed = time.time() - start_time
            num_masks = len(masks) if masks is not None else 0
            print(f"  Found {num_masks} masks in {elapsed:.1f}s")

        # Reset prompts for next iteration (keep image features)
        processor.reset_all_prompts(state)

    return results


class SAM3InferenceWrapper:
    """
    High-level wrapper for SAM3 inference.

    Example:
        wrapper = SAM3InferenceWrapper(device="cpu")
        results = wrapper.segment(image, prompts=["liver", "gallbladder", "tool"])
    """

    def __init__(
        self,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize SAM3 wrapper.

        Args:
            device: 'cpu' or 'cuda' (auto-detect if None)
            checkpoint_path: Path to checkpoint (downloads from HF if None)
            verbose: Print progress information
        """
        self.config = configure_device(device=device, verbose=verbose)
        self.verbose = verbose

        # Load model
        self.model = load_sam3_model(
            config=self.config,
            checkpoint_path=checkpoint_path,
            load_from_hf=checkpoint_path is None,
            verbose=verbose,
        )

        # Create processor
        self.processor = get_processor(self.model, device=self.config.device)

    def segment(self, image, prompts: list) -> dict:
        """
        Segment image with text prompts.

        Args:
            image: PIL Image or path to image
            prompts: List of text prompts

        Returns:
            Dictionary mapping prompt -> {"masks": tensor, "scores": tensor}
        """
        from PIL import Image

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        return run_text_prompt_inference(
            self.processor,
            image,
            prompts,
            verbose=self.verbose,
        )

    def segment_single(self, image, prompt: str):
        """
        Segment image with a single text prompt.

        Args:
            image: PIL Image or path
            prompt: Text prompt

        Returns:
            Tuple of (masks, scores)
        """
        results = self.segment(image, [prompt])
        return results[prompt]["masks"], results[prompt]["scores"]


if __name__ == "__main__":
    # Test loading
    print("Testing SAM3 model loading...")

    # Test CPU mode
    config = configure_device(device="cpu", verbose=True)
    print("\nNote: Full model loading requires checkpoint download from HuggingFace")
    print("Run with: python model_loader.py --load to test full loading")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Actually load the model")
    args = parser.parse_args()

    if args.load:
        model = load_sam3_model(config=config, verbose=True)
        print("\nModel loaded successfully!")
