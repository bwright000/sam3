"""Boundary refinement for painted tool masks.

Per research: the primary refinement is SAM3's `add_new_mask` with the painted
mask as dense prompt + optional correction points. This module provides the
pre-processing chain that prepares a crude brush-painted mask for SAM3 input.

Pre-processing steps (surgical-instrument specific):
  1. Morph close (3x3) — seal brush gaps
  2. Largest connected component — drop stray dabs
  3. Gaussian blur (5x5, sigma=1) — soft logits
  4. Specular-highlight inpainting on the underlying frame (optional) — reduces
     glare confusion on reflective metal.

The refined mask is then fed to SAM3 via sam3_service.add_mask_anchor().
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_painted_mask(mask: np.ndarray, min_component_area: int = 100) -> np.ndarray:
    """Clean a brush-painted binary mask for SAM3 consumption.

    mask: (H, W) uint8 with 0/1 values.
    min_component_area: drop connected components smaller than this many pixels
        (default 100 — kills mouse-jitter dabs but preserves multi-instrument
        masks where the user paints two or more separate tools).
    Returns: (H, W) uint8 cleaned mask.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {mask.shape}")

    # 1. Seal brush gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Drop only TINY components (stray dabs). Keep everything else so users
    # can paint multiple disjoint instruments under a single category.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return closed

    keep = np.zeros_like(closed)
    for i in range(1, n_labels):  # skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            keep[labels == i] = 1
    return keep if keep.sum() > 0 else closed  # fallback: keep everything if all CCs were tiny


def inpaint_specular_highlights(frame_rgb: np.ndarray, v_threshold: int = 245,
                                inpaint_radius: int = 3) -> np.ndarray:
    """Inpaint saturated specular highlights before refinement.

    frame_rgb: (H, W, 3) uint8 RGB frame.
    Returns: RGB frame with highlights inpainted.
    """
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    spec_mask = (v > v_threshold).astype(np.uint8) * 255
    # dilate slightly for better inpainting
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spec_mask = cv2.dilate(spec_mask, k, iterations=1)
    inpainted = cv2.inpaint(bgr, spec_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def mask_to_low_res_logits(mask: np.ndarray, target: int = 256) -> np.ndarray:
    """Downsample a binary mask to target x target soft logits for SAM3 mask prompt.

    SAM/SAM2/SAM3 expect low-res mask prompts at 256x256 (image_size/4 for 1024 input).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # float blur for soft logits
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
    resized = cv2.resize(blurred, (target, target), interpolation=cv2.INTER_AREA)
    # Scale to logit range: [-10, 10] so that mask > 0 -> positive logit
    logits = (resized * 20.0) - 10.0
    return logits.astype(np.float32)
