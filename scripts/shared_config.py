"""
Shared configuration for the CRCD segmentation pipeline.

Single source of truth for category definitions, color palettes,
and common rendering utilities. All scripts should import from here
instead of defining their own copies.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = ["Liver", "Gallbladder", "Meat", "Skin", "FBF", "PCH"]

# Maps GT annotation category names (title-case) to pipeline keys (lowercase)
GT_CAT_MAP = {
    "Liver": "liver",
    "Gallbladder": "gallbladder",
    "Meat": "meat",
    "Skin": "skin",
    "FBF": "fbf",
    "PCH": "pch",
}

# Object IDs for SAM3 tracker (tissue categories)
TISSUE_OBJ_IDS = {"Liver": 1, "Gallbladder": 2, "Meat": 3, "Skin": 4}
OBJ_ID_TO_KEY = {v: GT_CAT_MAP[k] for k, v in TISSUE_OBJ_IDS.items()}

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Canonical colors in BGR (OpenCV native format)
# Title-case keys match GT annotation category names
CATEGORY_COLORS_BGR = {
    "Liver": (0, 0, 255),        # Red
    "Gallbladder": (0, 255, 0),  # Green
    "Meat": (0, 165, 255),       # Orange
    "Skin": (203, 192, 255),     # Pink
    "FBF": (255, 0, 0),          # Blue
    "PCH": (255, 255, 0),        # Cyan
    "Tool": (255, 128, 0),       # Blue-ish
}
DEFAULT_COLOR_BGR = (0, 255, 255)  # Yellow

# Lowercase keys for pipeline scripts (generate_split_video, generate_tool_masks)
CATEGORY_COLORS_BGR_LOWER = {k.lower(): v for k, v in CATEGORY_COLORS_BGR.items()}
DEFAULT_COLOR_BGR_LOWER = DEFAULT_COLOR_BGR

# RGB versions for Gradio/PIL (auto-derived from BGR)
CATEGORY_COLORS_RGB = {k: (r, g, b) for k, (b, g, r) in CATEGORY_COLORS_BGR.items()}
DEFAULT_COLOR_RGB = tuple(reversed(DEFAULT_COLOR_BGR))

# RGB with lowercase keys
CATEGORY_COLORS_RGB_LOWER = {k.lower(): v for k, v in CATEGORY_COLORS_RGB.items()}


# ---------------------------------------------------------------------------
# Shared rendering utilities
# ---------------------------------------------------------------------------

def render_mask_overlay(frame, mask, color_bgr, alpha=0.25, contour_thickness=2,
                        contour_outline=False):
    """
    Render a single mask overlay on a BGR frame.

    Args:
        frame: BGR numpy array (modified in place)
        mask: binary mask (H, W), uint8 with values 0/1 or 0/255
        color_bgr: BGR color tuple
        alpha: overlay opacity (0=transparent, 1=opaque)
        contour_thickness: contour line width
        contour_outline: if True, draw white outline around contour

    Returns:
        frame with overlay applied
    """
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() == 1:
        mask_uint8 = mask_uint8 * 255

    mask_bool = mask_uint8 > 0
    if not mask_bool.any():
        return frame

    color_np = np.array(color_bgr, dtype=np.uint8)
    overlay = frame.copy()
    overlay[mask_bool] = color_np
    frame = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contour_outline:
        cv2.drawContours(frame, contours, -1, (255, 255, 255), contour_thickness + 2)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), contour_thickness + 1)
    cv2.drawContours(frame, contours, -1, color_bgr, contour_thickness)

    return frame


def draw_legend(frame, category_keys, masks_dict, colors=None, use_lowercase=True):
    """
    Draw a color legend on a BGR frame.

    Args:
        frame: BGR numpy array (modified in place)
        category_keys: list of category names to show
        masks_dict: dict {category: list_of_masks} for counts
        colors: color dict to use (defaults to BGR lowercase)
        use_lowercase: if True, use lowercase color lookup
    """
    if colors is None:
        colors = CATEGORY_COLORS_BGR_LOWER if use_lowercase else CATEGORY_COLORS_BGR
    default = DEFAULT_COLOR_BGR

    x, y = 10, 30
    for key in category_keys:
        masks = masks_dict.get(key, [])
        color = colors.get(key, default)
        if masks and isinstance(masks[0], dict) and masks[0].get("_gt"):
            label = f"{key}: GT"
        else:
            label = f"{key}: {len(masks)}"
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30
