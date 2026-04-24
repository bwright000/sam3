"""COCO RLE helpers for compact mask wire format."""

from typing import Any

import numpy as np
from pycocotools import mask as coco_mask


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Binary mask (H, W) uint8 -> COCO RLE dict with ascii-safe 'counts'."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    fortran = np.asfortranarray(mask)
    rle = coco_mask.encode(fortran)
    # pycocotools returns bytes for counts; make JSON-safe
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    """COCO RLE dict -> binary mask (H, W) uint8."""
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("ascii")
    return coco_mask.decode(r).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, min_area: int = 50, epsilon_frac: float = 0.002) -> list[list[float]]:
    """Binary mask -> list of COCO-format flat polygons [x,y,x,y,...].

    epsilon_frac controls Douglas-Peucker simplification (as fraction of contour arclength).
    """
    import cv2
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_frac * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        flat = approx.reshape(-1).astype(float).tolist()
        polygons.append(flat)
    return polygons


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """List of flat polygons [x,y,x,y,...] -> binary mask (H, W) uint8."""
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask
