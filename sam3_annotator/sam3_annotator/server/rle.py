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


def mask_to_polygons(mask: np.ndarray, min_area: int = 50,
                     epsilon_frac: float = 0.002,
                     epsilon_max_px: float | None = 2.0) -> list[list[float]]:
    """Binary mask -> list of COCO-format flat polygons [x,y,x,y,...].

    epsilon_frac controls Douglas-Peucker simplification (as fraction of contour
    arclength). For large-perimeter masks (e.g. tissue regions covering most of
    the frame) the raw 0.002 fraction produces aggressive smoothing — a
    10k-perim contour would get eps=20, leaving 8-vertex blobs that bear little
    resemblance to the original.

    `epsilon_max_px` clamps the absolute simplification error in pixels. The
    default of 2.0 is near-lossless visually (sub-pixel-pair detail is
    preserved). Pass a larger value (e.g. 5-10) only if you need a smaller
    polygon for storage or rendering at small scales, AND your downstream is
    tolerant of the simplification. Pass None to opt out entirely.
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
        if epsilon_max_px is not None:
            eps = min(eps, epsilon_max_px)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        flat = approx.reshape(-1).astype(float).tolist()
        polygons.append(flat)
    return polygons


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """List of flat polygons [x,y,x,y,...] -> binary mask (H, W) uint8.

    Each ring is filled solid independently. Use polygons_to_mask_even_odd
    when the input may contain hole rings that should subtract from outers.
    """
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygons_with_holes(mask: np.ndarray, min_area: int = 50,
                                 epsilon_frac: float = 0.002,
                                 epsilon_max_px: float | None = 2.0
                                 ) -> list[list[float]]:
    """Hole-aware variant of mask_to_polygons. Emits outer + child contours.

    Uses RETR_CCOMP so interior holes survive as separate rings. Rings come
    out in [outer1, outer1_hole_a, outer1_hole_b, outer2, ...] order; consumers
    should rasterise via polygons_to_mask_even_odd to recover the original
    binary mask (the parity-XOR semantics turn outer+hole into a donut).

    The outer ring is dropped when contourArea < min_area; holes are always
    kept (a 3-px hole still carries semantic meaning - e.g. tissue visible
    through a tool gap). Douglas-Peucker simplification uses the same eps
    logic as mask_to_polygons.
    """
    import cv2
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return []
    # hierarchy[0][i] = [next, prev, first_child, parent].
    # RETR_CCOMP gives a 2-level hierarchy: outers have parent=-1, holes have
    # parent>=0. Iterate by parent so each outer is followed by its holes.
    hier = hierarchy[0] if hierarchy is not None else []
    polygons: list[list[float]] = []

    def _simplify(cnt: np.ndarray) -> list[float] | None:
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, epsilon_frac * perim)
        if epsilon_max_px is not None:
            eps = min(eps, epsilon_max_px)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            return None
        return approx.reshape(-1).astype(float).tolist()

    for i, cnt in enumerate(contours):
        parent = int(hier[i][3]) if len(hier) > i else -1
        if parent != -1:
            continue  # holes are emitted with their outer below
        if cv2.contourArea(cnt) < min_area:
            continue
        flat = _simplify(cnt)
        if flat is None:
            continue
        polygons.append(flat)
        # walk children (holes) of this outer via the hierarchy
        child = int(hier[i][2]) if len(hier) > i else -1
        while child != -1:
            hole_flat = _simplify(contours[child])
            if hole_flat is not None:
                polygons.append(hole_flat)
            child = int(hier[child][0])
    return polygons


def polygons_to_mask_even_odd(polygons: list[list[float]], height: int,
                              width: int) -> np.ndarray:
    """Hole-aware rasteriser. Parity-XOR fill across rings.

    For a multi-ring polygon [outer, hole], each ring is rasterised in its
    own buffer and XOR'd into the accumulator. Outer adds; hole (drawn after)
    cancels the overlap region -> donut. This is the correct even-odd
    semantic; cv2.fillPoly in a single call across rings does NOT do this
    (it fills each ring solid independently).

    Single-ring inputs round-trip exactly the same as polygons_to_mask, so
    this is a safe drop-in for the legacy lossy polygon path too.
    """
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    tmp = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        tmp[:] = 0
        cv2.fillPoly(tmp, [pts], 1)
        mask ^= tmp
    return mask
