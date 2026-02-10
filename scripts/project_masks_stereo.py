"""
Project left-camera segmentation masks onto right-camera frames using
stereo calibration and disparity mapping.

The SAM3 segmentation pipeline produces masks only for the left camera.
For downstream stereo SLAM, masks are needed on both views. This script
projects left masks → right using dense disparity (computed via StereoSGBM
or pre-computed from SLAM).

Algorithm per frame:
  1. Load left + right images
  2. Compute or load dense disparity map
  3. For each left mask pixel (y, x): right pixel = (y, x - d)
  4. Morphological close to fill quantization gaps
  5. Convert back to COCO polygons

Usage:
    # Compute disparity internally (StereoSGBM)
    python scripts/project_masks_stereo.py \\
        --calibration stereo_calib.yaml \\
        --segments-dir data/Segments --episode C_1 --snippet 1

    # Use pre-computed disparity from SLAM
    python scripts/project_masks_stereo.py \\
        --calibration stereo_calib.yaml \\
        --segments-dir data/Segments --episode C_1 \\
        --disparity-dir outputs/slam_disparity/C_1/snippet_001

    # Test mode
    python scripts/project_masks_stereo.py \\
        --calibration stereo_calib.yaml \\
        --segments-dir data/Segments --episode C_1 --snippet 1 \\
        --test 5 --save-disparity
"""

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_tool_masks import (
    mask_to_coco_polygons,
    _render_overlay_from_results,
    _draw_legend,
    _stitch_snippet_video,
    CATEGORY_COLORS,
    DEFAULT_COLOR,
)


# ---------------------------------------------------------------------------
# Stereo calibration
# ---------------------------------------------------------------------------

@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters."""
    K1: np.ndarray          # 3x3 left intrinsic matrix
    K2: np.ndarray          # 3x3 right intrinsic matrix
    D1: np.ndarray          # left distortion coefficients
    D2: np.ndarray          # right distortion coefficients
    R: np.ndarray           # 3x3 rotation (left → right)
    T: np.ndarray           # 3x1 translation (left → right)
    image_size: Tuple[int, int]  # (width, height)
    # Derived from stereoRectify (populated if not assume_rectified)
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    map1_left: Optional[np.ndarray] = None
    map2_left: Optional[np.ndarray] = None
    map1_right: Optional[np.ndarray] = None
    map2_right: Optional[np.ndarray] = None
    is_rectified: bool = True


def load_stereo_calibration(calib_path: Path, assume_rectified: bool = True) -> StereoCalibration:
    """
    Load stereo calibration from a pickle, OpenCV YAML/JSON, or plain JSON file.

    Supported formats:
      - .pkl: CRCD calibration pickle with pre-computed rectification maps
             (ecm_left_rect_K, ecm_R, ecm_T, ecm_Q, ecm_map_left_x/y, etc.)
      - .yaml/.yml/.xml: OpenCV FileStorage with K1, K2, D1, D2, R, T
      - .json: Plain JSON with K1, K2, D1, D2, R, T arrays
    """
    calib_path = Path(calib_path)
    suffix = calib_path.suffix.lower()

    if suffix == ".pkl":
        # CRCD calibration pickle format
        with open(calib_path, "rb") as f:
            data = pickle.load(f)

        K1 = data["ecm_left_rect_K"]
        K2 = data["ecm_right_rect_K"]
        R = data["ecm_R"]
        T = data["ecm_T"]
        Q = data.get("ecm_Q")

        # Rectification maps are pre-computed in the pickle
        map1_left = data["ecm_map_left_x"]
        map2_left = data["ecm_map_left_y"]
        map1_right = data["ecm_map_right_x"]
        map2_right = data["ecm_map_right_y"]

        # Infer image size from map shape (H, W)
        h, w = map1_left.shape[:2]
        image_size = (w, h)

        # Extract distortion and projection matrices if available
        # (from mono calibration keys if present, otherwise None)
        D1 = data.get("caminfoL", {}).get("D", np.zeros(5))
        D2 = data.get("caminfoR", {}).get("D", np.zeros(5))
        if hasattr(D1, "flatten"):
            D1 = D1.flatten()
        if hasattr(D2, "flatten"):
            D2 = D2.flatten()

        P1 = data.get("ecm_left_rect_P")
        P2 = data.get("ecm_right_rect_P")

        calib = StereoCalibration(
            K1=K1, K2=K2, D1=D1, D2=D2, R=R, T=T,
            image_size=image_size,
            is_rectified=False,  # Raw frames need rectification via maps
            P1=P1, P2=P2, Q=Q,
            map1_left=map1_left, map2_left=map2_left,
            map1_right=map1_right, map2_right=map2_right,
        )
        print(f"  Loaded CRCD pickle with pre-computed rectification maps ({w}x{h})")
        return calib

    elif suffix in (".yaml", ".yml", ".xml"):
        # OpenCV FileStorage format
        fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
        K1 = fs.getNode("K1").mat()
        K2 = fs.getNode("K2").mat()
        D1 = fs.getNode("D1").mat()
        D2 = fs.getNode("D2").mat()
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat()
        sz_node = fs.getNode("image_size")
        if sz_node.isSeq():
            image_size = (int(sz_node.at(0).real()), int(sz_node.at(1).real()))
        else:
            image_size = (1280, 720)
        fs.release()
    else:
        # Plain JSON format
        with open(calib_path, "r") as f:
            data = json.load(f)
        K1 = np.array(data["K1"], dtype=np.float64).reshape(3, 3)
        K2 = np.array(data["K2"], dtype=np.float64).reshape(3, 3)
        D1 = np.array(data["D1"], dtype=np.float64).flatten()
        D2 = np.array(data["D2"], dtype=np.float64).flatten()
        R = np.array(data["R"], dtype=np.float64).reshape(3, 3)
        T = np.array(data["T"], dtype=np.float64).reshape(3, 1)
        image_size = tuple(data.get("image_size", [1280, 720]))

    # Validate dimensions
    assert K1.shape == (3, 3), f"K1 shape {K1.shape} != (3, 3)"
    assert K2.shape == (3, 3), f"K2 shape {K2.shape} != (3, 3)"
    assert R.shape == (3, 3), f"R shape {R.shape} != (3, 3)"
    assert T.shape == (3, 1), f"T shape {T.shape} != (3, 1)"

    calib = StereoCalibration(
        K1=K1, K2=K2, D1=D1, D2=D2, R=R, T=T,
        image_size=image_size, is_rectified=assume_rectified,
    )

    # Compute rectification maps if images are NOT already rectified
    if not assume_rectified:
        w, h = image_size
        calib.R1, calib.R2, calib.P1, calib.P2, calib.Q, _, _ = (
            cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, alpha=0)
        )
        calib.map1_left, calib.map2_left = cv2.initUndistortRectifyMap(
            K1, D1, calib.R1, calib.P1, (w, h), cv2.CV_32FC1
        )
        calib.map1_right, calib.map2_right = cv2.initUndistortRectifyMap(
            K2, D2, calib.R2, calib.P2, (w, h), cv2.CV_32FC1
        )
        print(f"  Computed stereoRectify maps for {w}x{h}")

    return calib


# ---------------------------------------------------------------------------
# Disparity computation
# ---------------------------------------------------------------------------

def compute_disparity_sgbm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """
    Compute dense disparity map using StereoSGBM.

    Returns float32 disparity map where invalid pixels = -1.0.
    """
    # SGBM parameters tuned for surgical endoscope (~5mm baseline, close field)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # SGBM returns int16 scaled by 16
    raw = stereo.compute(left_gray, right_gray)
    disparity = raw.astype(np.float32) / 16.0

    # Mark invalid pixels (SGBM returns minDisparity-1 for invalid)
    disparity[raw == -16] = -1.0
    disparity[disparity < 0] = -1.0

    # Post-filter: try WLS filter (opencv-contrib), fall back to median blur
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        raw_right = right_matcher.compute(right_gray, left_gray)
        wls = cv2.ximgproc.createDisparityWLSFilter(stereo)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        filtered = wls.filter(raw, left_gray, disparity_map_right=raw_right)
        disparity = filtered.astype(np.float32) / 16.0
        disparity[filtered == -16] = -1.0
        disparity[disparity < 0] = -1.0
    except AttributeError:
        # cv2.ximgproc not available — use median blur as fallback
        valid = disparity > 0
        if valid.any():
            med = cv2.medianBlur(
                disparity.astype(np.float32), 5
            )
            # Only apply median where we had valid disparity
            disparity[valid] = med[valid]

    return disparity


def load_precomputed_disparity(disparity_dir: Path, frame_name: str) -> Optional[np.ndarray]:
    """
    Load pre-computed disparity map from SLAM pipeline.

    Supports .npy (numpy) and .png (16-bit depth scaled by 256).
    Returns float32 disparity or None if not found.
    """
    # Try .npy first
    npy_path = disparity_dir / f"{frame_name}.npy"
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)

    # Try 16-bit PNG (disparity = pixel_value / 256.0)
    png_path = disparity_dir / f"{frame_name}.png"
    if png_path.exists():
        raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.dtype == np.uint16:
            disparity = raw.astype(np.float32) / 256.0
            disparity[raw == 0] = -1.0
            return disparity

    return None


# ---------------------------------------------------------------------------
# Mask projection
# ---------------------------------------------------------------------------

def project_mask_to_right(
    mask_binary: np.ndarray,
    disparity: np.ndarray,
    height: int,
    width: int,
    min_disparity: float = 0.5,
    morph_close_size: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Project a binary mask from left to right camera using disparity.

    For rectified stereo: right_x = left_x - disparity.

    Returns:
        right_mask: (H, W) uint8 binary mask in right camera frame
        stats: projection statistics
    """
    ys, xs = np.where(mask_binary > 0)
    total_pixels = len(ys)

    if total_pixels == 0:
        return np.zeros((height, width), dtype=np.uint8), {
            "total_pixels": 0, "valid_pixels": 0, "coverage": 0.0,
        }

    # Look up disparity at mask pixel locations
    ds = disparity[ys, xs]

    # Filter invalid disparity
    valid = ds >= min_disparity
    valid_count = int(valid.sum())

    if valid_count == 0:
        return np.zeros((height, width), dtype=np.uint8), {
            "total_pixels": total_pixels, "valid_pixels": 0, "coverage": 0.0,
        }

    # Compute right-image x coordinates
    xs_right = np.round(xs[valid] - ds[valid]).astype(np.int32)
    ys_valid = ys[valid]

    # Clip to image bounds
    in_bounds = (xs_right >= 0) & (xs_right < width)
    xs_right = xs_right[in_bounds]
    ys_valid = ys_valid[in_bounds]
    projected_count = len(xs_right)

    # Build right-camera binary mask
    right_mask = np.zeros((height, width), dtype=np.uint8)
    if projected_count > 0:
        right_mask[ys_valid, xs_right] = 1

    # Morphological close to fill small gaps from disparity quantization
    if morph_close_size > 0 and projected_count > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)
        )
        right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_CLOSE, kernel)

    coverage = valid_count / total_pixels if total_pixels > 0 else 0.0

    stats = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_count,
        "projected_pixels": projected_count,
        "coverage": round(coverage, 3),
    }

    return right_mask, stats


def project_frame_masks(
    frame_result: dict,
    disparity: np.ndarray,
    height: int,
    width: int,
    prompts: List[str],
    min_disparity: float = 0.5,
    morph_close_size: int = 3,
    min_area: int = 100,
) -> Tuple[dict, dict]:
    """
    Project all masks for a single frame from left to right camera.

    Returns:
        right_result: dict in the same format as frame_result but for right camera
        frame_stats: per-prompt projection statistics
    """
    right_result = {
        "frame": frame_result["frame"],
        "height": height,
        "width": width,
        "masks": {},
    }

    # Carry forward avg_confidence if present
    if "avg_confidence" in frame_result:
        right_result["avg_confidence"] = frame_result["avg_confidence"]

    frame_stats = {}

    for prompt in prompts:
        right_masks_list = []
        prompt_stats = []

        for md in frame_result.get("masks", {}).get(prompt, []):
            # Reconstruct binary mask from COCO polygons
            left_mask = np.zeros((height, width), dtype=np.uint8)
            for poly in md.get("segmentation", []):
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(left_mask, [pts], 1)

            # Project to right camera
            right_mask, stats = project_mask_to_right(
                left_mask, disparity, height, width,
                min_disparity, morph_close_size,
            )

            new_area = float(right_mask.sum())
            if new_area < min_area:
                prompt_stats.append({**stats, "kept": False, "area": new_area})
                continue

            # Convert to COCO polygons
            polygons = mask_to_coco_polygons(right_mask * 255)
            if not polygons:
                prompt_stats.append({**stats, "kept": False, "area": new_area})
                continue

            # Compute bbox from projected mask
            ys, xs = np.where(right_mask > 0)
            bbox = [
                float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min()),
            ]

            right_masks_list.append({
                "segmentation": polygons,
                "area": new_area,
                "score": md.get("score", 0.0),
                "bbox": bbox,
                "projection_coverage": stats["coverage"],
            })
            prompt_stats.append({**stats, "kept": True, "area": new_area})

        right_result["masks"][prompt] = right_masks_list
        frame_stats[prompt] = prompt_stats

    return right_result, frame_stats


# ---------------------------------------------------------------------------
# Frame matching
# ---------------------------------------------------------------------------

def build_frame_pairs(
    snippet_dir: Path,
    left_results: List[dict],
) -> List[Tuple[int, Path, Path, dict]]:
    """
    Match left-camera result frames to right-camera frames by frame number.

    Handles different file extensions and sampling rates (e.g. E_3 right
    has every-10th-frame sampling).

    Returns list of (frame_number, left_frame_path, right_frame_path, result_entry).
    """
    frames_left_dir = snippet_dir / "frames_left"
    frames_right_dir = snippet_dir / "frames_right"

    if not frames_right_dir.exists():
        print(f"  WARNING: frames_right/ not found in {snippet_dir}")
        return []

    # Build right-frame lookup: frame_number -> path
    right_lookup = {}
    for ext in ("*.webp", "*.jpg", "*.png"):
        for fp in frames_right_dir.glob(ext):
            try:
                fnum = int(fp.stem.split("_")[1])
                right_lookup[fnum] = fp
            except (IndexError, ValueError):
                continue

    # Build left-frame lookup
    left_lookup = {}
    for ext in ("*.webp", "*.jpg", "*.png"):
        for fp in frames_left_dir.glob(ext):
            try:
                fnum = int(fp.stem.split("_")[1])
                left_lookup[fnum] = fp
            except (IndexError, ValueError):
                continue

    # Match results to right frames
    pairs = []
    for result in left_results:
        frame_name = result["frame"]  # e.g. "frame_002130"
        try:
            fnum = int(frame_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        left_path = left_lookup.get(fnum)
        right_path = right_lookup.get(fnum)

        if left_path is not None and right_path is not None:
            pairs.append((fnum, left_path, right_path, result))

    print(f"  Matched {len(pairs)} of {len(left_results)} result frames "
          f"to right camera ({len(right_lookup)} right frames available)")

    return sorted(pairs, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Per-snippet processing
# ---------------------------------------------------------------------------

def process_snippet_stereo(
    snippet_dir: Path,
    left_results: List[dict],
    calib: StereoCalibration,
    output_dir: Path,
    prompts: List[str],
    disparity_dir: Optional[Path] = None,
    num_disparities: int = 128,
    block_size: int = 5,
    min_area: int = 100,
    morph_close_size: int = 3,
    max_frames: Optional[int] = None,
    render_overlays: bool = True,
    save_disparity: bool = False,
):
    """
    Project all left-camera masks onto right-camera frames for a snippet.
    """
    # Build frame pairs
    pairs = build_frame_pairs(snippet_dir, left_results)
    if not pairs:
        print("  No frame pairs found — skipping")
        return

    if max_frames:
        pairs = pairs[:max_frames]

    # Create output directories
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if save_disparity:
        disp_dir = output_dir / "disparity"
        disp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    right_results = []
    all_stats = []

    w, h = calib.image_size

    for i, (fnum, left_path, right_path, result) in enumerate(pairs):
        frame_start = time.time()

        # Load images
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        if left_img is None or right_img is None:
            print(f"  WARNING: Could not read frame pair {fnum}")
            continue

        # Apply rectification if needed
        if not calib.is_rectified and calib.map1_left is not None:
            left_img = cv2.remap(
                left_img, calib.map1_left, calib.map2_left, cv2.INTER_LINEAR
            )
            right_img = cv2.remap(
                right_img, calib.map1_right, calib.map2_right, cv2.INTER_LINEAR
            )

        # Get disparity
        frame_name = result["frame"]
        if disparity_dir:
            disparity = load_precomputed_disparity(disparity_dir, frame_name)
            if disparity is None:
                print(f"  WARNING: No disparity for {frame_name}, computing SGBM")
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                disparity = compute_disparity_sgbm(
                    left_gray, right_gray, num_disparities, block_size
                )
        else:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            disparity = compute_disparity_sgbm(
                left_gray, right_gray, num_disparities, block_size
            )

        # Save disparity if requested
        if save_disparity:
            np.save(str(disp_dir / f"{frame_name}.npy"), disparity)

        # Project masks
        img_h, img_w = left_img.shape[:2]
        right_result, frame_stats = project_frame_masks(
            result, disparity, img_h, img_w, prompts,
            min_disparity=0.5,
            morph_close_size=morph_close_size,
            min_area=min_area,
        )
        right_results.append(right_result)
        all_stats.append({"frame": frame_name, "stats": frame_stats})

        # Check for low coverage warnings
        for prompt, pstats in frame_stats.items():
            for ms in pstats:
                if ms.get("coverage", 1.0) < 0.3 and ms.get("total_pixels", 0) > 0:
                    print(f"  WARNING: {frame_name} {prompt} low disparity coverage "
                          f"({ms['coverage']:.0%})")

        # Render overlay on right frame
        if render_overlays:
            _, overlay, _ = _render_overlay_from_results(
                right_path, right_result, prompts
            )
            cv2.imwrite(str(overlays_dir / f"{frame_name}.jpg"), overlay)

        # Progress
        elapsed = time.time() - frame_start
        n_masks = sum(
            len(right_result["masks"].get(p, [])) for p in prompts
        )
        total_elapsed = time.time() - t0
        fps = (i + 1) / total_elapsed if total_elapsed > 0 else 0
        remaining = (len(pairs) - i - 1) / fps if fps > 0 else 0
        print(f"  [{i+1}/{len(pairs)}] {frame_name}: {n_masks} masks projected "
              f"({elapsed:.2f}s, ETA {remaining:.0f}s)")

    # Stitch overlay video
    if render_overlays and right_results:
        video_path = output_dir / f"{snippet_dir.name}_right_overlay.mp4"
        _stitch_snippet_video(overlays_dir, video_path, fps=60)

    # Save right-camera results JSON
    results_path = output_dir / f"{snippet_dir.name}_right_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "source": "stereo_projection",
            "snippet": snippet_dir.name,
            "prompts": prompts,
            "num_frames": len(right_results),
            "frames": right_results,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")

    # Save projection stats
    stats_path = output_dir / "projection_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Summary
    total_time = time.time() - t0
    print(f"\n  Projection done: {len(right_results)} frames in {total_time:.1f}s "
          f"({total_time/max(len(right_results),1):.2f}s/frame)")
    for prompt in prompts:
        total_masks = sum(
            len(r["masks"].get(prompt, [])) for r in right_results
        )
        frames_with = sum(
            1 for r in right_results if len(r["masks"].get(prompt, [])) > 0
        )
        print(f"    {prompt}: {total_masks} masks across "
              f"{frames_with}/{len(right_results)} frames")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Project left-camera masks onto right-camera frames "
                    "using stereo calibration and disparity"
    )

    # Required
    parser.add_argument(
        "--calibration", "-c", required=True,
        help="Path to stereo calibration file (OpenCV YAML/JSON or plain JSON)",
    )
    parser.add_argument(
        "--segments-dir", required=True,
        help="Path to Segments directory",
    )
    parser.add_argument(
        "--episode", required=True,
        help="Episode name (C_1, E_3, F_3)",
    )

    # Snippet selection
    parser.add_argument(
        "--snippet", type=int, default=None,
        help="Specific snippet number (default: all snippets)",
    )

    # Results source
    parser.add_argument(
        "--results-dir", default="outputs/segments_video",
        help="Directory containing left-camera results JSONs "
             "(default: outputs/segments_video)",
    )

    # Disparity source
    parser.add_argument(
        "--disparity-dir", default=None,
        help="Pre-computed disparity maps directory (default: compute via SGBM)",
    )

    # SGBM parameters
    parser.add_argument(
        "--num-disparities", type=int, default=128,
        help="SGBM max disparity range (default: 128, must be multiple of 16)",
    )
    parser.add_argument(
        "--block-size", type=int, default=5,
        help="SGBM block size (default: 5)",
    )

    # Rectification
    parser.add_argument(
        "--assume-rectified", action="store_true", default=True,
        help="Assume images are already rectified (default: True for da Vinci)",
    )
    parser.add_argument(
        "--no-assume-rectified", dest="assume_rectified", action="store_false",
        help="Apply stereoRectify before disparity computation",
    )

    # Projection parameters
    parser.add_argument(
        "--min-area", type=int, default=100,
        help="Minimum projected mask area to keep (default: 100)",
    )
    parser.add_argument(
        "--morph-close", type=int, default=3,
        help="Morphological close kernel size (default: 3, 0=disable)",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o", default="outputs/segments_stereo",
        help="Output directory (default: outputs/segments_stereo)",
    )
    parser.add_argument(
        "--render-overlays", action="store_true", default=True,
        help="Render overlay images on right frames (default: True)",
    )
    parser.add_argument(
        "--no-render-overlays", dest="render_overlays", action="store_false",
    )
    parser.add_argument(
        "--save-disparity", action="store_true",
        help="Save computed disparity maps as .npy files",
    )

    # Pipeline patterns
    parser.add_argument(
        "--test", type=int, default=None, metavar="N",
        help="Process only N frames per snippet (test mode)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip snippets with existing right-camera results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)

    # --- Load calibration ---
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        sys.exit(1)

    print("Loading stereo calibration...")
    calib = load_stereo_calibration(calib_path, args.assume_rectified)
    print(f"  Image size: {calib.image_size}")
    print(f"  Baseline: {np.linalg.norm(calib.T):.2f}mm")
    print(f"  Rectified: {'assumed' if args.assume_rectified else 'computed'}")

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

    # --- Header ---
    print("=" * 60)
    print("Stereo Mask Projection")
    print("=" * 60)
    print(f"Episode: {args.episode}")
    print(f"Snippets: {len(snippet_list)}")
    print(f"Results source: {results_dir}")
    print(f"Output: {output_dir}")
    disparity_mode = "pre-computed" if args.disparity_dir else "SGBM"
    print(f"Disparity: {disparity_mode}")
    if not args.disparity_dir:
        print(f"  num_disparities={args.num_disparities}, block_size={args.block_size}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")
    if args.resume:
        print(f"Resume: enabled")

    # --- Process each snippet ---
    t_total = time.time()
    for snip_dir in snippet_list:
        snippet_id = snip_dir.name

        snip_output = output_dir / args.episode / snippet_id

        # Resume check
        results_path = snip_output / f"{snippet_id}_right_results.json"
        if args.resume and results_path.exists():
            print(f"\n  Skipping {snippet_id} (results exist, --resume)")
            continue

        # Load left-camera results
        left_results_path = results_dir / args.episode / snippet_id / f"{snippet_id}_results.json"
        if not left_results_path.exists():
            print(f"\n  Skipping {snippet_id}: no left results at {left_results_path}")
            continue

        with open(left_results_path, "r") as f:
            left_data = json.load(f)

        # Handle both formats:
        #   Video pipeline: {"frames": [...], "prompts": [...], "tissue_categories": [...]}
        #   Image pipeline: [{frame1}, {frame2}, ...]
        if isinstance(left_data, list):
            left_results = left_data
            # Infer prompts from first frame's mask keys
            if left_results and "masks" in left_results[0]:
                prompts = list(left_results[0]["masks"].keys())
            else:
                prompts = ["tool"]
        else:
            left_results = left_data.get("frames", [])
            # Combine tool prompts + tissue categories for full projection
            prompts = left_data.get("prompts", ["tool"])
            tissue_cats = left_data.get("tissue_categories", [])
            # Auto-detect all mask keys across frames for complete coverage
            all_keys = set(prompts) | set(tissue_cats)
            for fr in left_results:
                all_keys.update(fr.get("masks", {}).keys())
            prompts = list(all_keys)

        if not left_results:
            print(f"\n  Skipping {snippet_id}: empty left results")
            continue

        print(f"\n{'=' * 60}")
        print(f"Episode: {args.episode} / {snippet_id}")
        print(f"{'=' * 60}")
        print(f"  Left results: {len(left_results)} frames, prompts={prompts}")

        # Disparity directory for this snippet
        disp_dir = None
        if args.disparity_dir:
            disp_dir = Path(args.disparity_dir)
            if not disp_dir.exists():
                print(f"  WARNING: disparity-dir not found, falling back to SGBM")
                disp_dir = None

        process_snippet_stereo(
            snippet_dir=snip_dir,
            left_results=left_results,
            calib=calib,
            output_dir=snip_output,
            prompts=prompts,
            disparity_dir=disp_dir,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            min_area=args.min_area,
            morph_close_size=args.morph_close,
            max_frames=args.test,
            render_overlays=args.render_overlays,
            save_disparity=args.save_disparity,
        )

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"All done! {len(snippet_list)} snippets in {total_time:.1f}s")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
