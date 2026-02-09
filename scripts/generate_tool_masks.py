"""
Generate segmentation masks using SAM3.

Two modes:
  1. Box-prompt mode (instrument datasets): Feed existing COCO bounding boxes
     as box prompts to generate tool segmentation polygons.
  2. Snippet mode (Segments clips): Run SAM3 text prompts on short video
     snippets to segment liver, gallbladder, and tools.

Works on CPU or GTX 970 (4GB VRAM) by processing one image at a time.

Usage:
    # --- Snippet mode (recommended) ---
    # Process a single snippet
    python scripts/generate_tool_masks.py --segments-dir data/Segments --episode C_1 --snippet 1

    # Process all snippets for an episode
    python scripts/generate_tool_masks.py --segments-dir data/Segments --episode C_1

    # Custom prompts
    python scripts/generate_tool_masks.py --segments-dir data/Segments --episode C_1 --snippet 1 --prompts liver gallbladder tool

    # --- Box-prompt mode (instrument datasets) ---
    python scripts/generate_tool_masks.py --dataset-dir "path/to/FBF_1"
    python scripts/generate_tool_masks.py --dataset-dir "path/to/instrument_keypoint" --all
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add parent directory to path for sam3 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.model_loader import load_sam3_model, get_processor
from scripts.device_config import configure_device
from scripts.check_annotations import COCOAnnotationLoader


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_images(dataset_dir: Path) -> Path:
    """Extract images.zip if not already extracted. Returns images directory path."""
    images_dir = dataset_dir / "images"
    zip_path = dataset_dir / "images.zip"

    if images_dir.exists() and any(images_dir.iterdir()):
        count = len(list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.png")))
        print(f"  Images already extracted: {count} files in {images_dir}")
        return images_dir

    if not zip_path.exists():
        print(f"  ERROR: No images.zip found at {zip_path}")
        sys.exit(1)

    print(f"  Extracting {zip_path.name}...")
    start = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    elapsed = time.time() - start
    print(f"  Extracted in {elapsed:.1f}s")

    # Check for nested extraction (images/images/)
    nested = images_dir / "images"
    if nested.exists():
        import shutil
        for f in nested.iterdir():
            shutil.move(str(f), str(images_dir / f.name))
        nested.rmdir()

    count = len(list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.png")))
    print(f"  {count} image files found")
    return images_dir


def load_instrument_annotations(json_path: Path) -> dict:
    """
    Load COCO-format instrument keypoint annotations.

    Returns dict with:
        images: {image_id: image_info}
        annotations: {image_id: [annotation_list]}
        categories: {category_id: category_info}
        raw: original JSON data
    """
    print(f"  Loading {json_path.name} ({json_path.stat().st_size / 1e6:.1f} MB)...")
    start = time.time()
    with open(json_path, "r") as f:
        data = json.load(f)
    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s")

    # Index images by id
    images = {img["id"]: img for img in data["images"]}

    # Index annotations by image_id
    annotations = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    # Index categories
    categories = {cat["id"]: cat for cat in data["categories"]}

    cat_names = ", ".join(c["name"] for c in data["categories"])
    print(f"  {len(images)} images, {len(data['annotations'])} annotations")
    print(f"  Categories: {cat_names}")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# Bbox conversion
# ---------------------------------------------------------------------------

def coco_bbox_to_sam3(bbox: list, img_w: int, img_h: int) -> list:
    """
    Convert COCO bbox to SAM3 normalized box prompt format.

    COCO:  [x, y, width, height] in absolute pixels (top-left corner)
    SAM3:  [center_x, center_y, width, height] normalized [0, 1]
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return [cx, cy, nw, nh]


# ---------------------------------------------------------------------------
# Mask to polygon conversion
# ---------------------------------------------------------------------------

def mask_to_coco_polygons(binary_mask: np.ndarray, min_area: int = 100) -> list:
    """
    Convert a binary mask to COCO polygon segmentation format.

    Args:
        binary_mask: 2D uint8 array (H, W), values 0 or 1/255
        min_area: Minimum contour area to keep (filters noise)

    Returns:
        List of polygon coordinate lists: [[x1,y1,x2,y2,...], ...]
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0.5).astype(np.uint8)
    if binary_mask.max() == 1:
        binary_mask = binary_mask * 255

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # Flatten to [x1, y1, x2, y2, ...] format
        coords = contour.flatten().tolist()
        if len(coords) >= 6:  # Need at least 3 points
            polygons.append(coords)

    return polygons


def compute_mask_area(binary_mask: np.ndarray) -> float:
    """Compute area of binary mask in pixels."""
    return float(np.sum(binary_mask > 0.5))


# ---------------------------------------------------------------------------
# SAM3 processing
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_dir: Path,
    processor,
    device: str,
    split: str = "train",
    output_dir: Path = None,
    confidence: float = 0.3,
    resume: bool = False,
    max_images: int = None,
):
    """
    Process a single instrument dataset with SAM3 box prompts.

    Args:
        max_images: If set, only process this many images (for --test mode).

    Returns the enriched COCO data with segmentation polygons added.
    """
    dataset_name = dataset_dir.name
    json_path = dataset_dir / f"{split}.json"

    if not json_path.exists():
        print(f"  WARNING: {json_path} not found, skipping")
        return None

    # Load annotations
    data = load_instrument_annotations(json_path)
    images_dir = extract_images(dataset_dir)

    # Set confidence threshold
    processor.set_confidence_threshold(confidence)

    # Check for resume
    output_json = None
    processed_ids = set()
    if output_dir and resume:
        output_json = output_dir / f"{dataset_name}_{split}_masks.json"
        if output_json.exists():
            with open(output_json, "r") as f:
                existing = json.load(f)
            # Find already-processed annotations (those with sam3_score)
            for ann in existing["annotations"]:
                if "sam3_score" in ann:
                    processed_ids.add(ann["id"])
            print(f"  Resuming: {len(processed_ids)} annotations already processed")

    # Process each image
    raw_annotations = data["raw"]["annotations"]
    total_all = len(raw_annotations)
    # Apply max_images limit (for --test mode)
    if max_images is not None:
        total_to_process = min(max_images, total_all)
    else:
        total_to_process = total_all
    skipped = 0
    failed = 0
    processed_count = 0
    start_time = time.time()

    print(f"\n  Processing {total_to_process} of {total_all} annotations from {dataset_name}/{split}...")
    if max_images is not None:
        print(f"  (Test mode: limited to {max_images} images)")

    for i, ann in enumerate(raw_annotations):
        # Stop if we've processed enough (test mode)
        if processed_count >= total_to_process:
            break

        ann_id = ann["id"]

        # Skip if already processed (resume mode)
        if ann_id in processed_ids:
            skipped += 1
            continue

        img_id = ann["image_id"]
        img_info = data["images"].get(img_id)
        if img_info is None:
            print(f"  WARNING: No image info for image_id={img_id}")
            failed += 1
            continue

        # Resolve image path
        file_name = img_info["file_name"]
        # Strip leading ./ if present
        if file_name.startswith("./"):
            file_name = file_name[2:]
        img_path = dataset_dir / file_name

        if not img_path.exists():
            # Try case-insensitive match
            parent = img_path.parent
            name_lower = img_path.name.lower()
            matches = [f for f in parent.iterdir() if f.name.lower() == name_lower]
            if matches:
                img_path = matches[0]
            else:
                if i < 3:  # Only warn for first few
                    print(f"  WARNING: Image not found: {img_path}")
                failed += 1
                continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        try:
            img_start = time.time()

            # Load image
            pil_image = Image.open(img_path).convert("RGB")

            # Convert bbox to SAM3 format
            sam3_box = coco_bbox_to_sam3(ann["bbox"], img_w, img_h)

            # Run SAM3
            state = processor.set_image(pil_image)
            state = processor.add_geometric_prompt(
                box=sam3_box,
                label=True,
                state=state,
            )

            masks = state.get("masks")
            scores = state.get("scores")

            if masks is not None and len(masks) > 0:
                # Take the best mask (highest score)
                best_idx = scores.argmax().item() if len(scores) > 1 else 0
                best_mask = masks[best_idx].cpu().numpy()
                best_score = scores[best_idx].item()

                # Squeeze to 2D
                if best_mask.ndim == 3:
                    best_mask = best_mask.squeeze()

                # Convert to polygons
                mask_uint8 = (best_mask > 0.5).astype(np.uint8)
                polygons = mask_to_coco_polygons(mask_uint8)
                area = compute_mask_area(mask_uint8)

                # Update annotation in-place
                ann["segmentation"] = polygons
                ann["area"] = area
                ann["sam3_score"] = round(best_score, 4)
            else:
                ann["segmentation"] = []
                ann["sam3_score"] = 0.0
                failed += 1

            # Reset for next image
            processor.reset_all_prompts(state)

            img_elapsed = time.time() - img_start

            # Per-image timing in test mode
            if max_images is not None:
                score_str = f"score={ann.get('sam3_score', 0):.3f}"
                print(f"  [{processed_count + 1}/{total_to_process}] {img_path.name} | {img_elapsed:.1f}s | {score_str}")

        except Exception as e:
            print(f"  ERROR processing image {img_path.name}: {e}")
            ann["segmentation"] = []
            ann["sam3_score"] = 0.0
            failed += 1

        processed_count += 1

        # Progress (batch mode only — test mode prints per-image above)
        if max_images is None and processed_count > 0 and (processed_count % 10 == 0 or processed_count == total_to_process):
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = total_to_process - processed_count
            eta = remaining / rate if rate > 0 else 0
            print(
                f"  [{processed_count}/{total_to_process}] "
                f"{100 * processed_count / total_to_process:.1f}% "
                f"| {rate:.1f} img/s "
                f"| ETA: {eta / 60:.1f}min"
            )

        # Periodic save (every 50 images, or every image in test mode)
        save_interval = 1 if max_images is not None else 50
        if output_dir and processed_count > 0 and processed_count % save_interval == 0:
            _save_output(data["raw"], output_dir, dataset_name, split)

    # Final stats
    elapsed = time.time() - start_time
    success = processed_count - failed
    print(f"\n  Done: {success} masks generated, {failed} failed, {skipped} skipped (resumed)")
    print(f"  Total time: {elapsed / 60:.1f} minutes")

    # Compute score stats
    scores_list = [
        a["sam3_score"] for a in raw_annotations if "sam3_score" in a and a["sam3_score"] > 0
    ]
    if scores_list:
        print(
            f"  SAM3 scores: mean={np.mean(scores_list):.3f}, "
            f"min={np.min(scores_list):.3f}, max={np.max(scores_list):.3f}"
        )

    return data["raw"]


def _save_output(coco_data: dict, output_dir: Path, dataset_name: str, split: str):
    """Save enriched COCO JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_{split}_masks.json"
    with open(output_path, "w") as f:
        json.dump(coco_data, f)
    print(f"  Saved progress to {output_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_sample_frames(
    dataset_dir: Path,
    coco_data: dict,
    output_dir: Path,
    num_samples: int = 10,
):
    """Render sample frames with generated masks overlaid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index images
    images = {img["id"]: img for img in coco_data["images"]}
    categories = {cat["id"]: cat for cat in coco_data["categories"]}

    # Filter annotations that have segmentation
    annotated = [a for a in coco_data["annotations"] if a.get("segmentation")]
    if not annotated:
        print("  No annotations with segmentation found for visualization")
        return

    # Sample evenly
    step = max(1, len(annotated) // num_samples)
    samples = annotated[::step][:num_samples]

    # Color palette (BGR)
    tool_colors = {
        "FBF": (255, 128, 0),    # Blue-ish
        "PCH": (0, 128, 255),    # Orange-ish
    }
    default_color = (255, 0, 0)  # Blue

    print(f"\n  Rendering {len(samples)} sample frames...")

    for idx, ann in enumerate(samples):
        img_info = images[ann["image_id"]]
        file_name = img_info["file_name"]
        if file_name.startswith("./"):
            file_name = file_name[2:]
        img_path = dataset_dir / file_name

        if not img_path.exists():
            # Case-insensitive fallback
            parent = img_path.parent
            name_lower = img_path.name.lower()
            matches = [f for f in parent.iterdir() if f.name.lower() == name_lower]
            if matches:
                img_path = matches[0]
            else:
                continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        # Get category color
        cat_name = categories.get(ann["category_id"], {}).get("name", "")
        color = tool_colors.get(cat_name, default_color)
        color_np = np.array(color, dtype=np.uint8)

        # Draw mask from segmentation polygons
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 255)

        mask_bool = mask > 0

        # Alpha blending (25% mask color)
        alpha = 0.25
        overlay = img_bgr.copy()
        overlay[mask_bool] = color_np
        frame = cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)

        # Triple-layer contours (SAM3 demo style)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 7)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(frame, contours, -1, color, 3)

        # Draw original bbox for comparison
        bx, by, bw, bh = [int(v) for v in ann["bbox"]]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

        # Label
        score = ann.get("sam3_score", 0)
        label = f"{cat_name} score={score:.2f}"
        cv2.putText(frame, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save
        out_path = output_dir / f"sample_{idx:03d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), frame)

    print(f"  Saved {len(samples)} visualizations to {output_dir}")


# ---------------------------------------------------------------------------
# GT annotation loading (tissue segmentation datasets)
# ---------------------------------------------------------------------------

def _load_episode_annotations(tissue_seg_dir: Path, episode: str):
    """
    Load and merge ALL GT annotation files for an episode (train + test + annotations).
    Returns a COCOAnnotationLoader with full continuous ID coverage, or None.
    """
    ep_dir = Path(tissue_seg_dir) / episode
    if not ep_dir.exists():
        return None

    json_files = sorted(ep_dir.glob("*.json"))
    if not json_files:
        return None

    # Load first file as base
    loader = COCOAnnotationLoader(str(json_files[0]), str(ep_dir))
    loader.load()

    # Merge remaining files
    for jf in json_files[1:]:
        _merge_annotations_into(loader, jf)

    print(f"  GT annotations for {episode}: {len(loader.images)} frames total")
    return loader


def _merge_annotations_into(loader: COCOAnnotationLoader, json_path: Path):
    """Merge an additional annotation JSON file into an existing loader."""
    print(f"    Merging {json_path.name}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    added_imgs = 0
    for img in data.get("images", []):
        if img["id"] not in loader.images:
            loader.images[img["id"]] = img
            loader.file_to_id[img["file_name"]] = img["id"]
            added_imgs += 1
    for ann in data.get("annotations", []):
        loader.annotations[ann["image_id"]].append(ann)
    print(f"      +{added_imgs} new images")


# ---------------------------------------------------------------------------
# Snippet processing (Segments mode)
# ---------------------------------------------------------------------------

# Color palette for text-prompted categories (BGR)
CATEGORY_COLORS = {
    "liver": (0, 0, 255),        # Red
    "gallbladder": (0, 255, 0),  # Green
    "tool": (255, 128, 0),       # Blue
}
DEFAULT_COLOR = (0, 255, 255)    # Yellow


def _run_frame_inference(processor, frame_path, prompts, min_area):
    """Run SAM3 text prompts on a single frame. Returns (frame_result, overlay_frame)."""
    pil_image = Image.open(frame_path).convert("RGB")
    img_w, img_h = pil_image.size

    with torch.inference_mode():
        state = processor.set_image(pil_image)

    frame_result = {
        "frame": frame_path.stem,
        "file": str(frame_path.name),
        "width": img_w,
        "height": img_h,
        "masks": {},
    }

    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    overlay_frame = img_bgr.copy()
    filtered = 0

    for prompt in prompts:
        with torch.inference_mode():
            state = processor.set_text_prompt(prompt, state)
        masks = state.get("masks")
        scores = state.get("scores")

        prompt_masks = []
        if masks is not None and len(masks) > 0:
            for m_idx in range(len(masks)):
                mask_np = masks[m_idx].cpu().numpy()
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()
                score = scores[m_idx].item()

                mask_uint8 = (mask_np > 0.5).astype(np.uint8)
                area = compute_mask_area(mask_uint8)

                # Min-area filter
                if area < min_area:
                    filtered += 1
                    continue

                polygons = mask_to_coco_polygons(mask_uint8)
                prompt_masks.append({
                    "score": round(score, 4),
                    "area": area,
                    "segmentation": polygons,
                })

                # Draw on overlay
                color = CATEGORY_COLORS.get(prompt, DEFAULT_COLOR)
                mask_bool = mask_uint8 > 0
                color_np = np.array(color, dtype=np.uint8)
                color_overlay = overlay_frame.copy()
                color_overlay[mask_bool] = color_np
                overlay_frame = cv2.addWeighted(overlay_frame, 0.75, color_overlay, 0.25, 0)

                contours, _ = cv2.findContours(
                    mask_uint8 * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(overlay_frame, contours, -1, (255, 255, 255), 7)
                cv2.drawContours(overlay_frame, contours, -1, (0, 0, 0), 5)
                cv2.drawContours(overlay_frame, contours, -1, color, 3)

        processor.reset_all_prompts(state)
        frame_result["masks"][prompt] = prompt_masks

    _draw_legend(overlay_frame, prompts, frame_result["masks"])
    return frame_result, overlay_frame, filtered


def _retry_dropped_frames(
    all_results,
    frame_files,
    processor,
    prompts,
    overlay_dir,
    expected_tools: int = 2,
    retry_confidences: list = None,
    min_area: int = 5000,
):
    """
    Re-run SAM3 at lower confidence levels on frames that detected fewer
    tools than expected. Tries up to 2 lower confidence levels before
    accepting the result.
    """
    if retry_confidences is None:
        retry_confidences = [0.3, 0.15]

    retried = 0

    for prompt in prompts:
        counts = [len(r["masks"].get(prompt, [])) for r in all_results]
        drop_indices = [i for i, c in enumerate(counts) if c < expected_tools]
        if not drop_indices:
            continue

        print(f"\n  Retry pass ({prompt}): {len(drop_indices)} frames below expected {expected_tools} tools")

        for idx in drop_indices:
            frame_path = frame_files[idx]
            frame_name = frame_path.stem
            old_count = counts[idx]
            recovered = False

            for conf in retry_confidences:
                processor.set_confidence_threshold(conf)
                retry_start = time.time()
                result, overlay, _ = _run_frame_inference(
                    processor, frame_path, [prompt], min_area
                )
                retry_elapsed = time.time() - retry_start
                new_count = len(result["masks"].get(prompt, []))

                if new_count >= expected_tools:
                    all_results[idx]["masks"][prompt] = result["masks"][prompt]

                    # Re-render full overlay
                    _, full_overlay, _ = _render_overlay_from_results(
                        frame_path, all_results[idx], prompts
                    )
                    out_path = overlay_dir / f"{frame_name}.jpg"
                    cv2.imwrite(str(out_path), full_overlay)

                    retried += 1
                    recovered = True
                    print(f"    [{frame_name}] recovered at conf={conf}: {old_count} -> {new_count} masks ({retry_elapsed:.1f}s)")
                    break
                elif new_count > old_count:
                    # Partial improvement, keep trying lower confidence
                    all_results[idx]["masks"][prompt] = result["masks"][prompt]
                    old_count = new_count
                    print(f"    [{frame_name}] partial at conf={conf}: -> {new_count} masks, trying lower ({retry_elapsed:.1f}s)")
                else:
                    print(f"    [{frame_name}] no improvement at conf={conf}: still {old_count} ({retry_elapsed:.1f}s)")

            if not recovered and old_count < expected_tools:
                # Re-render with best result so far
                _, full_overlay, _ = _render_overlay_from_results(
                    frame_path, all_results[idx], prompts
                )
                out_path = overlay_dir / f"{frame_name}.jpg"
                cv2.imwrite(str(out_path), full_overlay)
                print(f"    [{frame_name}] accepted {old_count} tools (below expected {expected_tools})")

    return retried


def _subtract_gt_tissue(all_results, frame_files, annotation_loader, prompts):
    """
    Subtract GT liver/gallbladder masks from SAM3 tool masks.
    Removes tool mask pixels that overlap with known tissue regions.
    GT tissue takes priority — any tool pixels on liver/gallbladder are removed.

    Returns count of frames where tool masks were modified.
    """
    cleaned = 0
    for i, (result, fpath) in enumerate(zip(all_results, frame_files)):
        # Extract frame number from filename: frame_001582.webp -> 1582
        frame_num = int(fpath.stem.split("_")[1])

        # Check if GT exists for this frame
        if frame_num not in annotation_loader.images:
            continue

        gt_masks = annotation_loader.get_frame_masks(frame_num)
        h, w = result["height"], result["width"]

        # Combined tissue mask (Liver + Gallbladder)
        tissue = np.zeros((h, w), dtype=np.uint8)
        for cat in ["Liver", "Gallbladder"]:
            if cat in gt_masks:
                tissue |= gt_masks[cat]
        if tissue.sum() == 0:
            continue

        # Subtract tissue from each tool mask
        modified = False
        for prompt in prompts:
            new_masks = []
            for md in result["masks"].get(prompt, []):
                # Reconstruct tool mask from stored polygons
                tool = np.zeros((h, w), dtype=np.uint8)
                for poly in md["segmentation"]:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(tool, [pts], 1)

                # Subtract: keep tool pixels NOT on tissue
                cleaned_mask = tool & (~tissue)
                new_area = float(np.sum(cleaned_mask))

                if new_area > 0:
                    old_area = md["area"]
                    md["area"] = new_area
                    md["segmentation"] = mask_to_coco_polygons(cleaned_mask * 255)
                    new_masks.append(md)
                    if new_area != old_area:
                        modified = True
                else:
                    # Mask was entirely on tissue — removed
                    modified = True
            result["masks"][prompt] = new_masks

        if modified:
            cleaned += 1

    return cleaned


def _render_overlay_from_results(frame_path, frame_result, prompts, annotation_loader=None):
    """Re-render an overlay image from stored mask results, with optional GT tissue."""
    pil_image = Image.open(frame_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    overlay_frame = img_bgr.copy()
    h, w = overlay_frame.shape[:2]

    # Map GT category names (title-case) to our color keys (lowercase)
    GT_CAT_MAP = {"Liver": "liver", "Gallbladder": "gallbladder"}
    gt_masks = {}

    # --- Layer 1: GT tissue masks (bottom layer) ---
    if annotation_loader is not None:
        frame_num = int(frame_path.stem.split("_")[1])
        if frame_num in annotation_loader.images:
            gt_masks = annotation_loader.get_frame_masks(frame_num)
            for gt_cat, color_key in GT_CAT_MAP.items():
                if gt_cat not in gt_masks:
                    continue
                mask = gt_masks[gt_cat].astype(np.uint8)
                color = CATEGORY_COLORS.get(color_key, DEFAULT_COLOR)
                color_np = np.array(color, dtype=np.uint8)

                mask_bool = mask > 0
                color_overlay = overlay_frame.copy()
                color_overlay[mask_bool] = color_np
                overlay_frame = cv2.addWeighted(overlay_frame, 0.75, color_overlay, 0.25, 0)

                contours, _ = cv2.findContours(
                    mask * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(overlay_frame, contours, -1, (255, 255, 255), 7)
                cv2.drawContours(overlay_frame, contours, -1, (0, 0, 0), 5)
                cv2.drawContours(overlay_frame, contours, -1, color, 3)

    # --- Layer 2: SAM3 tool masks (top layer) ---
    for prompt in prompts:
        color = CATEGORY_COLORS.get(prompt, DEFAULT_COLOR)
        color_np = np.array(color, dtype=np.uint8)

        for mask_data in frame_result["masks"].get(prompt, []):
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in mask_data["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)

            mask_bool = mask > 0
            color_overlay = overlay_frame.copy()
            color_overlay[mask_bool] = color_np
            overlay_frame = cv2.addWeighted(overlay_frame, 0.75, color_overlay, 0.25, 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay_frame, contours, -1, (255, 255, 255), 7)
            cv2.drawContours(overlay_frame, contours, -1, (0, 0, 0), 5)
            cv2.drawContours(overlay_frame, contours, -1, color, 3)

    # --- Legend ---
    # Build combined masks dict for legend (GT + SAM3)
    legend_masks = {}
    for gt_cat, color_key in GT_CAT_MAP.items():
        if gt_cat in gt_masks:
            legend_masks[color_key] = [{"_gt": True}]  # count = 1 per GT category
    legend_masks.update(frame_result["masks"])
    all_legend_keys = list(GT_CAT_MAP.values()) + prompts
    _draw_legend(overlay_frame, all_legend_keys, legend_masks)

    return frame_result, overlay_frame, 0


def process_snippet(
    snippet_dir: Path,
    processor,
    prompts: list,
    output_dir: Path,
    confidence: float = 0.5,
    max_frames: int = None,
    min_area: int = 5000,
    expected_tools: int = 2,
    annotation_loader=None,
):
    """
    Process a single snippet folder with SAM3 text prompts.

    Three-pass approach:
      Pass 1: Run all frames at normal confidence with min-area filter
      Pass 2: Retry dropped frames (count < expected) at lower confidence
      Pass 3: Subtract GT liver/gallbladder from tool masks (if annotations available)

    Args:
        snippet_dir: Path to snippet folder (contains frames_left/)
        processor: SAM3 processor
        prompts: List of text prompts
        output_dir: Where to save overlay images and mask JSON
        confidence: SAM3 confidence threshold for pass 1
        max_frames: Limit to N frames (for quick test)
        min_area: Minimum mask area in pixels (filters tiny noise)
        expected_tools: Expected number of tools per frame (drives retry logic)
        annotation_loader: COCOAnnotationLoader with GT tissue masks (optional)

    Returns:
        List of per-frame result dicts
    """
    frames_dir = snippet_dir / "frames_left"
    if not frames_dir.exists():
        print(f"  ERROR: No frames_left/ in {snippet_dir}")
        return None

    # Get sorted frame files
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print(f"  ERROR: No frame images found in {frames_dir}")
        return None

    total = len(frame_files)
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
        print(f"  Test mode: processing {len(frame_files)} of {total} frames")

    # Set confidence threshold
    processor.set_confidence_threshold(confidence)

    # Output dirs
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    snippet_name = snippet_dir.name
    print(f"\n  Processing {snippet_name}: {len(frame_files)} frames, prompts={prompts}")
    print(f"  Confidence: {confidence}, min_area: {min_area}, expected_tools: {expected_tools}")

    # --- Pass 1: Process all frames ---
    all_results = []
    total_filtered = 0
    start_time = time.time()

    for i, frame_path in enumerate(frame_files):
        frame_start = time.time()

        frame_result, overlay_frame, filtered = _run_frame_inference(
            processor, frame_path, prompts, min_area
        )
        total_filtered += filtered

        # Save overlay
        out_path = overlay_dir / f"{frame_result['frame']}.jpg"
        cv2.imwrite(str(out_path), overlay_frame)

        all_results.append(frame_result)

        # Per-frame timing
        elapsed = time.time() - frame_start
        mask_summary = ", ".join(
            f"{p}={len(frame_result['masks'][p])}" for p in prompts
        )
        filt_str = f" (filtered {filtered})" if filtered > 0 else ""
        print(f"  [{i+1}/{len(frame_files)}] {frame_result['frame']} | {elapsed:.1f}s | {mask_summary}{filt_str}")

    pass1_elapsed = time.time() - start_time
    print(f"\n  Pass 1 done: {len(all_results)} frames in {pass1_elapsed:.1f}s")
    if total_filtered > 0:
        print(f"  Filtered {total_filtered} small masks (area < {min_area})")

    # --- Pass 2: Retry dropped frames ---
    retried = _retry_dropped_frames(
        all_results, frame_files, processor, prompts,
        overlay_dir, expected_tools=expected_tools, min_area=min_area,
    )
    # Restore confidence for any subsequent processing
    processor.set_confidence_threshold(confidence)

    # --- Pass 3: GT tissue subtraction ---
    gt_cleaned = 0
    if annotation_loader is not None:
        gt_cleaned = _subtract_gt_tissue(
            all_results, frame_files, annotation_loader, prompts
        )
        if gt_cleaned > 0:
            print(f"\n  Pass 3: GT subtraction cleaned {gt_cleaned} frames")

        # Re-render ALL overlays with GT tissue + cleaned tool masks
        print(f"  Re-rendering overlays with GT tissue...")
        for i, (result, fpath) in enumerate(zip(all_results, frame_files)):
            _, full_overlay, _ = _render_overlay_from_results(
                fpath, result, prompts, annotation_loader=annotation_loader
            )
            cv2.imwrite(str(overlay_dir / f"{fpath.stem}.jpg"), full_overlay)

    # Save results JSON
    results_path = output_dir / f"{snippet_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Stitch overlays into video
    _stitch_snippet_video(overlay_dir, output_dir / f"{snippet_name}_overlay.mp4", fps=6)

    # Summary
    total_elapsed = time.time() - start_time
    print(f"\n  {snippet_name} done: {len(all_results)} frames in {total_elapsed:.1f}s ({total_elapsed/len(all_results):.1f}s/frame)")
    if retried > 0:
        print(f"  Retried and recovered {retried} dropped frames")
    if gt_cleaned > 0:
        print(f"  GT tissue subtraction cleaned {gt_cleaned} frames")
    for prompt in prompts:
        total_masks = sum(len(r["masks"][prompt]) for r in all_results)
        frames_with = sum(1 for r in all_results if len(r["masks"][prompt]) > 0)
        print(f"    {prompt}: {total_masks} masks across {frames_with}/{len(all_results)} frames")

    return all_results


def _draw_legend(frame, prompts, masks_dict):
    """Draw a legend on the overlay frame."""
    h, w = frame.shape[:2]
    x, y = 10, 30
    for prompt in prompts:
        masks = masks_dict.get(prompt, [])
        color = CATEGORY_COLORS.get(prompt, DEFAULT_COLOR)
        # Check if this is a GT category (marked with _gt key)
        if masks and isinstance(masks[0], dict) and masks[0].get("_gt"):
            label = f"{prompt}: GT"
        else:
            label = f"{prompt}: {len(masks)}"
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30


def _stitch_snippet_video(frames_dir: Path, output_path: Path, fps: int = 6):
    """Stitch overlay JPGs into a video."""
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return

    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            writer.write(img)
    writer.release()
    print(f"  Video saved: {output_path} ({len(frame_files)} frames @ {fps}fps)")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

INSTRUMENT_DATASETS = ["FBF_1", "FBF_2", "FBF_E_1", "PCH_1", "PCH_2", "PCH_E_1"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks using SAM3 (snippet or box-prompt mode)"
    )

    # --- Snippet mode args ---
    snippet_group = parser.add_argument_group("Snippet mode (Segments clips)")
    snippet_group.add_argument(
        "--segments-dir",
        default=None,
        help="Path to Segments directory (e.g., data/Segments)",
    )
    snippet_group.add_argument(
        "--episode",
        default=None,
        help="Episode to process (C_1, E_3, or F_3). If omitted, processes all.",
    )
    snippet_group.add_argument(
        "--snippet",
        type=int,
        default=None,
        help="Snippet number to process (e.g., 1). If omitted, processes all snippets.",
    )
    snippet_group.add_argument(
        "--prompts",
        nargs="+",
        default=["liver", "gallbladder", "tool"],
        help="Text prompts for SAM3 (default: liver gallbladder tool)",
    )

    # --- Box-prompt mode args ---
    box_group = parser.add_argument_group("Box-prompt mode (instrument datasets)")
    box_group.add_argument(
        "--dataset-dir", "-d",
        default=None,
        help="Path to dataset (FBF_1, etc.) or parent directory for --all",
    )
    box_group.add_argument(
        "--all",
        action="store_true",
        help="Process all 6 instrument datasets (dataset-dir should be parent)",
    )
    box_group.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "both"],
        help="Which split to process (default: train)",
    )
    box_group.add_argument(
        "--visualize",
        action="store_true",
        help="Render sample frames with generated masks",
    )
    box_group.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample frames to visualize (default: 10)",
    )
    box_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial output (skip already-processed images)",
    )

    # --- Shared args ---
    parser.add_argument(
        "--output-dir", "-o",
        default="./outputs/segments",
        help="Output directory (default: ./outputs/segments)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5000,
        help="Minimum mask area in pixels to keep (default: 5000, filters tiny noise)",
    )
    parser.add_argument(
        "--expected-tools",
        type=int,
        default=2,
        help="Expected number of tools per frame (default: 2). Frames with fewer trigger retry at lower confidence.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Quick test mode: process only N frames/images",
    )
    parser.add_argument(
        "--tissue-seg-dir",
        default=None,
        help="Path to tissue_segmentation directory with per-episode GT annotations "
             "(e.g., 'F:/2026 vibes/MPHY Project/annotated_dataset/tissue_segmentation')",
    )

    args = parser.parse_args()

    # Route to the right mode
    if args.segments_dir is not None:
        _main_snippets(args)
    elif args.dataset_dir is not None:
        _main_box_prompts(args)
    else:
        parser.error("Must specify either --segments-dir (snippet mode) or --dataset-dir (box-prompt mode)")


def _main_snippets(args):
    """Snippet mode: process Segments clips with text prompts."""
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)

    if not segments_dir.exists():
        print(f"ERROR: Segments directory not found: {segments_dir}")
        sys.exit(1)

    # Determine which episodes to process
    if args.episode:
        episodes = [segments_dir / args.episode]
        if not episodes[0].exists():
            print(f"ERROR: Episode not found: {episodes[0]}")
            sys.exit(1)
    else:
        episodes = sorted([d for d in segments_dir.iterdir() if d.is_dir()])

    # Collect snippet folders
    snippet_list = []
    for ep_dir in episodes:
        if args.snippet is not None:
            snip = ep_dir / f"snippet_{args.snippet:03d}"
            if snip.exists():
                snippet_list.append((ep_dir.name, snip))
            else:
                print(f"WARNING: {snip} not found")
        else:
            snips = sorted(ep_dir.glob("snippet_*"))
            snips = [s for s in snips if s.is_dir()]
            for s in snips:
                snippet_list.append((ep_dir.name, s))

    if not snippet_list:
        print("ERROR: No snippets found to process")
        sys.exit(1)

    # Count total frames
    total_frames = 0
    for _, snip_dir in snippet_list:
        frames_dir = snip_dir / "frames_left"
        if frames_dir.exists():
            total_frames += len(list(frames_dir.glob("frame_*")))

    print("=" * 60)
    print("SAM3 Segment Processor (Snippet Mode)")
    print("=" * 60)
    print(f"Snippets: {len(snippet_list)}")
    print(f"Total frames: {total_frames}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {output_dir}")
    if args.test:
        print(f"Test mode: {args.test} frames per snippet")

    # Load GT annotations per episode (if tissue-seg-dir provided)
    ann_loaders = {}
    if args.tissue_seg_dir:
        tissue_seg_dir = Path(args.tissue_seg_dir)
        if not tissue_seg_dir.exists():
            print(f"WARNING: tissue-seg-dir not found: {tissue_seg_dir}")
        else:
            print(f"\nLoading GT annotations from {tissue_seg_dir}...")
            for ep_name, _ in snippet_list:
                if ep_name not in ann_loaders:
                    loader = _load_episode_annotations(tissue_seg_dir, ep_name)
                    if loader:
                        ann_loaders[ep_name] = loader

    # Load SAM3 model
    print("\nLoading SAM3 model...")
    config = configure_device(device=args.device, verbose=True)
    model = load_sam3_model(config=config, verbose=True)
    processor = get_processor(model, device=config.device)

    # Process each snippet
    for ep_name, snip_dir in snippet_list:
        print(f"\n{'=' * 60}")
        print(f"Episode: {ep_name} / {snip_dir.name}")
        print(f"{'=' * 60}")

        snip_output = output_dir / ep_name / snip_dir.name
        process_snippet(
            snippet_dir=snip_dir,
            processor=processor,
            prompts=args.prompts,
            output_dir=snip_output,
            confidence=args.confidence,
            max_frames=args.test,
            min_area=args.min_area,
            expected_tools=args.expected_tools,
            annotation_loader=ann_loaders.get(ep_name),
        )

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


def _main_box_prompts(args):
    """Box-prompt mode: process instrument datasets with COCO bboxes."""
    # --test implies --visualize
    if args.test is not None:
        args.visualize = True
        args.num_samples = args.test

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Determine which datasets to process
    if args.all:
        datasets = []
        for name in INSTRUMENT_DATASETS:
            d = dataset_dir / name
            if d.exists():
                datasets.append(d)
            else:
                print(f"  WARNING: {name} not found in {dataset_dir}")
    else:
        datasets = [dataset_dir]

    if not datasets:
        print("ERROR: No datasets found to process")
        sys.exit(1)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    print("=" * 60)
    print("SAM3 Tool Mask Generator (Box-Prompt Mode)")
    print("=" * 60)
    print(f"Datasets: {[d.name for d in datasets]}")
    print(f"Splits: {splits}")
    print(f"Output: {output_dir}")

    # Load SAM3 model
    print("\nLoading SAM3 model...")
    config = configure_device(device=args.device, verbose=True)
    model = load_sam3_model(config=config, verbose=True)
    processor = get_processor(model, device=config.device)

    for ds_dir in datasets:
        for split in splits:
            print(f"\n{'=' * 60}")
            print(f"Processing: {ds_dir.name} / {split}")
            print(f"{'=' * 60}")

            coco_data = process_dataset(
                dataset_dir=ds_dir,
                processor=processor,
                device=config.device,
                split=split,
                output_dir=output_dir,
                confidence=args.confidence,
                resume=args.resume,
                max_images=args.test,
            )

            if coco_data is None:
                continue

            _save_output(coco_data, output_dir, ds_dir.name, split)

            if args.visualize:
                vis_dir = output_dir / f"{ds_dir.name}_{split}_vis"
                render_sample_frames(
                    dataset_dir=ds_dir,
                    coco_data=coco_data,
                    output_dir=vis_dir,
                    num_samples=args.num_samples,
                )

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
