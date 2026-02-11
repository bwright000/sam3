#!/usr/bin/env python3
"""
CRCD Annotation Quality Checker.

Loads COCO-format annotations for surgical video datasets (C_1, E_3, F_3)
and computes quality metrics: coverage, area distribution, temporal consistency,
boundary complexity, and inter-class overlap.

Usage:
    # Quick stats
    python scripts/check_annotations.py --dataset-dir path/to/C_1

    # With sample frame visualizations
    python scripts/check_annotations.py --dataset-dir path/to/C_1 --visualize --num-samples 10

    # Render a split as annotated video
    python scripts/check_annotations.py --dataset-dir path/to/C_1 --render-split 0
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# COCO Annotation Loader
# ---------------------------------------------------------------------------

class COCOAnnotationLoader:
    """Load and index COCO-format annotations."""

    def __init__(self, json_path: str, dataset_dir: str):
        self.json_path = Path(json_path)
        self.dataset_dir = Path(dataset_dir)
        self.images = {}          # image_id -> image info
        self.annotations = {}     # image_id -> list of annotations
        self.categories = {}      # category_id -> category name
        self.file_to_id = {}      # file_name -> image_id

    def load(self):
        """Load and index the annotation JSON."""
        print(f"Loading annotations from {self.json_path.name} ({self.json_path.stat().st_size / 1e6:.0f} MB)...")
        t0 = time.time()
        with open(self.json_path, "r") as f:
            data = json.load(f)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Index categories
        for cat in data.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        # Index images
        for img in data.get("images", []):
            self.images[img["id"]] = img
            self.file_to_id[img["file_name"]] = img["id"]

        # Index annotations by image_id
        self.annotations = defaultdict(list)
        for ann in data.get("annotations", []):
            self.annotations[ann["image_id"]].append(ann)

        print(f"  Images: {len(self.images)}")
        print(f"  Annotations: {sum(len(v) for v in self.annotations.values())}")
        print(f"  Categories: {self.categories}")

        self.build_frame_mapping()

    def polygon_to_mask(self, segmentation: list, height: int, width: int) -> np.ndarray:
        """Convert COCO polygon segmentation to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def get_frame_masks(self, image_id: int) -> Dict[str, np.ndarray]:
        """Get binary masks for a frame, keyed by category name."""
        anns = self.annotations.get(image_id, [])
        img_info = self.images[image_id]
        h, w = img_info["height"], img_info["width"]

        masks = {}
        for ann in anns:
            cat_name = self.categories.get(ann["category_id"], f"cat_{ann['category_id']}")
            seg = ann.get("segmentation", [])
            if seg:
                mask = self.polygon_to_mask(seg, h, w)
                masks[cat_name] = mask
        return masks

    def get_split_image_ids(self) -> Dict[str, List[int]]:
        """Group image IDs by split directory."""
        splits = defaultdict(list)
        for img_id, img_info in self.images.items():
            fname = img_info["file_name"]
            # Extract split name from path like ./split_imgs/split_0/00000.jpg
            # We want "split_0" (the numbered split), not "split_imgs"
            parts = Path(fname).parts
            split_name = None
            for p in parts:
                if re.match(r"split_\d+", p):
                    split_name = p
                    break
            if split_name:
                splits[split_name].append(img_id)

        # Sort image IDs within each split by filename
        for split_name in splits:
            splits[split_name].sort(
                key=lambda iid: self.images[iid]["file_name"]
            )
        return dict(splits)

    def build_frame_mapping(self):
        """Build video_frame_number -> image_id mapping from split filenames.

        Annotation splits encode video position: split_N/XXXXX.jpg = frame N*split_size+XXXXX.
        image_ids are sequential and DON'T match frame numbers when splits are missing.

        Split size is auto-detected from annotation filenames: max offset + 1.
        E_3 uses 120-frame splits, F_3 uses 100-frame splits, C_1 uses 120.

        When multiple JSONs are merged (train+test), the same video frame may have
        different image_ids with different category annotations. We collect ALL
        image_ids per frame so get_frame_masks_by_frame_num returns complete masks.
        """
        # Auto-detect split size from max filename offset
        max_offset = 0
        for img_info in self.images.values():
            stem = Path(img_info["file_name"]).stem
            try:
                max_offset = max(max_offset, int(stem))
            except ValueError:
                pass
        split_size = max_offset + 1 if max_offset > 0 else 120  # fallback
        self._split_size = split_size

        self._frame_to_all_ids = defaultdict(list)
        for img_id, img_info in self.images.items():
            fname = img_info["file_name"]
            m = re.search(r'split_(\d+)', fname)
            if not m:
                continue
            split_num = int(m.group(1))
            stem = Path(fname).stem
            try:
                offset = int(stem)
            except ValueError:
                # C_1 train.json has broken file_name like "./split_14/" (no filename).
                # For that dataset image_id = split*split_size+offset, so recover offset from id.
                if img_id // split_size == split_num:
                    offset = img_id % split_size
                else:
                    continue
            video_frame = split_num * split_size + offset
            self._frame_to_all_ids[video_frame].append(img_id)
        # Backward-compat: single image_id mapping (first per frame)
        self.frame_to_image_id = {
            vf: ids[0] for vf, ids in self._frame_to_all_ids.items()
        }
        if self.frame_to_image_id:
            frames = sorted(self.frame_to_image_id.keys())
            print(f"  Frame mapping: {len(self.frame_to_image_id)} video frames "
                  f"(range {frames[0]}-{frames[-1]}, split_size={split_size})")

    def get_frame_masks_by_frame_num(self, frame_num: int) -> Optional[Dict[str, np.ndarray]]:
        """Get masks using video frame number (not COCO image_id).

        Collects masks from ALL image_ids that map to this frame (handles
        merged train+test JSONs where different categories are on different IDs).
        """
        image_ids = getattr(self, '_frame_to_all_ids', {}).get(frame_num, [])
        if not image_ids:
            image_id = self.frame_to_image_id.get(frame_num)
            if image_id is None:
                return None
            return self.get_frame_masks(image_id)
        masks = {}
        for image_id in image_ids:
            masks.update(self.get_frame_masks(image_id))
        return masks if masks else None

    def resolve_frame_path(self, image_id: int) -> Optional[Path]:
        """Resolve the actual file path for a frame."""
        img_info = self.images[image_id]
        fname = img_info["file_name"]

        # Try direct path relative to dataset dir
        path = self.dataset_dir / fname
        if path.exists():
            return path

        # Handle double-nesting (E_3 has split_imgs/split_imgs/)
        path2 = self.dataset_dir / "split_imgs" / fname
        if path2.exists():
            return path2

        # Try stripping leading ./
        clean = fname.lstrip("./")
        path3 = self.dataset_dir / clean
        if path3.exists():
            return path3

        return None


# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------

class QualityMetrics:
    """Compute annotation quality metrics."""

    def __init__(self, loader: COCOAnnotationLoader):
        self.loader = loader

    def coverage_stats(self) -> Dict:
        """Compute annotation coverage per category."""
        total_images = len(self.loader.images)
        cat_counts = Counter()
        for anns in self.loader.annotations.values():
            for ann in anns:
                cat_name = self.loader.categories.get(ann["category_id"], f"cat_{ann['category_id']}")
                cat_counts[cat_name] += 1

        images_with_anns = len([iid for iid, anns in self.loader.annotations.items() if len(anns) > 0])
        images_without = total_images - images_with_anns

        return {
            "total_images": total_images,
            "images_with_annotations": images_with_anns,
            "images_without_annotations": images_without,
            "coverage_pct": 100 * images_with_anns / max(total_images, 1),
            "per_category": {
                name: {
                    "count": count,
                    "pct_of_images": 100 * count / max(total_images, 1),
                }
                for name, count in cat_counts.items()
            },
        }

    def area_stats(self, sample_size: int = 500) -> Dict:
        """Compute mask area statistics per category."""
        cat_areas = defaultdict(list)
        all_anns = []
        for anns in self.loader.annotations.values():
            all_anns.extend(anns)

        # Sample if too many annotations
        if len(all_anns) > sample_size:
            indices = np.linspace(0, len(all_anns) - 1, sample_size, dtype=int)
            sampled = [all_anns[i] for i in indices]
        else:
            sampled = all_anns

        for ann in sampled:
            cat_name = self.loader.categories.get(ann["category_id"], f"cat_{ann['category_id']}")
            img_info = self.loader.images[ann["image_id"]]
            img_area = img_info["width"] * img_info["height"]
            area_pct = 100 * ann["area"] / img_area
            cat_areas[cat_name].append(area_pct)

        result = {}
        for cat_name, areas in cat_areas.items():
            arr = np.array(areas)
            result[cat_name] = {
                "mean_pct": float(np.mean(arr)),
                "std_pct": float(np.std(arr)),
                "min_pct": float(np.min(arr)),
                "max_pct": float(np.max(arr)),
                "median_pct": float(np.median(arr)),
            }
        return result

    def boundary_complexity(self, sample_size: int = 500) -> Dict:
        """Compute polygon boundary complexity (point count)."""
        cat_points = defaultdict(list)
        all_anns = []
        for anns in self.loader.annotations.values():
            all_anns.extend(anns)

        if len(all_anns) > sample_size:
            indices = np.linspace(0, len(all_anns) - 1, sample_size, dtype=int)
            sampled = [all_anns[i] for i in indices]
        else:
            sampled = all_anns

        for ann in sampled:
            cat_name = self.loader.categories.get(ann["category_id"], f"cat_{ann['category_id']}")
            seg = ann.get("segmentation", [])
            total_points = sum(len(poly) // 2 for poly in seg)
            cat_points[cat_name].append(total_points)

        result = {}
        for cat_name, points in cat_points.items():
            arr = np.array(points)
            result[cat_name] = {
                "mean_points": float(np.mean(arr)),
                "std_points": float(np.std(arr)),
                "min_points": int(np.min(arr)),
                "max_points": int(np.max(arr)),
            }
        return result

    def temporal_consistency(self, split_name: str, max_pairs: int = 50) -> Dict:
        """Compute IoU between consecutive frames in a split."""
        splits = self.loader.get_split_image_ids()
        if split_name not in splits:
            return {"error": f"Split {split_name} not found"}

        image_ids = splits[split_name]
        if len(image_ids) < 2:
            return {"error": "Need at least 2 frames"}

        cat_ious = defaultdict(list)
        pairs = min(max_pairs, len(image_ids) - 1)
        step = max(1, (len(image_ids) - 1) // pairs)

        for i in range(0, len(image_ids) - 1, step):
            masks_a = self.loader.get_frame_masks(image_ids[i])
            masks_b = self.loader.get_frame_masks(image_ids[i + 1])

            for cat_name in masks_a:
                if cat_name in masks_b:
                    iou = self._compute_iou(masks_a[cat_name], masks_b[cat_name])
                    cat_ious[cat_name].append(iou)

        result = {}
        for cat_name, ious in cat_ious.items():
            arr = np.array(ious)
            result[cat_name] = {
                "mean_iou": float(np.mean(arr)),
                "std_iou": float(np.std(arr)),
                "min_iou": float(np.min(arr)),
                "num_pairs": len(ious),
            }
        return result

    def overlap_check(self, sample_size: int = 200) -> Dict:
        """Check for overlap between different category masks."""
        image_ids = list(self.loader.images.keys())
        if len(image_ids) > sample_size:
            indices = np.linspace(0, len(image_ids) - 1, sample_size, dtype=int)
            image_ids = [image_ids[i] for i in indices]

        overlap_counts = 0
        overlap_areas = []
        total_checked = 0

        for image_id in image_ids:
            masks = self.loader.get_frame_masks(image_id)
            cat_names = list(masks.keys())
            for i in range(len(cat_names)):
                for j in range(i + 1, len(cat_names)):
                    overlap = masks[cat_names[i]] & masks[cat_names[j]]
                    overlap_area = overlap.sum()
                    total_checked += 1
                    if overlap_area > 0:
                        overlap_counts += 1
                        img_info = self.loader.images[image_id]
                        img_area = img_info["width"] * img_info["height"]
                        overlap_areas.append(100 * overlap_area / img_area)

        return {
            "frames_checked": len(image_ids),
            "pairs_with_overlap": overlap_counts,
            "total_pairs_checked": total_checked,
            "overlap_pct": 100 * overlap_counts / max(total_checked, 1),
            "mean_overlap_area_pct": float(np.mean(overlap_areas)) if overlap_areas else 0,
            "max_overlap_area_pct": float(np.max(overlap_areas)) if overlap_areas else 0,
        }

    def area_jump_detection(self, split_name: str, threshold: float = 0.5) -> Dict:
        """Detect frames where mask area changes dramatically (potential annotation errors)."""
        splits = self.loader.get_split_image_ids()
        if split_name not in splits:
            return {"error": f"Split {split_name} not found"}

        image_ids = splits[split_name]
        all_anns_flat = []
        for anns in self.loader.annotations.values():
            all_anns_flat.extend(anns)

        # Build quick lookup: image_id -> {cat_id: area}
        area_by_frame = defaultdict(dict)
        for ann in all_anns_flat:
            if ann["image_id"] in set(image_ids):
                cat_name = self.loader.categories.get(ann["category_id"], f"cat_{ann['category_id']}")
                area_by_frame[ann["image_id"]][cat_name] = ann["area"]

        jumps = defaultdict(list)
        for i in range(len(image_ids) - 1):
            areas_a = area_by_frame.get(image_ids[i], {})
            areas_b = area_by_frame.get(image_ids[i + 1], {})
            for cat_name in areas_a:
                if cat_name in areas_b:
                    a, b = areas_a[cat_name], areas_b[cat_name]
                    if a > 0:
                        change = abs(b - a) / a
                        if change > threshold:
                            jumps[cat_name].append({
                                "frame_idx": i,
                                "area_before": a,
                                "area_after": b,
                                "change_pct": 100 * change,
                            })

        result = {}
        for cat_name, jump_list in jumps.items():
            result[cat_name] = {
                "num_jumps": len(jump_list),
                "worst_jumps": sorted(jump_list, key=lambda x: x["change_pct"], reverse=True)[:5],
            }
        return result

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = (mask_a & mask_b).sum()
        union = (mask_a | mask_b).sum()
        return float(intersection / max(union, 1))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Colors: BGR for OpenCV
CATEGORY_COLORS = {
    "Liver": (0, 0, 255),        # Red
    "Gallbladder": (0, 255, 0),  # Green
    "Meat": (0, 165, 255),       # Orange
    "Skin": (203, 192, 255),     # Pink
    "FBF": (255, 0, 0),          # Blue
    "PCH": (255, 255, 0),        # Cyan
}
DEFAULT_COLOR = (0, 255, 255)    # Yellow


def render_annotation_frame(
    image_bgr: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.25,
) -> np.ndarray:
    """Render masks on a frame with SAM3 demo-style triple contours."""
    frame = image_bgr.copy()

    for cat_name, mask in masks.items():
        color = CATEGORY_COLORS.get(cat_name, DEFAULT_COLOR)
        color_np = np.array(color, dtype=np.uint8)

        # Alpha blending
        mask_bool = mask.astype(bool)
        colored = np.where(mask_bool[..., None], color_np, frame)
        frame = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)

        # Triple-layer contours
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 7)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(frame, contours, -1, color, 3)

    # Add legend
    y_offset = 30
    for cat_name, mask in masks.items():
        color = CATEGORY_COLORS.get(cat_name, DEFAULT_COLOR)
        area_pct = 100 * mask.sum() / max(mask.size, 1)
        label = f"{cat_name}: {area_pct:.1f}%"
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    return frame


def render_sample_frames(
    loader: COCOAnnotationLoader,
    num_samples: int = 10,
    output_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """Render evenly-spaced sample frames with annotations."""
    image_ids = sorted(loader.images.keys())
    indices = np.linspace(0, len(image_ids) - 1, num_samples, dtype=int)
    sampled_ids = [image_ids[i] for i in indices]

    frames = []
    for i, img_id in enumerate(sampled_ids):
        frame_path = loader.resolve_frame_path(img_id)
        if frame_path is None:
            print(f"  Warning: Could not find frame for image_id {img_id}")
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            print(f"  Warning: Could not read {frame_path}")
            continue

        masks = loader.get_frame_masks(img_id)
        rendered = render_annotation_frame(img_bgr, masks)

        # Add frame info
        img_info = loader.images[img_id]
        label = f"ID:{img_id} | {img_info['file_name']}"
        cv2.putText(rendered, label, (10, rendered.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(rendered, label, (10, rendered.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        frames.append(rendered)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"sample_{i:03d}_id{img_id}.jpg"), rendered)

    print(f"  Rendered {len(frames)} sample frames")
    return frames


def render_split_video(
    loader: COCOAnnotationLoader,
    split_name: str,
    output_dir: Path,
    fps: int = 30,
):
    """Render an entire split as a video with annotation overlays."""
    splits = loader.get_split_image_ids()
    if split_name not in splits:
        print(f"Error: Split {split_name} not found")
        return

    image_ids = splits[split_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame to determine dimensions
    first_path = loader.resolve_frame_path(image_ids[0])
    if first_path is None:
        print(f"Error: Could not find first frame")
        return

    first_frame = cv2.imread(str(first_path))
    h, w = first_frame.shape[:2]

    output_path = output_dir / f"{split_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"  Rendering {split_name} ({len(image_ids)} frames)...")
    for i, img_id in enumerate(image_ids):
        frame_path = loader.resolve_frame_path(img_id)
        if frame_path is None:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue

        masks = loader.get_frame_masks(img_id)
        rendered = render_annotation_frame(img_bgr, masks)

        # Frame counter
        cv2.putText(rendered, f"Frame {i}/{len(image_ids)}",
                     (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(rendered)

        if (i + 1) % 20 == 0 or i == len(image_ids) - 1:
            print(f"    Frame {i + 1}/{len(image_ids)}")

    out.release()
    print(f"  Video saved to: {output_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(dataset_name: str, coverage: Dict, areas: Dict, complexity: Dict,
                 temporal: Dict, overlap: Dict, jumps: Dict):
    """Print a formatted quality report."""
    print("\n" + "=" * 70)
    print(f"  ANNOTATION QUALITY REPORT: {dataset_name}")
    print("=" * 70)

    # Coverage
    print("\n--- Coverage ---")
    print(f"  Total images:              {coverage['total_images']}")
    print(f"  With annotations:          {coverage['images_with_annotations']} ({coverage['coverage_pct']:.1f}%)")
    print(f"  Without annotations:       {coverage['images_without_annotations']}")
    for cat, stats in coverage["per_category"].items():
        print(f"  {cat:20s}:       {stats['count']} annotations ({stats['pct_of_images']:.1f}% of images)")

    # Area
    print("\n--- Mask Area (% of image) ---")
    for cat, stats in areas.items():
        print(f"  {cat:20s}:  mean={stats['mean_pct']:.1f}%  std={stats['std_pct']:.1f}%  "
              f"range=[{stats['min_pct']:.1f}%, {stats['max_pct']:.1f}%]  median={stats['median_pct']:.1f}%")

    # Boundary complexity
    print("\n--- Boundary Complexity (polygon points) ---")
    for cat, stats in complexity.items():
        print(f"  {cat:20s}:  mean={stats['mean_points']:.0f}  std={stats['std_points']:.0f}  "
              f"range=[{stats['min_points']}, {stats['max_points']}]")

    # Temporal consistency
    print("\n--- Temporal Consistency (consecutive frame IoU) ---")
    if "error" in temporal:
        print(f"  {temporal['error']}")
    else:
        for cat, stats in temporal.items():
            print(f"  {cat:20s}:  mean_IoU={stats['mean_iou']:.3f}  std={stats['std_iou']:.3f}  "
                  f"min={stats['min_iou']:.3f}  ({stats['num_pairs']} pairs)")

    # Overlap
    print("\n--- Inter-Category Overlap ---")
    print(f"  Frames checked:            {overlap['frames_checked']}")
    print(f"  Pairs with overlap:        {overlap['pairs_with_overlap']} / {overlap['total_pairs_checked']} "
          f"({overlap['overlap_pct']:.1f}%)")
    if overlap["mean_overlap_area_pct"] > 0:
        print(f"  Mean overlap area:         {overlap['mean_overlap_area_pct']:.3f}% of image")
        print(f"  Max overlap area:          {overlap['max_overlap_area_pct']:.3f}% of image")

    # Area jumps
    print("\n--- Area Jumps (>50% change between consecutive frames) ---")
    if not jumps:
        print("  No significant area jumps detected")
    else:
        for cat, stats in jumps.items():
            print(f"  {cat:20s}:  {stats['num_jumps']} jumps detected")
            for j in stats["worst_jumps"][:3]:
                print(f"    Frame {j['frame_idx']}: {j['area_before']:.0f} -> {j['area_after']:.0f} "
                      f"({j['change_pct']:.0f}% change)")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def natural_sort_key(name: str) -> tuple:
    parts = re.split(r'(\d+)', name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def find_annotation_file(dataset_dir: Path) -> Optional[Path]:
    """Auto-detect the annotation JSON file."""
    for name in ["annotations.json", "train.json"]:
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Check CRCD annotation quality")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset root (C_1, E_3, or F_3)")
    parser.add_argument("--annotation-file", default=None, help="Path to annotation JSON (auto-detect if not given)")
    parser.add_argument("--output-dir", default="./outputs/annotation_check", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Render sample frames")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample frames to render")
    parser.add_argument("--render-split", type=int, default=None, help="Render a specific split as video")
    parser.add_argument("--fps", type=int, default=30, help="FPS for rendered video")
    parser.add_argument("--save-report", action="store_true", help="Save metrics as JSON")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_name = dataset_dir.name

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Find annotation file
    if args.annotation_file:
        ann_path = Path(args.annotation_file)
    else:
        ann_path = find_annotation_file(dataset_dir)
    if ann_path is None or not ann_path.exists():
        print(f"Error: No annotation file found in {dataset_dir}")
        sys.exit(1)

    # Load annotations
    loader = COCOAnnotationLoader(str(ann_path), str(dataset_dir))
    loader.load()

    # Find splits
    splits = loader.get_split_image_ids()
    sorted_splits = sorted(splits.keys(), key=natural_sort_key)
    print(f"\nSplits found: {len(sorted_splits)}")
    if sorted_splits:
        print(f"  First: {sorted_splits[0]} ({len(splits[sorted_splits[0]])} frames)")
        print(f"  Last:  {sorted_splits[-1]} ({len(splits[sorted_splits[-1]])} frames)")

    # Compute metrics
    print("\nComputing quality metrics...")
    metrics = QualityMetrics(loader)

    coverage = metrics.coverage_stats()
    areas = metrics.area_stats()
    complexity = metrics.boundary_complexity()
    overlap = metrics.overlap_check()

    # Temporal consistency on first split
    first_split = sorted_splits[0] if sorted_splits else None
    if first_split:
        temporal = metrics.temporal_consistency(first_split)
        jumps = metrics.area_jump_detection(first_split)
    else:
        temporal = {"error": "No splits found"}
        jumps = {}

    # Print report
    print_report(dataset_name, coverage, areas, complexity, temporal, overlap, jumps)

    # Save report JSON
    output_dir = Path(args.output_dir)
    if args.save_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": dataset_name,
            "coverage": coverage,
            "area_stats": areas,
            "boundary_complexity": complexity,
            "temporal_consistency": temporal,
            "overlap": overlap,
            "area_jumps": jumps,
        }
        report_path = output_dir / f"{dataset_name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

    # Visualize samples
    if args.visualize:
        print(f"\nRendering {args.num_samples} sample frames...")
        render_sample_frames(loader, args.num_samples, output_dir / dataset_name)

    # Render split video
    if args.render_split is not None:
        split_name = f"split_{args.render_split}"
        if split_name in splits:
            print(f"\nRendering {split_name} as video...")
            render_split_video(loader, split_name, output_dir / dataset_name, args.fps)
        else:
            print(f"Error: {split_name} not found in dataset")


if __name__ == "__main__":
    main()
