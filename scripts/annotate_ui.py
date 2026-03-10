"""
Interactive SAM3 Annotation UI with Confidence Tracker.

Gradio-based tool for annotating surgical video snippets:
  1. Click to segment at GT keyframes (points/boxes)
  2. Review masks before propagation
  3. Propagate with SAM3 tracker
  4. Confidence tracker monitors quality, prompts re-annotation when degraded
  5. Export COCO-format annotations

Usage:
    python scripts/annotate_ui.py --segments-dir data/Segments --episode F_3
    python scripts/annotate_ui.py --segments-dir data/Segments --episode F_3 \\
        --categories Liver Gallbladder Meat Skin
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_annotations import COCOAnnotationLoader
from scripts.generate_tool_masks import mask_to_coco_polygons
from scripts.generate_tool_masks_video import _load_frames_for_tracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from scripts.shared_config import (
    CATEGORY_COLORS_RGB as CATEGORY_COLORS,
    DEFAULT_COLOR_RGB as DEFAULT_COLOR,
)

# ---------------------------------------------------------------------------
# Global state (inference_state is not deepcopyable)
# ---------------------------------------------------------------------------

g_state = {
    "tracker": None,           # Sam3TrackerPredictor
    "tracker_state": None,     # inference_state dict
    "predictor": None,         # Sam3VideoPredictor
    "images_tensor": None,     # loaded frames tensor for tracker
    "frame_files": [],         # sorted frame file paths
    "frame_images": [],        # loaded numpy frames (RGB) for display
    "video_height": 0,
    "video_width": 0,
    "categories": [],          # list of category names
    "cat_to_objid": {},        # category name -> obj_id
    "objid_to_cat": {},        # obj_id -> category name
    "points": {},              # category -> list of (x, y, label)
    "preview_masks": {},       # frame_idx -> {category: binary_mask}
    "approved_masks": {},      # frame_idx -> {category: binary_mask}
    "propagated_masks": {},    # frame_idx -> {category: binary_mask}
    "confidence_log": [],      # list of (frame_idx, category, obj_score, mean_logit)
    "low_conf_frames": [],     # sorted list of (frame_idx, category, score)
    "annotation_loader": None, # COCOAnnotationLoader for existing GT
    "snippet_dir": None,       # current snippet directory
    "split_size": None,        # episode split size
    "snippet_meta": None,      # snippet metadata from snippets.json
    "episode": None,           # current episode name
    "snippet_id": None,        # current snippet id
}


# ---------------------------------------------------------------------------
# Autosave / restore
# ---------------------------------------------------------------------------

def _autosave_path():
    """Get autosave file path for current snippet."""
    if g_state["snippet_dir"] is None:
        return None
    return g_state["snippet_dir"] / "session_autosave.json"


def _masks_to_serializable(masks_dict):
    """Convert {frame_idx: {cat: np.ndarray}} to JSON-serializable format."""
    result = {}
    for fidx, cats in masks_dict.items():
        result[str(fidx)] = {}
        for cat, mask in cats.items():
            polys = mask_to_coco_polygons(mask, min_area=10)
            area = int(mask.sum())
            result[str(fidx)][cat] = {"polygons": polys, "area": area}
    return result


def _masks_from_serializable(data, height, width):
    """Convert serialized masks back to {frame_idx: {cat: np.ndarray}}."""
    result = {}
    for fidx_str, cats in data.items():
        fidx = int(fidx_str)
        result[fidx] = {}
        for cat, mask_data in cats.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in mask_data.get("polygons", []):
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            result[fidx][cat] = mask
    return result


def _autosave():
    """Save current session state to disk."""
    path = _autosave_path()
    if path is None:
        return

    data = {
        "episode": g_state["episode"],
        "snippet_id": g_state["snippet_id"],
        "categories": g_state["categories"],
        "points": {cat: pts for cat, pts in g_state["points"].items()},
        "approved_masks": _masks_to_serializable(g_state["approved_masks"]),
        "propagated_masks": _masks_to_serializable(g_state["propagated_masks"]),
        "confidence_log": g_state["confidence_log"],
        "low_conf_frames": g_state["low_conf_frames"],
        "timestamp": time.time(),
    }

    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        tmp_path.replace(path)  # atomic rename
    except Exception as e:
        print(f"Autosave failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()


def _restore_autosave(height, width):
    """Restore session state from autosave if it exists. Returns status message."""
    path = _autosave_path()
    if path is None or not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        backup = path.with_suffix(".corrupt")
        try:
            path.rename(backup)
        except OSError:
            pass
        return f"Autosave was corrupted (backed up to {backup.name})"
    except Exception:
        return None

    # Validate it's for the same snippet
    if (data.get("episode") != g_state["episode"] or
            data.get("snippet_id") != g_state["snippet_id"]):
        return None

    # Restore points
    for cat, pts in data.get("points", {}).items():
        if cat in g_state["points"]:
            g_state["points"][cat] = pts

    # Restore masks
    g_state["approved_masks"] = _masks_from_serializable(
        data.get("approved_masks", {}), height, width
    )
    g_state["propagated_masks"] = _masks_from_serializable(
        data.get("propagated_masks", {}), height, width
    )
    g_state["confidence_log"] = data.get("confidence_log", [])
    g_state["low_conf_frames"] = data.get("low_conf_frames", [])

    n_approved = sum(len(v) for v in g_state["approved_masks"].values())
    n_propagated = sum(len(v) for v in g_state["propagated_masks"].values())
    ts = data.get("timestamp", 0)
    age = time.time() - ts
    age_str = f"{age / 60:.0f}min" if age < 3600 else f"{age / 3600:.1f}hr"
    return f"Restored session ({age_str} old): {n_approved} approved, {n_propagated} propagated masks"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load SAM3 model (once at startup)."""
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    print("Loading SAM3 video model...")
    t0 = time.time()
    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    g_state["predictor"] = predictor
    g_state["tracker"] = predictor.model.tracker
    # Share backbone for standalone tracker usage
    g_state["tracker"].backbone = predictor.model.detector.backbone


# ---------------------------------------------------------------------------
# Snippet loading
# ---------------------------------------------------------------------------

def get_snippets(segments_dir, episode):
    """List available snippets for an episode."""
    ep_dir = Path(segments_dir) / episode
    snippets_json = ep_dir / f"{episode}_snippets.json"
    if not snippets_json.exists():
        return []
    with open(snippets_json) as f:
        snippets = json.load(f)
    return [s["snippet_id"] for s in snippets]


def load_snippet(segments_dir, episode, snippet_id, categories):
    """Load a snippet's frames and existing annotations."""
    ep_dir = Path(segments_dir) / episode
    snippet_dir = ep_dir / f"snippet_{snippet_id}"
    frames_dir = snippet_dir / "frames_left"

    if not frames_dir.exists():
        return None, f"frames_left/ not found in {snippet_dir}"

    # Load snippet metadata
    snippets_json = ep_dir / f"{episode}_snippets.json"
    with open(snippets_json) as f:
        all_snippets = json.load(f)
    snippet_meta = next((s for s in all_snippets if s["snippet_id"] == snippet_id), None)
    if not snippet_meta:
        return None, f"Snippet {snippet_id} not found in metadata"

    split_size = snippet_meta.get("split_size", 120)

    # Load frames
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return None, "No frame files found"

    # Load frame images for display (RGB numpy)
    frame_images = []
    for fp in frame_files:
        img = np.array(Image.open(fp).convert("RGB"))
        frame_images.append(img)

    # Load existing annotations if available
    ann_path = snippet_dir / "snippet_annotations.json"
    annotation_loader = None
    if ann_path.exists():
        annotation_loader = COCOAnnotationLoader(
            str(ann_path), str(snippet_dir), split_size=split_size
        )
        annotation_loader.load()

    # Set up category -> obj_id mapping
    cat_to_objid = {}
    objid_to_cat = {}
    for i, cat in enumerate(categories):
        oid = i + 1
        cat_to_objid[cat] = oid
        objid_to_cat[oid] = cat

    # Update global state
    g_state["frame_files"] = frame_files
    g_state["frame_images"] = frame_images
    g_state["snippet_dir"] = snippet_dir
    g_state["split_size"] = split_size
    g_state["snippet_meta"] = snippet_meta
    g_state["annotation_loader"] = annotation_loader
    g_state["categories"] = categories
    g_state["cat_to_objid"] = cat_to_objid
    g_state["objid_to_cat"] = objid_to_cat
    g_state["points"] = {cat: [] for cat in categories}
    g_state["preview_masks"] = {}
    g_state["approved_masks"] = {}
    g_state["propagated_masks"] = {}
    g_state["confidence_log"] = []
    g_state["low_conf_frames"] = []
    g_state["video_height"] = frame_images[0].shape[0]
    g_state["video_width"] = frame_images[0].shape[1]
    g_state["episode"] = episode
    g_state["snippet_id"] = snippet_id

    # Clean up old tracker state and free GPU memory
    if g_state["tracker_state"] is not None:
        del g_state["tracker_state"]
        g_state["tracker_state"] = None
    if g_state["images_tensor"] is not None:
        del g_state["images_tensor"]
        g_state["images_tensor"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_frames = len(frame_files)
    gt_info = f"GT annotations: {'loaded' if annotation_loader else 'none'}"
    status = f"Loaded {n_frames} frames. {gt_info}. Split size: {split_size}"

    # Try to restore autosave
    restore_msg = _restore_autosave(frame_images[0].shape[0], frame_images[0].shape[1])
    if restore_msg:
        status += f"\n{restore_msg}"

    return n_frames, status


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(frame_idx, show_gt=True, show_points=True, show_approved=True,
                 show_propagated=True, show_preview=True):
    """Render a frame with overlays."""
    if frame_idx < 0 or frame_idx >= len(g_state["frame_images"]):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    frame = g_state["frame_images"][frame_idx].copy()
    h, w = frame.shape[:2]

    # Layer 1: Existing GT annotations (from snippet_annotations.json)
    if show_gt and g_state["annotation_loader"] is not None:
        fpath = g_state["frame_files"][frame_idx]
        frame_num = int(fpath.stem.split("_")[1])
        gt_masks = g_state["annotation_loader"].get_frame_masks_by_frame_num(frame_num)
        if gt_masks is not None:
            for cat_name, mask in gt_masks.items():
                color = np.array(CATEGORY_COLORS.get(cat_name, DEFAULT_COLOR), dtype=np.uint8)
                mask_bool = mask.astype(bool)
                overlay = frame.copy()
                overlay[mask_bool] = color
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, color.tolist(), 2)

    # Layer 2: Propagated masks
    if show_propagated and frame_idx in g_state["propagated_masks"]:
        for cat, mask in g_state["propagated_masks"][frame_idx].items():
            color = np.array(CATEGORY_COLORS.get(cat, DEFAULT_COLOR), dtype=np.uint8)
            mask_bool = mask.astype(bool)
            overlay = frame.copy()
            overlay[mask_bool] = color
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color.tolist(), 2)

    # Layer 3: Preview masks (unapproved — shown with dashed/thin contour)
    if show_preview and frame_idx in g_state["preview_masks"]:
        for cat, mask in g_state["preview_masks"][frame_idx].items():
            color = np.array(CATEGORY_COLORS.get(cat, DEFAULT_COLOR), dtype=np.uint8)
            mask_bool = mask.astype(bool)
            overlay = frame.copy()
            overlay[mask_bool] = color
            frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Thin dashed-style contour to distinguish from approved
            cv2.drawContours(frame, contours, -1, color.tolist(), 1)

    # Layer 4: Approved masks (user-confirmed — thick white+color contour)
    if show_approved and frame_idx in g_state["approved_masks"]:
        for cat, mask in g_state["approved_masks"][frame_idx].items():
            color = np.array(CATEGORY_COLORS.get(cat, DEFAULT_COLOR), dtype=np.uint8)
            mask_bool = mask.astype(bool)
            overlay = frame.copy()
            overlay[mask_bool] = color
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 3)
            cv2.drawContours(frame, contours, -1, color.tolist(), 2)

    # Layer 5: Current click points
    if show_points:
        for cat, pts in g_state["points"].items():
            color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
            for x, y, label in pts:
                marker_color = color if label == 1 else (128, 128, 128)
                if label == 1:
                    cv2.circle(frame, (int(x), int(y)), 8, marker_color, -1)
                    cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (int(x), int(y)), 8, marker_color, 2)
                    cv2.line(frame, (int(x) - 5, int(y) - 5),
                             (int(x) + 5, int(y) + 5), marker_color, 2)
                    cv2.line(frame, (int(x) - 5, int(y) + 5),
                             (int(x) + 5, int(y) - 5), marker_color, 2)

    # Legend
    y_pos = 25
    for cat in g_state["categories"]:
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        n_pts = len(g_state["points"].get(cat, []))
        has_approved = frame_idx in g_state["approved_masks"] and cat in g_state["approved_masks"][frame_idx]
        has_preview = frame_idx in g_state["preview_masks"] and cat in g_state["preview_masks"][frame_idx]
        has_prop = frame_idx in g_state["propagated_masks"] and cat in g_state["propagated_masks"][frame_idx]
        status = ""
        if has_approved:
            status = " [APPROVED]"
        elif has_preview:
            status = " [preview]"
        elif has_prop:
            status = " [propagated]"
        elif n_pts > 0:
            status = f" [{n_pts} pts]"
        label = f"{cat}{status}"
        cv2.putText(frame, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3)
        cv2.putText(frame, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        y_pos += 22

    # Frame info
    fpath = g_state["frame_files"][frame_idx]
    frame_num = int(fpath.stem.split("_")[1])
    split_num = frame_num // g_state["split_size"]
    offset = frame_num % g_state["split_size"]
    is_gt = offset == 0
    info = f"Frame {frame_num} (split {split_num}, offset {offset})"
    if is_gt:
        info += " [GT KEYFRAME]"
    cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3)
    cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# Annotation actions
# ---------------------------------------------------------------------------

def preview_mask(frame_idx, category):
    """Run single-frame SAM3 inference with current points to preview mask."""
    tracker = g_state["tracker"]
    if tracker is None:
        return render_frame(frame_idx), "Model not loaded"

    pts = g_state["points"].get(category, [])
    if not pts:
        return render_frame(frame_idx), "No points placed for this category"

    # Always reinitialize tracker state for preview to avoid state lock
    # (propagate_in_video_preflight locks the state, preventing new obj_ids)
    if g_state["tracker_state"] is not None:
        del g_state["tracker_state"]
        g_state["tracker_state"] = None
    if g_state["images_tensor"] is not None:
        del g_state["images_tensor"]
        g_state["images_tensor"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _init_tracker_state()

    obj_id = g_state["cat_to_objid"][category]
    state = g_state["tracker_state"]

    # Convert points
    points_xy = [[p[0], p[1]] for p in pts]
    labels = [p[2] for p in pts]

    try:
        tracker.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points_xy,
            labels=labels,
            rel_coordinates=False,
            clear_old_points=True,
        )

        # Run single frame inference
        tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

        preview_mask_np = None
        for fidx, obj_ids, _, video_res_masks, obj_scores in tracker.propagate_in_video(
            state, start_frame_idx=frame_idx, max_frame_num_to_track=0
        ):
            for i, oid in enumerate(obj_ids):
                if int(oid) == obj_id:
                    mask_logits = video_res_masks[i]
                    if isinstance(mask_logits, torch.Tensor):
                        preview_mask_np = (mask_logits.squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                    else:
                        preview_mask_np = (mask_logits > 0).astype(np.uint8)

        if preview_mask_np is not None:
            # Store in preview (NOT approved)
            if frame_idx not in g_state["preview_masks"]:
                g_state["preview_masks"][frame_idx] = {}
            g_state["preview_masks"][frame_idx][category] = preview_mask_np

            area = int(preview_mask_np.sum())
            return render_frame(frame_idx), f"Preview: {category} mask, area={area}px. Click Approve to confirm."
        else:
            return render_frame(frame_idx), "No mask generated"

    except Exception as e:
        return render_frame(frame_idx), f"Error: {e}"


def _init_tracker_state():
    """Initialize tracker state with all frames."""
    tracker = g_state["tracker"]
    frame_files = g_state["frame_files"]
    n = len(frame_files)

    print(f"Loading {n} frames for tracker...")
    images, vh, vw = _load_frames_for_tracker(frame_files, n, tracker.image_size)

    state = tracker.init_state(
        video_height=vh, video_width=vw, num_frames=n
    )
    state["images"] = images

    g_state["tracker_state"] = state
    g_state["images_tensor"] = images
    g_state["video_height"] = vh
    g_state["video_width"] = vw
    print(f"Tracker state initialized: {n} frames, {vh}x{vw}")


def approve_mask(frame_idx, category):
    """Approve preview mask -> moves it to approved state."""
    # Check preview first
    if frame_idx in g_state["preview_masks"] and category in g_state["preview_masks"][frame_idx]:
        mask = g_state["preview_masks"][frame_idx].pop(category)
        if not g_state["preview_masks"][frame_idx]:
            del g_state["preview_masks"][frame_idx]

        if frame_idx not in g_state["approved_masks"]:
            g_state["approved_masks"][frame_idx] = {}
        g_state["approved_masks"][frame_idx][category] = mask

        _autosave()
        return render_frame(frame_idx), f"Approved {category} mask at frame {frame_idx}"

    # Check if already approved
    if frame_idx in g_state["approved_masks"] and category in g_state["approved_masks"][frame_idx]:
        return render_frame(frame_idx), f"{category} already approved at frame {frame_idx}"

    return render_frame(frame_idx), "No preview mask to approve. Click Preview Mask first."


def undo_last_point(category, frame_idx):
    """Remove the last point added for this category."""
    pts = g_state["points"].get(category, [])
    if not pts:
        return render_frame(int(frame_idx)), f"No points to undo for {category}"
    removed = pts.pop()
    return render_frame(int(frame_idx)), f"Undid point at ({removed[0]}, {removed[1]}) for {category}. {len(pts)} remaining."


def clear_points(category):
    """Clear all points for a category."""
    g_state["points"][category] = []
    return "Points cleared"


def clear_all_points():
    """Clear points for all categories."""
    for cat in g_state["categories"]:
        g_state["points"][cat] = []
    return "All points cleared"


def _get_anchor_frames():
    """Build sorted list of anchor frame indices from approved masks."""
    return sorted(g_state["approved_masks"].keys())


def _propagate_chunk(tracker, anchor_frames, chunk_start, chunk_end,
                     conf_threshold, progress_fn, progress_offset, progress_total):
    """Propagate a single chunk between anchor boundaries.

    Adds approved masks at anchor frames within/adjacent to the chunk,
    then propagates bidirectionally within the chunk bounds.
    Returns (low_conf_list, frames_processed).
    """
    approved = g_state["approved_masks"]
    n_frames = len(g_state["frame_files"])
    low_conf = []

    # Fresh tracker state for this chunk
    if g_state["tracker_state"] is not None:
        del g_state["tracker_state"]
        g_state["tracker_state"] = None
    if g_state["images_tensor"] is not None:
        del g_state["images_tensor"]
        g_state["images_tensor"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _init_tracker_state()
    state = g_state["tracker_state"]

    # Add approved masks at anchors that are within or at the boundaries of this chunk
    prompts_added = 0
    for fidx in anchor_frames:
        if fidx < chunk_start or fidx > chunk_end:
            continue
        if fidx not in approved:
            continue
        for cat, mask in approved[fidx].items():
            obj_id = g_state["cat_to_objid"][cat]
            mask_tensor = torch.from_numpy(mask).float()
            try:
                tracker.add_new_mask(
                    inference_state=state,
                    frame_idx=fidx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )
                prompts_added += 1
            except Exception as e:
                print(f"WARNING: Failed to add mask at frame {fidx} for {cat}: {e}")

    if prompts_added == 0:
        return low_conf, 0

    # Preflight
    tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

    # Propagate bidirectionally within chunk bounds
    chunk_size = chunk_end - chunk_start + 1
    frames_processed = 0

    for reverse in [False, True]:
        direction = "backward" if reverse else "forward"
        max_track = chunk_size

        for frame_idx, obj_ids, _, video_res_masks, obj_scores in tracker.propagate_in_video(
            state, start_frame_idx=None, max_frame_num_to_track=max_track, reverse=reverse
        ):
            if frame_idx < chunk_start or frame_idx > chunk_end:
                continue
            if frame_idx >= n_frames:
                continue

            progress_fn(
                (progress_offset + frames_processed) / max(progress_total, 1),
                desc=f"Chunk {chunk_start}-{chunk_end} {direction}: frame {frame_idx}"
            )

            if frame_idx not in g_state["propagated_masks"]:
                g_state["propagated_masks"][frame_idx] = {}

            for i, oid in enumerate(obj_ids):
                cat = g_state["objid_to_cat"].get(int(oid))
                if cat is None:
                    continue

                # Skip if already approved (don't overwrite user masks)
                if frame_idx in approved and cat in approved[frame_idx]:
                    continue

                score_val = obj_scores[i]
                obj_score = float(score_val.squeeze().cpu().item()) if isinstance(score_val, torch.Tensor) else float(score_val)

                mask_logits = video_res_masks[i]
                if isinstance(mask_logits, torch.Tensor):
                    mask_np = mask_logits.squeeze(0).cpu().numpy()
                else:
                    mask_np = mask_logits

                positive_mask = mask_np > 0.0
                mean_logit = float(mask_np[positive_mask].mean()) if positive_mask.any() else 0.0
                binary_mask = positive_mask.astype(np.uint8)

                existing = g_state["propagated_masks"][frame_idx].get(cat)
                if existing is None or obj_score > 0:
                    g_state["propagated_masks"][frame_idx][cat] = binary_mask

                g_state["confidence_log"].append((frame_idx, cat, obj_score, mean_logit))

                if obj_score < conf_threshold:
                    low_conf.append((frame_idx, cat, obj_score))

            frames_processed += 1

    return low_conf, frames_processed


def propagate(conf_threshold, progress=gr.Progress()):
    """Run chunked bidirectional SAM3 propagation from all approved masks.

    Propagation is split into chunks bounded by anchor frames (approved masks).
    Each chunk re-initializes the tracker with the nearest anchors, preventing
    drift over long sequences. Chunks are at most max_chunk_size frames (default
    matches split_size, typically 120). If no anchor exists within a chunk's
    range, the nearest approved mask from an adjacent chunk boundary is used.
    """
    tracker = g_state["tracker"]
    if tracker is None:
        return "Model not loaded"

    approved = g_state["approved_masks"]
    if not approved:
        return "No approved masks to propagate from"

    n_frames = len(g_state["frame_files"])
    max_chunk = g_state["split_size"] or 120
    anchor_frames = _get_anchor_frames()

    # Build chunk boundaries: split the video at each anchor frame
    # so every chunk has at least one anchor at its start or end
    boundaries = set()
    boundaries.add(0)
    boundaries.add(n_frames - 1)
    for af in anchor_frames:
        boundaries.add(af)
        # Also break at max_chunk intervals from each anchor
        # Forward from anchor
        pos = af + max_chunk
        while pos < n_frames:
            boundaries.add(min(pos, n_frames - 1))
            pos += max_chunk
        # Backward from anchor
        pos = af - max_chunk
        while pos > 0:
            boundaries.add(max(pos, 0))
            pos -= max_chunk

    boundaries = sorted(boundaries)

    # Build non-overlapping chunks from boundaries
    chunks = []
    for i in range(len(boundaries) - 1):
        c_start = boundaries[i]
        c_end = boundaries[i + 1]
        # Each chunk must have at least one anchor within range to propagate from
        has_anchor = any(c_start <= af <= c_end for af in anchor_frames)
        if has_anchor:
            chunks.append((c_start, c_end))
        else:
            # Extend the previous chunk to cover this range
            if chunks:
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, c_end)

    if not chunks:
        # Fallback: single chunk covering everything
        chunks = [(0, n_frames - 1)]

    g_state["propagated_masks"] = {}
    g_state["confidence_log"] = []
    all_low_conf = []
    total_frames_processed = 0

    progress(0, desc=f"Propagating in {len(chunks)} chunks (max {max_chunk} frames each)...")
    print(f"Chunked propagation: {len(chunks)} chunks, {len(anchor_frames)} anchors, max_chunk={max_chunk}")

    for ci, (c_start, c_end) in enumerate(chunks):
        chunk_anchors = [af for af in anchor_frames if c_start <= af <= c_end]
        print(f"  Chunk {ci + 1}/{len(chunks)}: frames {c_start}-{c_end} ({c_end - c_start + 1} frames, {len(chunk_anchors)} anchors)")

        low_conf, frames_done = _propagate_chunk(
            tracker, anchor_frames, c_start, c_end,
            conf_threshold, progress,
            progress_offset=total_frames_processed,
            progress_total=n_frames * 2,  # *2 for bidirectional
        )
        all_low_conf.extend(low_conf)
        total_frames_processed += frames_done

    # Sort and deduplicate low confidence frames
    all_low_conf.sort(key=lambda x: x[2])
    seen = set()
    unique_low = []
    for fidx, cat, score in all_low_conf:
        key = (fidx, cat)
        if key not in seen:
            seen.add(key)
            unique_low.append((fidx, cat, score))
    g_state["low_conf_frames"] = unique_low

    # Free GPU memory after propagation (masks already stored as numpy)
    if g_state["tracker_state"] is not None:
        del g_state["tracker_state"]
        g_state["tracker_state"] = None
    if g_state["images_tensor"] is not None:
        del g_state["images_tensor"]
        g_state["images_tensor"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _autosave()

    total_propagated = sum(len(v) for v in g_state["propagated_masks"].values())
    status = (f"Propagation complete: {total_propagated} masks across "
              f"{len(g_state['propagated_masks'])} frames ({len(chunks)} chunks)")
    if unique_low:
        status += f"\n{len(unique_low)} low-confidence frames (threshold={conf_threshold})"
        status += f"\nWorst: frame {unique_low[0][0]} ({unique_low[0][1]}, score={unique_low[0][2]:.3f})"
    return status


def jump_to_low_conf(current_idx):
    """Jump to next low-confidence frame. Returns (frame_idx, category, status)."""
    low_conf = g_state["low_conf_frames"]
    if not low_conf:
        return current_idx, None, "No low-confidence frames"

    # Find next after current
    for fidx, cat, score in low_conf:
        if fidx > current_idx:
            return fidx, cat, f"Low confidence: frame {fidx}, {cat}, score={score:.3f}"

    # Wrap around
    fidx, cat, score = low_conf[0]
    return fidx, cat, f"Low confidence (wrapped): frame {fidx}, {cat}, score={score:.3f}"


def jump_to_gt(current_idx, direction="next"):
    """Jump to nearest GT keyframe (offset == 0)."""
    if not g_state["frame_files"]:
        return current_idx

    split_size = g_state["split_size"]
    n = len(g_state["frame_files"])

    if direction == "next":
        for i in range(current_idx + 1, n):
            fpath = g_state["frame_files"][i]
            frame_num = int(fpath.stem.split("_")[1])
            if frame_num % split_size == 0:
                return i
        return current_idx  # no GT found ahead
    else:
        for i in range(current_idx - 1, -1, -1):
            fpath = g_state["frame_files"][i]
            frame_num = int(fpath.stem.split("_")[1])
            if frame_num % split_size == 0:
                return i
        return current_idx


def save_annotations():
    """Export annotations in COCO format."""
    if g_state["snippet_dir"] is None:
        return "No snippet loaded"

    # Merge approved + propagated masks
    all_masks = {}
    for fidx, cats in g_state["propagated_masks"].items():
        all_masks[fidx] = dict(cats)
    # Approved masks override propagated
    for fidx, cats in g_state["approved_masks"].items():
        if fidx not in all_masks:
            all_masks[fidx] = {}
        all_masks[fidx].update(cats)

    if not all_masks:
        return "No masks to save"

    # Build COCO format
    categories_list = []
    for cat in g_state["categories"]:
        categories_list.append({
            "id": g_state["cat_to_objid"][cat],
            "name": cat,
        })

    images_list = []
    annotations_list = []
    ann_id = 1

    for fidx in sorted(all_masks.keys()):
        if fidx >= len(g_state["frame_files"]):
            continue

        fpath = g_state["frame_files"][fidx]
        frame_num = int(fpath.stem.split("_")[1])
        split_num = frame_num // g_state["split_size"]
        offset = frame_num % g_state["split_size"]

        image_entry = {
            "id": frame_num,
            "file_name": f"./split_imgs/split_{split_num}/{offset:05d}.jpg",
            "height": g_state["video_height"],
            "width": g_state["video_width"],
        }
        images_list.append(image_entry)

        for cat, mask in all_masks[fidx].items():
            if mask.sum() == 0:
                continue
            polygons = mask_to_coco_polygons(mask, min_area=50)
            if not polygons:
                continue

            ys, xs = np.where(mask > 0)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            area = float(mask.sum())

            annotations_list.append({
                "id": ann_id,
                "image_id": frame_num,
                "category_id": g_state["cat_to_objid"][cat],
                "segmentation": polygons,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_output = {
        "categories": categories_list,
        "images": images_list,
        "annotations": annotations_list,
    }

    out_path = g_state["snippet_dir"] / "annotated_masks.json"
    try:
        tmp_path = out_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        tmp_path.replace(out_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"SAVE FAILED: {e}. Your work is preserved in autosave."

    _autosave()

    return (f"Saved {len(annotations_list)} annotations across "
            f"{len(images_list)} frames to {out_path}")


def load_gt_masks_for_frame(frame_idx):
    """Load existing GT masks as approved masks for a frame."""
    if g_state["annotation_loader"] is None:
        return "No GT annotations loaded"

    fpath = g_state["frame_files"][frame_idx]
    frame_num = int(fpath.stem.split("_")[1])
    gt_masks = g_state["annotation_loader"].get_frame_masks_by_frame_num(frame_num)

    if gt_masks is None:
        return f"No GT masks for frame {frame_num}"

    if frame_idx not in g_state["approved_masks"]:
        g_state["approved_masks"][frame_idx] = {}

    loaded = []
    for cat_name, mask in gt_masks.items():
        if cat_name in g_state["cat_to_objid"]:
            g_state["approved_masks"][frame_idx][cat_name] = mask.astype(np.uint8)
            loaded.append(cat_name)

    if loaded:
        _autosave()

    return f"Loaded GT masks: {', '.join(loaded)}" if loaded else "No matching categories in GT"


def load_all_gt_previews():
    """Scan all GT keyframes and load their masks as previews for review.

    GT keyframes are at offset 0 of each split (frame_num % split_size == 0).
    Masks are loaded into preview_masks so the user can review and approve them
    before propagation.
    """
    if g_state["annotation_loader"] is None:
        return "No GT annotations loaded"

    split_size = g_state["split_size"]
    n_frames = len(g_state["frame_files"])
    loaded_count = 0
    gt_frames_found = []

    for fidx in range(n_frames):
        fpath = g_state["frame_files"][fidx]
        frame_num = int(fpath.stem.split("_")[1])

        # Only GT keyframes (offset 0)
        if frame_num % split_size != 0:
            continue

        gt_masks = g_state["annotation_loader"].get_frame_masks_by_frame_num(frame_num)
        if gt_masks is None or len(gt_masks) == 0:
            continue

        if fidx not in g_state["preview_masks"]:
            g_state["preview_masks"][fidx] = {}

        cats_loaded = []
        for cat_name, mask in gt_masks.items():
            if cat_name in g_state["cat_to_objid"]:
                # Skip if already approved at this frame
                if fidx in g_state["approved_masks"] and cat_name in g_state["approved_masks"][fidx]:
                    continue
                g_state["preview_masks"][fidx][cat_name] = mask.astype(np.uint8)
                cats_loaded.append(cat_name)
                loaded_count += 1

        if cats_loaded:
            gt_frames_found.append((fidx, frame_num, cats_loaded))

    if not gt_frames_found:
        return "No GT keyframes with annotations found"

    return (f"Loaded {loaded_count} GT masks as previews across {len(gt_frames_found)} keyframes. "
            f"Navigate to each GT frame (use GT> button) to review and Approve.")


def approve_all_previews():
    """Approve all current preview masks at once (batch approve after review)."""
    count = 0
    for fidx in list(g_state["preview_masks"].keys()):
        for cat in list(g_state["preview_masks"][fidx].keys()):
            mask = g_state["preview_masks"][fidx].pop(cat)
            if fidx not in g_state["approved_masks"]:
                g_state["approved_masks"][fidx] = {}
            g_state["approved_masks"][fidx][cat] = mask
            count += 1
        if not g_state["preview_masks"][fidx]:
            del g_state["preview_masks"][fidx]

    if count:
        _autosave()
    return f"Approved {count} preview masks" if count else "No preview masks to approve"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

KEYBOARD_JS = """
() => {
    document.addEventListener('keydown', (e) => {
        // Don't trigger when typing in text inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        const slider = document.querySelector('input[type="range"]');
        if (!slider) return;

        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            slider.value = Math.max(Number(slider.min), Number(slider.value) - 1);
            slider.dispatchEvent(new Event('input', {bubbles: true}));
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            slider.value = Math.min(Number(slider.max), Number(slider.value) + 1);
            slider.dispatchEvent(new Event('input', {bubbles: true}));
        }
    });
}
"""


def build_ui(args, episodes=None):
    """Build the Gradio interface."""
    segments_dir = args.segments_dir
    default_episode = args.episode
    categories = args.categories
    episode_choices = episodes or ([default_episode] if default_episode else [])

    with gr.Blocks(title="SAM3 Annotation Tool") as demo:
        gr.Markdown("# SAM3 Interactive Annotation Tool")
        gr.Markdown("Arrow keys: navigate frames | Click image: place points")

        # --- Top row: episode/snippet selection ---
        with gr.Row():
            episode_dd = gr.Dropdown(
                label="Episode",
                choices=episode_choices,
                value=default_episode or (episode_choices[0] if episode_choices else None),
            )
            snippet_dd = gr.Dropdown(label="Snippet", choices=[])
            load_btn = gr.Button("Load Snippet", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=3)

        # --- Frame display ---
        frame_display = gr.Image(
            label="Frame (click to place points)",
            interactive=True,
            type="numpy",
            height=600,
        )

        # --- Frame navigation ---
        with gr.Row():
            prev_btn = gr.Button("< Prev")
            frame_slider = gr.Slider(
                minimum=0, maximum=0, step=1, value=0, label="Frame Index"
            )
            next_btn = gr.Button("Next >")
            prev_gt_btn = gr.Button("< GT")
            next_gt_btn = gr.Button("GT >")
            frame_info = gr.Textbox(label="Frame", interactive=False, scale=0)

        # --- Annotation controls ---
        with gr.Row():
            category_dd = gr.Dropdown(
                label="Category", choices=categories, value=categories[0] if categories else None
            )
            point_type = gr.Radio(
                choices=["Positive", "Negative"], value="Positive", label="Point Type"
            )
            conf_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                label="Confidence Threshold"
            )

        # --- Action buttons ---
        # --- Action buttons ---
        with gr.Row():
            preview_btn = gr.Button("Preview Mask")
            undo_btn = gr.Button("Undo Point")
            clear_btn = gr.Button("Clear Points")
            clear_all_btn = gr.Button("Clear All")
            approve_btn = gr.Button("Approve Mask", variant="primary")
            load_gt_btn = gr.Button("Load GT (frame)")

        # --- GT + Propagation ---
        with gr.Row():
            load_all_gt_btn = gr.Button("Load All GT as Previews")
            approve_all_btn = gr.Button("Approve All Previews", variant="primary")
            propagate_btn = gr.Button("Propagate (chunked)", variant="primary")
            jump_low_btn = gr.Button("Next Low Confidence")
            save_btn = gr.Button("Save Annotations", variant="stop")

        # --- Event handlers ---

        def on_episode_change(episode):
            snippets = get_snippets(segments_dir, episode)
            return gr.update(choices=snippets, value=snippets[0] if snippets else None)

        def on_load(episode, snippet_id):
            if not episode or not snippet_id:
                return (gr.update(), gr.update(), gr.update(),
                        "Select episode and snippet first")
            n_frames, msg = load_snippet(segments_dir, episode, snippet_id, categories)
            if n_frames is None:
                return gr.update(), gr.update(), gr.update(), msg

            frame = render_frame(0)
            return (
                frame,
                gr.update(maximum=n_frames - 1, value=0),
                f"0 / {n_frames}",
                msg,
            )

        def on_slider_change(idx):
            idx = int(idx)
            frame = render_frame(idx)
            n = len(g_state["frame_files"])
            return frame, f"{idx} / {n}"

        def on_prev(idx):
            return max(0, int(idx) - 1)

        def on_next(idx):
            return min(len(g_state["frame_files"]) - 1, int(idx) + 1)

        def on_prev_gt(idx):
            return jump_to_gt(int(idx), "prev")

        def on_next_gt(idx):
            return jump_to_gt(int(idx), "next")

        def on_click(evt: gr.SelectData, frame_idx, category, pt_type):
            if not g_state["frame_files"]:
                return render_frame(0), "No snippet loaded"
            x, y = evt.index[0], evt.index[1]
            label = 1 if pt_type == "Positive" else 0
            g_state["points"][category].append((x, y, label))
            frame = render_frame(int(frame_idx))
            n_pts = len(g_state["points"][category])
            return frame, f"{'+'if label == 1 else '-'} ({x}, {y}) for {category}. Total: {n_pts}"

        def on_preview(frame_idx, category):
            return preview_mask(int(frame_idx), category)

        def on_approve(frame_idx, category):
            return approve_mask(int(frame_idx), category)

        def on_undo(category, frame_idx):
            return undo_last_point(category, frame_idx)

        def on_clear(category, frame_idx):
            msg = clear_points(category)
            return render_frame(int(frame_idx)), msg

        def on_clear_all(frame_idx):
            msg = clear_all_points()
            return render_frame(int(frame_idx)), msg

        def on_propagate(conf_threshold):
            return propagate(conf_threshold)

        def on_jump_low(current_idx):
            new_idx, cat, msg = jump_to_low_conf(int(current_idx))
            if cat:
                return new_idx, cat, msg
            return new_idx, gr.update(), msg

        def on_load_gt(frame_idx):
            msg = load_gt_masks_for_frame(int(frame_idx))
            return render_frame(int(frame_idx)), msg

        def on_load_all_gt(frame_idx):
            msg = load_all_gt_previews()
            return render_frame(int(frame_idx)), msg

        def on_approve_all(frame_idx):
            msg = approve_all_previews()
            return render_frame(int(frame_idx)), msg

        def on_save():
            return save_annotations()

        # --- Wire up events ---

        episode_dd.change(on_episode_change, [episode_dd], [snippet_dd])

        load_btn.click(
            on_load, [episode_dd, snippet_dd],
            [frame_display, frame_slider, frame_info, status_box],
        )

        frame_slider.change(on_slider_change, [frame_slider], [frame_display, frame_info])

        prev_btn.click(on_prev, [frame_slider], [frame_slider])
        next_btn.click(on_next, [frame_slider], [frame_slider])
        prev_gt_btn.click(on_prev_gt, [frame_slider], [frame_slider])
        next_gt_btn.click(on_next_gt, [frame_slider], [frame_slider])

        frame_display.select(
            on_click,
            [frame_slider, category_dd, point_type],
            [frame_display, status_box],
        )

        preview_btn.click(on_preview, [frame_slider, category_dd], [frame_display, status_box])
        approve_btn.click(on_approve, [frame_slider, category_dd], [frame_display, status_box])
        undo_btn.click(on_undo, [category_dd, frame_slider], [frame_display, status_box])
        clear_btn.click(on_clear, [category_dd, frame_slider], [frame_display, status_box])
        clear_all_btn.click(on_clear_all, [frame_slider], [frame_display, status_box])
        load_gt_btn.click(on_load_gt, [frame_slider], [frame_display, status_box])
        load_all_gt_btn.click(on_load_all_gt, [frame_slider], [frame_display, status_box])
        approve_all_btn.click(on_approve_all, [frame_slider], [frame_display, status_box])
        propagate_btn.click(on_propagate, [conf_slider], [status_box])
        jump_low_btn.click(on_jump_low, [frame_slider], [frame_slider, category_dd, status_box])
        save_btn.click(on_save, [], [status_box])

        # Keyboard shortcuts
        demo.load(js=KEYBOARD_JS)

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive SAM3 annotation tool with confidence tracking"
    )
    parser.add_argument(
        "--segments-dir", required=True,
        help="Path to data/Segments directory"
    )
    parser.add_argument(
        "--episode", default=None,
        help="Default episode to load"
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=["Liver", "Gallbladder", "Meat", "Skin"],
        help="Tissue categories to annotate (default: Liver Gallbladder Meat Skin)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Gradio server port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip model loading (for UI testing only)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.no_model:
        load_model()

    # Populate episode list from segments dir
    segments_path = Path(args.segments_dir)
    if segments_path.exists():
        episodes = sorted([d.name for d in segments_path.iterdir()
                          if d.is_dir() and (d / f"{d.name}_snippets.json").exists()])
    else:
        episodes = []

    demo = build_ui(args, episodes=episodes)

    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(
        server_port=args.port,
        share=args.share,
        max_threads=1,
    )
