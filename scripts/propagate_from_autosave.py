#!/usr/bin/env python3
"""Replay sam3_annotator anchors and propagate them across a snippet.

Companion to the annotator's lightweight mode. The lightweight server saves
painted anchors to `session_autosave.json` but never engages SAM3's video
tracker. This script:

  1. Loads the snippet via the annotator's storage helper (so it handles both
     the legacy `frames_left/` and the cluster-format `rgb/` layout).
  2. Reads `session_autosave.json` and rasterises each approved-mask entry
     (per frame, per category) back to a binary mask.
  3. Boots SAM3's video predictor, initialises tracker state on the snippet's
     frames, calls `add_new_mask` for every anchor, runs a single
     bidirectional propagation, and writes `annotated_masks.json` (or a
     suffixed variant) into the snippet directory.

Designed to run on Colab A100 / any GPU that can hold the snippet's frames.
Local GTX 970 will TDR for any non-trivial snippet.

Example:
    python scripts/propagate_from_autosave.py \\
        --data-dir 'F:/Datasets/CRCD' \\
        --episode F_1 --snippet 001 \\
        --output-suffix .replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _find_load_frames():
    """propagate_gap_fill.py and generate_tool_masks_video.py both expose this.
    Prefer the gap_fill copy since it's pure (no side imports)."""
    try:
        from scripts.propagate_gap_fill import _load_frames_for_tracker
        return _load_frames_for_tracker
    except Exception:
        from scripts.generate_tool_masks_video import _load_frames_for_tracker
        return _load_frames_for_tracker


def rasterise_polygons(polygons: list[list[float]], h: int, w: int) -> np.ndarray:
    """Flat polygon list -> binary uint8 mask. Mirrors storage.polygons_to_mask
    so we don't have to import cv2 in two places."""
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_anchors(autosave: dict, snippet_h: int, snippet_w: int
                 ) -> dict[int, dict[str, np.ndarray]]:
    """Return {frame_idx: {category: binary mask}}."""
    out: dict[int, dict[str, np.ndarray]] = {}
    approved = autosave.get("approved_masks", {}) or {}
    for fidx_s, cats in approved.items():
        try:
            fidx = int(fidx_s)
        except (ValueError, TypeError):
            continue
        for cat, payload in cats.items():
            polys = (payload or {}).get("polygons", []) if isinstance(payload, dict) else []
            if not polys:
                continue
            mask = rasterise_polygons(polys, snippet_h, snippet_w)
            if mask.sum() == 0:
                continue
            out.setdefault(fidx, {})[cat] = mask
    return out


def assign_obj_ids(anchors: dict[int, dict[str, np.ndarray]],
                   autosave_cat_to_objid: dict) -> dict[str, int]:
    """Reuse autosave cat_to_objid where present, allocate fresh otherwise.
    obj_ids must be positive integers and unique per category."""
    cat_to_objid: dict[str, int] = {}
    used: set[int] = set()
    # First pass: trust the autosave assignment
    for cat, oid in (autosave_cat_to_objid or {}).items():
        try:
            ioid = int(oid)
        except (TypeError, ValueError):
            continue
        if ioid > 0 and ioid not in used:
            cat_to_objid[cat] = ioid
            used.add(ioid)
    # Second pass: allocate ids for any anchored category we missed
    next_id = (max(used) + 1) if used else 1
    for cats in anchors.values():
        for cat in cats:
            if cat in cat_to_objid:
                continue
            while next_id in used:
                next_id += 1
            cat_to_objid[cat] = next_id
            used.add(next_id)
            next_id += 1
    return cat_to_objid


def build_coco(snippet, per_frame_per_cat: dict[int, dict[str, np.ndarray]],
               cat_to_objid: dict[str, int], min_area: int) -> dict:
    """Re-uses storage.export_coco's layout logic but takes already-merged masks.

    Emits RLE in the segmentation field (COCO-native dict form). RLE is
    lossless and preserves interior holes; the legacy polygon path used
    cv2.RETR_EXTERNAL which silently dropped tissue-through-tool gaps.
    """
    from sam3_annotator.server.rle import mask_to_rle

    images_list: list[dict] = []
    ann_list: list[dict] = []
    ann_id = 1
    for fidx in sorted(per_frame_per_cat.keys()):
        if fidx >= len(snippet.frame_files):
            continue
        fpath = snippet.frame_files[fidx]
        frame_num = int(fpath.stem.split("_")[1])
        if snippet.split_size:
            split_num = frame_num // snippet.split_size
            offset = frame_num % snippet.split_size
            file_name = f"./split_imgs/split_{split_num}/{offset:05d}.jpg"
        else:
            file_name = f"./{snippet.frames_dir_name}/{fpath.name}"
        images_list.append({
            "id": frame_num,
            "file_name": file_name,
            "height": snippet.height,
            "width": snippet.width,
        })
        for cat, mask in per_frame_per_cat[fidx].items():
            mask_area = int(mask.sum())
            if mask_area < min_area:
                continue
            ys, xs = np.where(mask > 0)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            ann_list.append({
                "id": ann_id,
                "image_id": frame_num,
                "category_id": cat_to_objid.get(cat, 0),
                "segmentation": mask_to_rle(mask),
                "bbox": bbox,
                "area": float(mask_area),
                "iscrowd": 0,
            })
            ann_id += 1

    categories_list = [{"id": cat_to_objid[c], "name": c}
                       for c in sorted(cat_to_objid.keys())]
    return {
        "categories": categories_list,
        "images": images_list,
        "annotations": ann_list,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="Segments root: contains {EP}/{EP}_snippets.json + snippet_NNN/")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True,
                    help="snippet_id (string, e.g. '001')")
    ap.add_argument("--output-suffix", default="",
                    help="suffix added to annotated_masks.json (default: overwrite)")
    ap.add_argument("--min-area", type=int, default=50,
                    help="minimum polygon area to retain in COCO output")
    ap.add_argument("--max-frames-per-direction", type=int, default=10000,
                    help="cap propagation frames per direction (default: cover the whole snippet)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse autosave + count anchors, do not load SAM3 or write")
    args = ap.parse_args()

    # Tell sam3_annotator.storage where to look
    os.environ["SAM3_ANNOT_DATA_DIR"] = str(Path(args.data_dir).resolve())
    from sam3_annotator.server import storage

    snippet = storage.load_snippet(args.episode, args.snippet)
    autosave_path = snippet.root / "session_autosave.json"
    if not autosave_path.exists():
        print(f"[err] no autosave at {autosave_path}", file=sys.stderr)
        sys.exit(2)
    with open(autosave_path) as f:
        autosave = json.load(f)

    anchors = load_anchors(autosave, snippet.height, snippet.width)
    n_anchors = sum(len(v) for v in anchors.values())
    print(f"snippet: {snippet.episode}/snippet_{snippet.snippet_id}")
    print(f"  frames_dir: {snippet.frames_dir_name}/  ({len(snippet.frame_files)} files)")
    print(f"  resolution: {snippet.width}x{snippet.height}")
    print(f"  split_size: {snippet.split_size}")
    print(f"  anchors: {n_anchors} mask(s) across {len(anchors)} frame(s)")
    if n_anchors == 0:
        print("[err] no anchors in session_autosave.json — nothing to propagate")
        sys.exit(2)

    cat_to_objid = assign_obj_ids(anchors, autosave.get("cat_to_objid") or {})
    print(f"  cat_to_objid: {cat_to_objid}")

    if args.dry_run:
        print("dry-run: stopping before SAM3 load")
        return

    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    print(f"\nLoading SAM3 video predictor (~25-30s)…")
    t0 = time.time()
    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    tracker = predictor.model.tracker
    tracker.backbone = predictor.model.detector.backbone
    print(f"  loaded in {time.time() - t0:.1f}s")

    n = len(snippet.frame_files)
    print(f"\nLoading {n} frames into tracker (image_size={tracker.image_size})…")
    _load = _find_load_frames()
    t0 = time.time()
    images, vh, vw = _load(snippet.frame_files, n, tracker.image_size)
    state = tracker.init_state(video_height=vh, video_width=vw, num_frames=n)
    state["images"] = images
    print(f"  init_state in {time.time() - t0:.1f}s")

    print(f"\nRegistering {n_anchors} anchor(s)…")
    t0 = time.time()
    for fidx in sorted(anchors.keys()):
        for cat, mask in anchors[fidx].items():
            obj_id = cat_to_objid[cat]
            mask_t = torch.from_numpy(mask).float()
            tracker.add_new_mask(
                inference_state=state,
                frame_idx=int(fidx),
                obj_id=int(obj_id),
                mask=mask_t,
            )
            print(f"  + f{fidx:>5} {cat:<14} obj_id={obj_id} area={int(mask.sum())}")
    print(f"  registered in {time.time() - t0:.1f}s")

    print("\nPreflight + bidirectional propagation…")
    objid_to_cat = {v: k for k, v in cat_to_objid.items()}
    per_frame: dict[int, dict[str, np.ndarray]] = {}

    tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
    t0 = time.time()
    for reverse in (False, True):
        direction = "backward" if reverse else "forward"
        seen = 0
        for fidx, obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
            state,
            start_frame_idx=None,
            max_frame_num_to_track=args.max_frames_per_direction,
            reverse=reverse,
        ):
            if fidx >= n:
                continue
            seen += 1
            for i, oid in enumerate(obj_ids):
                cat = objid_to_cat.get(int(oid))
                if cat is None:
                    continue
                ml = video_res_masks[i]
                if isinstance(ml, torch.Tensor):
                    mnp = ml.squeeze(0).cpu().numpy()
                else:
                    mnp = ml
                bm = (mnp > 0.0).astype(np.uint8)
                per_frame.setdefault(fidx, {})[cat] = bm
        print(f"  {direction}: {seen} frames")
    print(f"  propagated in {time.time() - t0:.1f}s")

    # Re-seed approved anchor masks AFTER propagation, overwriting SAM3's
    # cond-frame reconstruction. SAM3's mask decoder produces a smoothed
    # reconstruction at every anchor frame (cond_frame_outputs.pred_masks),
    # losing small interior holes (≤3 px) and sub-pixel boundary detail. The
    # user-painted RLE in the autosave is the authoritative anchor — it must
    # appear verbatim at the anchor frame in the output JSON. A pre-loop
    # seed + in-loop guard was attempted but did not preserve the anchors
    # reliably; an unconditional post-loop overwrite is the robust fix.
    anchor_cells = sum(len(cats) for cats in anchors.values())
    for fidx, cats in anchors.items():
        per_frame.setdefault(fidx, {}).update(cats)
    print(f"  re-seeded {anchor_cells} anchor cells (autosave wins at anchor frames)")

    # Build + write COCO
    coco = build_coco(snippet, per_frame, cat_to_objid, args.min_area)
    out_path = snippet.root / f"annotated_masks{args.output_suffix}.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(coco, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nwrote {out_path}")
    print(f"  images: {len(coco['images'])}, annotations: {len(coco['annotations'])}, "
          f"categories: {[c['name'] for c in coco['categories']]}")

    # Cleanup
    del state, images
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
