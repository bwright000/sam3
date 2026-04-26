#!/usr/bin/env python3
"""Headless bidirectional chunked propagation from session_autosave.json.

Intended for cluster / A100 use: record clicks locally in annotate_ui.py, copy
session_autosave.json + frames to the cluster, then run this script to
propagate and write annotated_masks.json (same schema as the UI export).

Usage:
    # single snippet
    python scripts/propagate_snippet_cli.py --episode E_3 --snippet 001

    # all snippets in episode with a session file
    python scripts/propagate_snippet_cli.py --episode E_3

    # override max chunk size
    python scripts/propagate_snippet_cli.py --episode F_3 --snippet 001 --max-chunk 100

    # confidence threshold for flagging low-conf frames (log only)
    python scripts/propagate_snippet_cli.py --episode E_3 --snippet 001 --conf-threshold 0.3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
SEGMENTS = BASE / "data" / "Segments"
sys.path.insert(0, str(BASE))

# Reuse helpers that don't drag in gradio
from scripts.generate_tool_masks_video import _load_frames_for_tracker  # noqa: E402
from scripts.generate_tool_masks import mask_to_coco_polygons  # noqa: E402


def _masks_from_serializable(serialized: dict, h: int, w: int) -> dict:
    """Mirror of annotate_ui._masks_from_serializable. Polygon-based rehydrate.
    fidx -> {cat: uint8 mask (h, w)}.
    """
    out = {}
    for fidx_str, cats in serialized.items():
        fidx = int(fidx_str)
        out[fidx] = {}
        for cat, mask_data in cats.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in mask_data.get("polygons", []):
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            out[fidx][cat] = mask
    return out


def _frame_hw(frame_file: Path) -> tuple[int, int]:
    with Image.open(frame_file) as im:
        return im.height, im.width


def _build_chunks(anchor_frames, n_frames, max_chunk):
    """Replicate annotate_ui.propagate chunk-builder."""
    if not anchor_frames:
        return []
    boundaries = {0, n_frames - 1}
    for af in anchor_frames:
        boundaries.add(af)
        pos = af + max_chunk
        while pos < n_frames:
            boundaries.add(min(pos, n_frames - 1))
            pos += max_chunk
        pos = af - max_chunk
        while pos > 0:
            boundaries.add(max(pos, 0))
            pos -= max_chunk
    boundaries = sorted(boundaries)

    chunks = []
    for i in range(len(boundaries) - 1):
        c_start, c_end = boundaries[i], boundaries[i + 1]
        has_anchor = any(c_start <= af <= c_end for af in anchor_frames)
        if has_anchor:
            chunks.append((c_start, c_end))
        else:
            if chunks:
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, c_end)
    if not chunks:
        chunks = [(0, n_frames - 1)]
    return chunks


def _propagate_chunk(tracker, images, vh, vw, n_frames,
                     anchor_frames, approved_masks, cat_to_objid, objid_to_cat,
                     chunk_start, chunk_end, conf_threshold, propagated):
    """Run bidirectional propagation for one chunk. Mutates `propagated`."""
    state = tracker.init_state(video_height=vh, video_width=vw, num_frames=n_frames)
    state["images"] = images

    prompts_added = 0
    for fidx in anchor_frames:
        if fidx < chunk_start or fidx > chunk_end:
            continue
        if fidx not in approved_masks:
            continue
        for cat, mask in approved_masks[fidx].items():
            obj_id = cat_to_objid[cat]
            mask_tensor = torch.from_numpy(mask).float()
            try:
                tracker.add_new_mask(
                    inference_state=state, frame_idx=fidx,
                    obj_id=obj_id, mask=mask_tensor,
                )
                prompts_added += 1
            except Exception as e:
                print(f"  WARN: add_new_mask failed f={fidx} cat={cat}: {e}")

    if prompts_added == 0:
        return 0, []

    tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

    low_conf = []
    frames_processed = 0
    chunk_size = chunk_end - chunk_start + 1

    for reverse in (False, True):
        for frame_idx, obj_ids, _, video_res_masks, obj_scores in tracker.propagate_in_video(
            state, start_frame_idx=None, max_frame_num_to_track=chunk_size, reverse=reverse
        ):
            if frame_idx < chunk_start or frame_idx > chunk_end or frame_idx >= n_frames:
                continue
            if frame_idx not in propagated:
                propagated[frame_idx] = {}
            for i, oid in enumerate(obj_ids):
                cat = objid_to_cat.get(int(oid))
                if cat is None:
                    continue
                if frame_idx in approved_masks and cat in approved_masks[frame_idx]:
                    continue
                sv = obj_scores[i]
                obj_score = float(sv.squeeze().cpu().item()) if isinstance(sv, torch.Tensor) else float(sv)
                ml = video_res_masks[i]
                mnp = ml.squeeze(0).cpu().numpy() if isinstance(ml, torch.Tensor) else ml
                binary = (mnp > 0.0).astype(np.uint8)
                existing = propagated[frame_idx].get(cat)
                if existing is None or obj_score > 0:
                    propagated[frame_idx][cat] = binary
                if obj_score < conf_threshold:
                    low_conf.append((frame_idx, cat, obj_score))
            frames_processed += 1

    # Tear down chunk state
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frames_processed, low_conf


def _export_coco(snip_dir: Path, approved, propagated, split_size, frame_files,
                 vh, vw, cat_to_objid, categories, min_area=50):
    """Mirror of annotate_ui export_coco. Writes annotated_masks.json."""
    all_masks = {}
    for fidx, cats in propagated.items():
        all_masks[fidx] = dict(cats)
    for fidx, cats in approved.items():
        all_masks.setdefault(fidx, {}).update(cats)

    if not all_masks:
        return None

    categories_list = [{"id": cat_to_objid[c], "name": c} for c in categories]
    images_list, annotations_list = [], []
    ann_id = 1
    for fidx in sorted(all_masks.keys()):
        if fidx >= len(frame_files):
            continue
        fpath = frame_files[fidx]
        frame_num = int(fpath.stem.split("_")[1])
        split_num = frame_num // split_size
        offset = frame_num % split_size
        images_list.append({
            "id": frame_num,
            "file_name": f"./split_imgs/split_{split_num}/{offset:05d}.jpg",
            "height": vh, "width": vw,
        })
        for cat, mask in all_masks[fidx].items():
            if mask.sum() == 0:
                continue
            polys = mask_to_coco_polygons(mask, min_area=min_area)
            if not polys:
                continue
            ys, xs = np.where(mask > 0)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            annotations_list.append({
                "id": ann_id, "image_id": frame_num,
                "category_id": cat_to_objid[cat],
                "segmentation": polys, "bbox": bbox,
                "area": float(mask.sum()), "iscrowd": 0,
            })
            ann_id += 1

    out = {"categories": categories_list, "images": images_list, "annotations": annotations_list}
    out_path = snip_dir / "annotated_masks.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.replace(out_path)
    return out_path


def process_snippet(predictor, snip_dir: Path, max_chunk_override=None, conf_threshold=0.3):
    session_path = snip_dir / "session_autosave.json"
    if not session_path.exists():
        print(f"[{snip_dir.name}] no session_autosave.json — skip")
        return False

    with open(session_path) as f:
        data = json.load(f)

    categories = data.get("categories", [])
    if not categories:
        print(f"[{snip_dir.name}] session has no categories — skip")
        return False
    cat_to_objid = {c: i + 1 for i, c in enumerate(categories)}
    objid_to_cat = {v: k for k, v in cat_to_objid.items()}

    # Load frames
    frames_dir = snip_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        print(f"[{snip_dir.name}] no frames_left/*.webp — skip")
        return False
    n = len(frame_files)

    # Load split_size
    ep = snip_dir.parent.name
    snippets_json = snip_dir.parent / f"{ep}_snippets.json"
    split_size = 120
    with open(snippets_json) as f:
        for s in json.load(f):
            if s["snippet_id"] == snip_dir.name.split("_")[-1]:
                split_size = s.get("split_size", 120)
                break
    max_chunk = max_chunk_override or split_size

    tracker = predictor.model.tracker
    tracker.backbone = predictor.model.detector.backbone

    orig_h, orig_w = _frame_hw(frame_files[0])

    print(f"[{snip_dir.name}] loading {n} frames for tracker at image_size={tracker.image_size}")
    t0 = time.time()
    images, vh, vw = _load_frames_for_tracker(frame_files, n, tracker.image_size)
    print(f"[{snip_dir.name}] frames loaded ({time.time()-t0:.1f}s) vh={vh} vw={vw}  orig=({orig_h},{orig_w})")

    # Approved masks are stored at original image resolution (polygons).
    approved = _masks_from_serializable(data.get("approved_masks", {}), orig_h, orig_w)
    if not approved:
        print(f"[{snip_dir.name}] no approved masks in session — nothing to propagate")
        return False

    anchor_frames = sorted(approved.keys())
    chunks = _build_chunks(anchor_frames, n, max_chunk)
    print(f"[{snip_dir.name}] anchors={len(anchor_frames)} chunks={len(chunks)} "
          f"max_chunk={max_chunk}")

    propagated = {}
    all_low = []
    total_processed = 0
    for ci, (cs, ce) in enumerate(chunks, 1):
        t1 = time.time()
        frames_in_chunk, low = _propagate_chunk(
            tracker, images, vh, vw, n,
            anchor_frames, approved, cat_to_objid, objid_to_cat,
            cs, ce, conf_threshold, propagated,
        )
        total_processed += frames_in_chunk
        all_low.extend(low)
        print(f"[{snip_dir.name}]   chunk {ci}/{len(chunks)} [{cs},{ce}] "
              f"frames={frames_in_chunk} low_conf={len(low)} ({time.time()-t1:.1f}s)")

    out_path = _export_coco(
        snip_dir, approved, propagated, split_size, frame_files,
        vh, vw, cat_to_objid, categories,
    )
    print(f"[{snip_dir.name}] wrote {out_path} (propagated={total_processed} frames, "
          f"low_conf={len(all_low)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", default=None, help="e.g. 001; omit to process all")
    ap.add_argument("--max-chunk", type=int, default=None)
    ap.add_argument("--conf-threshold", type=float, default=0.3)
    args = ap.parse_args()

    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    print("Loading SAM3 video predictor...")
    t0 = time.time()
    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    print(f"Predictor loaded in {time.time()-t0:.1f}s")

    ep_dir = SEGMENTS / args.episode
    if args.snippet:
        targets = [ep_dir / f"snippet_{args.snippet}"]
    else:
        targets = sorted(ep_dir.glob("snippet_*"))

    ok = 0
    for snip in targets:
        if not snip.exists():
            print(f"skip missing {snip}")
            continue
        try:
            if process_snippet(predictor, snip, args.max_chunk, args.conf_threshold):
                ok += 1
        except Exception as e:
            print(f"[{snip.name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    print(f"\ndone: {ok}/{len(targets)} snippets produced annotated_masks.json")


if __name__ == "__main__":
    main()
