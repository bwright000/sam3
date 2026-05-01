#!/usr/bin/env python3
"""Stage 3 of the auto-gap-fill pipeline.

For each snippet listed in gap_manifest.json:
  1. Load SAM3 video model.
  2. Build a tracker state on the snippet's frames_left/.
  3. For each seed PNG (gap-fill or full-rerun) listed in anchor_index.json:
       - Convert PNG to binary tensor.
       - tracker.add_new_mask(state, frame_idx=anchor_idx, obj_id=N, mask=tensor)
  4. Run a single bidirectional propagation over the whole snippet (forward,
     then backward).
  5. For every frame, union the masks from all seed obj_ids -> single Tool mask.
  6. Write annotated_masks.gapfill.json + tool_detection_stats.gapfill.json.

This script must run on Colab A100 (heavy SAM3 compute). Local GTX 970 will TDR.

Usage (Colab):
    python scripts/propagate_gap_fill.py \\
        --manifest /content/outputs/gap_manifest.json \\
        --anchors-dir /content/outputs/anchors \\
        --data-dir '/content/data/To Be Annotated' \\
        --output-suffix .gapfill
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.generate_tool_masks import mask_to_coco_polygons


TOOL_CAT = {"id": 6, "name": "Tool", "supercategory": "Tool"}


def get_split_size(snip_dir, snip_id):
    ep_dir = snip_dir.parent
    p = ep_dir / f"{ep_dir.name}_snippets.json"
    if not p.exists():
        return 120
    try:
        with open(p) as f:
            for s in json.load(f):
                if s.get("snippet_id") == snip_id:
                    return int(s.get("split_size", 120))
    except Exception:
        pass
    return 120


def _load_frames_for_tracker(frame_files, num_frames, image_size):
    """Mirror of generate_tool_masks_video._load_frames_for_tracker."""
    import torch
    from sam3.model.io_utils import _load_img_as_tensor

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    video_h = video_w = None
    for n, fpath in enumerate(frame_files[:num_frames]):
        images[n], video_h, video_w = _load_img_as_tensor(str(fpath), image_size)
    if torch.cuda.is_available():
        images = images.cuda()
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16)[:, None, None]
    if torch.cuda.is_available():
        mean = mean.cuda(); std = std.cuda()
    images -= mean
    images /= std
    return images, video_h, video_w


def load_seed_mask_png(path, target_h, target_w):
    """Load a binary PNG and resize to (target_h, target_w) preserving binary."""
    arr = np.array(Image.open(path).convert("L"))
    if arr.shape != (target_h, target_w):
        # Resize via nearest-neighbour to keep binary edges
        import cv2
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (arr > 127).astype(np.uint8)


def collect_seeds_for_snippet(anchor_index, ep, snippet_name):
    """Return [{anchor_idx, png_path, kind, gap_range}] for matching entry."""
    for s in anchor_index.get("snippets", []):
        if s["ep"] == ep and s["snippet"] == snippet_name:
            seeds = []
            for seed in s["seeds"]:
                if seed.get("status") != "ok":
                    continue
                if seed.get("kind") == "full_rerun_seed":
                    seeds.append({
                        "kind": "full_rerun_seed",
                        "anchor_idx": seed["snip_idx"],
                        "png": seed["out"],
                        "gap_start": None,
                        "gap_end": None,
                    })
                else:
                    seeds.append({
                        "kind": "gap_seed",
                        "anchor_idx": seed["anchor_idx"],
                        "png": seed["out"],
                        "gap_start": seed["snip_idx_start"],
                        "gap_end": seed["snip_idx_end"],
                    })
            return seeds
    return []


def process_snippet(predictor, snip_dir, seeds, output_suffix, expected_tools,
                    full_rerun, min_area=100):
    """Run targeted re-prop with the given seeds.

    Returns a result dict (status, frames_with_tool, etc.)."""
    import torch

    snip_id = snip_dir.name.split("_")[-1].split(" ")[0]
    ep = snip_dir.parent.name
    frames_dir = snip_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        return {"status": "skip", "reason": "no_frames"}
    n = len(frame_files)
    split_size = get_split_size(snip_dir, snip_id)

    tracker = predictor.model.tracker

    print(f"  Loading {n} frames into tracker...")
    images, video_h, video_w = _load_frames_for_tracker(frame_files, n, tracker.image_size)
    state = tracker.init_state(video_height=video_h, video_width=video_w, num_frames=n)
    state["images"] = images

    # Add seeds
    seed_specs = []
    for i, seed in enumerate(seeds):
        obj_id = i + 1
        frame_idx = int(seed["anchor_idx"])
        if frame_idx >= n:
            print(f"    seed {i}: frame_idx {frame_idx} >= n {n}, skipping")
            continue
        mask = load_seed_mask_png(seed["png"], video_h, video_w)
        if mask.sum() == 0:
            print(f"    seed {i}: empty mask {seed['png']}, skipping")
            continue
        mask_t = torch.from_numpy(mask).float()
        try:
            tracker.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask_t,
            )
            seed_specs.append({**seed, "obj_id": obj_id})
            print(f"    + seed obj_id={obj_id} frame_idx={frame_idx} "
                  f"(area={int(mask.sum())})")
        except Exception as e:
            print(f"    seed {i}: add_new_mask failed: {e}")
            traceback.print_exc()

    if not seed_specs:
        del state, images
        torch.cuda.empty_cache()
        return {"status": "skip", "reason": "no_seeds_added"}

    tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

    per_frame_union = {}  # fidx -> uint8 (H, W) at video res
    per_frame_per_seed_count = {}  # fidx -> { obj_id: 0/1 }

    t0 = time.time()
    for reverse in (False, True):
        seen = 0
        direction = "backward" if reverse else "forward"
        for fidx, obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
            state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=reverse,
        ):
            if fidx >= n:
                continue
            seen += 1
            for i, oid in enumerate(obj_ids):
                ml = video_res_masks[i]
                if isinstance(ml, torch.Tensor):
                    mnp = ml.squeeze(0).cpu().numpy()
                else:
                    mnp = ml
                bm = (mnp > 0.0).astype(np.uint8)
                if bm.sum() < min_area:
                    continue
                per_frame_per_seed_count.setdefault(fidx, {})[int(oid)] = 1
                if fidx not in per_frame_union:
                    per_frame_union[fidx] = np.zeros_like(bm)
                np.maximum(per_frame_union[fidx], bm, out=per_frame_union[fidx])
        print(f"    propagate {direction}: {seen} frames")

    elapsed = time.time() - t0

    # Build COCO output
    images_out, anns_out = [], []
    ann_id = 1
    for fidx in sorted(per_frame_union.keys()):
        if fidx >= len(frame_files):
            continue
        fp = frame_files[fidx]
        frame_num = int(fp.stem.split("_")[1])
        split_n = frame_num // split_size
        offset = frame_num % split_size
        images_out.append({
            "id": frame_num,
            "width": video_w,
            "height": video_h,
            "file_name": f"./split_imgs/split_{split_n}/{offset:05d}.jpg",
        })
        mask = per_frame_union[fidx]
        polys = mask_to_coco_polygons(mask, min_area=50)
        if not polys:
            continue
        ys, xs = np.where(mask > 0)
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        anns_out.append({
            "id": ann_id,
            "image_id": frame_num,
            "category_id": TOOL_CAT["id"],
            "segmentation": polys,
            "bbox": bbox,
            "area": float(mask.sum()),
            "iscrowd": 0,
        })
        ann_id += 1

    out = {
        "categories": [TOOL_CAT],
        "images": images_out,
        "annotations": anns_out,
        "_gapfill_meta": {
            "full_rerun": full_rerun,
            "seeds": [
                {k: v for k, v in s.items() if k != "png"} | {"png": str(Path(s["png"]).name)}
                for s in seed_specs
            ],
        },
    }
    out_path = snip_dir / f"annotated_masks{output_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)

    # Build per-frame count and stats
    counts = {}
    for fidx in range(n):
        counts[fidx] = sum(per_frame_per_seed_count.get(fidx, {}).values())
    from collections import Counter
    hist = Counter(counts.values())
    n_zero = hist.get(0, 0)
    matches_pct = None
    if expected_tools is not None:
        matches_pct = round(100 * sum(1 for v in counts.values()
                                      if v == expected_tools) / max(1, n), 1)

    stats_path = snip_dir / f"tool_detection_stats{output_suffix}.json"
    with open(stats_path, "w") as f:
        json.dump({
            "snippet": f"{ep}/{snip_dir.name}",
            "frames": n,
            "histogram": dict(sorted(hist.items())),
            "expected_tools": expected_tools,
            "matches_expected_pct": matches_pct,
            "per_frame_count": {str(k): v for k, v in sorted(counts.items())},
            "_gapfill_meta": {"full_rerun": full_rerun,
                              "n_seeds": len(seed_specs)},
        }, f, indent=2)

    del state, images
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "snippet": f"{ep}/{snip_dir.name}",
        "frames": n,
        "frames_with_tool": len(per_frame_union),
        "coverage_pct": round(100 * len(per_frame_union) / max(1, n), 1),
        "histogram": dict(sorted(hist.items())),
        "expected_tools": expected_tools,
        "matches_expected_pct": matches_pct,
        "n_seeds": len(seed_specs),
        "elapsed_s": round(elapsed, 1),
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Stage 1 gap_manifest.json")
    ap.add_argument("--anchors-dir", required=True, help="Stage 2 anchors directory")
    ap.add_argument("--data-dir", required=True,
                    help="staged 'To Be Annotated' root")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to ep/snippet (matches manifest 'ep'/'snippet')")
    ap.add_argument("--output-suffix", default=".gapfill")
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--score-threshold", type=float, default=None)
    ap.add_argument("--new-det-thresh", type=float, default=None)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    anchor_index = json.load(open(Path(args.anchors_dir) / "anchor_index.json"))
    data_dir = Path(args.data_dir)

    print("Loading SAM3 video predictor (~25-30s)...")
    t0 = time.time()
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    print(f"  loaded in {time.time()-t0:.1f}s")

    if args.score_threshold is not None:
        predictor.model.score_threshold_detection = float(args.score_threshold)
    if args.new_det_thresh is not None:
        predictor.model.new_det_thresh = float(args.new_det_thresh)

    summary = []
    for entry in manifest.get("snippets", []):
        if entry.get("skipped"):
            continue
        ep = entry["ep"]
        snippet_name = entry["snippet"]
        rel = f"{ep}/{snippet_name}"
        rel_stripped = f"{ep}/{snippet_name.replace(' tbd', '').strip()}"
        if args.only and rel not in args.only and rel_stripped not in args.only:
            continue
        full_rerun = bool(entry.get("full_rerun"))
        if not full_rerun and not entry.get("gaps"):
            print(f"\n=== {rel} ===  no gaps, skipping")
            continue

        snip_dir = data_dir / ep / snippet_name
        if not snip_dir.is_dir():
            print(f"\n=== {rel} ===  not a directory: {snip_dir}")
            continue

        seeds = collect_seeds_for_snippet(anchor_index, ep, snippet_name)
        if not seeds:
            print(f"\n=== {rel} ===  no seeds in anchor_index, skipping")
            continue

        print(f"\n=== {rel} ===  full_rerun={full_rerun} seeds={len(seeds)}")
        try:
            info = process_snippet(
                predictor, snip_dir, seeds, args.output_suffix,
                expected_tools=entry.get("expected_tools"),
                full_rerun=full_rerun,
                min_area=args.min_area,
            )
            summary.append(info)
            if info.get("status") == "ok":
                print(f"  -> {info['frames_with_tool']}/{info['frames']} "
                      f"({info['coverage_pct']}%) match={info['matches_expected_pct']}%  "
                      f"in {info['elapsed_s']}s")
            else:
                print(f"  -> {info.get('status')}: {info.get('reason')}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append({"status": "error", "snippet": rel, "error": str(e)})

    print("\n=== SUMMARY ===")
    for s in summary:
        if s.get("status") == "ok":
            print(f"  ok    {s['snippet']:<22} {s['frames_with_tool']:>4}/{s['frames']:<4} "
                  f"({s['coverage_pct']:>5.1f}%) match={s.get('matches_expected_pct')}%")
        else:
            print(f"  {s.get('status'):<6} {s.get('snippet','?')} "
                  f"{s.get('error', s.get('reason',''))}")


if __name__ == "__main__":
    main()
