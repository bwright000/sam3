#!/usr/bin/env python3
"""Batch tool annotation via text prompt + SAM3 video propagation.

For each snippet:
  1. Open SAM3 video session on frames_left/
  2. add_prompt(text="tool") at the first GT keyframe (or frame 0)
  3. Propagate bidirectionally
  4. Union all detected tool masks per frame into a single Tool mask
  5. Write annotated_masks.json (same format as the new annotator's Export)

Designed to run headlessly while the interactive annotator is being debugged.
After this finishes, run `scripts/merge_tool_masks.py` to merge into
snippet_annotations.json.

Usage:
    python scripts/batch_text_propagate.py \\
        --data-dir '/content/data/To Be Annotated' \\
        --episodes E_3 F_3 \\
        --prompt tool

    # Single snippet
    python scripts/batch_text_propagate.py \\
        --data-dir '/content/data/To Be Annotated' \\
        --episode E_3 --snippet 001
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
from scripts.generate_tool_masks_video import _convert_video_output


# COCO category ids from shared_config
TOOL_CAT = {"id": 6, "name": "Tool", "supercategory": "Tool"}


def find_keyframes(frame_files: list[Path], split_size: int) -> list[int]:
    """Local indices of frames where frame_n % split_size == 0."""
    out = []
    for i, p in enumerate(frame_files):
        try:
            fn = int(p.stem.split("_")[1])
            if fn % split_size == 0:
                out.append(i)
        except (ValueError, IndexError):
            pass
    return out


def get_split_size(snip_dir: Path, snip_id: str) -> int:
    ep = snip_dir.parent.name
    p = snip_dir.parent / f"{ep}_snippets.json"
    if not p.exists():
        return 120
    with open(p) as f:
        for s in json.load(f):
            if s["snippet_id"] == snip_id:
                return int(s.get("split_size", 120))
    return 120


def union_masks_per_frame(per_frame_results: dict[int, dict]) -> dict[int, np.ndarray]:
    """For each frame, union all detected tool masks into one binary mask."""
    unions: dict[int, np.ndarray] = {}
    for fidx, result in per_frame_results.items():
        masks = result.get("masks", {})
        tool_masks = masks.get("tool", []) or masks.get("Tool", [])
        if not tool_masks:
            continue
        h = result.get("height", 0)
        w = result.get("width", 0)
        if h == 0 or w == 0:
            continue
        u = np.zeros((h, w), dtype=np.uint8)
        for m in tool_masks:
            polys = m.get("segmentation", [])
            if not polys:
                continue
            import cv2
            for poly in polys:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(u, [pts], 1)
        if u.sum() > 0:
            unions[fidx] = u
    return unions


def write_annotated_masks(snip_dir: Path, frame_files: list[Path], split_size: int,
                          unions: dict[int, np.ndarray], h: int, w: int) -> Path:
    images = []
    annotations = []
    ann_id = 1
    for fidx in sorted(unions.keys()):
        if fidx >= len(frame_files):
            continue
        fp = frame_files[fidx]
        frame_num = int(fp.stem.split("_")[1])
        split_n = frame_num // split_size
        offset = frame_num % split_size
        images.append({
            "id": frame_num,
            "width": w,
            "height": h,
            "file_name": f"./split_imgs/split_{split_n}/{offset:05d}.jpg",
        })
        mask = unions[fidx]
        polys = mask_to_coco_polygons(mask, min_area=50)
        if not polys:
            continue
        ys, xs = np.where(mask > 0)
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        annotations.append({
            "id": ann_id,
            "image_id": frame_num,
            "category_id": TOOL_CAT["id"],
            "segmentation": polys,
            "bbox": bbox,
            "area": float(mask.sum()),
            "iscrowd": 0,
        })
        ann_id += 1

    out = {"categories": [TOOL_CAT], "images": images, "annotations": annotations}
    out_path = snip_dir / "annotated_masks.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f)
    tmp.replace(out_path)
    return out_path


def process_snippet(predictor, snip_dir: Path, prompt: str = "tool",
                    min_area: int = 100) -> dict:
    snip_id = snip_dir.name.split("_")[-1]
    ep = snip_dir.parent.name
    frames_dir = snip_dir / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        return {"status": "skip", "reason": "no frames"}

    split_size = get_split_size(snip_dir, snip_id)
    keyframes = find_keyframes(frame_files, split_size)
    if not keyframes:
        keyframes = [0]  # fallback

    with Image.open(frame_files[0]) as im:
        h, w = im.height, im.width
    n = len(frame_files)

    print(f"  {n} frames, split_size={split_size}, "
          f"keyframes_local={keyframes[:5]}{'...' if len(keyframes)>5 else ''}")

    t0 = time.time()
    session = predictor.start_session(resource_path=str(frames_dir))
    sid = session["session_id"]
    per_frame: dict[int, dict] = {}

    try:
        # Text prompt is global — one call sets it for all frames.
        # frame_idx is just where to anchor the initial inference.
        anchor = keyframes[0]
        print(f"  add_prompt text='{prompt}' @ local frame {anchor}")
        predictor.add_prompt(session_id=sid, frame_idx=anchor, text=prompt)

        # Forward then backward
        for direction in ("forward", "backward"):
            seen = 0
            for response in predictor.propagate_in_video(
                session_id=sid,
                propagation_direction=direction,
                start_frame_idx=None,
                max_frame_num_to_track=n,
            ):
                fidx = response["frame_index"]
                outputs = response["outputs"]
                if fidx >= n:
                    continue
                result = _convert_video_output(outputs, frame_files[fidx], prompt, min_area)
                per_frame[fidx] = result
                seen += 1
            print(f"  {direction}: {seen} frames")
    finally:
        predictor.close_session(session_id=sid)

    unions = union_masks_per_frame(per_frame)
    out_path = write_annotated_masks(snip_dir, frame_files, split_size, unions, h, w)
    elapsed = time.time() - t0

    return {
        "status": "ok",
        "snippet": f"{ep}/{snip_dir.name}",
        "frames": n,
        "frames_with_tool": len(unions),
        "coverage_pct": round(100 * len(unions) / max(1, n), 1),
        "elapsed_s": round(elapsed, 1),
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="e.g. --episodes E_3 F_3")
    ap.add_argument("--episode", default=None, help="single episode")
    ap.add_argument("--snippet", default=None, help="single snippet (with --episode)")
    ap.add_argument("--prompt", default="tool")
    ap.add_argument("--min-area", type=int, default=100,
                    help="drop detections with mask area below this many pixels")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="skip snippets that already have annotated_masks.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    if args.episode:
        eps_to_run = [args.episode]
    elif args.episodes:
        eps_to_run = args.episodes
    else:
        eps_to_run = ["E_3", "F_3"]

    print(f"Loading SAM3 video predictor (this takes ~25-30s)...")
    t0 = time.time()
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    predictor = Sam3VideoPredictor(apply_temporal_disambiguation=True)
    print(f"Loaded in {time.time()-t0:.1f}s")

    summary = []
    for ep in eps_to_run:
        ep_dir = data_dir / ep
        if not ep_dir.is_dir():
            print(f"\n!! {ep_dir} not found")
            continue
        snippets = (
            [ep_dir / f"snippet_{args.snippet}"]
            if args.snippet else sorted(ep_dir.glob("snippet_*"))
        )
        for snip_dir in snippets:
            if not snip_dir.is_dir():
                continue
            print(f"\n=== {ep}/{snip_dir.name} ===")
            ann_path = snip_dir / "annotated_masks.json"
            if args.skip_if_exists and ann_path.exists():
                print(f"  skip (annotated_masks.json exists)")
                continue
            try:
                info = process_snippet(predictor, snip_dir, args.prompt, args.min_area)
                summary.append(info)
                print(f"  -> {info['frames_with_tool']}/{info['frames']} frames "
                      f"({info['coverage_pct']}%) in {info['elapsed_s']}s")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                summary.append({"status": "error", "snippet": f"{ep}/{snip_dir.name}",
                                "error": str(e)})

    print("\n\n=== SUMMARY ===")
    for s in summary:
        if s.get("status") == "ok":
            print(f"  {s['snippet']:<22} {s['frames_with_tool']:>4}/{s['frames']:<4} "
                  f"({s['coverage_pct']:>5.1f}%) {s['elapsed_s']:>6.1f}s")
        else:
            print(f"  {s.get('snippet','?'):<22} {s.get('status','?')} "
                  f"{s.get('error', s.get('reason', ''))}")


if __name__ == "__main__":
    main()
