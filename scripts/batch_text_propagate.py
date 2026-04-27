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


def union_masks_per_frame(per_frame_results: dict[int, dict]) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """For each frame, union all detected tool masks into one binary mask.

    Returns:
      unions: {frame_idx: binary mask (H, W)}
      counts: {frame_idx: number_of_distinct_tool_detections}
    """
    import cv2
    unions: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for fidx, result in per_frame_results.items():
        masks = result.get("masks", {})
        tool_masks = masks.get("tool", []) or masks.get("Tool", [])
        counts[fidx] = len(tool_masks) if tool_masks else 0
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
            for poly in polys:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(u, [pts], 1)
        if u.sum() > 0:
            unions[fidx] = u
    return unions, counts


def render_overlays(snip_dir: Path, frame_files: list[Path],
                    per_frame_results: dict[int, dict], counts: dict[int, int]) -> int:
    """Render per-frame overlays showing detected tool masks + counts."""
    import cv2
    out_dir = snip_dir / "overlays_tool"
    out_dir.mkdir(exist_ok=True)
    written = 0
    for fidx, result in per_frame_results.items():
        if fidx >= len(frame_files):
            continue
        fp = frame_files[fidx]
        img = cv2.imread(str(fp))
        if img is None:
            continue
        masks = result.get("masks", {})
        tool_masks = masks.get("tool", []) or masks.get("Tool", [])
        n = counts.get(fidx, 0)
        # Draw each detection in a different color
        palette = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
                   (255, 255, 80), (255, 80, 255), (80, 255, 255)]
        for i, m in enumerate(tool_masks):
            color = palette[i % len(palette)]
            for poly in m.get("segmentation", []):
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], True, color, 2)
                # filled overlay at 30% alpha
                fill = img.copy()
                cv2.fillPoly(fill, [pts], color)
                img = cv2.addWeighted(fill, 0.3, img, 0.7, 0)
            score = m.get("score", 0)
            if poly := m.get("segmentation", [[]])[0]:
                if len(poly) >= 2:
                    x, y = int(poly[0]), int(poly[1])
                    cv2.putText(img, f"#{i} {score:.2f}", (x, max(15, y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Frame counter top-left
        cv2.putText(img, f"tools: {n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{fp.stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        written += 1
    return written


def write_annotated_masks(snip_dir: Path, frame_files: list[Path], split_size: int,
                          unions: dict[int, np.ndarray], h: int, w: int,
                          suffix: str = "") -> Path:
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
    fname = f"annotated_masks{suffix}.json"
    out_path = snip_dir / fname
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f)
    tmp.replace(out_path)
    return out_path


def process_snippet(predictor, snip_dir: Path, prompt: str = "tool",
                    min_area: int = 100, do_render_overlays: bool = False,
                    expected_tools: int | None = None,
                    output_suffix: str = "") -> dict:
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

    unions, counts = union_masks_per_frame(per_frame)
    out_path = write_annotated_masks(snip_dir, frame_files, split_size, unions, h, w,
                                     suffix=output_suffix)

    # Histogram of detection counts
    from collections import Counter
    hist = Counter(counts.values())
    n_zero = hist.get(0, 0)
    # If user told us how many tools to expect, count "matching" frames
    matches_pct = None
    if expected_tools is not None:
        n_match = sum(1 for v in counts.values() if v == expected_tools)
        matches_pct = round(100 * n_match / max(1, n), 1)

    rendered = 0
    if do_render_overlays:
        rendered = render_overlays(snip_dir, frame_files, per_frame, counts)

    elapsed = time.time() - t0

    # Persist count stats next to annotated_masks.json for later analysis
    stats_path = snip_dir / f"tool_detection_stats{output_suffix}.json"
    with open(stats_path, "w") as f:
        json.dump({
            "snippet": f"{ep}/{snip_dir.name}",
            "frames": n,
            "histogram": dict(sorted(hist.items())),  # {count: n_frames}
            "expected_tools": expected_tools,
            "matches_expected_pct": matches_pct,
            "per_frame_count": {str(k): v for k, v in sorted(counts.items())},
        }, f, indent=2)

    return {
        "status": "ok",
        "snippet": f"{ep}/{snip_dir.name}",
        "frames": n,
        "frames_with_tool": len(unions),
        "coverage_pct": round(100 * len(unions) / max(1, n), 1),
        "histogram": dict(sorted(hist.items())),
        "expected_tools": expected_tools,
        "matches_expected_pct": matches_pct,
        "overlays_rendered": rendered,
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
    ap.add_argument("--render-overlays", action="store_true",
                    help="render per-frame overlay JPGs to snippet/overlays_tool/")
    ap.add_argument("--expected-json", default=None,
                    help="path to JSON {ep/snippet_id: expected_tool_count}")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="override Sam3VideoInference score_threshold_detection "
                         "(default 0.5; lower = more permissive detector)")
    ap.add_argument("--new-det-thresh", type=float, default=None,
                    help="override Sam3VideoInference new_det_thresh "
                         "(default 0.7; lower = spawns new tracks more eagerly)")
    ap.add_argument("--output-suffix", default="",
                    help="append to output filenames, e.g. '.lowthresh' for Pass 2")
    args = ap.parse_args()

    expected_map = {}
    if args.expected_json:
        with open(args.expected_json) as f:
            expected_map = json.load(f)

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
    # Cast model weights to bf16 to match autocast inputs.
    # Without this, SAM3 raises "Input type BFloat16 and bias type float should be the same"
    # at sam_mask_decoder.conv_s0 (the conv layer's bias stays fp32 by default).
    if torch.cuda.is_available():
        predictor.model.to(dtype=torch.bfloat16)
        print("  cast model to bfloat16")

    # Optional threshold overrides (for Pass 2 / low-threshold reruns)
    if args.score_threshold is not None:
        old = getattr(predictor.model, "score_threshold_detection", None)
        predictor.model.score_threshold_detection = float(args.score_threshold)
        print(f"  score_threshold_detection: {old} -> {args.score_threshold}")
    if args.new_det_thresh is not None:
        old = getattr(predictor.model, "new_det_thresh", None)
        predictor.model.new_det_thresh = float(args.new_det_thresh)
        print(f"  new_det_thresh: {old} -> {args.new_det_thresh}")

    import torch, gc

    def _free_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            return f"VRAM free {free/1e9:.1f}/{total/1e9:.1f} GB"
        return ""

    print(f"  startup {_free_gpu()}")

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
            print(f"\n=== {ep}/{snip_dir.name} ===  {_free_gpu()}")
            ann_path = snip_dir / f"annotated_masks{args.output_suffix}.json"
            if args.skip_if_exists and ann_path.exists():
                print(f"  skip ({ann_path.name} exists)")
                continue
            try:
                key = f"{ep}/{snip_dir.name.split('_')[-1]}"
                expected = expected_map.get(key) or expected_map.get(f"{ep}/{snip_dir.name}")
                info = process_snippet(predictor, snip_dir, args.prompt, args.min_area,
                                       do_render_overlays=args.render_overlays,
                                       expected_tools=expected,
                                       output_suffix=args.output_suffix)
                summary.append(info)
                hist_str = " ".join(f"{k}:{v}" for k, v in info["histogram"].items())
                exp_str = (f"  expected={info['expected_tools']} matches={info['matches_expected_pct']}%"
                           if info['expected_tools'] is not None else "")
                print(f"  -> {info['frames_with_tool']}/{info['frames']} frames "
                      f"({info['coverage_pct']}%) hist[{hist_str}]{exp_str}  "
                      f"in {info['elapsed_s']}s")
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM: {e}")
                summary.append({"status": "oom", "snippet": f"{ep}/{snip_dir.name}"})
                _free_gpu()
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                summary.append({"status": "error", "snippet": f"{ep}/{snip_dir.name}",
                                "error": str(e)})
            finally:
                # Free per-snippet GPU state so the next snippet starts clean
                _free_gpu()

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
