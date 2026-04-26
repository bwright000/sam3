#!/usr/bin/env python3
"""Render per-frame GT overlay JPGs into a snippet's overlays/ dir.

Matches the visual style of the existing pipeline overlays:
  - filled polygons, semi-transparent
  - colored outline per category (Liver=red, Gallbladder=green, etc.)
  - top-left counter showing category counts for the frame

Reads snippet_annotations.json (the canonical GT) and writes
data/Segments/{EP}/snippet_NNN/overlays/frame_NNNNNN.jpg.

Memory: processes one frame at a time. Backs up existing overlays/ to
overlays.bak_pre_render/ if not already present.

Usage:
    python scripts/render_gt_overlays.py --episode E_3 --snippet 004
    python scripts/render_gt_overlays.py --episode F_3 --snippet 005 --episode F_3 --snippet 006
    python scripts/render_gt_overlays.py --all-missing   # render only snippets without overlays/
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SEG = BASE / "data" / "Segments"

# BGR colors per category (matches existing overlays)
CAT_COLORS = {
    "Liver": (0, 0, 255),         # red
    "Gallbladder": (0, 255, 0),   # green
    "Meat": (0, 165, 255),        # orange
    "Skin": (203, 192, 255),      # pink
    "FBF": (255, 0, 0),           # blue
    "PCH": (255, 255, 0),         # cyan
    "Tool": (255, 128, 0),        # blue-ish
}
DEFAULT_COLOR = (0, 255, 255)
FILL_ALPHA = 0.35


def render_frame_overlay(frame_bgr: np.ndarray,
                         polys_by_cat: dict[str, list[list[float]]],
                         counts: dict[str, int]) -> np.ndarray:
    """Draw filled+outlined polys with category counter."""
    overlay = frame_bgr.copy()
    out = frame_bgr.copy()

    # Draw fills onto overlay
    for cat, polys in polys_by_cat.items():
        col = CAT_COLORS.get(cat, DEFAULT_COLOR)
        for poly_flat in polys:
            if len(poly_flat) < 6:
                continue
            pts = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], col)

    # Blend overlay with original for semi-transparent fill
    cv2.addWeighted(overlay, FILL_ALPHA, out, 1 - FILL_ALPHA, 0, out)

    # Draw outlines (full opacity) on top
    for cat, polys in polys_by_cat.items():
        col = CAT_COLORS.get(cat, DEFAULT_COLOR)
        for poly_flat in polys:
            if len(poly_flat) < 6:
                continue
            pts = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=col, thickness=2)

    # Counter top-left
    y = 22
    for cat in sorted(counts.keys()):
        col = CAT_COLORS.get(cat, DEFAULT_COLOR)
        cv2.putText(out, f"{cat.lower()}: {counts[cat]}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += 18
    return out


def process_snippet(ep: str, snip_id: str) -> dict:
    snip_dir = SEG / ep / f"snippet_{snip_id}"
    if not snip_dir.is_dir():
        return {"status": "skip", "reason": "missing dir"}

    ann_path = snip_dir / "snippet_annotations.json"
    if not ann_path.exists():
        return {"status": "skip", "reason": "no snippet_annotations.json"}

    frames_dir = snip_dir / "frames_left"
    if not frames_dir.is_dir():
        return {"status": "skip", "reason": "no frames_left/"}

    overlays_dir = snip_dir / "overlays"
    if overlays_dir.exists() and any(overlays_dir.glob("frame_*.jpg")):
        # Backup before re-render
        bak = snip_dir / "overlays.bak_pre_render"
        if not bak.exists():
            shutil.move(str(overlays_dir), str(bak))
            print(f"  backed up existing overlays/ -> overlays.bak_pre_render/")
    overlays_dir.mkdir(exist_ok=True)

    with open(ann_path) as f:
        ann = json.load(f)
    cat_by_id = {c["id"]: c["name"] for c in ann.get("categories", [])}

    # Index annotations by frame_num (= image_id)
    by_frame: dict[int, dict[str, list[list[float]]]] = {}
    for a in ann.get("annotations", []):
        cat_name = cat_by_id.get(a.get("category_id"))
        if cat_name is None:
            continue
        seg = a.get("segmentation") or []
        if not isinstance(seg, list):
            continue
        polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
        if not polys:
            continue
        by_frame.setdefault(a["image_id"], {}).setdefault(cat_name, []).extend(polys)

    n_written = 0
    n_skipped = 0
    t0 = time.time()
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    for fp in frame_files:
        try:
            fn = int(fp.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            n_skipped += 1
            continue
        polys = by_frame.get(fn, {})
        counts = {cat: len(p) for cat, p in polys.items()}
        out = render_frame_overlay(img, polys, counts)
        out_path = overlays_dir / f"frame_{fn:06d}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_written += 1
        # Free large arrays
        del img, out
    elapsed = time.time() - t0
    return {
        "status": "ok",
        "frames_written": n_written,
        "frames_skipped": n_skipped,
        "anns_by_frame": len(by_frame),
        "elapsed_s": round(elapsed, 1),
    }


def find_missing_overlays() -> list[tuple[str, str]]:
    out = []
    for ep_dir in sorted(SEG.iterdir()):
        if not ep_dir.is_dir():
            continue
        for snip in sorted(ep_dir.glob("snippet_*")):
            ov = snip / "overlays"
            ann = snip / "snippet_annotations.json"
            has_ov = ov.exists() and any(ov.glob("frame_*.jpg"))
            if not has_ov and ann.exists():
                out.append((ep_dir.name, snip.name.split("_")[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="append", default=[])
    ap.add_argument("--snippet", action="append", default=[])
    ap.add_argument("--all-missing", action="store_true",
                    help="render only snippets that have a snippet_annotations.json but no overlays/")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    if args.all_missing:
        targets = find_missing_overlays()
    elif args.episode and args.snippet:
        if len(args.episode) != len(args.snippet):
            sys.exit("ERROR: --episode and --snippet must have matching counts")
        targets = list(zip(args.episode, args.snippet))
    else:
        sys.exit("Specify --all-missing OR --episode + --snippet pairs")

    if not targets:
        print("nothing to do.")
        return

    print(f"will render {len(targets)} snippets:")
    for ep, sid in targets:
        print(f"  {ep}/snippet_{sid}")
    print()

    for ep, sid in targets:
        print(f"== {ep}/snippet_{sid} ==")
        info = process_snippet(ep, sid)
        if info["status"] == "ok":
            print(f"  wrote {info['frames_written']} overlays "
                  f"(GT in {info['anns_by_frame']} frames) "
                  f"in {info['elapsed_s']}s")
        else:
            print(f"  skip: {info.get('reason')}")


if __name__ == "__main__":
    main()
