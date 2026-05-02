#!/usr/bin/env python3
"""Render per-frame overlay JPGs from a COCO annotations file.

Decouples visualisation from the SAM3 propagation pipeline: re-renders any
mask-state (canonical / .gapfill / .merged / hand-painted) without touching
the model. Use this whenever you want to *see* what's in an `annotated_masks*.json`
file without re-running propagation.

Polygon colours per category are drawn from a fixed palette (Tool=cyan,
Liver=red, Gallbladder=green, anything else cycles a fallback palette).
A frame counter and per-frame mask count are stamped top-left.

Frame discovery follows the same priority as sam3_annotator.storage:
`frames_left/` first, then `rgb/`, then `frames/`. Supports webp / png / jpg.

Examples:

    # Render the merged gap-fill output for one snippet
    python scripts/render_overlays_from_coco.py \\
        --snippet-dir '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated/E_3/snippet_003 tbd' \\
        --coco annotated_masks.merged.json \\
        --out-dir overlays_tool.merged

    # Render every variant present in every tbd snippet under a root
    python scripts/render_overlays_from_coco.py \\
        --data-dir '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated' \\
        --coco-suffixes '.merged' '.gapfill' '' \\
        --tbd-only

    # Visualise hand-painted anchors from session_autosave.json (uses
    # propagate_from_autosave's COCO export pathway by writing a temporary
    # annotated_masks.painted.json first; see that script)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Match the annotator frontend palette
CAT_COLORS_BGR = {
    "Tool": (216, 180, 0),       # cyan-ish
    "tool": (216, 180, 0),
    "Liver": (70, 57, 230),      # red
    "Gallbladder": (135, 183, 82),  # green
    "Meat": (0, 127, 247),
    "Skin": (162, 180, 255),
    "FBF": (183, 9, 114),
    "PCH": (10, 214, 255),
}
FALLBACK_PALETTE_BGR = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
]
FRAMES_DIR_CANDIDATES = ("frames_left", "rgb", "frames")
FRAME_EXTS = ("webp", "png", "jpg")


def find_frames_dir(snip_dir: Path) -> tuple[Optional[Path], list[Path]]:
    """Return (frames_dir, frame_files) using the first dir that has files."""
    for cand in FRAMES_DIR_CANDIDATES:
        d = snip_dir / cand
        if not d.is_dir():
            continue
        for ext in FRAME_EXTS:
            files = sorted(d.glob(f"frame_*.{ext}"))
            if files:
                return d, files
    return None, []


def cat_color(cat: str, fallback_idx: int) -> tuple[int, int, int]:
    """BGR triple for a given category name; cycles a fallback palette for unknowns."""
    if cat in CAT_COLORS_BGR:
        return CAT_COLORS_BGR[cat]
    return FALLBACK_PALETTE_BGR[fallback_idx % len(FALLBACK_PALETTE_BGR)]


def render_one(coco: dict, frame_files: list[Path], out_dir: Path,
               jpg_quality: int = 85, draw_outlines: bool = True,
               fill_alpha: float = 0.35) -> dict:
    """Render every annotated frame to out_dir. Returns summary dict."""
    cat_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # Map frame_num (image_id from COCO) -> [annotations]
    by_frame_num: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        by_frame_num[int(ann["image_id"])].append(ann)

    # Map frame_num -> frame_files index by parsing the filename stem
    num_to_idx: dict[int, int] = {}
    for i, fp in enumerate(frame_files):
        try:
            num_to_idx[int(fp.stem.split("_")[1])] = i
        except (ValueError, IndexError):
            continue

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_frame = 0
    by_frame_count: dict[int, int] = defaultdict(int)

    # Render every frame in the snippet (so empty frames also appear and
    # the strip is gap-free), even when they have no annotations. The user
    # explicitly wants to see "where there's nothing" too.
    for i, fp in enumerate(frame_files):
        try:
            frame_num = int(fp.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        anns = by_frame_num.get(frame_num, [])
        img = cv2.imread(str(fp))
        if img is None:
            skipped_no_frame += 1
            continue
        n_ann = len(anns)
        by_frame_count[n_ann] += 1

        # Composite each annotation (filled overlay + optional outline + label)
        for j, ann in enumerate(anns):
            cat_name = cat_by_id.get(ann.get("category_id"), f"cat_{ann.get('category_id')}")
            color = cat_color(cat_name, j)
            seg = ann.get("segmentation") or []
            if not isinstance(seg, list):
                continue
            # Build a composite fill overlay first, then alpha-blend
            overlay = img.copy()
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)
            if draw_outlines:
                for poly in seg:
                    if not isinstance(poly, list) or len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(img, [pts], True, color, 2)
                # Label at first vertex
                first_poly = seg[0] if seg else None
                if isinstance(first_poly, list) and len(first_poly) >= 2:
                    x, y = int(first_poly[0]), int(first_poly[1])
                    cv2.putText(img, cat_name, (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Top-left counters: annotations on this frame + frame_num for cross-ref
        cv2.putText(img, f"f{i} #{frame_num}  anns:{n_ann}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{fp.stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        written += 1

    return {
        "written": written,
        "skipped_no_frame": skipped_no_frame,
        "n_frames_total": len(frame_files),
        "frames_with_n_anns": dict(sorted(by_frame_count.items())),
    }


def render_snippet(snip_dir: Path, coco_filename: str, out_dir_name: str,
                   **kwargs) -> dict:
    """Top-level: validate snippet, load COCO, render."""
    coco_path = snip_dir / coco_filename
    if not coco_path.exists():
        return {"status": "skip", "reason": f"no_{coco_filename}",
                "snip": str(snip_dir)}
    frames_dir, frame_files = find_frames_dir(snip_dir)
    if not frame_files:
        return {"status": "skip", "reason": "no_frames",
                "snip": str(snip_dir)}
    coco = json.load(open(coco_path))
    out_dir = snip_dir / out_dir_name
    summary = render_one(coco, frame_files, out_dir, **kwargs)
    summary.update({
        "status": "ok",
        "snip": str(snip_dir),
        "coco": coco_filename,
        "out_dir": str(out_dir),
        "frames_dir": frames_dir.name,
    })
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snippet-dir", type=Path,
                   help="render a single snippet")
    g.add_argument("--data-dir", type=Path,
                   help="render every snippet under this root (one tbd subdir per episode)")
    ap.add_argument("--coco", default="annotated_masks.merged.json",
                    help="COCO filename inside each snippet dir")
    ap.add_argument("--coco-suffixes", nargs="+", default=None,
                    help="alternative: render annotated_masks{SUFFIX}.json for each suffix; "
                         "out_dir becomes overlays_tool{SUFFIX}/. Empty string = canonical.")
    ap.add_argument("--out-dir", default=None,
                    help="output dir name under snippet root (default: derived from coco filename)")
    ap.add_argument("--tbd-only", action="store_true",
                    help="when used with --data-dir, only descend into snippet dirs ending in 'tbd'")
    ap.add_argument("--jpg-quality", type=int, default=85)
    ap.add_argument("--no-outlines", action="store_true",
                    help="omit polygon outlines + labels (only filled translucent overlay)")
    ap.add_argument("--fill-alpha", type=float, default=0.35)
    args = ap.parse_args()

    render_kwargs = dict(
        jpg_quality=args.jpg_quality,
        draw_outlines=not args.no_outlines,
        fill_alpha=args.fill_alpha,
    )

    def _resolve_out_dir(coco_filename: str) -> str:
        if args.out_dir:
            return args.out_dir
        # annotated_masks.merged.json -> overlays_tool.merged
        # annotated_masks.json        -> overlays_tool.canonical
        stem = Path(coco_filename).stem  # annotated_masks.merged
        if stem == "annotated_masks":
            return "overlays_tool.canonical"
        suffix = stem.replace("annotated_masks", "", 1)
        return f"overlays_tool{suffix}"

    targets: list[Path] = []
    if args.snippet_dir:
        targets = [args.snippet_dir]
    else:
        # Walk data-dir for snippet dirs (any directory whose name starts with snippet_)
        for ep in sorted(args.data_dir.iterdir()):
            if not ep.is_dir():
                continue
            for snip in sorted(ep.iterdir()):
                if not snip.is_dir() or not snip.name.startswith("snippet_"):
                    continue
                if args.tbd_only and not snip.name.endswith("tbd"):
                    continue
                targets.append(snip)

    coco_list: list[str]
    if args.coco_suffixes is not None:
        coco_list = [
            ("annotated_masks.json" if s == "" else f"annotated_masks{s}.json")
            for s in args.coco_suffixes
        ]
    else:
        coco_list = [args.coco]

    summaries: list[dict] = []
    for snip in targets:
        for coco_filename in coco_list:
            out_dir_name = _resolve_out_dir(coco_filename)
            summary = render_snippet(snip, coco_filename, out_dir_name, **render_kwargs)
            summaries.append(summary)
            if summary["status"] == "ok":
                hist = summary["frames_with_n_anns"]
                rel = snip.relative_to(snip.parents[1]) if len(snip.parents) >= 2 else snip
                print(f"  ok    {rel} {coco_filename} -> {out_dir_name}/  "
                      f"({summary['written']}/{summary['n_frames_total']} frames, hist={hist})")
            else:
                rel = snip.relative_to(snip.parents[1]) if len(snip.parents) >= 2 else snip
                print(f"  skip  {rel} {coco_filename}  ({summary['reason']})")

    if not summaries:
        print("nothing to render", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
