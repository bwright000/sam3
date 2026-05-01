#!/usr/bin/env python3
"""Combine GT (Liver + Gallbladder) and Tool masks into priority-resolved
per-frame label maps + a non-overlapping COCO JSON.

Priority (highest paints last, wins overlaps):
    Gallbladder > Liver > Tool

Tool handling:
    All tool polygons in annotated_masks.json are unioned into a single
    binary mask, small connected components (< --min-tool-area px) dropped
    to remove SAM3 noise specks. Spatially separate genuine tool instances
    are kept (multi-tool surgical scenes are real).

Output per snippet:
    combined_masks/frame_NNNNNN.png    paletted label map
                                        0=bg, 1=tool, 2=liver, 3=gallbladder
    combined_annotations.json          COCO with non-overlapping polygons

Usage:
    # All 7 priority snippets in one run
    python scripts/combine_priority_masks.py \\
        --data-dir 'data/Segments/.../To Be Annotated' \\
        --snippets E_3/001 E_3/002 E_3/004 F_3/001 F_3/004 F_3/005 F_3/007

    # Single snippet
    python scripts/combine_priority_masks.py \\
        --data-dir '...' --episode F_3 --snippet 001

    # Dry run (skip writing)
    python scripts/combine_priority_masks.py --data-dir '...' --episode F_3 --snippet 001 --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LABEL_BG = 0
LABEL_TOOL = 1
LABEL_LIVER = 2
LABEL_GALLBLADDER = 3

# Order matters: lower priority painted first, higher overwrites.
PAINT_ORDER = [
    ("Tool", LABEL_TOOL),
    ("Liver", LABEL_LIVER),
    ("Gallbladder", LABEL_GALLBLADDER),
]

# Output COCO category ids (compact, independent of source IDs).
OUT_CAT_IDS = {
    "Tool": 1,
    "Liver": 2,
    "Gallbladder": 3,
}

# PNG palette (RGB) for visual inspection. Values match LABEL_* indices.
PALETTE_RGB = [
    (0, 0, 0),         # 0 background
    (0, 128, 255),     # 1 tool — blue
    (255, 0, 0),       # 2 liver — red
    (0, 255, 0),       # 3 gallbladder — green
]


def build_palette() -> list[int]:
    flat = []
    for r, g, b in PALETTE_RGB:
        flat.extend([r, g, b])
    flat.extend([0] * (3 * (256 - len(PALETTE_RGB))))
    return flat


def polys_to_mask(polys: list, h: int, w: int) -> np.ndarray:
    """List of flat polygons [x,y,x,y,...] -> (H, W) uint8 binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop CCs below min_area pixels. Keeps multiple separate large ones."""
    if min_area <= 0 or mask.sum() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def mask_to_polys(mask: np.ndarray, min_area: int = 30, eps_frac: float = 0.002) -> list[list[float]]:
    """Binary mask -> list of flat COCO polygons. Drops contours < min_area."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, eps_frac * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        polys.append(approx.reshape(-1).astype(float).tolist())
    return polys


def collect_polys_by_image(coco: dict, target_cat_names: set) -> dict[int, dict[str, list[list[float]]]]:
    """image_id -> {cat_name: [poly, poly, ...]} for given target categories."""
    catid_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    out: dict[int, dict[str, list[list[float]]]] = {}
    for a in coco.get("annotations", []):
        cat_name = catid_to_name.get(a["category_id"])
        if cat_name not in target_cat_names:
            continue
        seg = a.get("segmentation")
        if not isinstance(seg, list):
            continue  # RLE not handled here; expand if needed
        img_polys = out.setdefault(a["image_id"], {})
        cat_polys = img_polys.setdefault(cat_name, [])
        cat_polys.extend(seg)
    return out


def process_snippet(snip_dir: Path, min_tool_area: int, dry_run: bool,
                    fill_tool_gaps: bool = False) -> dict:
    gt_path = snip_dir / "snippet_annotations.json"
    tool_path = snip_dir / "annotated_masks.json"

    if not gt_path.exists():
        return {"error": "missing snippet_annotations.json"}
    if not tool_path.exists():
        return {"error": "missing annotated_masks.json"}

    gt = json.load(open(gt_path))
    tool = json.load(open(tool_path))

    img_meta = {img["id"]: img for img in gt.get("images", [])}
    # union with tool images in case some image_ids appear only in tool file
    for img in tool.get("images", []):
        img_meta.setdefault(img["id"], img)

    gt_polys = collect_polys_by_image(gt, {"Liver", "Gallbladder"})
    tool_polys = collect_polys_by_image(tool, {"Tool"})

    out_dir = snip_dir / "combined_masks"
    if not dry_run:
        out_dir.mkdir(exist_ok=True)

    palette = build_palette()
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_ann_id = 1

    counts = {"frames_written": 0, "tool_frames": 0, "liver_frames": 0,
              "gallbladder_frames": 0, "tool_filled_from_prev": 0}
    last_tool_polys: list = []  # for fill_tool_gaps

    for image_id in sorted(img_meta.keys()):
        meta = img_meta[image_id]
        h = int(meta["height"])
        w = int(meta["width"])

        label_map = np.zeros((h, w), dtype=np.uint8)

        # 1. Tool (lowest priority) — union, then drop small CCs
        tpolys = tool_polys.get(image_id, {}).get("Tool", [])
        if not tpolys and fill_tool_gaps and last_tool_polys:
            tpolys = last_tool_polys
            counts["tool_filled_from_prev"] += 1
        if tpolys:
            tool_mask = polys_to_mask(tpolys, h, w)
            tool_mask = drop_small_components(tool_mask, min_tool_area)
            label_map[tool_mask > 0] = LABEL_TOOL
            if tool_mask.any():
                counts["tool_frames"] += 1
                last_tool_polys = tpolys
            elif tool_polys.get(image_id, {}).get("Tool"):
                # had source polys, but they got entirely filtered — don't poison cache
                pass

        # 2. Liver overwrites tool
        lpolys = gt_polys.get(image_id, {}).get("Liver", [])
        if lpolys:
            liver_mask = polys_to_mask(lpolys, h, w)
            label_map[liver_mask > 0] = LABEL_LIVER
            counts["liver_frames"] += 1

        # 3. Gallbladder overwrites everything
        gpolys = gt_polys.get(image_id, {}).get("Gallbladder", [])
        if gpolys:
            gb_mask = polys_to_mask(gpolys, h, w)
            label_map[gb_mask > 0] = LABEL_GALLBLADDER
            counts["gallbladder_frames"] += 1

        # Write paletted PNG
        out_name = f"frame_{image_id:06d}.png"
        if not dry_run:
            img = Image.fromarray(label_map, mode="P")
            img.putpalette(palette)
            img.save(out_dir / out_name, optimize=True)

        out_images.append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": f"./combined_masks/{out_name}",
        })

        # Build per-class non-overlapping polygons from the resolved label map
        for cat_name, label_val in (("Tool", LABEL_TOOL),
                                     ("Liver", LABEL_LIVER),
                                     ("Gallbladder", LABEL_GALLBLADDER)):
            class_mask = (label_map == label_val).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
            polys = mask_to_polys(class_mask)
            if not polys:
                continue
            ys, xs = np.where(class_mask > 0)
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            out_annotations.append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": OUT_CAT_IDS[cat_name],
                "segmentation": polys,
                "bbox": [float(x_min), float(y_min),
                         float(x_max - x_min + 1), float(y_max - y_min + 1)],
                "area": float(class_mask.sum()),
                "iscrowd": 0,
            })
            next_ann_id += 1

        counts["frames_written"] += 1

    coco_out = {
        "info": {
            "description": "Priority-resolved combined masks",
            "priority_order": ["Gallbladder", "Liver", "Tool"],
            "label_map": {
                "background": LABEL_BG,
                "tool": LABEL_TOOL,
                "liver": LABEL_LIVER,
                "gallbladder": LABEL_GALLBLADDER,
            },
            "min_tool_area_px": min_tool_area,
        },
        "categories": [
            {"id": OUT_CAT_IDS["Tool"], "name": "Tool", "supercategory": "Tool"},
            {"id": OUT_CAT_IDS["Liver"], "name": "Liver", "supercategory": "Liver"},
            {"id": OUT_CAT_IDS["Gallbladder"], "name": "Gallbladder", "supercategory": "Gallbladder"},
        ],
        "images": out_images,
        "annotations": out_annotations,
    }

    out_json = snip_dir / "combined_annotations.json"
    if not dry_run:
        tmp = out_json.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(coco_out, f)
        tmp.replace(out_json)

    return {
        "frames": counts["frames_written"],
        "tool_frames": counts["tool_frames"],
        "liver_frames": counts["liver_frames"],
        "gallbladder_frames": counts["gallbladder_frames"],
        "tool_filled": counts["tool_filled_from_prev"],
        "annotations": len(out_annotations),
    }


def parse_snippets_arg(snippets: list[str]) -> list[tuple[str, str]]:
    """E_3/001 -> (E_3, 001). Accepts also bare 'E_3/snippet_001'."""
    out = []
    for s in snippets:
        s = s.strip()
        if "/" not in s:
            print(f"  SKIP malformed snippet spec: {s}", file=sys.stderr)
            continue
        ep, sid = s.split("/", 1)
        sid = sid.replace("snippet_", "")
        out.append((ep, sid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="path to 'To Be Annotated' root")
    ap.add_argument("--snippets", nargs="+", default=None,
                    help="explicit list, e.g. E_3/001 F_3/004")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--snippet", default=None)
    ap.add_argument("--min-tool-area", type=int, default=200,
                    help="drop tool CCs smaller than this many pixels (default 200)")
    ap.add_argument("--fill-tool-gaps", action="store_true",
                    help="if a frame has no tool detection, copy previous frame's tool mask")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    targets: list[tuple[str, str]] = []

    if args.snippets:
        targets.extend(parse_snippets_arg(args.snippets))
    if args.episode and args.snippet:
        targets.append((args.episode, args.snippet))
    if not targets:
        print("ERROR: provide --snippets or --episode/--snippet", file=sys.stderr)
        sys.exit(2)

    print(f"  combining {len(targets)} snippets, min_tool_area={args.min_tool_area}px"
          + (" [DRY-RUN]" if args.dry_run else ""))
    print()

    for ep, sid in targets:
        snip_dir = root / ep / f"snippet_{sid}"
        if not snip_dir.exists():
            print(f"  SKIP {ep}/{sid}: directory not found ({snip_dir})")
            continue
        t0 = time.time()
        try:
            result = process_snippet(snip_dir, args.min_tool_area, args.dry_run,
                                     fill_tool_gaps=args.fill_tool_gaps)
        except Exception as e:
            print(f"  ERROR {ep}/{sid}: {e}")
            continue
        dt = time.time() - t0
        if "error" in result:
            print(f"  SKIP {ep}/{sid}: {result['error']}")
            continue
        fill = f"  filled={result['tool_filled']}" if result.get("tool_filled") else ""
        print(f"  {ep}/{sid:<6} {result['frames']:>4} frames  "
              f"tool={result['tool_frames']:>4}  liver={result['liver_frames']:>4}  "
              f"gb={result['gallbladder_frames']:>4}  anns={result['annotations']:>5}{fill}  "
              f"({dt:.1f}s)")

    print()
    print("  Outputs per snippet:")
    print("    combined_masks/frame_NNNNNN.png   (paletted label map)")
    print("    combined_annotations.json         (non-overlapping COCO)")


if __name__ == "__main__":
    main()
