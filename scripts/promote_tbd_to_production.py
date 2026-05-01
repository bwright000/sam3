#!/usr/bin/env python3
"""Stage 5 of the auto-gap-fill pipeline.

Take the merged staging output (annotated_masks.merged.json) for each tbd
snippet and propagate it into production at:
    data/Segments/{EP}/snippet_NNN/

Behaviours:
  * Slice merged annotations to the production snippet's image_id range
    (handles F_3/006: staging covers 387 frames, production now 200).
  * Add Tool category to production's snippet_annotations.json if absent.
  * Append/update Tool annotations per frame.
  * Rasterise Tool polygons into existing semantic_instance/*.png at
    pixel value = Tool_COCO_id + 1 = 7. Tissue pixels (Liver=3, Gallbladder=4)
    are NEVER overwritten — priority is Gallbladder > Liver > Tool.
  * Update info_semantic.json to declare the Tool class.

Usage:
    python scripts/promote_tbd_to_production.py \\
        --manifest outputs/gap_manifest.json \\
        --production-root c:/Users/benli/sam3facebook/data/Segments \\
        --merged-suffix .merged \\
        --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


TOOL_COCO_ID = 6
TOOL_PIXEL = TOOL_COCO_ID + 1   # 7
TOOL_INFO_SEMANTIC_ID = 7       # COCO id + 1


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(p, d, indent=None):
    tmp = Path(p).with_suffix(Path(p).suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(d, f, indent=indent)
    tmp.replace(p)


def slice_annotations_to_range(merged, image_id_lo, image_id_hi):
    """Filter merged COCO to images/annotations within [lo, hi] inclusive."""
    keep_imgs = [i for i in merged.get("images", [])
                 if image_id_lo <= i["id"] <= image_id_hi]
    keep_ann = [a for a in merged.get("annotations", [])
                if image_id_lo <= a["image_id"] <= image_id_hi]
    return keep_imgs, keep_ann


def merge_into_production_coco(prod_coco, tool_anns_to_add):
    """Inject Tool annotations into the production snippet_annotations.json.

    Adds the Tool category if missing, replaces any existing Tool annotations
    on the touched image_ids, and renumbers annotation ids contiguously."""
    cats = prod_coco.setdefault("categories", [])
    if not any(c.get("id") == TOOL_COCO_ID for c in cats):
        cats.append({"id": TOOL_COCO_ID, "name": "Tool", "supercategory": "Tool"})

    # Drop any existing Tool annotations on the same images we're about to write
    touched_image_ids = {a["image_id"] for a in tool_anns_to_add}
    kept = [a for a in prod_coco.get("annotations", [])
            if not (a.get("category_id") == TOOL_COCO_ID
                    and a["image_id"] in touched_image_ids)]
    kept.extend(tool_anns_to_add)
    for i, a in enumerate(kept, 1):
        a["id"] = i
    prod_coco["annotations"] = kept
    return prod_coco


def update_info_semantic(info_path):
    """Add Tool class (id=7) if missing."""
    if not info_path.exists():
        info = {
            "classes": [],
            "background_id": 0,
            "note": "Pixel value in semantic_instance/*.png = category_id + 1. 0 = background.",
        }
    else:
        info = load_json(info_path)
    classes = info.setdefault("classes", [])
    if not any(c.get("name") == "Tool" for c in classes):
        classes.append({
            "id": TOOL_INFO_SEMANTIC_ID,
            "name": "Tool",
            "supercategory": "Tool",
        })
    save_json(info_path, info, indent=2)


def paint_tool_into_semantic_pngs(semantic_dir, tool_anns, image_id_to_frame_n):
    """For each Tool annotation, paint pixel TOOL_PIXEL into the existing PNG
    *only where it is currently background (0)*. Returns count of frames
    actually written.
    """
    written = 0
    skipped_no_tissue_clash = 0
    for a in tool_anns:
        image_id = a["image_id"]
        frame_n = image_id_to_frame_n.get(image_id, image_id)
        png_path = semantic_dir / f"frame_{frame_n:06d}.png"
        if not png_path.exists():
            continue
        canvas = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if canvas is None:
            continue
        if canvas.dtype != np.uint16:
            canvas = canvas.astype(np.uint16)
        # Build the tool mask (binary)
        tool_mask = np.zeros_like(canvas, dtype=np.uint8)
        for poly in a.get("segmentation") or []:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(tool_mask, [pts], 1)
        if tool_mask.sum() == 0:
            continue
        # Apply only where canvas is currently background (priority preserves tissue)
        bg = canvas == 0
        write_mask = np.logical_and(tool_mask.astype(bool), bg)
        if not write_mask.any():
            skipped_no_tissue_clash += 1
            continue
        canvas[write_mask] = TOOL_PIXEL
        cv2.imwrite(str(png_path), canvas)
        written += 1
    return written, skipped_no_tissue_clash


def build_image_id_to_frame_n(rgb_dir):
    """Map COCO image_id (= original frame number) -> frame_n filename digit."""
    out = {}
    for p in sorted(rgb_dir.glob("frame_*.png")):
        try:
            fr = int(p.stem.split("_")[1])
            out[fr] = fr
        except (IndexError, ValueError):
            continue
    return out


def process_snippet(entry, manifest_data_dir, prod_root, merged_suffix,
                    backup_suffix, dry_run, verbose=True):
    ep = entry["ep"]
    snip_name_tbd = entry["snippet"]
    snip_name_prod = snip_name_tbd.replace(" tbd", "").strip()

    tbd_dir = manifest_data_dir / ep / snip_name_tbd
    prod_dir = prod_root / ep / snip_name_prod
    rel = f"{ep}/{snip_name_prod}"

    if not prod_dir.is_dir():
        return {"status": "skip", "rel": rel, "reason": "no_production_dir"}
    merged_path = tbd_dir / f"annotated_masks{merged_suffix}.json"
    if not merged_path.exists():
        return {"status": "skip", "rel": rel, "reason": f"no_merged_at_{merged_path.name}"}

    prod_ann_path = prod_dir / "snippet_annotations.json"
    prod_info_path = prod_dir / "info_semantic.json"
    prod_sem_dir = prod_dir / "semantic_instance"
    prod_rgb_dir = prod_dir / "rgb"
    if not prod_ann_path.exists() or not prod_sem_dir.is_dir():
        return {"status": "skip", "rel": rel, "reason": "production_layout_incomplete"}

    prod_coco = load_json(prod_ann_path)
    prod_image_ids = sorted([i["id"] for i in prod_coco.get("images", [])])
    if not prod_image_ids:
        return {"status": "skip", "rel": rel, "reason": "no_production_images"}
    lo, hi = prod_image_ids[0], prod_image_ids[-1]

    merged_coco = load_json(merged_path)
    keep_imgs, keep_ann = slice_annotations_to_range(merged_coco, lo, hi)
    tool_anns = [a for a in keep_ann if a.get("category_id") == TOOL_COCO_ID]

    n_in_range = len(keep_imgs)
    n_tool_total = sum(1 for a in merged_coco.get("annotations", [])
                       if a.get("category_id") == TOOL_COCO_ID)
    n_tool_in_range = len(tool_anns)

    if verbose:
        print(f"  {rel}: prod_range=[{lo}..{hi}] ({len(prod_image_ids)} frames), "
              f"tool_anns_total={n_tool_total} in_range={n_tool_in_range} "
              f"sliced={n_tool_total - n_tool_in_range}")

    if dry_run:
        return {
            "status": "dry-run",
            "rel": rel,
            "tool_anns_total": n_tool_total,
            "tool_anns_in_range": n_tool_in_range,
            "frames_in_range": n_in_range,
            "production_frames": len(prod_image_ids),
        }

    # Backup originals
    if backup_suffix:
        if not (prod_ann_path.with_suffix(f".json{backup_suffix}").exists()):
            shutil.copy2(prod_ann_path, str(prod_ann_path) + backup_suffix)
        if prod_info_path.exists() and not (
                Path(str(prod_info_path) + backup_suffix).exists()):
            shutil.copy2(prod_info_path, str(prod_info_path) + backup_suffix)

    # Update production COCO
    merged_prod = merge_into_production_coco(prod_coco, tool_anns)
    save_json(prod_ann_path, merged_prod)

    # Update info_semantic
    update_info_semantic(prod_info_path)

    # Rasterise Tool into semantic_instance/
    image_id_to_frame_n = build_image_id_to_frame_n(prod_rgb_dir)
    if not image_id_to_frame_n:
        # Fallback: assume image_id == frame_n (the project-wide convention)
        image_id_to_frame_n = {a["image_id"]: a["image_id"] for a in tool_anns}

    written, clashed = paint_tool_into_semantic_pngs(
        prod_sem_dir, tool_anns, image_id_to_frame_n
    )

    return {
        "status": "ok",
        "rel": rel,
        "tool_anns_in_range": n_tool_in_range,
        "frames_painted": written,
        "frames_skipped_no_clear_pixels": clashed,
        "production_frames": len(prod_image_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--production-root", required=True,
                    help="root of canonical TUM/Replica snippets, "
                         "e.g. c:/Users/benli/sam3facebook/data/Segments")
    ap.add_argument("--merged-suffix", default=".merged")
    ap.add_argument("--backup-suffix", default=".bak_pre_tool_promote",
                    help="suffix for backup of original snippet_annotations.json "
                         "before tool injection (set empty to skip)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to ep/snippet (e.g. F_3/snippet_001)")
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    manifest_data_dir = Path(manifest["data_dir"])
    prod_root = Path(args.production_root)

    summary = []
    for entry in manifest.get("snippets", []):
        if entry.get("skipped"):
            continue
        rel = f"{entry['ep']}/{entry['snippet'].replace(' tbd','').strip()}"
        if args.only and rel not in args.only:
            continue
        info = process_snippet(entry, manifest_data_dir, prod_root,
                               args.merged_suffix, args.backup_suffix,
                               args.dry_run)
        summary.append(info)

    print("\n=== Promotion summary ===")
    for s in summary:
        if s["status"] == "ok":
            print(f"  ok       {s['rel']:<22} tool_in_range={s['tool_anns_in_range']:>4}, "
                  f"painted={s['frames_painted']}/{s['production_frames']}, "
                  f"clashed={s['frames_skipped_no_clear_pixels']}")
        elif s["status"] == "dry-run":
            print(f"  dry-run  {s['rel']:<22} tool_in_range={s['tool_anns_in_range']:>4}/"
                  f"{s['tool_anns_total']:<4}  prod_frames={s['production_frames']}")
        else:
            print(f"  {s['status']:<8} {s['rel']:<22} {s.get('reason','')}")


if __name__ == "__main__":
    main()
