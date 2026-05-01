#!/usr/bin/env python3
"""Stage 2 of the auto-gap-fill pipeline.

Read gap_manifest.json (output of build_gap_manifest.py), look up the polygon
of the *anchor source* frame for each gap (and the *full-rerun seed frame*
for snippets flagged full_rerun), and rasterise it to a binary PNG that the
patched batch_text_propagate.py can pass to SAM3's add_new_mask.

Output layout:
    {anchors-dir}/
        {ep}/
            {snippet}/
                gap_{snip_idx_start}_{snip_idx_end}_seed_{anchor_idx}_polysrc_{src_idx}.png
                full_rerun_seed_{seed_idx}.png
        anchor_index.json    # what was extracted, for downstream stages

Each PNG is a uint8 binary mask {0, 255} at the snippet's image height/width.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_polys_for_frame(snip_dir, best_variant, image_id):
    """Return list of polygon lists for `image_id`, sourced from the BEST
    variant's annotated_masks file when available, else from the merged
    snippet_annotations.json (used by F_3/001 full-rerun seeding)."""
    candidates = []
    if best_variant is not None:
        candidates.append(snip_dir / f"annotated_masks{best_variant}.json")
    candidates.append(snip_dir / "snippet_annotations.json")

    polys_out = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        tool_id = next((k for k, v in cats.items() if v == "Tool"), 6)
        polys_out = []
        for a in d.get("annotations", []):
            if a.get("image_id") != image_id:
                continue
            if a.get("category_id") != tool_id:
                continue
            for poly in a.get("segmentation", []) or []:
                if isinstance(poly, list) and len(poly) >= 6:
                    polys_out.append(poly)
        if polys_out:
            return polys_out, p.name
    return [], None


def rasterise(polys, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def get_image_dims(snip_dir, image_id):
    """Pull image width/height from snippet_annotations.json's images entry."""
    sa = json.load(open(snip_dir / "snippet_annotations.json"))
    for img in sa.get("images", []):
        if img["id"] == image_id:
            return int(img["height"]), int(img["width"])
    if sa["images"]:
        first = sa["images"][0]
        return int(first.get("height", 720)), int(first.get("width", 1280))
    return 720, 1280


def process_snippet(entry, data_dir, anchors_dir):
    ep = entry["ep"]
    snip_name = entry["snippet"]
    snip_dir = data_dir / ep / snip_name
    out_dir = anchors_dir / ep / snip_name.replace(" tbd", "").strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = []

    if entry.get("full_rerun"):
        seed = entry.get("full_rerun_seed") or {}
        seed_image_id = seed.get("seed_image_id")
        if seed_image_id is None:
            extracted.append({
                "kind": "full_rerun_seed",
                "status": "no_seed_paint_required",
                "snip_idx": None,
                "image_id": None,
                "polygon_source": None,
            })
            return extracted
        h, w = get_image_dims(snip_dir, seed_image_id)
        polys, src = load_polys_for_frame(snip_dir, entry.get("best_variant"), seed_image_id)
        mask = rasterise(polys, h, w)
        out_name = f"full_rerun_seed_{seed['seed_snip_idx']}.png"
        cv2.imwrite(str(out_dir / out_name), mask)
        extracted.append({
            "kind": "full_rerun_seed",
            "status": "ok" if polys else "no_polys",
            "snip_idx": seed["seed_snip_idx"],
            "image_id": seed_image_id,
            "polygon_source": src,
            "components": seed.get("components"),
            "area": seed.get("area"),
            "out": str(out_dir / out_name),
        })
        return extracted

    best = entry.get("best_variant")
    for gap in entry.get("gaps", []):
        src_idx = gap.get("anchor_polygon_source_idx")
        src_image_id = gap.get("anchor_polygon_source_image_id")
        if src_idx is None or src_image_id is None:
            extracted.append({
                "kind": "gap_seed",
                "status": "no_neighbour_paint_required",
                "snip_idx_start": gap["snip_idx_start"],
                "snip_idx_end": gap["snip_idx_end"],
                "anchor_idx": gap["anchor_idx"],
            })
            continue
        h, w = get_image_dims(snip_dir, src_image_id)
        polys, src = load_polys_for_frame(snip_dir, best, src_image_id)
        if not polys:
            extracted.append({
                "kind": "gap_seed",
                "status": "no_polys_paint_required",
                "snip_idx_start": gap["snip_idx_start"],
                "snip_idx_end": gap["snip_idx_end"],
                "anchor_idx": gap["anchor_idx"],
                "polygon_source_image_id": src_image_id,
            })
            continue
        mask = rasterise(polys, h, w)
        out_name = (
            f"gap_{gap['snip_idx_start']}_{gap['snip_idx_end']}"
            f"_seed_{gap['anchor_idx']}_polysrc_{src_idx}.png"
        )
        cv2.imwrite(str(out_dir / out_name), mask)
        extracted.append({
            "kind": "gap_seed",
            "status": "ok",
            "snip_idx_start": gap["snip_idx_start"],
            "snip_idx_end": gap["snip_idx_end"],
            "anchor_idx": gap["anchor_idx"],
            "anchor_image_id": gap["anchor_image_id"],
            "polygon_source_idx": src_idx,
            "polygon_source_image_id": src_image_id,
            "polygon_source_file": src,
            "kind_of_gap": gap["kind"],
            "length": gap["length"],
            "out": str(out_dir / out_name),
        })
    return extracted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="gap_manifest.json from Stage 1")
    ap.add_argument("--anchors-dir", required=True, help="output dir for seed PNGs")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    data_dir = Path(manifest["data_dir"])
    anchors_dir = Path(args.anchors_dir)
    anchors_dir.mkdir(parents=True, exist_ok=True)

    index = {"manifest": str(Path(args.manifest).resolve()), "snippets": []}

    n_ok = n_paint = n_other = 0
    paint_required_snippets = []
    for entry in manifest.get("snippets", []):
        if entry.get("skipped"):
            continue
        seeds = process_snippet(entry, data_dir, anchors_dir)
        snip_record = {
            "ep": entry["ep"],
            "snippet": entry["snippet"],
            "full_rerun": entry.get("full_rerun", False),
            "expected_tools": entry.get("expected_tools"),
            "best_variant": entry.get("best_variant"),
            "structural_undermask_warning": entry.get("structural_undermask_warning", False),
            "asymmetry_score": entry.get("asymmetry_score"),
            "healthy_anchor_count": entry.get("healthy_anchor_count"),
            "seeds": seeds,
        }
        index["snippets"].append(snip_record)
        snip_paint_count = 0
        for s in seeds:
            if s.get("status") == "ok":
                n_ok += 1
            elif "paint_required" in s.get("status", ""):
                n_paint += 1
                snip_paint_count += 1
            else:
                n_other += 1
        if snip_paint_count == len(seeds) and snip_paint_count > 0:
            paint_required_snippets.append(f"{entry['ep']}/{entry['snippet']}")
    index["paint_required_snippets"] = paint_required_snippets

    out_index = anchors_dir / "anchor_index.json"
    with open(out_index, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {out_index}")
    print(f"Seeds: ok={n_ok}, paint_required={n_paint}, other={n_other}")
    for s in index["snippets"]:
        tag = "RERUN" if s["full_rerun"] else "GAPFL"
        print(f"  {tag} {s['ep']}/{s['snippet']:<14} expected={s['expected_tools']} "
              f"best_variant={s['best_variant']!r} seeds={len(s['seeds'])}")
        for seed in s["seeds"]:
            kind = seed.get("kind_of_gap") or seed["kind"]
            sf = seed.get("out", "(no file)").split("\\")[-1].split("/")[-1]
            extra = ""
            if seed.get("snip_idx_start") is not None:
                extra = f" gap[{seed['snip_idx_start']}..{seed['snip_idx_end']}] anchor@{seed['anchor_idx']}"
            print(f"      {seed['status']:<26} {kind:<16}{extra}  -> {sf}")


if __name__ == "__main__":
    sys.exit(main() or 0)
