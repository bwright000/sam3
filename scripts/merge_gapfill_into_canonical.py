#!/usr/bin/env python3
"""Stage 4 of the auto-gap-fill pipeline.

Given gap_manifest.json + the output of propagate_gap_fill.py, merge the
.gapfill annotations into the canonical annotated_masks.json per snippet,
producing annotated_masks.merged.json. Strategy:

  full_rerun=true  -> annotated_masks.merged.json := annotated_masks.gapfill.json
                       (the rerun is canonical for the whole snippet)

  gap-fill         -> for each gap [snip_idx_start..snip_idx_end]:
                        * decide per-frame which source wins:
                            - new (gapfill) if it has >= expected_tools
                              components AND the original had < expected
                            - new if original had no entry for this frame
                            - keep original otherwise
                        * splice winning frame into merged file

After merge, write tool_detection_stats.merged.json with a recomputed
histogram + match%, then update expected_tools where it was None
(filled from scene_motion).

Usage:
    python scripts/merge_gapfill_into_canonical.py \\
        --manifest outputs/gap_manifest.json \\
        --gapfill-suffix .gapfill \\
        --merged-suffix .merged
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def index_by_image_id(coco):
    return {img["id"]: img for img in coco.get("images", [])}, \
           {a["image_id"]: a for a in coco.get("annotations", [])}


def count_components(ann):
    if not ann:
        return 0
    polys = ann.get("segmentation") or []
    return sum(1 for p in polys if isinstance(p, list) and len(p) >= 6)


def total_area(ann):
    if not ann:
        return 0.0
    return float(ann.get("area", 0) or 0)


def merge_canonical_with_gapfill(canonical, gapfill, gaps, expected_tools,
                                 area_improve_ratio=1.15):
    """Return merged COCO + per-frame action log.

    Per-frame decision ladder for image_ids that fall in any gap range:
      1. If canon has < expected components and gap has >= expected -> gap.
      2. If canon already has >= expected AND its area is >= gap's area
         (or gap not present) -> keep canon ("canon_kept_already_good").
      3. If canon has expected components but gap has *more area* (by
         `area_improve_ratio`), use gap. This is the undermask-fix path —
         e.g. canon detected only the gripper tip, gap recovered the shaft.
      4. If canon empty but gap has any annotation -> gap.
      5. Else keep canon.
    """
    images_canon, anns_canon = index_by_image_id(canonical)
    images_gap, anns_gap = index_by_image_id(gapfill)

    gap_image_ids = set()
    gap_kinds_by_image = {}
    for g in gaps:
        for img_id in range(g["image_id_start"], (g["image_id_end"] or g["image_id_start"]) + 1):
            gap_image_ids.add(img_id)
            gap_kinds_by_image.setdefault(img_id, set()).add(g.get("kind", "?"))

    merged_images = []
    merged_anns = []
    actions = Counter()

    all_image_ids = sorted(set(images_canon.keys()) | set(images_gap.keys()))
    for img_id in all_image_ids:
        in_gap = img_id in gap_image_ids
        a_canon = anns_canon.get(img_id)
        a_gap = anns_gap.get(img_id)
        chosen = None
        chosen_kind = None

        if in_gap:
            comp_canon = count_components(a_canon)
            comp_gap = count_components(a_gap)
            area_canon = total_area(a_canon)
            area_gap = total_area(a_gap)
            kinds = gap_kinds_by_image.get(img_id, set())

            if expected_tools is not None:
                # 1. canon under-count, gap matches expected: gap wins
                if comp_gap >= expected_tools and comp_canon < expected_tools:
                    chosen, chosen_kind = a_gap, "gapfill_filled_under"
                # 2. canon already at/above expected and gap is empty
                elif comp_canon >= expected_tools and not a_gap:
                    chosen, chosen_kind = a_canon, "canon_kept_already_good"
                # 3. both at expected but gap has measurably more area
                elif (comp_canon >= expected_tools and comp_gap >= expected_tools
                      and area_gap > area_canon * area_improve_ratio):
                    chosen, chosen_kind = a_gap, "gapfill_better_area"
                # 4. canon undermask, gap improvements (any direction)
                elif "undermask" in kinds and a_gap and area_gap > area_canon:
                    chosen, chosen_kind = a_gap, "gapfill_undermask_recovered"
                # 5. canon empty but gap exists
                elif a_gap and not a_canon:
                    chosen, chosen_kind = a_gap, "gapfill_filled_empty"
                # 6. canon kept (default)
                else:
                    chosen = a_canon if a_canon else a_gap
                    chosen_kind = "canon_kept_no_clear_improvement" if a_canon else "gapfill_only"
            else:
                if a_gap and not a_canon:
                    chosen, chosen_kind = a_gap, "gapfill_filled_empty"
                elif a_canon and not a_gap:
                    chosen, chosen_kind = a_canon, "canon_only"
                elif a_canon and a_gap:
                    if comp_gap > comp_canon:
                        chosen, chosen_kind = a_gap, "gapfill_more_components"
                    elif area_gap > area_canon * area_improve_ratio:
                        chosen, chosen_kind = a_gap, "gapfill_better_area"
                    else:
                        chosen, chosen_kind = a_canon, "canon_kept"
        else:
            chosen = a_canon if a_canon else a_gap
            chosen_kind = "canon_kept_outside_gap" if a_canon else "gapfill_outside_gap"

        actions[chosen_kind] += 1

        img_entry = images_canon.get(img_id) or images_gap.get(img_id)
        if img_entry:
            merged_images.append(img_entry)
        if chosen:
            merged_anns.append(dict(chosen, image_id=img_id))

    # Renumber annotation IDs
    for i, a in enumerate(merged_anns, 1):
        a["id"] = i

    merged = {
        "categories": canonical.get("categories") or gapfill.get("categories"),
        "images": merged_images,
        "annotations": merged_anns,
        "_merge_meta": {
            "actions": dict(actions),
            "expected_tools": expected_tools,
            "gap_image_id_count": len(gap_image_ids),
        },
    }
    return merged, dict(actions)


def expected_from_scene_motion(snip_dir):
    sm = snip_dir / "scene_motion.json"
    if not sm.exists():
        return None
    try:
        d = json.load(open(sm))
    except Exception:
        return None
    p1 = bool(d.get("psm1_motion", False))
    p2 = bool(d.get("psm2_motion", False))
    if p1 and p2:
        return 2
    if p1 or p2:
        return 1
    return None


def recompute_stats(snip_dir, merged, expected_tools):
    """Build a fresh tool_detection_stats blob from a merged COCO file by
    walking the snippet's image list and counting components per frame."""
    sa_path = snip_dir / "snippet_annotations.json"
    sa = json.load(open(sa_path))
    image_ids = sorted([i["id"] for i in sa["images"]])
    n = len(image_ids)
    by_image = {}
    for a in merged.get("annotations", []):
        if a.get("category_id") != 6:
            continue
        by_image[a["image_id"]] = count_components(a)

    counts = {}
    for i, img_id in enumerate(image_ids):
        counts[i] = by_image.get(img_id, 0)
    hist = Counter(counts.values())
    matches_pct = None
    if expected_tools is not None:
        matches_pct = round(100 * sum(1 for v in counts.values()
                                      if v == expected_tools) / max(1, n), 1)
    stats = {
        "snippet": f"{snip_dir.parent.name}/{snip_dir.name}",
        "frames": n,
        "histogram": dict(sorted(hist.items())),
        "expected_tools": expected_tools,
        "matches_expected_pct": matches_pct,
        "per_frame_count": {str(k): v for k, v in sorted(counts.items())},
    }
    return stats


def process_snippet(entry, data_dir, gapfill_suffix, merged_suffix):
    ep = entry["ep"]
    snip_name = entry["snippet"]
    snip_dir = data_dir / ep / snip_name

    if entry.get("skipped"):
        return {"status": "skip", "reason": entry["skipped"], "rel": f"{ep}/{snip_name}"}

    expected = entry.get("expected_tools")
    if expected is None:
        expected = expected_from_scene_motion(snip_dir)

    full_rerun = bool(entry.get("full_rerun"))
    gapfill_path = snip_dir / f"annotated_masks{gapfill_suffix}.json"
    if not gapfill_path.exists():
        return {"status": "skip", "reason": "no_gapfill_output",
                "rel": f"{ep}/{snip_name}"}

    if full_rerun:
        merged = json.load(open(gapfill_path))
        merged.setdefault("_merge_meta", {})["full_rerun"] = True
        merged["_merge_meta"]["expected_tools"] = expected
        actions = {"full_rerun_replaced_canonical": len(merged.get("annotations", []))}
    else:
        canon_path = snip_dir / "annotated_masks.json"
        if not canon_path.exists():
            return {"status": "skip", "reason": "no_canonical",
                    "rel": f"{ep}/{snip_name}"}
        canonical = json.load(open(canon_path))
        gapfill = json.load(open(gapfill_path))
        merged, actions = merge_canonical_with_gapfill(
            canonical, gapfill, entry.get("gaps", []), expected
        )

    out_merged = snip_dir / f"annotated_masks{merged_suffix}.json"
    with open(out_merged, "w") as f:
        json.dump(merged, f)

    stats = recompute_stats(snip_dir, merged, expected)
    out_stats = snip_dir / f"tool_detection_stats{merged_suffix}.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "status": "ok",
        "rel": f"{ep}/{snip_name}",
        "expected_tools": expected,
        "match_pct_after": stats["matches_expected_pct"],
        "histogram_after": stats["histogram"],
        "actions": actions,
        "merged_path": str(out_merged),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gapfill-suffix", default=".gapfill")
    ap.add_argument("--merged-suffix", default=".merged")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    data_dir = Path(manifest["data_dir"])

    summary = []
    for entry in manifest.get("snippets", []):
        info = process_snippet(entry, data_dir, args.gapfill_suffix, args.merged_suffix)
        summary.append(info)

    print("=== Merge results ===")
    for s in summary:
        if s["status"] == "ok":
            print(f"  ok    {s['rel']:<22} expected={s['expected_tools']} "
                  f"match_after={s['match_pct_after']}% hist={s['histogram_after']} "
                  f"actions={s['actions']}")
        else:
            print(f"  {s['status']:<6} {s['rel']:<22} {s.get('reason','')}")


if __name__ == "__main__":
    main()
