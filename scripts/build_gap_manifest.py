#!/usr/bin/env python3
"""Stage 1 of the auto-gap-fill pipeline.

Scan staged ('To Be Annotated') snippets and emit a manifest describing where
SAM3 propagation produced gaps that need targeted re-prop.

Outputs `gap_manifest.json` at --out path with this shape:

{
  "data_dir": "...",
  "default_strategy": "single_anchor_bidirectional",
  "snippets": [
    {
      "ep": "E_3",
      "snippet": "snippet_003",
      "frames": 263,
      "image_id_start": 16777,
      "image_id_end": 17039,
      "expected_tools": 1,
      "expected_source": "stats",            # or "scene_motion" or "manual"
      "best_variant": "" | ".lowthresh" | "...",
      "match_pct": 87.5,
      "full_rerun": false,
      "gaps": [
        {
          "kind": "zero" | "under",
          "snip_idx_start": 23, "snip_idx_end": 55,
          "image_id_start": 16800, "image_id_end": 16832,
          "length": 33,
          "anchor_idx": 39,                  # midpoint
          "anchor_image_id": 16816,
          "anchor_source": "neighbour_pre",  # or _post / midpoint_paint_required
          "anchor_polygon_source_idx": 22    # nearest good frame whose polygon we'll use
        }
      ]
    }
  ]
}

For F_3/snippet_001 — flagged `full_rerun: true` because user asked for a
full re-pass. The "best existing frame" is reported for reference but the
re-prop will be seeded from a freshly painted mask if gap-fill drifts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def runs(indices):
    if not indices:
        return []
    indices = sorted(indices)
    out = []
    s = e = indices[0]
    for i in indices[1:]:
        if i == e + 1:
            e = i
        else:
            out.append((s, e))
            s = e = i
    out.append((s, e))
    return out


def load_stats_variants(snip_dir):
    """Return list of (suffix, dict) for every tool_detection_stats*.json."""
    out = []
    for p in sorted(snip_dir.glob("tool_detection_stats*.json")):
        suffix = p.stem.replace("tool_detection_stats", "")
        try:
            out.append((suffix, json.load(open(p))))
        except Exception:
            continue
    return out


def _shoelace_area(poly_flat):
    if not isinstance(poly_flat, list) or len(poly_flat) < 6:
        return 0.0
    xs = poly_flat[0::2]
    ys = poly_flat[1::2]
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def per_frame_tool_stats_from_coco(coco):
    """Return per-frame stats keyed by snip_idx:
        { idx: {
            "total_area": float,
            "comps": int,
            "sub_areas": [float, ...],     # one per polygon component
            "min_comp_area": float | None,
            "max_comp_area": float | None,
        }, ... }
    """
    image_ids = sorted([i["id"] for i in coco.get("images", [])])
    idx_of = {gid: i for i, gid in enumerate(image_ids)}
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    tool_id = next((k for k, v in cats.items() if v == "Tool"), 6)

    stats = {i: {"total_area": 0.0, "comps": 0, "sub_areas": [],
                 "min_comp_area": None, "max_comp_area": None}
             for i in range(len(image_ids))}
    for a in coco.get("annotations", []):
        if a.get("category_id") != tool_id:
            continue
        gid = a["image_id"]
        if gid not in idx_of:
            continue
        idx = idx_of[gid]
        stats[idx]["total_area"] += float(a.get("area", 0))
        for poly in a.get("segmentation", []) or []:
            ar = _shoelace_area(poly)
            if ar <= 0:
                continue
            stats[idx]["comps"] += 1
            stats[idx]["sub_areas"].append(ar)
    for s in stats.values():
        if s["sub_areas"]:
            s["min_comp_area"] = min(s["sub_areas"])
            s["max_comp_area"] = max(s["sub_areas"])
    return stats


def pick_best_variant(snip_dir, variants):
    """Pick the variant with the highest mask-file coverage / match%.

    Mirrors promote_best_pass.py's score function so manifest reflects what
    promote will choose.
    """
    if not variants:
        return None, None
    best = None
    best_score = -1
    for suffix, d in variants:
        mask_path = snip_dir / f"annotated_masks{suffix}.json"
        n_imgs = 0
        if mask_path.exists():
            try:
                n_imgs = len(json.load(open(mask_path)).get("images", []))
            except Exception:
                pass
        total = d.get("frames", 1) or 1
        mp = d.get("matches_expected_pct")
        if mp is not None:
            score = mp
        elif n_imgs:
            score = 100 * n_imgs / total
        else:
            hist = d.get("histogram", {})
            n_zero = int(hist.get("0", 0))
            score = 100 * (1 - n_zero / total)
        if score > best_score:
            best_score = score
            best = (suffix, d)
    return best, round(best_score, 1)


def expected_from_scene_motion(snip_dir):
    """Infer expected tool count from PSM motion flags."""
    sm_path = snip_dir / "scene_motion.json"
    if not sm_path.exists():
        return None, "missing_scene_motion"
    try:
        sm = json.load(open(sm_path))
    except Exception:
        return None, "unreadable_scene_motion"
    p1 = bool(sm.get("psm1_motion", False))
    p2 = bool(sm.get("psm2_motion", False))
    if p1 and p2:
        return 2, "scene_motion(p1+p2)"
    if p1 or p2:
        return 1, "scene_motion(single_psm)"
    return None, "scene_motion(no_psm_motion)"


def find_anchor(snip_idx_start, snip_idx_end, good_indices):
    """Pick the SAM3 add_new_mask seed frame for a gap.

    Strategy (a) — single_anchor_bidirectional: the seed must be placed at a
    frame where the mask actually overlaps the tool in pixel space. So we
    seed at the *closest good neighbour frame* (one tick before gap start,
    falling back to one tick after) and let bidirectional propagation pull
    the seed mask into the gap.
    """
    pre = max((g for g in good_indices if g < snip_idx_start), default=None)
    post = min((g for g in good_indices if g > snip_idx_end), default=None)
    if pre is not None and (post is None or (snip_idx_start - pre) <= (post - snip_idx_end)):
        return pre, pre, "neighbour_pre"
    if post is not None:
        return post, post, "neighbour_post"
    return (snip_idx_start + snip_idx_end) // 2, None, "no_neighbour_paint_required"


def best_full_rerun_seed_from_coco(snip_dir, expected_tools):
    """For F_3/001 full-rerun: pick a frame whose existing Tool polygon looks
    cleanest (component count == expected_tools, area near median) to use as
    the SAM3 add_new_mask seed for the fresh full pass.
    """
    sa_path = snip_dir / "snippet_annotations.json"
    if not sa_path.exists():
        return None
    sa = json.load(open(sa_path))
    cats = {c["id"]: c["name"] for c in sa["categories"]}
    tool_id = next((k for k, v in cats.items() if v == "Tool"), None)
    if tool_id is None:
        return None

    img_ids = sorted([i["id"] for i in sa["images"]])
    idx_of = {gid: i for i, gid in enumerate(img_ids)}

    by_frame = {}
    for a in sa["annotations"]:
        if a["category_id"] != tool_id:
            continue
        polys = a.get("segmentation", [])
        area = a.get("area", 0)
        gid = a["image_id"]
        rec = by_frame.setdefault(gid, {"comps": 0, "area": 0})
        rec["comps"] += len(polys)
        rec["area"] += area

    if not by_frame:
        return None

    candidates = [
        (gid, rec) for gid, rec in by_frame.items()
        if (expected_tools is None or rec["comps"] == expected_tools)
    ]
    if not candidates:
        candidates = list(by_frame.items())

    areas = sorted(rec["area"] for _, rec in candidates)
    median = areas[len(areas) // 2]
    candidates.sort(key=lambda x: abs(x[1]["area"] - median))
    seed_gid, seed_rec = candidates[0]
    return {
        "seed_image_id": seed_gid,
        "seed_snip_idx": idx_of[seed_gid],
        "components": seed_rec["comps"],
        "area": seed_rec["area"],
    }


def detect_undermask_runs(per_frame_count, per_frame_stats, expected,
                          area_floor_ratio=0.7, min_run_length=2,
                          tiny_component_ratio=0.3):
    """Flag runs where component count >= expected but the mask quality is
    suspect.  Two triggers:

      A. Total tool area is below `area_floor_ratio * median(healthy total)`.
      B. The smallest component in a frame is < `tiny_component_ratio` of the
         snippet's median *largest-component* area — i.e. one of the two tools
         is detected only as a tiny tip while the other is full-body.

    Trigger (B) is critical for snippets like F_3/002 where SAM3 consistently
    captures only a tool's gripper tip and misses the shaft.
    """
    eligible = []
    for i, c in per_frame_count.items():
        if expected is not None and c < expected:
            continue
        if c == 0:
            continue
        eligible.append(i)
    if not eligible:
        return [], 0.0, 0.0

    healthy_totals = sorted(per_frame_stats.get(i, {}).get("total_area", 0.0)
                            for i in eligible)
    median_total = healthy_totals[len(healthy_totals) // 2] if healthy_totals else 0.0

    healthy_max_comp = sorted(per_frame_stats.get(i, {}).get("max_comp_area") or 0.0
                              for i in eligible)
    healthy_max_comp = [a for a in healthy_max_comp if a > 0]
    median_max_comp = (healthy_max_comp[len(healthy_max_comp) // 2]
                       if healthy_max_comp else 0.0)

    flagged = set()
    if median_total > 0:
        threshold_total = median_total * area_floor_ratio
        for i in eligible:
            ta = per_frame_stats.get(i, {}).get("total_area", 0.0)
            if ta < threshold_total:
                flagged.add(i)

    if median_max_comp > 0:
        threshold_min = median_max_comp * tiny_component_ratio
        for i in eligible:
            mc = per_frame_stats.get(i, {}).get("min_comp_area")
            if mc is None:
                continue
            comps = per_frame_stats.get(i, {}).get("comps", 0)
            # Only meaningful when the frame has multiple components and the
            # smallest one is suspiciously tiny.
            if comps >= 2 and mc < threshold_min:
                flagged.add(i)

    out_runs = []
    for s, e in runs(sorted(flagged)):
        if e - s + 1 >= min_run_length:
            out_runs.append((s, e))
    return out_runs, median_total, median_max_comp


def build_snippet_entry(ep, snip_dir, full_rerun_set):
    rel = f"{ep}/{snip_dir.name}"
    rel_stripped = f"{ep}/{snip_dir.name.replace(' tbd', '').strip()}"
    sa_path = snip_dir / "snippet_annotations.json"
    if not sa_path.exists():
        return {"ep": ep, "snippet": snip_dir.name, "skipped": "no_snippet_annotations"}
    sa = json.load(open(sa_path))
    sa_img_ids = sorted([i["id"] for i in sa["images"]])
    # Prefer annotated_masks.*.json image_ids when available — they're aligned
    # with tool_detection_stats per_frame_count (which is what gap detection
    # walks). snippet_annotations.json may be a post-trim subset.
    am_path = snip_dir / "annotated_masks.json"
    img_ids = sa_img_ids
    if am_path.exists():
        try:
            am = json.load(open(am_path))
            am_ids = sorted([i["id"] for i in am.get("images", [])])
            if len(am_ids) >= len(sa_img_ids):
                img_ids = am_ids
        except Exception:
            pass
    if not img_ids:
        return {"ep": ep, "snippet": snip_dir.name, "skipped": "no_images"}

    entry = {
        "ep": ep,
        "snippet": snip_dir.name,
        "frames": len(img_ids),
        "image_id_start": img_ids[0],
        "image_id_end": img_ids[-1],
        "full_rerun": rel in full_rerun_set or rel_stripped in full_rerun_set,
    }

    variants = load_stats_variants(snip_dir)

    expected = None
    expected_source = None
    for _, d in variants:
        if d.get("expected_tools") is not None:
            expected = d["expected_tools"]
            expected_source = "stats"
            break
    if expected is None:
        expected, expected_source = expected_from_scene_motion(snip_dir)
    entry["expected_tools"] = expected
    entry["expected_source"] = expected_source

    if entry["full_rerun"]:
        entry["best_variant"] = None
        entry["match_pct"] = None
        entry["gaps"] = []
        seed_info = best_full_rerun_seed_from_coco(snip_dir, expected)
        entry["full_rerun_seed"] = seed_info
        entry["full_rerun_strategy"] = (
            "extract_seed_from_existing_GT_then_full_bidirectional_reprop"
            if seed_info else "no_existing_GT_text_prompt_required"
        )
        return entry

    if not variants:
        entry["best_variant"] = None
        entry["match_pct"] = None
        entry["gaps"] = []
        entry["note"] = "no_stats_files_propagation_never_ran"
        return entry

    (best_suffix, best_d), best_score = pick_best_variant(snip_dir, variants)
    entry["best_variant"] = best_suffix
    entry["match_pct"] = best_score

    pfc = best_d.get("per_frame_count", {})
    pairs = sorted((int(k), int(v)) for k, v in pfc.items())
    pfc_dict = dict(pairs)
    n = len(pairs)
    zero_idx = [i for i, c in pairs if c == 0]
    under_idx = []
    if expected is not None:
        under_idx = [i for i, c in pairs if 0 < c < expected]

    # Load per-frame mask stats from the best variant's annotated_masks file
    pfs = {}
    if best_suffix is not None:
        ann_path = snip_dir / f"annotated_masks{best_suffix}.json"
        if ann_path.exists():
            try:
                pfs = per_frame_tool_stats_from_coco(json.load(open(ann_path)))
            except Exception:
                pfs = {}

    undermask_runs, median_total, median_max_comp = detect_undermask_runs(
        pfc_dict, pfs, expected,
        area_floor_ratio=0.7, min_run_length=2, tiny_component_ratio=0.3,
    )
    undermask_idx_set = {i for s, e in undermask_runs for i in range(s, e + 1)}

    # Snippet-level asymmetry diagnostic: in many frames is the smaller
    # component much smaller than the larger? Indicates structural undermask
    # (e.g. SAM3 catching gripper tip but missing shaft) — in which case
    # gap-fill alone can't fix it; manual paint is required.
    ratios = []
    for i in range(n):
        s = pfs.get(i, {})
        if s.get("comps", 0) >= 2 and s.get("max_comp_area"):
            mn, mx = s["min_comp_area"], s["max_comp_area"]
            if mx and mx > 0:
                ratios.append(mn / mx)
    asymmetry_score = (sorted(ratios)[len(ratios) // 2]
                       if ratios else None)
    structural_undermask = (asymmetry_score is not None and
                            asymmetry_score < 0.30 and
                            len(ratios) >= 0.5 * n)

    # A "good" anchor source must have count==expected AND total area
    # >= 0.7*median AND not be in the structural-undermask warning zone.
    good_idx = []
    for i, c in pairs:
        if c == 0:
            continue
        if expected is not None and c < expected:
            continue
        if i in undermask_idx_set:
            continue
        ta = pfs.get(i, {}).get("total_area", 0.0)
        if median_total > 0 and ta < 0.7 * median_total:
            continue
        s = pfs.get(i, {})
        if s.get("comps", 0) >= 2 and s.get("max_comp_area"):
            r = s["min_comp_area"] / s["max_comp_area"] if s["max_comp_area"] > 0 else 0
            if r < 0.30:
                continue
        good_idx.append(i)

    entry["median_tool_total_area"] = round(median_total, 1)
    entry["median_max_component_area"] = round(median_max_comp, 1)
    entry["asymmetry_score"] = (round(asymmetry_score, 3)
                                if asymmetry_score is not None else None)
    entry["structural_undermask_warning"] = bool(structural_undermask)
    entry["healthy_anchor_count"] = len(good_idx)
    if structural_undermask and len(good_idx) == 0:
        entry["recommended_action"] = (
            "manual_paint_seed_required: every frame is structurally undermasked, "
            "no clean neighbour exists; paint a corrected mask in the new annotator "
            "and run propagate_gap_fill.py with that as the full_rerun seed."
        )

    gaps = []
    for kind, indices in [
        ("zero", zero_idx),
        ("under", under_idx),
        ("undermask", sorted(undermask_idx_set)),
    ]:
        for s, e in runs(indices):
            if kind == "undermask" and e - s + 1 < 2:
                continue
            anchor_idx, poly_src_idx, source = find_anchor(s, e, good_idx)
            poly_src_image_id = (
                img_ids[poly_src_idx]
                if poly_src_idx is not None and poly_src_idx < len(img_ids)
                else None
            )
            gap_entry = {
                "kind": kind,
                "snip_idx_start": s,
                "snip_idx_end": e,
                "image_id_start": img_ids[s] if s < len(img_ids) else None,
                "image_id_end": img_ids[e] if e < len(img_ids) else None,
                "length": e - s + 1,
                "anchor_idx": anchor_idx,
                "anchor_image_id": img_ids[anchor_idx] if anchor_idx < len(img_ids) else None,
                "anchor_source": source,
                "anchor_polygon_source_idx": poly_src_idx,
                "anchor_polygon_source_image_id": poly_src_image_id,
            }
            if kind == "undermask":
                worst_ratio = min(
                    (pfs.get(i, {}).get("total_area", 0.0) / median_total)
                    if median_total > 0 else 1.0
                    for i in range(s, e + 1)
                )
                gap_entry["worst_area_ratio"] = round(worst_ratio, 2)
                gap_entry["median_total_area_px"] = round(median_total, 1)
            gaps.append(gap_entry)
    entry["gaps"] = gaps
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="staged snippets root, e.g. .../To Be Annotated")
    ap.add_argument("--out", default="gap_manifest.json")
    ap.add_argument("--full-rerun", nargs="*", default=["F_3/snippet_001"],
                    help="list of ep/snippet that need a full re-pass instead of gap-fill")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    full_rerun_set = set(args.full_rerun or [])

    entries = []
    for ep_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for snip_dir in sorted(ep_dir.glob("snippet_*")):
            if not snip_dir.is_dir():
                continue
            entry = build_snippet_entry(ep_dir.name, snip_dir, full_rerun_set)
            entries.append(entry)

    manifest = {
        "data_dir": str(data_dir),
        "default_strategy": "single_anchor_bidirectional",
        "snippets": entries,
    }
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = Counter()
    for e in entries:
        if e.get("skipped"):
            summary["skipped"] += 1
            continue
        if e.get("full_rerun"):
            summary["full_rerun"] += 1
        elif not e.get("gaps"):
            summary["no_gaps_or_no_stats"] += 1
        else:
            summary["with_gaps"] += 1
        summary["total_gaps"] += len(e.get("gaps") or [])

    print(f"Wrote {out_path}")
    print(f"Snippets: {len(entries)} (full_rerun={summary['full_rerun']}, "
          f"with_gaps={summary['with_gaps']}, "
          f"no_gaps_or_no_stats={summary['no_gaps_or_no_stats']}, "
          f"skipped={summary['skipped']})")
    print(f"Total gap runs to fix: {summary['total_gaps']}")
    print()
    for e in entries:
        if e.get("skipped"):
            print(f"  skip   {e['ep']}/{e['snippet']}  ({e['skipped']})")
            continue
        tag = "RERUN" if e.get("full_rerun") else (
            "GAPS " if e.get("gaps") else "OK   "
        )
        gap_str = ""
        if e.get("gaps"):
            gap_str = ", ".join(
                f"{g['kind']}[{g['snip_idx_start']}..{g['snip_idx_end']}]"
                for g in e["gaps"]
            )
        print(f"  {tag} {e['ep']}/{e['snippet']:<14} expected={e.get('expected_tools')} "
              f"match={e.get('match_pct')}%  gaps=[{gap_str}]")


if __name__ == "__main__":
    main()
