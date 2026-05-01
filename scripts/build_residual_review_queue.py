#!/usr/bin/env python3
"""Stage 6 of the auto-gap-fill pipeline.

After merge + promotion, identify frames that are STILL under-detected and
write a review queue for the new annotator (sam3_annotator/) to consume.

Inputs:
    --merged-stats-glob     pattern for tool_detection_stats.merged.json
    --manifest              gap_manifest.json (for expected_tools / scene_motion)

Output:
    review_queue.json       [{ep, snippet, snip_idx, image_id, current_count,
                               expected_count, severity, suggested_action}, ...]

Severity levels:
    "zero"  — current_count == 0
    "under" — current_count < expected_count
    "over"  — current_count > expected_count (likely false positive)
"""

import argparse
import json
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stats-suffix", default=".merged")
    ap.add_argument("--out", default="review_queue.json")
    ap.add_argument("--include-over", action="store_true",
                    help="flag frames where count > expected (false positives)")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    data_dir = Path(manifest["data_dir"])

    queue = []
    per_snip = {}
    manual_paint_snippets = []

    for entry in manifest.get("snippets", []):
        if entry.get("skipped"):
            continue
        ep = entry["ep"]
        snip_name = entry["snippet"]
        snip_dir = data_dir / ep / snip_name

        # Snippet-level manual-paint warning (asymmetry-detected by Stage 1)
        if entry.get("structural_undermask_warning") or \
           entry.get("recommended_action", "").startswith("manual_paint_seed_required"):
            manual_paint_snippets.append({
                "ep": ep,
                "snippet": snip_name,
                "asymmetry_score": entry.get("asymmetry_score"),
                "healthy_anchor_count": entry.get("healthy_anchor_count", 0),
                "recommended_action": entry.get("recommended_action", "manual_paint_seed_required"),
                "reason": (
                    "smaller-component / larger-component ratio is consistently low "
                    "across the snippet -> SAM3 captures only a tool's tip / partial body "
                    "and there is no clean reference frame to seed re-propagation from. "
                    "Paint a corrected mask in the new annotator at one well-lit frame "
                    "and feed it as the full_rerun seed."
                ),
            })

        stats_path = snip_dir / f"tool_detection_stats{args.stats_suffix}.json"
        if not stats_path.exists():
            continue
        stats = json.load(open(stats_path))
        expected = stats.get("expected_tools") or entry.get("expected_tools") \
                   or expected_from_scene_motion(snip_dir)
        # Use annotated_masks image_ids if available (handles trimmed-production mismatch)
        image_ids = []
        am = snip_dir / "annotated_masks.json"
        if am.exists():
            try:
                image_ids = sorted([i["id"] for i in json.load(open(am)).get("images", [])])
            except Exception:
                pass
        if not image_ids:
            sa = json.load(open(snip_dir / "snippet_annotations.json"))
            image_ids = sorted([i["id"] for i in sa["images"]])
        pfc = stats.get("per_frame_count", {})

        flagged_here = 0
        for k, v in pfc.items():
            idx = int(k)
            if idx >= len(image_ids):
                continue
            severity = None
            if v == 0:
                severity = "zero"
            elif expected is not None and v < expected:
                severity = "under"
            elif args.include_over and expected is not None and v > expected:
                severity = "over"
            if severity is None:
                continue
            queue.append({
                "ep": ep,
                "snippet": snip_name,
                "snip_idx": idx,
                "image_id": image_ids[idx],
                "current_count": v,
                "expected_count": expected,
                "severity": severity,
                "suggested_action": (
                    "paint_anchor_then_propagate" if severity == "zero"
                    else "add_correction_click_for_missing_tool" if severity == "under"
                    else "review_for_false_positive"
                ),
            })
            flagged_here += 1
        per_snip[f"{ep}/{snip_name}"] = flagged_here

    out = {
        "manifest": str(Path(args.manifest).resolve()),
        "stats_suffix": args.stats_suffix,
        "total_flagged_frames": len(queue),
        "by_snippet": per_snip,
        "manual_paint_required_snippets": manual_paint_snippets,
        "queue": queue,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out}")
    if manual_paint_snippets:
        print()
        print(f"!! Manual paint required ({len(manual_paint_snippets)} snippets) !!")
        for m in manual_paint_snippets:
            print(f"   {m['ep']}/{m['snippet']:<22} "
                  f"asymmetry={m['asymmetry_score']} healthy_anchors={m['healthy_anchor_count']}")
        print(f"   -> open each in sam3_annotator/, paint a corrected Tool mask at one")
        print(f"      good frame, then feed that PNG as a fresh anchor seed.")
    print()
    print(f"Total flagged frames in residual queue: {len(queue)}")
    for k, v in sorted(per_snip.items()):
        print(f"  {k:<22}  {v} frames flagged")


if __name__ == "__main__":
    main()
