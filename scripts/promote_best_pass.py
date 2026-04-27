#!/usr/bin/env python3
"""For each snippet, pick the best Pass (1 vs 2 lowthresh) by match-expected,
and copy the winning files to be the canonical annotated_masks.json /
tool_detection_stats.json. The losers are kept under their suffix for forensic.

Usage:
    python scripts/promote_best_pass.py --data-dir '/content/data/To Be Annotated'
    python scripts/promote_best_pass.py --data-dir '/content/data/To Be Annotated' --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--suffix", default=".lowthresh",
                    help="Pass 2 suffix, e.g. '.lowthresh'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    promoted = []
    kept = []
    skipped = []

    for stats_p1 in sorted(root.glob("*/snippet_*/tool_detection_stats.json")):
        snip_dir = stats_p1.parent
        stats_p2 = snip_dir / f"tool_detection_stats{args.suffix}.json"
        if not stats_p2.exists():
            skipped.append(snip_dir.name + " (no Pass 2)")
            continue

        d1 = json.load(open(stats_p1))
        d2 = json.load(open(stats_p2))

        # Score by match-expected if available, else by frames-with-any-detection
        def score(d):
            mp = d.get("matches_expected_pct")
            if mp is not None:
                return mp
            hist = d.get("histogram", {})
            n_zero = int(hist.get("0", 0))
            total = d.get("frames", 1) or 1
            # any-detection rate as fallback
            return 100 * (1 - n_zero / total)

        s1 = score(d1)
        s2 = score(d2)

        rel = f"{snip_dir.parent.name}/{snip_dir.name}"
        if s2 > s1:
            promoted.append((rel, s1, s2))
            if not args.dry_run:
                shutil.copy2(stats_p2, stats_p1)
                ann_p2 = snip_dir / f"annotated_masks{args.suffix}.json"
                ann_p1 = snip_dir / "annotated_masks.json"
                if ann_p2.exists():
                    shutil.copy2(ann_p2, ann_p1)
        else:
            kept.append((rel, s1, s2))

    print(f"\n{'snippet':<22} {'pass1':<8} {'pass2':<8} {'winner'}")
    print("-" * 50)
    for rel, s1, s2 in promoted:
        print(f"  {rel:<22} {s1:>6.1f}% {s2:>6.1f}%  Pass 2")
    for rel, s1, s2 in kept:
        print(f"  {rel:<22} {s1:>6.1f}% {s2:>6.1f}%  Pass 1")
    if skipped:
        print("\nSkipped (no Pass 2):")
        for s in skipped:
            print(f"  {s}")
    print(f"\nPromoted {len(promoted)} snippets to Pass 2."
          f" Kept Pass 1 on {len(kept)}.")
    if args.dry_run:
        print("(dry-run; re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
