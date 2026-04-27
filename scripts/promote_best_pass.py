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
    ap.add_argument("--suffixes", nargs="+", default=[".lowthresh", ".multiprompt", ".multiprompt_default"],
                    help="Suffixes to compare against the canonical Pass 1; "
                         "best across all wins.")
    ap.add_argument("--suffix", default=None,
                    help="(legacy single-suffix mode)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    promoted = []
    kept = []
    skipped = []

    suffixes = [args.suffix] if args.suffix else args.suffixes

    def score(d):
        mp = d.get("matches_expected_pct")
        if mp is not None:
            return mp
        hist = d.get("histogram", {})
        n_zero = int(hist.get("0", 0))
        total = d.get("frames", 1) or 1
        return 100 * (1 - n_zero / total)

    for stats_p1 in sorted(root.glob("*/snippet_*/tool_detection_stats.json")):
        snip_dir = stats_p1.parent
        rel = f"{snip_dir.parent.name}/{snip_dir.name}"
        d1 = json.load(open(stats_p1))
        s1 = score(d1)
        candidates = [("", d1, s1)]
        for suf in suffixes:
            ps = snip_dir / f"tool_detection_stats{suf}.json"
            if not ps.exists():
                continue
            d = json.load(open(ps))
            candidates.append((suf, d, score(d)))

        # winner = highest score
        candidates.sort(key=lambda c: -c[2])
        winner_suf, winner_d, winner_score = candidates[0]
        runner = candidates[1][2] if len(candidates) > 1 else None

        if winner_suf == "":
            kept.append((rel, candidates))
        else:
            promoted.append((rel, candidates, winner_suf))
            if not args.dry_run:
                shutil.copy2(snip_dir / f"tool_detection_stats{winner_suf}.json", stats_p1)
                ann_w = snip_dir / f"annotated_masks{winner_suf}.json"
                ann_can = snip_dir / "annotated_masks.json"
                if ann_w.exists():
                    shutil.copy2(ann_w, ann_can)

    print(f"\n{'snippet':<22} {'scores per suffix (best first)':<60} {'winner'}")
    print("-" * 100)
    for rel, candidates, winner_suf in promoted:
        ss = "  ".join(f"{c[0] or 'pass1'}={c[2]:.1f}%" for c in candidates)
        print(f"  {rel:<22} {ss:<60} {winner_suf or 'pass1'}")
    for rel, candidates in kept:
        ss = "  ".join(f"{c[0] or 'pass1'}={c[2]:.1f}%" for c in candidates)
        print(f"  {rel:<22} {ss:<60} pass1")
    print(f"\nPromoted {len(promoted)} snippets. Kept pass1 on {len(kept)}.")
    if args.dry_run:
        print("(dry-run; re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
