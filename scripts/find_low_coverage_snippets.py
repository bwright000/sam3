#!/usr/bin/env python3
"""Identify snippets that need a Pass 2 (low-threshold) re-run.

A snippet is flagged when:
  - it has an expected_tools count (not None)
  - tool_detection_stats.json shows matches_expected_pct below the cutoff
    (default 98%)

Outputs a list of `episode/snippet_id` strings, ready to feed back into
batch_text_propagate.py via shell looping.

Usage:
    python scripts/find_low_coverage_snippets.py \\
        --data-dir '/content/data/To Be Annotated' \\
        --threshold 98

    # In a one-liner — re-run flagged snippets at lower thresholds:
    for snip in $(python scripts/find_low_coverage_snippets.py --data-dir "$DATA"); do
        ep=${snip%%/*}; sid=${snip##*/}
        python scripts/batch_text_propagate.py --data-dir "$DATA" \\
            --episode "$ep" --snippet "$sid" \\
            --score-threshold 0.3 --new-det-thresh 0.5 \\
            --output-suffix .lowthresh \\
            --render-overlays \\
            --expected-json scripts/expected_tool_counts.json
    done
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--threshold", type=float, default=98.0,
                    help="match-expected percentage cutoff; below this is flagged")
    ap.add_argument("--stats-suffix", default="",
                    help="e.g. '.lowthresh' to look at Pass 2 stats files")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    fname = f"tool_detection_stats{args.stats_suffix}.json"

    rows = []
    for stats_path in sorted(root.glob(f"*/snippet_*/{fname}")):
        try:
            with open(stats_path) as f:
                d = json.load(f)
        except Exception:
            continue
        rel = stats_path.parent.relative_to(root)  # e.g. E_3/snippet_001
        ep = rel.parts[0]
        sid = rel.parts[1].replace("snippet_", "")
        expected = d.get("expected_tools")
        match_pct = d.get("matches_expected_pct")
        rows.append({
            "key": f"{ep}/{sid}",
            "expected": expected,
            "match_pct": match_pct,
            "histogram": d.get("histogram"),
            "frames": d.get("frames"),
        })

    flagged = []
    if args.verbose:
        print(f"{'snippet':<14} {'expected':<10} {'match%':<8} {'hist'}")
        print("-" * 60)
    for r in rows:
        is_flagged = (r["expected"] is not None
                      and r["match_pct"] is not None
                      and r["match_pct"] < args.threshold)
        marker = "FLAG" if is_flagged else "ok  "
        if is_flagged:
            flagged.append(r["key"])
        if args.verbose:
            exp = r["expected"] if r["expected"] is not None else "var"
            mp = f"{r['match_pct']:.1f}%" if r["match_pct"] is not None else "-"
            hist = r["histogram"] or {}
            print(f"  {marker} {r['key']:<14} {str(exp):<10} {mp:<8} {hist}")

    if args.verbose:
        print(f"\n{len(flagged)} of {len(rows)} snippets flagged for Pass 2 (< {args.threshold}%)")
    else:
        for k in flagged:
            print(k)


if __name__ == "__main__":
    main()
