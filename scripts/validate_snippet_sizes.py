#!/usr/bin/env python3
"""Validate every snippet: frames_left, frames_right, overlays, poses, results all same size.

Reports OK / MISMATCH per snippet.
"""

import json
from pathlib import Path

SEGMENTS = Path("data/Segments")
EPISODES = ["C_1", "E_3", "F_3"]


def count_dir(p: Path) -> int | None:
    if not p.exists():
        return None
    return sum(1 for _ in p.iterdir() if _.is_file())


def count_lines(p: Path) -> int | None:
    if not p.exists():
        return None
    with open(p) as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def count_results_frames(p: Path) -> int | None:
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            return len(data["frames"])
        if "num_frames" in data:
            return int(data["num_frames"])
    return None


def main():
    print(f"{'snippet':<20} {'left':>6} {'right':>6} {'overlay':>8} {'poses':>7} {'results':>8}  status")
    print("-" * 90)

    total = 0
    ok = 0
    mismatches = []
    missing_parts = []

    for ep in EPISODES:
        ep_dir = SEGMENTS / ep
        if not ep_dir.exists():
            continue
        for snip in sorted(ep_dir.glob("snippet_*")):
            total += 1
            snip_id = snip.name.split("_")[-1]
            left = count_dir(snip / "frames_left")
            right = count_dir(snip / "frames_right")
            overlay = count_dir(snip / "overlays")
            poses = count_lines(snip / "poses.txt")
            results = count_results_frames(snip / f"snippet_{snip_id}_results.json")

            vals = {"left": left, "right": right, "overlay": overlay, "poses": poses, "results": results}
            present = {k: v for k, v in vals.items() if v is not None}
            missing = [k for k, v in vals.items() if v is None]

            unique = set(present.values())
            if len(unique) == 1:
                status = "OK"
                if missing:
                    status = f"OK (missing: {','.join(missing)})"
                    missing_parts.append((f"{ep}/{snip.name}", missing))
                else:
                    ok += 1
            else:
                status = f"MISMATCH max-min={max(unique)-min(unique)}"
                mismatches.append((f"{ep}/{snip.name}", vals))

            def fmt(v):
                return "-" if v is None else str(v)

            print(
                f"{ep}/{snip.name:<12} "
                f"{fmt(left):>6} {fmt(right):>6} {fmt(overlay):>8} {fmt(poses):>7} {fmt(results):>8}  "
                f"{status}"
            )

    print("-" * 90)
    print(f"Total snippets: {total}  |  fully OK: {ok}  |  mismatched: {len(mismatches)}  |  missing parts: {len(missing_parts)}")

    if mismatches:
        print("\nMismatches:")
        for name, vals in mismatches:
            print(f"  {name}: {vals}")


if __name__ == "__main__":
    main()
