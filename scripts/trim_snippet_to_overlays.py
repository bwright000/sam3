#!/usr/bin/env python3
"""Trim a snippet's frames_left/, frames_right/, poses.txt to exactly match
the frame set present in overlays/.

Usage:
    python scripts/trim_snippet_to_overlays.py --episode F_3 --snippet 006
    python scripts/trim_snippet_to_overlays.py --episode F_3 --snippet 006 --dry-run
"""

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = BASE_DIR / "data" / "Segments"


def frame_n(path: Path) -> int:
    return int(path.stem.split("_")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True, help="snippet id e.g. 006")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snip = SEGMENTS_DIR / args.episode / f"snippet_{args.snippet}"
    if not snip.exists():
        raise SystemExit(f"no such snippet: {snip}")

    overlays = snip / "overlays"
    left = snip / "frames_left"
    right = snip / "frames_right"
    poses = snip / "poses.txt"

    if not overlays.exists():
        raise SystemExit(f"no overlays dir: {overlays}")

    keep = {frame_n(p) for p in overlays.glob("frame_*.jpg")}
    print(f"overlay frames to keep: {len(keep)}  range=[{min(keep)},{max(keep)}]")

    drop_left = [p for p in left.glob("frame_*.webp") if frame_n(p) not in keep]
    drop_right = [p for p in right.glob("frame_*.webp") if frame_n(p) not in keep]
    print(f"frames_left to drop:  {len(drop_left)}")
    print(f"frames_right to drop: {len(drop_right)}")

    # Parse existing poses.txt
    pose_lines = []
    headers = []
    with open(poses) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                headers.append(line)
                continue
            pose_lines.append(line)
    # Line n corresponds to nth kept frame. poses.txt was regenerated in sorted-by-frame
    # order, but frame_n isn't embedded in each line. We rebuild by assuming current
    # poses.txt contains one line per sorted(left) frame (which was 500 lines).
    # So we need frame_n per line. Recover by matching against sorted(left) before deletion.
    current_left_frames = sorted(frame_n(p) for p in left.glob("frame_*.webp"))
    if len(current_left_frames) != len(pose_lines):
        raise SystemExit(
            f"poses line count {len(pose_lines)} != current frames_left count "
            f"{len(current_left_frames)}; cannot trim safely"
        )
    kept_pose_lines = [
        line for fn, line in zip(current_left_frames, pose_lines) if fn in keep
    ]
    print(f"poses to keep: {len(kept_pose_lines)} of {len(pose_lines)}")

    if args.dry_run:
        print("(dry-run — no changes)")
        return

    for p in drop_left:
        p.unlink()
    for p in drop_right:
        p.unlink()

    # Rewrite poses.txt
    kept_sorted = sorted(keep)
    with open(poses, "w") as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write(f"# Episode snippet from frame {kept_sorted[0]} to {kept_sorted[-1]} "
                f"(trimmed to overlays set)\n")
        for line in kept_pose_lines:
            f.write(line)

    print("DONE")


if __name__ == "__main__":
    main()
