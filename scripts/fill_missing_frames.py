"""
Extract missing frames from parquet files to fill GT-bounded snippet ranges.

Reads the current snippets JSON for each episode, compares against existing
frames in frames_left/, and pulls only the missing frames from the CRCD
parquet dataset.

Usage:
    python scripts/fill_missing_frames.py                # all episodes
    python scripts/fill_missing_frames.py --episode F_3  # single episode
    python scripts/fill_missing_frames.py --dry-run      # show what would be extracted
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = Path(r"f:\2026 vibes\MPHY Project\CRCD_manual\hub\datasets--SITL-Eng--CRCD\snapshots\f597d230356f4e6d46516b83c2baa4f52c923358\data")
SEGMENTS_DIR = BASE_DIR / "data" / "Segments"

EPISODES = ["C_1", "E_3", "F_3"]


def get_missing_frames(ep, snippet):
    """Return set of frame numbers that need extracting."""
    sid = snippet["snippet_id"]
    sf, ef = snippet["start_frame"], snippet["end_frame"]
    needed = set(range(sf, ef + 1))

    frames_dir = SEGMENTS_DIR / ep / f"snippet_{sid}" / "frames_left"
    if frames_dir.exists():
        for f in frames_dir.glob("frame_*.webp"):
            try:
                vf = int(f.stem.split("_")[1])
                needed.discard(vf)
            except (ValueError, IndexError):
                pass

    return needed


def load_parquet_lazy(ep):
    """Return list of parquet file paths for an episode."""
    ep_dir = PARQUET_DIR / ep
    return sorted(ep_dir.glob("*.parquet"))


def extract_frames_from_parquet(parquet_path, target_frames, output_dirs):
    """Extract specific frames from a single parquet file.

    target_frames: set of frame_n values to extract
    output_dirs: dict of snippet_id -> frames_left Path

    Returns number of frames extracted.
    """
    df = pd.read_parquet(parquet_path, columns=["frame_n", "frame_left"])
    count = 0

    for _, row in df.iterrows():
        fn = row["frame_n"]
        if fn not in target_frames:
            continue

        left_data = row["frame_left"]
        if left_data is None or "bytes" not in left_data:
            continue

        # Find which snippet this frame belongs to
        for sid, (sf, ef, out_dir) in output_dirs.items():
            if sf <= fn <= ef:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"frame_{fn:06d}.webp"
                with open(out_path, "wb") as f:
                    f.write(left_data["bytes"])
                target_frames.discard(fn)
                count += 1
                break

    del df
    return count


def process_episode(ep, dry_run=False):
    """Extract all missing frames for an episode."""
    snippets_path = SEGMENTS_DIR / ep / f"{ep}_snippets.json"
    if not snippets_path.exists():
        print(f"  No snippets JSON found")
        return 0

    with open(snippets_path) as f:
        snippets = json.load(f)

    # Collect all missing frames and their output directories
    all_missing = set()
    output_dirs = {}  # snippet_id -> (start, end, output_path)

    for s in snippets:
        sid = s["snippet_id"]
        missing = get_missing_frames(ep, s)
        if missing:
            frames_dir = SEGMENTS_DIR / ep / f"snippet_{sid}" / "frames_left"
            output_dirs[sid] = (s["start_frame"], s["end_frame"], frames_dir)
            all_missing.update(missing)
            mn, mx = min(missing), max(missing)
            print(f"  snippet_{sid}: {len(missing)} missing frames ({mn}-{mx})")

    if not all_missing:
        print(f"  All frames present")
        return 0

    print(f"  Total missing: {len(all_missing)} frames")

    if dry_run:
        return len(all_missing)

    # Load parquet files one at a time to limit memory
    parquet_files = load_parquet_lazy(ep)
    total_extracted = 0

    for i, pf in enumerate(parquet_files):
        if not all_missing:
            break
        size_mb = pf.stat().st_size / 1e6
        print(f"  Reading {pf.name} ({size_mb:.0f} MB)...", end="", flush=True)
        t0 = time.time()
        count = extract_frames_from_parquet(pf, all_missing, output_dirs)
        dt = time.time() - t0
        print(f" {count} frames in {dt:.1f}s, {len(all_missing)} remaining")
        total_extracted += count

    if all_missing:
        print(f"  WARNING: {len(all_missing)} frames not found in parquet files")

    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Fill missing snippet frames from parquet")
    parser.add_argument("--episode", nargs="+", default=EPISODES,
                        help="Episodes to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be extracted without doing it")
    args = parser.parse_args()

    if not PARQUET_DIR.exists():
        print(f"ERROR: Parquet directory not found: {PARQUET_DIR}")
        sys.exit(1)

    total = 0
    for ep in args.episode:
        print(f"\n{'=' * 60}")
        print(f"  Episode: {ep}")
        print(f"{'=' * 60}")
        count = process_episode(ep, dry_run=args.dry_run)
        total += count

    action = "would extract" if args.dry_run else "extracted"
    print(f"\n{'=' * 60}")
    print(f"  Total {action}: {total} frames")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
