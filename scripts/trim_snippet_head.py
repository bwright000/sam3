#!/usr/bin/env python3
"""Remove the first N frames of a snippet and align all sibling structures.

Updates (in order):
  frames_left/ + frames_right/   delete N earliest-numbered files
  poses.txt                       drop first N data lines, keep headers, rewrite range
  overlays/                       delete first N .jpg (if exists)
  snippet_NNN_results.json        trim 'frames' list (if exists)
  {EP}_snippets.json             update start_frame, num_frames, motion_start/end_frame, gt_start_split
  session_autosave.json          delete (indices shift → stale)

Backups (metadata only; frame bytes are re-derivable from parquet):
  poses.txt.bak_trim, snippets.json.bak_trim, results.json.bak_trim

Usage:
  python scripts/trim_snippet_head.py --episode E_3 --snippet 001 --count 30           # dry-run
  python scripts/trim_snippet_head.py --episode E_3 --snippet 001 --count 30 --yes     # apply
"""

import argparse
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SEGMENTS = BASE / "data" / "Segments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--count", type=int, required=True, help="frames to remove from head")
    ap.add_argument("--yes", action="store_true", help="actually apply; without this, dry-run only")
    args = ap.parse_args()

    snip_dir = SEGMENTS / args.episode / f"snippet_{args.snippet}"
    if not snip_dir.exists():
        raise SystemExit(f"no snippet dir: {snip_dir}")

    # Identify frames to remove (first N by frame number)
    left_files = sorted(snip_dir.joinpath("frames_left").glob("frame_*.webp"))
    right_files = sorted(snip_dir.joinpath("frames_right").glob("frame_*.webp"))
    overlay_files = sorted(snip_dir.joinpath("overlays").glob("frame_*.jpg")) if (snip_dir / "overlays").exists() else []

    if args.count > len(left_files):
        raise SystemExit(f"count={args.count} exceeds frames_left ({len(left_files)})")

    drop_left = left_files[: args.count]
    drop_right_frames = {int(p.stem.split("_")[1]) for p in drop_left}
    drop_right = [p for p in right_files if int(p.stem.split("_")[1]) in drop_right_frames]
    drop_overlay = [p for p in overlay_files if int(p.stem.split("_")[1]) in drop_right_frames]

    new_first = int(left_files[args.count].stem.split("_")[1])
    old_first = int(left_files[0].stem.split("_")[1])

    print(f"snippet dir: {snip_dir}")
    print(f"count to drop: {args.count}")
    print(f"frames_left before: {len(left_files)}  remaining after: {len(left_files) - args.count}")
    print(f"frame range before: {old_first}-{int(left_files[-1].stem.split('_')[1])}")
    print(f"frame range after:  {new_first}-{int(left_files[-1].stem.split('_')[1])}")
    print(f"drop left: {len(drop_left)} files  (e.g. {drop_left[0].name}..{drop_left[-1].name})")
    print(f"drop right: {len(drop_right)} files")
    print(f"drop overlays: {len(drop_overlay)} files")

    # poses.txt trim preview
    poses_path = snip_dir / "poses.txt"
    new_poses_lines = None
    if poses_path.exists():
        headers, data_lines = [], []
        for line in poses_path.read_text().splitlines():
            if not line.strip():
                continue
            (headers if line.startswith("#") else data_lines).append(line)
        if args.count > len(data_lines):
            raise SystemExit(f"count={args.count} exceeds pose lines ({len(data_lines)})")
        trimmed_headers = ["# TUM format: timestamp tx ty tz qx qy qz qw",
                           f"# Episode snippet from frame {new_first} to "
                           f"{int(left_files[-1].stem.split('_')[1])} (trimmed head)"]
        new_poses_lines = trimmed_headers + data_lines[args.count:]
        print(f"poses.txt lines: {len(data_lines)} -> {len(new_poses_lines) - len(trimmed_headers)}")
    else:
        print("poses.txt: NOT FOUND — skipping")

    # results.json trim preview
    results_path = snip_dir / f"snippet_{args.snippet}_results.json"
    new_results = None
    if results_path.exists():
        with open(results_path) as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("frames"), list):
            orig = len(d["frames"])
            d["frames"] = [fr for fr in d["frames"]
                           if int((fr.get("frame") or Path(fr.get("file","")).stem).split("_")[1])
                           not in drop_right_frames]
            d["num_frames"] = len(d["frames"])
            print(f"results.json frames: {orig} -> {len(d['frames'])}")
            new_results = d
        else:
            print("results.json: unexpected shape — skipping")
    else:
        print("results.json: NOT FOUND — skipping")

    # snippets.json preview
    snippets_json = snip_dir.parent / f"{args.episode}_snippets.json"
    with open(snippets_json) as f:
        snippets = json.load(f)
    snip_meta = None
    for s in snippets:
        if s["snippet_id"] == args.snippet:
            snip_meta = s
            break
    if snip_meta is None:
        raise SystemExit(f"snippet {args.snippet} not in {snippets_json.name}")
    new_start = new_first
    new_end = int(left_files[-1].stem.split("_")[1])
    new_num = new_end - new_start + 1
    split_size = snip_meta.get("split_size", 120)
    new_motion_start = max(snip_meta.get("motion_start_frame", new_start), new_start)
    new_motion_end = snip_meta.get("motion_end_frame", new_end)
    new_gt_start_split = new_start // split_size

    print("\nsnippets.json update:")
    print(f"  start_frame:        {snip_meta.get('start_frame')} -> {new_start}")
    print(f"  end_frame:          {snip_meta.get('end_frame')} (unchanged)")
    print(f"  num_frames:         {snip_meta.get('num_frames')} -> {new_num}")
    print(f"  motion_start_frame: {snip_meta.get('motion_start_frame')} -> {new_motion_start}")
    print(f"  gt_start_split:     {snip_meta.get('gt_start_split')} -> {new_gt_start_split}")

    autosave_path = snip_dir / "session_autosave.json"
    print(f"\nsession_autosave.json present: {autosave_path.exists()} (will be deleted — indices shift)")

    if not args.yes:
        print("\n[dry-run] pass --yes to apply.")
        return

    print("\nAPPLYING...")

    # Backup metadata
    for src, bak in [
        (poses_path, poses_path.with_suffix(".txt.bak_trim")),
        (snippets_json, snippets_json.with_suffix(".json.bak_trim")),
        (results_path, results_path.with_suffix(".json.bak_trim") if results_path.exists() else None),
    ]:
        if src and src.exists() and bak and not bak.exists():
            shutil.copy2(src, bak)
            print(f"backed up {src.name} -> {bak.name}")

    # Delete files
    for p in drop_left + drop_right + drop_overlay:
        p.unlink()
    print(f"deleted {len(drop_left)+len(drop_right)+len(drop_overlay)} frame files")

    # Write poses
    if new_poses_lines is not None:
        with open(poses_path, "w") as f:
            f.write("\n".join(new_poses_lines) + "\n")
        print(f"rewrote {poses_path.name}")

    # Write results.json
    if new_results is not None:
        tmp = results_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(new_results, f)
        tmp.replace(results_path)
        print(f"rewrote {results_path.name}")

    # Update snippets.json
    snip_meta["start_frame"] = new_start
    snip_meta["num_frames"] = new_num
    snip_meta["motion_start_frame"] = new_motion_start
    snip_meta["gt_start_split"] = new_gt_start_split
    tmp = snippets_json.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(snippets, f, indent=2)
    tmp.replace(snippets_json)
    print(f"updated {snippets_json.name}")

    # Delete autosave (indices shift)
    if autosave_path.exists():
        autosave_path.unlink()
        print(f"deleted {autosave_path.name}")

    print("\nDONE")


if __name__ == "__main__":
    main()
