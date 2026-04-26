#!/usr/bin/env python3
"""Backfill frames_right/ and regenerate poses.txt from parquet to match frames_left/.

Uses frames_left/ as source of truth for each snippet's frame range, then:
  1. Extracts missing frame_right bytes from parquet -> frame_NNNNNN.webp
  2. Rewrites poses.txt (TUM: timestamp tx ty tz qx qy qz qw) from
     `timestamp` + `/ECM/custom/setpoint_cp` for the same range.

Original poses.txt preserved as poses.txt.bak.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = Path(
    r"f:\2026 vibes\MPHY Project\CRCD_manual\hub\datasets--SITL-Eng--CRCD"
    r"\snapshots\f597d230356f4e6d46516b83c2baa4f52c923358\data"
)
SEGMENTS_DIR = BASE_DIR / "data" / "Segments"
EPISODES = ["C_1", "E_3", "F_3"]


def scan_frames_left_range(snip_dir: Path) -> set[int]:
    fl = snip_dir / "frames_left"
    if not fl.exists():
        return set()
    out = set()
    for f in fl.glob("frame_*.webp"):
        try:
            out.add(int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return out


def scan_frames_right(snip_dir: Path) -> set[int]:
    fr = snip_dir / "frames_right"
    if not fr.exists():
        return set()
    out = set()
    for f in fr.glob("frame_*.webp"):
        try:
            out.add(int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return out


def plan_episode(ep: str) -> dict:
    """Per snippet determine target frame set and missing-right set."""
    ep_seg = SEGMENTS_DIR / ep
    plan = {}
    for snip in sorted(ep_seg.glob("snippet_*")):
        left = scan_frames_left_range(snip)
        right = scan_frames_right(snip)
        if not left:
            continue
        target = left
        missing_right = target - right
        plan[snip.name] = {
            "dir": snip,
            "target": target,
            "missing_right": missing_right,
            "fn_min": min(target),
            "fn_max": max(target),
        }
    return plan


def process_episode(ep: str, dry_run: bool = False) -> None:
    print(f"\n=== {ep} ===")
    plan = plan_episode(ep)
    if not plan:
        print("  no snippets to process")
        return

    any_missing_right = sum(len(p["missing_right"]) for p in plan.values())
    total_poses_frames = sum(len(p["target"]) for p in plan.values())
    ep_min = min(p["fn_min"] for p in plan.values())
    ep_max = max(p["fn_max"] for p in plan.values())
    print(
        f"  snippets={len(plan)}  missing_right_total={any_missing_right}  "
        f"poses_frames_total={total_poses_frames}  ep_range=[{ep_min},{ep_max}]"
    )
    for name, p in plan.items():
        print(
            f"    {name}: target={len(p['target'])} missing_right={len(p['missing_right'])} "
            f"range=[{p['fn_min']},{p['fn_max']}]"
        )

    if dry_run:
        return

    # Collect timestamp + ECM pose for every target frame across the episode
    # Map fn -> (timestamp, ecm_pose np.ndarray shape (7,))
    pose_rows: dict[int, tuple[float, np.ndarray]] = {}
    all_targets: set[int] = set().union(*(p["target"] for p in plan.values()))

    parquets = sorted((PARQUET_DIR / ep).glob("*.parquet"))
    if not parquets:
        print(f"  ERROR: no parquet files in {PARQUET_DIR / ep}")
        return

    right_writes = 0
    for i, pq in enumerate(parquets, 1):
        t0 = time.time()
        cols = ["frame_n", "timestamp", "frame_right", "/ECM/custom/setpoint_cp"]
        df = pd.read_parquet(pq, columns=cols)
        # Filter to frames we care about
        df_sub = df[df["frame_n"].isin(all_targets)]
        if df_sub.empty:
            del df
            print(f"  [{i}/{len(parquets)}] {pq.name}: no target frames  ({time.time()-t0:.1f}s)")
            continue

        # Write right frames + record pose
        for _, row in df_sub.iterrows():
            fn = int(row["frame_n"])
            ts = float(row["timestamp"])
            ecm = row["/ECM/custom/setpoint_cp"]
            if ecm is not None:
                pose_rows[fn] = (ts, np.asarray(ecm, dtype=float))

            # Find snippet owning this frame (first hit — ranges don't overlap across snippets)
            for name, p in plan.items():
                if fn in p["missing_right"]:
                    right_data = row["frame_right"]
                    if right_data is None:
                        continue
                    b = right_data.get("bytes") if isinstance(right_data, dict) else None
                    if not b:
                        continue
                    fr_dir = p["dir"] / "frames_right"
                    fr_dir.mkdir(parents=True, exist_ok=True)
                    (fr_dir / f"frame_{fn:06d}.webp").write_bytes(b)
                    p["missing_right"].discard(fn)
                    right_writes += 1
                    break
        del df, df_sub
        print(
            f"  [{i}/{len(parquets)}] {pq.name}: "
            f"poses_so_far={len(pose_rows)} right_writes={right_writes}  "
            f"({time.time()-t0:.1f}s)"
        )

    # Rewrite poses.txt per snippet (with .bak backup)
    for name, p in plan.items():
        target = sorted(p["target"])
        missing_poses = [fn for fn in target if fn not in pose_rows]
        if missing_poses:
            print(f"  WARN: {name} missing pose data for {len(missing_poses)} frames "
                  f"(e.g. {missing_poses[:5]})")
        poses_path = p["dir"] / "poses.txt"
        if poses_path.exists():
            bak = poses_path.with_suffix(".txt.bak")
            if not bak.exists():
                shutil.copy2(poses_path, bak)

        with open(poses_path, "w") as f:
            f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
            if target:
                f.write(f"# Episode snippet from frame {target[0]} to {target[-1]}\n")
            for fn in target:
                if fn not in pose_rows:
                    continue
                ts, pose = pose_rows[fn]
                tx, ty, tz, qx, qy, qz, qw = pose[:7]
                f.write(
                    f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
                )

    still_missing = sum(len(p["missing_right"]) for p in plan.values())
    print(f"  DONE  right_writes={right_writes}  still_missing_right={still_missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="single episode e.g. F_3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eps = [args.episode] if args.episode else EPISODES
    for ep in eps:
        process_episode(ep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
