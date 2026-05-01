#!/usr/bin/env python3
"""Trim a snippet to keep only frames in [keep_start, keep_end] (inclusive).

Handles head trims, tail trims, and arbitrary-range trims uniformly.
Updates all sibling structures and backs up metadata before mutating.

Touches (when present):
  frames_left/, frames_right/, overlays/, overlays_tool/,
  overlays.bak_pre_render/, combined_masks/        (delete files outside range)

  poses.txt                                         (drop lines by index, rewrite header)
  snippet_annotations.json                          (filter images + annotations)
  annotated_masks.json [+ all .lowthresh, .lowest, .multiprompt, .multiprompt_default,
    .lowest_multi sidecars]                          (filter images + annotations)
  combined_annotations.json                         (filter images + annotations)
  snippet_NNN_results.json                          (filter frames list)

Deletes (stale after trim — re-derivable):
  session_autosave.json
  snippet_NNN_overlay.mp4, video_left.mp4, video_stereo.mp4
  velocity.png, visualization.html, scene_motion.json
  tool_detection_stats.json + sidecars              (re-run promote_best_pass to rebuild)

Backups (with .bak_trim suffix, once-only):
  poses.txt, snippet_annotations.json, annotated_masks.json,
  combined_annotations.json, snippet_NNN_results.json

Usage:
    # Head-trim: drop 8 head frames
    python scripts/trim_snippet_range.py \\
        --snip-dir 'data/.../E_3/snippet_004' \\
        --keep-start 23760 --keep-end 24119 --yes

    # Tail-trim: keep only the larger half
    python scripts/trim_snippet_range.py \\
        --snip-dir 'data/.../F_3/snippet_005' \\
        --keep-start 36500 --keep-end 36699 --yes

    # Dry-run (no --yes)
    python scripts/trim_snippet_range.py --snip-dir '...' --keep-start X --keep-end Y
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


FRAME_DIRS = ["frames_left", "frames_right", "overlays", "overlays_tool",
              "overlays.bak_pre_render", "combined_masks"]
STALE_DELETE = ["session_autosave.json", "video_left.mp4", "video_stereo.mp4",
                "velocity.png", "visualization.html", "scene_motion.json"]
STATS_PATTERNS = ["tool_detection_stats.json", "tool_detection_stats.lowthresh.json",
                  "tool_detection_stats.lowest.json", "tool_detection_stats.multiprompt.json",
                  "tool_detection_stats.multiprompt_default.json",
                  "tool_detection_stats.lowest_multi.json"]


def parse_frame_id(p: Path) -> int | None:
    try:
        return int(p.stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def filter_coco(coco: dict, keep_set: set[int]) -> tuple[dict, int, int]:
    """Filter images and annotations to keep_set image_ids. Returns (new_coco, n_imgs_dropped, n_anns_dropped)."""
    imgs = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    new_imgs = [img for img in imgs if img["id"] in keep_set]
    new_anns = [a for a in anns if a["image_id"] in keep_set]
    out = dict(coco)
    out["images"] = new_imgs
    out["annotations"] = new_anns
    return out, len(imgs) - len(new_imgs), len(anns) - len(new_anns)


def write_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


def backup_once(src: Path):
    bak = src.with_suffix(src.suffix + ".bak_trim")
    if not bak.exists() and src.exists():
        shutil.copy2(src, bak)


def trim(snip_dir: Path, keep_start: int, keep_end: int, dry_run: bool) -> dict:
    fl_dir = snip_dir / "frames_left"
    if not fl_dir.is_dir():
        return {"error": f"no frames_left at {fl_dir}"}

    # Snapshot ORIGINAL frame list (before any deletions)
    fl_files = sorted(fl_dir.glob("frame_*.webp"))
    orig_ids = [int(p.stem.split("_")[1]) for p in fl_files]
    if not orig_ids:
        return {"error": "frames_left is empty"}
    keep_set = {fid for fid in orig_ids if keep_start <= fid <= keep_end}
    drop_count = len(orig_ids) - len(keep_set)

    summary = {
        "orig_frames": len(orig_ids),
        "orig_range": (orig_ids[0], orig_ids[-1]),
        "keep_range": (keep_start, keep_end),
        "kept": len(keep_set),
        "dropped": drop_count,
        "files_deleted": 0,
        "json_filtered": [],
        "stats_deleted": [],
        "stale_deleted": [],
    }

    if drop_count == 0:
        summary["status"] = "noop"
        return summary

    # ===== 1. Frame-keyed files: delete those outside keep range =====
    for sub in FRAME_DIRS:
        d = snip_dir / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            fid = parse_frame_id(f)
            if fid is None:
                continue
            if fid not in keep_set:
                summary["files_deleted"] += 1
                if not dry_run:
                    f.unlink()

    # ===== 2. poses.txt by line index =====
    poses_path = snip_dir / "poses.txt"
    if poses_path.exists():
        text_lines = poses_path.read_text().splitlines()
        data_lines = [l for l in text_lines if l.strip() and not l.startswith("#")]
        if len(data_lines) != len(orig_ids):
            summary["poses_warning"] = (f"data lines={len(data_lines)} vs frames={len(orig_ids)} "
                                        f"— line-to-frame mapping ambiguous; skipping")
        else:
            keep_idx = [i for i, fid in enumerate(orig_ids) if fid in keep_set]
            new_first = min(keep_set)
            new_last = max(keep_set)
            new_headers = ["# TUM format: timestamp tx ty tz qx qy qz qw",
                           f"# Episode snippet from frame {new_first} to {new_last} (trimmed)"]
            new_text = "\n".join(new_headers + [data_lines[i] for i in keep_idx]) + "\n"
            summary["poses"] = f"{len(data_lines)} -> {len(keep_idx)} lines"
            if not dry_run:
                backup_once(poses_path)
                poses_path.write_text(new_text)

    # ===== 3. JSON files (COCO-shaped): filter images + annotations =====
    json_targets = [
        ("snippet_annotations.json", True),
        ("annotated_masks.json", True),
        ("annotated_masks.lowthresh.json", True),
        ("annotated_masks.lowest.json", True),
        ("annotated_masks.multiprompt.json", True),
        ("annotated_masks.multiprompt_default.json", True),
        ("annotated_masks.lowest_multi.json", True),
        ("combined_annotations.json", True),
    ]
    for name, do_backup in json_targets:
        path = snip_dir / name
        if not path.exists():
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            summary[f"json_error_{name}"] = str(e)
            continue
        if not isinstance(data, dict) or "images" not in data:
            continue
        new_data, n_img_drop, n_ann_drop = filter_coco(data, keep_set)
        if n_img_drop == 0 and n_ann_drop == 0:
            continue
        summary["json_filtered"].append(f"{name} (-{n_img_drop} imgs, -{n_ann_drop} anns)")
        if not dry_run:
            if do_backup and name in ("snippet_annotations.json", "annotated_masks.json",
                                       "combined_annotations.json"):
                backup_once(path)
            write_atomic(path, new_data)

    # ===== 4. snippet_NNN_results.json: filter 'frames' list =====
    sid = snip_dir.name.replace("snippet_", "")
    res_path = snip_dir / f"snippet_{sid}_results.json"
    if res_path.exists():
        try:
            d = json.load(open(res_path))
            if isinstance(d, dict) and isinstance(d.get("frames"), list):
                orig_n = len(d["frames"])
                d["frames"] = [
                    fr for fr in d["frames"]
                    if (lambda x: x is not None and x in keep_set)(
                        _safe_int_after_underscore(fr.get("frame") or fr.get("file", ""))
                    )
                ]
                d["num_frames"] = len(d["frames"])
                summary["results"] = f"frames {orig_n} -> {len(d['frames'])}"
                if not dry_run and orig_n != len(d["frames"]):
                    backup_once(res_path)
                    write_atomic(res_path, d)
        except Exception as e:
            summary["results_error"] = str(e)

    # ===== 5. Stats files: delete (re-run promote_best_pass to rebuild canonical) =====
    for name in STATS_PATTERNS:
        p = snip_dir / name
        if p.exists():
            summary["stats_deleted"].append(name)
            if not dry_run:
                p.unlink()

    # ===== 6. Stale derivable artifacts =====
    for name in STALE_DELETE + [f"snippet_{sid}_overlay.mp4"]:
        p = snip_dir / name
        if p.exists():
            summary["stale_deleted"].append(name)
            if not dry_run:
                p.unlink()

    summary["status"] = "trimmed" if not dry_run else "would_trim"
    return summary


def _safe_int_after_underscore(s: str) -> int | None:
    if not s:
        return None
    base = s.split("/")[-1].split(".")[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snip-dir", required=True, type=Path)
    ap.add_argument("--keep-start", required=True, type=int, help="lowest image_id to keep (inclusive)")
    ap.add_argument("--keep-end", required=True, type=int, help="highest image_id to keep (inclusive)")
    ap.add_argument("--yes", action="store_true", help="apply (default = dry-run)")
    args = ap.parse_args()

    snip_dir: Path = args.snip_dir
    if not snip_dir.is_dir():
        print(f"ERROR: not a directory: {snip_dir}", file=sys.stderr)
        sys.exit(2)

    info = trim(snip_dir, args.keep_start, args.keep_end, dry_run=not args.yes)
    print(f"snip_dir: {snip_dir}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not args.yes and info.get("status") == "would_trim":
        print("\n[dry-run] pass --yes to apply.")


if __name__ == "__main__":
    main()
