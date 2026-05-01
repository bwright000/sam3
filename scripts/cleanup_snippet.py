#!/usr/bin/env python3
"""Finalize a snippet for distribution: merge Tool annotations into
snippet_annotations.json and delete intermediate scratch.

Reference structure (C_1/snippet_001):
  frames_left/  frames_right/  overlays/
  poses.txt  poses.txt.bak
  scene_motion.json
  snippet_NNN_overlay.mp4   snippet_NNN_results.json
  snippet_annotations.json
  velocity.png
  video_left.mp4  video_stereo.mp4
  visualization.html

What this script does:
  1. Reads combined_annotations.json (priority-resolved Tool category)
     and merges its Tool annotations into snippet_annotations.json
     (Tool category id = 6, continuing CRCD's 0-5 scheme).
  2. Deletes intermediates: annotated_masks*, tool_detection_stats*,
     combined_masks/, combined_annotations.json, overlays_tool/,
     overlays.bak_pre_render/, session_autosave.json, and any *.bak_*
     files except the canonical poses.txt.bak.
  3. Audits the snippet against the reference set; reports anything missing.

Backs up snippet_annotations.json once to .json.bak_pre_merge before merging.

Usage:
    python scripts/cleanup_snippet.py --snip-dir 'data/.../F_3/snippet_001' --yes
    python scripts/cleanup_snippet.py --snip-dir '...' --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


TOOL_CAT_ID = 6
TOOL_CATEGORY = {"id": TOOL_CAT_ID, "name": "Tool", "supercategory": "Tool"}


# Files / dirs to delete (relative to snip_dir). Globs allowed.
DELETE_PATTERNS = [
    "annotated_masks*.json",
    "tool_detection_stats*.json",
    "combined_annotations.json",
    "session_autosave.json",
    "snippet_annotations.json.bak_pre_regen",
    "snippet_annotations.json.bak_pre_merge",  # only if older run
    "*.bak_pre_tool",
    "*.bak_pre_render",
    "*.bak_pre_regen",
    "*.bak_trim",
    "*.bak_meta_update",
    "*.json.tmp",
    "*.txt.tmp",
]
DELETE_DIRS = [
    "combined_masks",
    "overlays_tool",
    "overlays.bak_pre_render",
]

# Reference structure expected after cleanup
REF_DIRS = ["frames_left", "frames_right", "overlays"]
REF_FILES_FIXED = [
    "poses.txt",
    "poses.txt.bak",
    "scene_motion.json",
    "snippet_annotations.json",
    "velocity.png",
    "video_left.mp4",
    "video_stereo.mp4",
    "visualization.html",
]
# These contain the snippet id: snippet_NNN_overlay.mp4, snippet_NNN_results.json
REF_FILES_TEMPLATE = ["snippet_{sid}_overlay.mp4", "snippet_{sid}_results.json"]


def merge_tool(snip_dir: Path, dry_run: bool) -> dict:
    ann_path = snip_dir / "snippet_annotations.json"
    comb_path = snip_dir / "combined_annotations.json"

    if not ann_path.exists():
        return {"merge": "skip", "reason": "no snippet_annotations.json"}
    if not comb_path.exists():
        return {"merge": "skip", "reason": "no combined_annotations.json — nothing to merge"}

    ann = json.load(open(ann_path))
    comb = json.load(open(comb_path))

    # Find Tool category id in combined (we wrote it as id=1 there; recover by name)
    comb_tool_catid = None
    for c in comb.get("categories", []):
        if c.get("name") == "Tool":
            comb_tool_catid = c["id"]
            break
    if comb_tool_catid is None:
        return {"merge": "skip", "reason": "no Tool category in combined_annotations.json"}

    # Ensure Tool category present in snippet_annotations.json with TOOL_CAT_ID
    cats = ann.get("categories", [])
    has_tool = any(c.get("name") == "Tool" for c in cats)
    if not has_tool:
        cats.append(TOOL_CATEGORY)
        ann["categories"] = cats
    else:
        # If already present but with different id, normalize to TOOL_CAT_ID
        for c in cats:
            if c.get("name") == "Tool":
                if c.get("id") != TOOL_CAT_ID:
                    # remap any existing tool annotations
                    old_id = c["id"]
                    for a in ann.get("annotations", []):
                        if a["category_id"] == old_id:
                            a["category_id"] = TOOL_CAT_ID
                    c["id"] = TOOL_CAT_ID

    # Drop any pre-existing Tool annotations (avoid double-merge)
    pre_n = len(ann.get("annotations", []))
    ann["annotations"] = [a for a in ann.get("annotations", []) if a["category_id"] != TOOL_CAT_ID]
    dropped_pre = pre_n - len(ann["annotations"])

    # Append Tool annotations from combined, with fresh IDs
    existing_ann_ids = {a["id"] for a in ann["annotations"]}
    next_id = (max(existing_ann_ids) + 1) if existing_ann_ids else 1
    added = 0
    for a in comb.get("annotations", []):
        if a.get("category_id") != comb_tool_catid:
            continue
        while next_id in existing_ann_ids:
            next_id += 1
        new_a = dict(a)
        new_a["id"] = next_id
        new_a["category_id"] = TOOL_CAT_ID
        existing_ann_ids.add(next_id)
        ann["annotations"].append(new_a)
        next_id += 1
        added += 1

    # Union image entries by id (combined may have images snippet_annotations doesn't,
    # though for our pipeline the image_id space is the same)
    existing_img_ids = {img["id"] for img in ann.get("images", [])}
    new_imgs = [img for img in comb.get("images", []) if img["id"] not in existing_img_ids]
    if new_imgs:
        # only add the ones whose id is referenced by tool anns
        ref_ids = {a["image_id"] for a in ann["annotations"] if a["category_id"] == TOOL_CAT_ID}
        new_imgs = [img for img in new_imgs if img["id"] in ref_ids]
        if new_imgs:
            ann.setdefault("images", []).extend(new_imgs)

    if dry_run:
        return {"merge": "would_merge", "tool_anns_added": added,
                "pre_existing_tool_dropped": dropped_pre,
                "imgs_added": len(new_imgs)}

    bak = ann_path.with_suffix(".json.bak_pre_merge")
    if not bak.exists():
        shutil.copy2(ann_path, bak)
    tmp = ann_path.with_suffix(".json.tmp")
    json.dump(ann, open(tmp, "w"))
    tmp.replace(ann_path)

    return {"merge": "merged", "tool_anns_added": added,
            "pre_existing_tool_dropped": dropped_pre,
            "imgs_added": len(new_imgs)}


def cleanup_intermediates(snip_dir: Path, dry_run: bool) -> dict:
    deleted_files = []
    deleted_dirs = []

    # Remove any pre-merge backup we just made if dry-running? No, keep it; user will see it next.
    # Skip the backup we ourselves create in merge_tool().
    skip_names = {"snippet_annotations.json.bak_pre_merge"}

    for pat in DELETE_PATTERNS:
        for p in snip_dir.glob(pat):
            if not p.is_file():
                continue
            if p.name in skip_names:
                continue
            deleted_files.append(p.name)
            if not dry_run:
                p.unlink()

    for sub in DELETE_DIRS:
        d = snip_dir / sub
        if d.is_dir():
            deleted_dirs.append(sub)
            if not dry_run:
                shutil.rmtree(d)

    return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs}


def audit_against_reference(snip_dir: Path, sid: str) -> dict:
    present = set(p.name for p in snip_dir.iterdir())
    expected = set(REF_DIRS) | set(REF_FILES_FIXED) | {f.format(sid=sid) for f in REF_FILES_TEMPLATE}
    missing = sorted(expected - present)
    extra = sorted(p for p in (present - expected)
                   if not (p.startswith(".") or p.endswith(".tmp")))
    return {"missing": missing, "extra": extra}


def cleanup_snippet(snip_dir: Path, dry_run: bool) -> dict:
    if not snip_dir.is_dir():
        return {"error": f"not a dir: {snip_dir}"}
    sid = snip_dir.name.replace("snippet_", "")

    out = {}
    out.update(merge_tool(snip_dir, dry_run))
    # Re-cleanup the .bak_pre_merge we just created if running with --remove-merge-bak
    out.update(cleanup_intermediates(snip_dir, dry_run))
    out.update(audit_against_reference(snip_dir, sid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snip-dir", required=True, type=Path)
    ap.add_argument("--yes", action="store_true",
                    help="apply (default = dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dry = args.dry_run or not args.yes
    info = cleanup_snippet(args.snip_dir, dry)

    print(f"snip_dir: {args.snip_dir}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if dry:
        print("\n[dry-run] pass --yes to apply.")


if __name__ == "__main__":
    main()
