#!/usr/bin/env python3
"""Merge tool masks from annotate_ui.py output (annotated_masks.json) into the
existing snippet_annotations.json for a snippet.

Safe: backs up snippet_annotations.json to snippet_annotations.json.bak (once)
before modifying. Atomic write via .tmp + replace().

Usage:
    # one snippet
    python scripts/merge_tool_masks.py --episode E_3 --snippet 001

    # all snippets in an episode that have annotated_masks.json
    python scripts/merge_tool_masks.py --episode E_3

    # all episodes
    python scripts/merge_tool_masks.py

    # dry-run — show what would change
    python scripts/merge_tool_masks.py --episode E_3 --snippet 001 --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SEGMENTS = BASE / "data" / "Segments"


def merge_one(snip_dir: Path, dry_run: bool) -> tuple[bool, str]:
    ann_path = snip_dir / "snippet_annotations.json"
    tool_path = snip_dir / "annotated_masks.json"

    if not tool_path.exists():
        return False, "no annotated_masks.json"
    if not ann_path.exists():
        return False, "no snippet_annotations.json"

    with open(ann_path) as f:
        ann = json.load(f)
    with open(tool_path) as f:
        tool = json.load(f)

    # 1) Merge categories by name; keep existing IDs, allocate fresh IDs for new
    existing_cats = {c["name"]: c for c in ann.get("categories", [])}
    existing_ids = {c["id"] for c in ann.get("categories", [])}
    next_cat_id = (max(existing_ids) + 1) if existing_ids else 0
    tool_catid_to_final = {}
    added_cats = []
    for c in tool.get("categories", []):
        name = c["name"]
        if name in existing_cats:
            tool_catid_to_final[c["id"]] = existing_cats[name]["id"]
        else:
            new_id = next_cat_id
            next_cat_id += 1
            while new_id in existing_ids:
                new_id = next_cat_id
                next_cat_id += 1
            existing_ids.add(new_id)
            tool_catid_to_final[c["id"]] = new_id
            new_cat = {"id": new_id, "name": name,
                       "supercategory": c.get("supercategory", name)}
            added_cats.append(new_cat)
            existing_cats[name] = new_cat

    # 2) Union images by id (the UI uses frame_num as image_id — same scheme as GT)
    existing_img_ids = {img["id"] for img in ann.get("images", [])}
    new_images = [img for img in tool.get("images", []) if img["id"] not in existing_img_ids]

    # 3) Append annotations with fresh unique IDs
    existing_ann_ids = {a["id"] for a in ann.get("annotations", [])}
    next_ann_id = (max(existing_ann_ids) + 1) if existing_ann_ids else 1
    added_anns = []
    for a in tool.get("annotations", []):
        while next_ann_id in existing_ann_ids:
            next_ann_id += 1
        new_a = dict(a)
        new_a["id"] = next_ann_id
        new_a["category_id"] = tool_catid_to_final[a["category_id"]]
        existing_ann_ids.add(next_ann_id)
        added_anns.append(new_a)
        next_ann_id += 1

    summary = (f"cats+{len(added_cats)} imgs+{len(new_images)} "
               f"anns+{len(added_anns)} (tool file had {len(tool.get('annotations',[]))})")

    if dry_run:
        return True, f"DRY {summary}"

    # 4) Backup once
    bak = ann_path.with_suffix(".json.bak_pre_tool")
    if not bak.exists():
        shutil.copy2(ann_path, bak)

    # 5) Apply merge
    ann.setdefault("categories", []).extend(added_cats)
    ann.setdefault("images", []).extend(new_images)
    ann.setdefault("annotations", []).extend(added_anns)

    tmp = ann_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(ann, f)
    tmp.replace(ann_path)

    return True, summary


def iter_targets(episode=None, snippet=None):
    if episode:
        eps = [episode]
    else:
        eps = sorted(d.name for d in SEGMENTS.iterdir() if d.is_dir())
    for ep in eps:
        ep_dir = SEGMENTS / ep
        if not ep_dir.is_dir():
            continue
        if snippet:
            yield ep_dir / f"snippet_{snippet}"
        else:
            for s in sorted(ep_dir.glob("snippet_*")):
                yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None)
    ap.add_argument("--snippet", default=None, help="snippet id e.g. 001")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    any_action = False
    for snip_dir in iter_targets(args.episode, args.snippet):
        if not snip_dir.exists():
            print(f"  SKIP {snip_dir}: not found")
            continue
        ok, msg = merge_one(snip_dir, args.dry_run)
        marker = "MERGED" if ok else "skip  "
        rel = snip_dir.relative_to(BASE)
        print(f"  {marker} {rel}: {msg}")
        any_action = any_action or ok

    if not any_action:
        print("\nNothing to merge.")


if __name__ == "__main__":
    main()
