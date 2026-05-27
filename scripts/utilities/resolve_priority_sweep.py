#!/usr/bin/env python3
"""General per-pixel category-priority sweep over session_autosave.json.

Given a priority order (highest first), make every category's mask
pixel-disjoint: higher-priority cats keep their pixels, lower-priority
cats lose any pixel claimed by a higher one. Applied to EVERY frame in
approved_masks (not just anchors).

Algorithm per frame:
    claimed = empty
    for cat in priority_order (high -> low):
        cat_mask &= ~claimed       # drop pixels already claimed
        claimed  |= cat_mask        # this cat now owns its remaining pixels

Example - Gallbladder > Tool > Liver:
    Gallbladder kept as-is
    Tool  := Tool  - Gallbladder
    Liver := Liver - Gallbladder - Tool

Operates only on the `rle` field (the slim RLE-only autosave schema).
Backs up to session_autosave.json.bak_pre_priority_sweep before write.

Usage:
    python scripts/utilities/resolve_priority_sweep.py \\
        --snippet-dir F:/Datasets/CRCD/C_1/snippet_001 \\
        --priority Gallbladder Tool Liver
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from sam3_annotator.server.rle import mask_to_rle, rle_to_mask


def resolve(snippet_dir: Path, priority: list[str], dry_run: bool = False) -> dict:
    autosave_path = snippet_dir / "session_autosave.json"
    if not autosave_path.exists():
        print(f"  [SKIP] no autosave at {snippet_dir}")
        return {}

    with open(autosave_path) as f:
        save = json.load(f)
    approved = save["approved_masks"]

    # Tally subtractions per cat
    subtracted = {c: 0 for c in priority}
    frames_touched = 0

    for fidx_s in sorted(approved.keys(), key=int):
        cell = approved[fidx_s]
        # Decode the cats present at this frame that are in the priority list
        masks: dict[str, np.ndarray] = {}
        size = None
        for cat in priority:
            if cat in cell and "rle" in cell[cat]:
                m = rle_to_mask(cell[cat]["rle"])
                masks[cat] = m
                size = m.shape
        if not masks or size is None:
            continue

        claimed = np.zeros(size, dtype=np.uint8)
        frame_changed = False
        for cat in priority:           # high -> low
            if cat not in masks:
                continue
            m = masks[cat]
            before = int(m.sum())
            # Drop pixels already claimed by a higher-priority cat
            cleaned = (m & (1 - claimed)).astype(np.uint8)
            removed = before - int(cleaned.sum())
            if removed > 0:
                subtracted[cat] += removed
                frame_changed = True
                if not dry_run:
                    cell[cat]["rle"] = mask_to_rle(cleaned)
                    cell[cat]["area"] = int(cleaned.sum())
                    cell[cat]["priority_swept"] = True
            masks[cat] = cleaned
            claimed |= cleaned          # this cat owns its kept pixels
        if frame_changed:
            frames_touched += 1

    total = sum(subtracted.values())
    if total == 0:
        print(f"  No overlaps to resolve in {snippet_dir.name} "
              f"(priority {' > '.join(priority)}).")
        return {"frames_touched": 0, "subtracted": subtracted}

    print(f"  priority {' > '.join(priority)}: removed " +
          ", ".join(f"{c}=-{n}px" for c, n in subtracted.items() if n) +
          f"  across {frames_touched} frames")

    if dry_run:
        print(f"  [DRY-RUN] no write")
        return {"frames_touched": frames_touched, "subtracted": subtracted}

    bak = autosave_path.with_suffix(".json.bak_pre_priority_sweep")
    if not bak.exists():
        shutil.copy(autosave_path, bak)
        print(f"  backed up -> {bak.name}")
    save["timestamp"] = time.time()
    with open(autosave_path, "w") as f:
        json.dump(save, f)
    print(f"  wrote {autosave_path.name}")
    return {"frames_touched": frames_touched, "subtracted": subtracted}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snippet-dir", required=True, type=Path)
    ap.add_argument("--priority", nargs="+", required=True,
                    help="Category names, HIGHEST priority first "
                         "(e.g. --priority Gallbladder Tool Liver)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"[{args.snippet_dir}]")
    resolve(args.snippet_dir, args.priority, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
