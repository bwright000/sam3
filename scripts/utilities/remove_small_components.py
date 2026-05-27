#!/usr/bin/env python3
"""Drop small connected components from every mask in session_autosave.json.

Removes SAM3 propagation noise (stray singular / few-pixel detections)
while preserving real structures. Operates on the slim RLE schema across
all approved_masks frames.

For each (frame, cat): decode RLE -> connectedComponents -> drop any
component with area < min_area -> re-encode. A component is kept iff its
pixel area >= min_area.

Backs up to session_autosave.json.bak_pre_small_cleanup before write.

Usage:
    python scripts/utilities/remove_small_components.py \\
        --snippet-dir F:/Datasets/CRCD/E_3/snippet_002 --min-area 50
    # preview only:
    python scripts/utilities/remove_small_components.py \\
        --snippet-dir ... --min-area 50 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from sam3_annotator.server.rle import mask_to_rle, rle_to_mask


def clean(snippet_dir: Path, min_area: int, dry_run: bool = False) -> dict:
    autosave = snippet_dir / "session_autosave.json"
    if not autosave.exists():
        print(f"  [SKIP] no autosave at {snippet_dir}")
        return {}
    save = json.load(open(autosave))
    approved = save["approved_masks"]

    removed_comp = {}     # cat -> count of components dropped
    removed_px = {}       # cat -> total px dropped
    frames_touched = 0

    for fs in sorted(approved.keys(), key=int):
        cell = approved[fs]
        frame_changed = False
        for cat, payload in list(cell.items()):
            if "rle" not in payload:
                continue
            m = rle_to_mask(payload["rle"])
            nc, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if nc <= 1:
                continue
            keep = np.zeros_like(m)
            dropped = 0
            dropped_px = 0
            for i in range(1, nc):
                a = int(stats[i, cv2.CC_STAT_AREA])
                if a >= min_area:
                    keep[lab == i] = 1
                else:
                    dropped += 1
                    dropped_px += a
            if dropped == 0:
                continue
            removed_comp[cat] = removed_comp.get(cat, 0) + dropped
            removed_px[cat] = removed_px.get(cat, 0) + dropped_px
            frame_changed = True
            new_area = int(keep.sum())
            if not dry_run:
                if new_area == 0:
                    # entire cat was noise at this frame -> drop the cell
                    del cell[cat]
                else:
                    payload["rle"] = mask_to_rle(keep)
                    payload["area"] = new_area
        if frame_changed:
            frames_touched += 1

    total_comp = sum(removed_comp.values())
    total_px = sum(removed_px.values())
    print(f"  min_area={min_area}: removed {total_comp} components "
          f"({total_px}px) across {frames_touched} frames")
    for cat in sorted(removed_comp):
        print(f"    {cat}: -{removed_comp[cat]} comps, -{removed_px[cat]}px")

    if dry_run:
        print("  [DRY-RUN] no write")
        return {"removed_components": total_comp, "removed_px": total_px}

    bak = autosave.with_suffix(".json.bak_pre_small_cleanup")
    if not bak.exists():
        shutil.copy(autosave, bak)
        print(f"  backed up -> {bak.name}")
    save["timestamp"] = time.time()
    json.dump(save, open(autosave, "w"))
    print(f"  wrote {autosave.name}")
    return {"removed_components": total_comp, "removed_px": total_px,
            "frames_touched": frames_touched}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snippet-dir", required=True, type=Path)
    ap.add_argument("--min-area", type=int, default=50,
                    help="drop connected components smaller than this many pixels")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"[{args.snippet_dir}]")
    clean(args.snippet_dir, args.min_area, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
