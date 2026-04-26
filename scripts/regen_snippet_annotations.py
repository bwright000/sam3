#!/usr/bin/env python3
"""Rebuild snippet_annotations.json so its image_ids match the actual frames_left/.

Source priority per snippet:
  1. Master train.json + test.json from data/{EP}/{EP}/  (or data/{train,test}.json for C_1)
  2. Fallback: extract GT (score=None entries) from snippet_NNN_results.json

For each (episode, snippet):
  - Determine actual frame range from frames_left/ (source of truth)
  - Build a fresh COCO subset for that range
  - Backup existing snippet_annotations.json (.bak_pre_regen)
  - Atomic write

Usage:
    python scripts/regen_snippet_annotations.py --audit
    python scripts/regen_snippet_annotations.py --episode E_3 --snippet 001 --yes
    python scripts/regen_snippet_annotations.py --yes        # apply to all
"""

import argparse
import json
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SEGMENTS = BASE / "data" / "Segments"
DATA = BASE / "data"

# Standard CRCD category IDs (from shared_config.py)
CAT_ID_BY_NAME = {"Meat": 0, "Skin": 1, "Liver": 2, "Gallbladder": 3, "FBF": 4, "PCH": 5}
CATEGORIES = [{"id": v, "name": k, "supercategory": k} for k, v in CAT_ID_BY_NAME.items()]


def load_master(ep: str) -> dict:
    """Load + merge train.json + test.json from data/{EP}/{EP}/.

    For C_1 only, falls back to the flat data/{train,test}.json layout (legacy).
    Other episodes return {} if no nested layout exists — caller falls back to
    snippet_NNN_results.json.
    """
    candidates = [(DATA / ep / ep / "train.json", DATA / ep / ep / "test.json")]
    if ep == "C_1":
        candidates.append((DATA / "train.json", DATA / "test.json"))

    for tr, te in candidates:
        if tr.exists() or te.exists():
            merged: dict = {"categories": [], "images": [], "annotations": []}
            for p in (tr, te):
                if not p.exists():
                    continue
                with open(p) as f:
                    d = json.load(f)
                merged["categories"] = d.get("categories", merged["categories"])
                merged["images"].extend(d.get("images", []))
                merged["annotations"].extend(d.get("annotations", []))
            return merged
    return {}


def get_actual_frame_range(snippet_dir: Path) -> tuple[int, int, int]:
    fl = snippet_dir / "frames_left"
    fnums = sorted(int(p.stem.split("_")[1]) for p in fl.glob("frame_*.webp"))
    if not fnums:
        raise FileNotFoundError(f"no frames in {fl}")
    return fnums[0], fnums[-1], len(fnums)


def get_split_size(ep: str, snip_id: str) -> int:
    p = SEGMENTS / ep / f"{ep}_snippets.json"
    if not p.exists():
        return 120
    with open(p) as f:
        for s in json.load(f):
            if s["snippet_id"] == snip_id:
                return int(s.get("split_size", 120))
    return 120


def _parse_true_frame_n(image_id: int, file_name: str, split_size: int) -> int | None:
    """Recover true video frame_n from a master image entry.

    file_name forms:
      - "./split_imgs/split_N/OFFSET.jpg"  → frame_n = N*split_size + OFFSET
      - "./split_N/" (C_1 train broken)    → recover OFFSET from image_id
    """
    m = re.search(r"split_(\d+)/(\d+)", file_name or "")
    if m:
        return int(m.group(1)) * split_size + int(m.group(2))
    m = re.search(r"split_(\d+)/?$", file_name or "")
    if m:
        split_n = int(m.group(1))
        # C_1 broken: offset isn't in filename, derive from image_id
        if image_id is not None:
            offset = image_id % split_size
            return split_n * split_size + offset
    return None


def from_master(merged: dict, frame_ids: set[int], split_size: int) -> dict:
    """Subset master annotations to frame_ids using FILE_NAME → true frame_n.

    Rewrites image_id to be true frame_n so consumers can match by frame number.
    """
    # Build map: original image_id → true frame_n
    id_to_fn: dict[int, int] = {}
    img_by_orig_id: dict[int, dict] = {}
    for img in merged.get("images", []):
        oid = img.get("id")
        if oid is None:
            continue
        fn = _parse_true_frame_n(oid, img.get("file_name", ""), split_size)
        if fn is None:
            continue
        id_to_fn[oid] = fn
        img_by_orig_id[oid] = img

    # Filter images by true frame_n
    out_images: list[dict] = []
    seen_fn: set[int] = set()
    for oid, fn in id_to_fn.items():
        if fn not in frame_ids or fn in seen_fn:
            continue
        new_img = dict(img_by_orig_id[oid])
        new_img["id"] = fn  # canonical: image_id == frame_n in our snippet files
        out_images.append(new_img)
        seen_fn.add(fn)

    # Filter + remap annotations
    out_anns: list[dict] = []
    next_aid = 1
    for a in merged.get("annotations", []):
        oid = a.get("image_id")
        fn = id_to_fn.get(oid)
        if fn is None or fn not in frame_ids:
            continue
        new_a = dict(a)
        new_a["image_id"] = fn
        new_a["id"] = next_aid
        next_aid += 1
        out_anns.append(new_a)

    cats = merged.get("categories", []) or CATEGORIES
    return {"categories": cats, "images": out_images, "annotations": out_anns}


def from_results(snippet_dir: Path, snip_id: str, ep: str,
                 split_size: int) -> dict:
    """Extract GT (score=None entries) from snippet_NNN_results.json."""
    rp = snippet_dir / f"snippet_{snip_id}_results.json"
    if not rp.exists():
        return {}
    with open(rp) as f:
        d = json.load(f)
    if not isinstance(d, dict) or "frames" not in d:
        return {}

    images: list[dict] = []
    anns: list[dict] = []
    ann_id = 1
    for frame in d.get("frames", []):
        frame_label = frame.get("frame", "")
        try:
            frame_num = int(frame_label.split("_")[1])
        except (IndexError, ValueError):
            continue
        h = int(frame.get("height", 720))
        w = int(frame.get("width", 1280))
        split_n = frame_num // split_size
        offset = frame_num % split_size
        images.append({
            "id": frame_num,
            "width": w,
            "height": h,
            "file_name": f"./split_imgs/split_{split_n}/{offset:05d}.jpg",
        })
        for cat_key, mask_entries in (frame.get("masks") or {}).items():
            cat_name = cat_key[:1].upper() + cat_key[1:]  # "liver" -> "Liver"
            cat_id = CAT_ID_BY_NAME.get(cat_name)
            if cat_id is None:
                continue
            for entry in (mask_entries or []):
                # GT only: score is null/None
                if entry.get("score") is not None:
                    continue
                seg = entry.get("segmentation") or []
                if not seg:
                    continue
                # Compute bbox
                xs: list[float] = []
                ys: list[float] = []
                for poly in seg:
                    if not isinstance(poly, list):
                        continue
                    for i in range(0, len(poly), 2):
                        xs.append(poly[i])
                        ys.append(poly[i + 1])
                if not xs:
                    continue
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                anns.append({
                    "id": ann_id,
                    "image_id": frame_num,
                    "category_id": cat_id,
                    "segmentation": seg,
                    "bbox": bbox,
                    "area": float(entry.get("area", 0)),
                    "iscrowd": 0,
                })
                ann_id += 1
    return {"categories": CATEGORIES, "images": images, "annotations": anns}


def regen_one(ep: str, snip_id: str, dry_run: bool = True,
              master_cache: dict | None = None) -> dict:
    snippet_dir = SEGMENTS / ep / f"snippet_{snip_id}"
    if not snippet_dir.is_dir():
        return {"status": "skip", "reason": f"missing dir"}
    try:
        sf, ef, n = get_actual_frame_range(snippet_dir)
    except FileNotFoundError as e:
        return {"status": "skip", "reason": str(e)}

    target_range = set(range(sf, ef + 1))
    ann_path = snippet_dir / "snippet_annotations.json"
    existing_ids: set[int] = set()
    existing_count = 0
    if ann_path.exists():
        with open(ann_path) as f:
            existing = json.load(f)
        existing_ids = {a["image_id"] for a in existing.get("annotations", [])}
        existing_count = len(existing.get("annotations", []))

    overlap = len(existing_ids & target_range)
    fully_aligned = (existing_count > 0 and existing_ids.issubset(target_range)
                     and len(existing_ids) >= 1)

    info = {
        "frames": n,
        "frame_range": (sf, ef),
        "existing_anns": existing_count,
        "existing_overlap": overlap,
        "source": None,
    }

    # Build new
    split_size = get_split_size(ep, snip_id)
    new = {}
    if master_cache and master_cache.get(ep):
        new = from_master(master_cache[ep], target_range, split_size)
        if new.get("annotations"):
            info["source"] = "master"
    if not new.get("annotations"):
        new = from_results(snippet_dir, snip_id, ep, split_size)
        if new.get("annotations"):
            info["source"] = "results.json"
    if not new.get("annotations") and not new.get("images"):
        info["status"] = "skip"
        info["reason"] = "no master + no GT in results.json"
        return info

    info["new_images"] = len(new.get("images", []))
    info["new_anns"] = len(new.get("annotations", []))

    # Decide whether to write: only if changed
    changed = (overlap == 0 and existing_count > 0) or (existing_count == 0) or \
              (info["new_anns"] != existing_count)

    if not changed:
        info["status"] = "ok"
        return info

    if dry_run:
        info["status"] = "would_regen"
        return info

    # Backup + atomic write
    if ann_path.exists():
        bak = ann_path.with_suffix(".json.bak_pre_regen")
        if not bak.exists():
            shutil.copy2(ann_path, bak)
    tmp = ann_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(new, f)
    tmp.replace(ann_path)
    info["status"] = "regenerated"
    return info


def iter_snippets(ep: str | None = None, snip: str | None = None):
    eps = [ep] if ep else sorted(d.name for d in SEGMENTS.iterdir() if d.is_dir())
    for e in eps:
        ep_dir = SEGMENTS / e
        if not ep_dir.is_dir():
            continue
        if snip:
            yield e, snip
        else:
            for s in sorted(ep_dir.glob("snippet_*")):
                yield e, s.name.split("_")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None)
    ap.add_argument("--snippet", default=None, help="e.g. 001")
    ap.add_argument("--yes", action="store_true", help="apply changes (default = dry-run)")
    ap.add_argument("--audit", action="store_true",
                    help="report mismatches across all snippets, don't change anything")
    args = ap.parse_args()

    dry = (not args.yes) or args.audit

    # Pre-load master annotations per episode
    master_cache: dict[str, dict] = {}
    eps_to_check = ([args.episode] if args.episode
                    else sorted(d.name for d in SEGMENTS.iterdir() if d.is_dir()))
    for e in eps_to_check:
        m = load_master(e)
        if m.get("annotations"):
            master_cache[e] = m
            print(f"  master {e}: {len(m['images'])} imgs, {len(m['annotations'])} anns")
        else:
            print(f"  master {e}: not found, will use results.json fallback")

    rows = []
    for ep, sid in iter_snippets(args.episode, args.snippet):
        info = regen_one(ep, sid, dry_run=dry, master_cache=master_cache)
        rows.append((f"{ep}/snippet_{sid}", info))

    print(f"\n{'snippet':<22} {'status':<14} {'frames':<7} {'range':<14} "
          f"{'old':<5} {'overlap':<8} {'new':<5} {'source':<11}")
    print("-" * 100)
    changed = 0
    for name, info in rows:
        st = info.get("status", "?")
        n = info.get("frames", "")
        rng = f"{info['frame_range'][0]}-{info['frame_range'][1]}" if "frame_range" in info else ""
        old = info.get("existing_anns", "")
        overlap = info.get("existing_overlap", "")
        new = info.get("new_anns", "")
        src = info.get("source") or "—"
        if st in ("would_regen", "regenerated"):
            changed += 1
        print(f"{name:<22} {st:<14} {str(n):<7} {rng:<14} {str(old):<5} "
              f"{str(overlap):<8} {str(new):<5} {src:<11}")
    print(f"\n{changed}/{len(rows)} snippets {'would change' if dry else 'rebuilt'}")
    if dry and changed:
        print("Re-run with --yes to apply.")


if __name__ == "__main__":
    main()
