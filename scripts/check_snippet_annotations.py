#!/usr/bin/env python3
"""Per-snippet tissue annotation audit.

For each snippet in C_1, E_3, F_3:
- Does snippet_annotations.json exist?
- Does it contain tissue categories (Liver, Gallbladder)?
- How many annotated images and per-category mask counts?
- Fraction of snippet's frame range with GT coverage.
"""

import json
from pathlib import Path
from collections import Counter

SEGMENTS = Path("data/Segments")
TISSUE = {"Liver", "Gallbladder", "Meat", "Skin"}
EPISODES = ["C_1", "E_3", "F_3"]


def audit_snippet(snippet_dir: Path, meta: dict) -> dict:
    ann = snippet_dir / "snippet_annotations.json"
    result = {
        "snippet": snippet_dir.name,
        "has_file": ann.exists(),
        "start": meta.get("start_frame"),
        "end": meta.get("end_frame"),
        "num_frames": meta.get("num_frames"),
    }
    if not ann.exists():
        return result
    with open(ann) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    anns = data.get("annotations", [])
    imgs = data.get("images", [])

    per_cat = Counter()
    imgs_with_tissue = set()
    for a in anns:
        name = cats.get(a["category_id"], f"id{a['category_id']}")
        per_cat[name] += 1
        if name in TISSUE:
            imgs_with_tissue.add(a["image_id"])

    tissue_cats = {n: per_cat[n] for n in per_cat if n in TISSUE}
    tool_cats = {n: per_cat[n] for n in per_cat if n not in TISSUE}

    result.update({
        "images_in_ann": len(imgs),
        "total_annotations": len(anns),
        "tissue_annotations": tissue_cats,
        "tool_annotations": tool_cats,
        "unique_images_with_tissue": len(imgs_with_tissue),
        "tissue_coverage_pct": round(100 * len(imgs_with_tissue) / max(1, len(imgs)), 1),
    })
    return result


def main():
    for ep in EPISODES:
        ep_dir = SEGMENTS / ep
        meta_path = ep_dir / f"{ep}_snippets.json"
        if not meta_path.exists():
            print(f"\n=== {ep}: NO METADATA ({meta_path}) ===")
            continue
        with open(meta_path) as f:
            snippets_meta = json.load(f)
        meta_by_id = {s["snippet_id"]: s for s in snippets_meta}

        print(f"\n=== {ep} ===")
        for snip_dir in sorted(ep_dir.glob("snippet_*")):
            snip_id = snip_dir.name.split("_")[-1]
            meta = meta_by_id.get(snip_id, {})
            r = audit_snippet(snip_dir, meta)
            has = "YES" if r["has_file"] else "NO "
            if not r["has_file"]:
                print(f"  {r['snippet']}: ann_file={has}  frames={r['start']}-{r['end']}")
                continue
            t = r["tissue_annotations"]
            tool = r["tool_annotations"]
            tissue_str = ", ".join(f"{k}={v}" for k, v in sorted(t.items())) or "NONE"
            tool_str = ", ".join(f"{k}={v}" for k, v in sorted(tool.items())) or "none"
            print(
                f"  {r['snippet']}: frames={r['start']}-{r['end']} ({r['num_frames']}f) | "
                f"ann_imgs={r['images_in_ann']} ({r['tissue_coverage_pct']}% have tissue) | "
                f"tissue: {tissue_str} | tool: {tool_str}"
            )


if __name__ == "__main__":
    main()
