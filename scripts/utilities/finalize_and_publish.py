#!/usr/bin/env python3
"""End-to-end finalize + publish for one snippet.

Orchestrates the publication flow we settled on this session:

  1. Load the propagated replay (from the Colab drop outputs) into the
     working autosave's approved_masks (all frames). Backs up the prior
     autosave once.
  2. (optional) Remove small connected components (SAM3 noise) below
     --min-area.
  3. Hole-aware priority sweep in --priority order (highest first).
  4. Build a clean workspace (rgb/rgbright/depth + TUM txts + intrinsics +
     scene_motion + video; NO snippet_annotations.json so promote runs
     fresh) with the swept masks as annotated_masks.replay.json.
  5. promote_replay (fresh) -> snippet_annotations.json + semantic_instance.
  6. finalize_snippet --priority <...> (hole-aware).
  7. generate_manifest.
  8. publication_gate -> must PASS or we abort before publishing.
  9. (unless --no-publish) rename the existing CRCD-Published copy aside
     and copy the validated workspace in; re-gate the published copy.
 10. Clean up workspace + scratch COCO.

Usage:
    python scripts/utilities/finalize_and_publish.py \\
        --episode E_3 --snippet 005 \\
        --replay-root "F:/Datasets/CRCD/run_2026_05_27_10ready-20260527T113710Z-3-001/run_2026_05_27_10ready/outputs" \\
        --priority Gallbladder Liver Tool [--min-area 50] [--no-publish]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CRCD = Path("F:/Datasets/CRCD")
PUBLISHED = Path("F:/Datasets/CRCD-Published")
PY = sys.executable

STATIC_FILES = ["associations.txt", "depth.txt", "groundtruth.txt", "rgb.txt",
                "rgbright.txt", "intrinsics.yaml", "scene_motion.json", "video_left.mp4"]
STATIC_DIRS = ["rgb", "rgbright", "depth"]


def _run(mod_args: list[str]) -> None:
    """Run a python -m module, streaming output, raise on failure."""
    cmd = [PY, "-m"] + mod_args
    print(f"  $ {' '.join(mod_args)}")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise RuntimeError(f"step failed (exit {r.returncode}): {' '.join(mod_args)}")


def load_replay_into_autosave(ep: str, sn: str, replay_path: Path) -> int:
    os.environ["SAM3_ANNOT_DATA_DIR"] = str(CRCD)
    from sam3_annotator.server import storage
    snip = storage.load_snippet(ep, sn)
    # frame_num -> frame_idx (replay image_id is a frame_num; autosave keys
    # are frame_idx strings)
    n2idx = {int(p.stem.split("_")[1]): i for i, p in enumerate(snip.frame_files)}
    autosave = snip.root / "session_autosave.json"
    bak = autosave.with_suffix(".json.bak_pre_finalization_load")
    if autosave.exists() and not bak.exists():
        shutil.copy(autosave, bak)
        print(f"  backed up anchor autosave -> {bak.name}")
    cur = json.loads(autosave.read_text()) if autosave.exists() else {
        "episode": ep, "snippet_id": sn,
        "categories": ["Tool", "Liver", "Gallbladder"],
        "cat_to_objid": {"Tool": 1, "Liver": 2, "Gallbladder": 3},
    }
    rep = json.loads(replay_path.read_text())
    cat_by_id = {c["id"]: c["name"] for c in rep["categories"]}
    new_ap: dict = {}
    for ann in rep["annotations"]:
        fn = ann["image_id"]
        if fn not in n2idx:
            continue
        seg = ann.get("segmentation")
        if not isinstance(seg, dict):
            continue
        new_ap.setdefault(str(n2idx[fn]), {})[cat_by_id[ann["category_id"]]] = {
            "rle": seg, "area": int(ann.get("area", 0))}
    cur["approved_masks"] = new_ap
    cur["propagated_masks"] = {}
    cur["timestamp"] = time.time()
    cur["_finalization_load"] = f"{ep}/{sn} replay loaded {time.strftime('%Y-%m-%d')}"
    autosave.write_text(json.dumps(cur))
    return len(new_ap)


def build_coco_from_autosave(ep: str, sn: str, out_path: Path) -> int:
    os.environ["SAM3_ANNOT_DATA_DIR"] = str(CRCD)
    from sam3_annotator.server import storage
    snip = storage.load_snippet(ep, sn)
    f2n = {i: int(p.stem.split("_")[1]) for i, p in enumerate(snip.frame_files)}
    d = json.loads((snip.root / "session_autosave.json").read_text())
    cti = d.get("cat_to_objid") or {"Tool": 1, "Liver": 2, "Gallbladder": 3}
    cats = [{"id": o, "name": c} for c, o in cti.items()]
    imgs, anns, aid, seen = [], [], 1, set()
    for fs, cell in d["approved_masks"].items():
        fn = f2n.get(int(fs))
        if fn is None:
            continue
        if fn not in seen:
            imgs.append({"id": fn, "file_name": f"./rgb/frame_{fn:06d}.png",
                         "height": snip.height, "width": snip.width})
            seen.add(fn)
        for cat, p in cell.items():
            if "rle" in p:
                anns.append({"id": aid, "image_id": fn, "category_id": cti.get(cat, 0),
                             "segmentation": p["rle"], "area": float(p.get("area", 0)),
                             "iscrowd": 0})
                aid += 1
    out_path.write_text(json.dumps({"categories": cats, "images": imgs, "annotations": anns}))
    return len(anns)


def build_workspace(src: Path, ws: Path, coco: Path) -> None:
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for d in STATIC_DIRS:
        if (src / d).is_dir():
            shutil.copytree(src / d, ws / d)
    for f in STATIC_FILES:
        if (src / f).is_file():
            shutil.copy(src / f, ws / f)
    shutil.copy(coco, ws / "annotated_masks.replay.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--snippet", required=True)
    ap.add_argument("--replay-root", required=True, type=Path,
                    help="Colab drop outputs root containing {EP}/snippet_{NNN}/annotated_masks.replay.json")
    ap.add_argument("--priority", nargs="+", default=["Gallbladder", "Liver", "Tool"],
                    help="Priority order, highest first")
    ap.add_argument("--min-area", type=int, default=0,
                    help="If >0, remove connected components smaller than this (px)")
    ap.add_argument("--no-publish", action="store_true",
                    help="Build + gate only; do not replace the published copy")
    args = ap.parse_args()

    ep, sn = args.episode, args.snippet
    sn_name = f"snippet_{sn}"
    src = CRCD / ep / sn_name
    replay = args.replay_root / ep / sn_name / "annotated_masks.replay.json"
    if not replay.is_file():
        print(f"ERROR: replay not found: {replay}", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(f"ERROR: snippet dir not found: {src}", file=sys.stderr)
        return 1

    print(f"=== finalize+publish {ep}/{sn_name}  priority={' > '.join(args.priority)} ===")

    print("[1] load replay -> approved_masks")
    n = load_replay_into_autosave(ep, sn, replay)
    print(f"    {n} frames loaded")

    if args.min_area > 0:
        print(f"[2] remove small components (<{args.min_area}px)")
        _run(["scripts.utilities.remove_small_components",
              "--snippet-dir", str(src), "--min-area", str(args.min_area)])
    else:
        print("[2] skip small-component removal (--min-area 0)")

    print(f"[3] priority sweep {' > '.join(args.priority)}")
    _run(["scripts.utilities.resolve_priority_sweep",
          "--snippet-dir", str(src), "--priority"] + args.priority)

    print("[4] build COCO + workspace")
    coco = src / "_swept_review.coco.json"
    n_anns = build_coco_from_autosave(ep, sn, coco)
    ws = CRCD / "_republish_ws" / ep / sn_name
    build_workspace(src, ws, coco)
    print(f"    COCO {n_anns} anns; workspace {ws}")

    print("[5] promote_replay (fresh)")
    _run(["scripts.propagation.promote_replay", "--snippet-dir", str(ws),
          "--replay", "annotated_masks.replay.json"])

    print(f"[6] finalize_snippet --priority {' '.join(args.priority)}")
    _run(["scripts.dataloading.finalize_snippet", "--snippet-dir", str(ws),
          "--priority"] + args.priority)

    print("[7] generate_manifest")
    _run(["scripts.evaluation.generate_manifest", "--snippet-dir", str(ws)])

    print("[8] publication_gate")
    _run(["scripts.evaluation.publication_gate", "--snippet-dir", str(ws)])
    print("    GATE PASS")

    if args.no_publish:
        print(f"[9] --no-publish: workspace left at {ws} for review")
        return 0

    print("[9] publish -> CRCD-Published")
    pub = PUBLISHED / ep / sn_name
    bak = PUBLISHED / ep / f"{sn_name}.bak_pre_republish_{time.strftime('%Y_%m_%d')}"
    if pub.exists():
        if bak.exists():
            shutil.rmtree(bak)
        shutil.move(str(pub), str(bak))
        print(f"    old -> {bak.name}")
    pub.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ws, pub)
    print(f"    published -> {pub}")

    print("[10] re-gate published copy")
    _run(["scripts.evaluation.publication_gate", "--snippet-dir", str(pub)])

    print("[11] cleanup")
    shutil.rmtree(CRCD / "_republish_ws", ignore_errors=True)
    coco.unlink(missing_ok=True)
    print(f"=== DONE {ep}/{sn_name} published ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
