"""Disk I/O: snippet loading, session autosave, COCO export."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .rle import mask_to_rle, rle_to_mask, mask_to_polygons


@dataclass
class SnippetInfo:
    episode: str
    snippet_id: str
    root: Path              # snippet dir
    frames_dir: Path        # frames_left/
    frame_files: list[Path]
    width: int
    height: int
    split_size: int
    start_frame: int
    end_frame: int


def data_dir() -> Path:
    p = os.environ.get("SAM3_ANNOT_DATA_DIR")
    if not p:
        raise RuntimeError("SAM3_ANNOT_DATA_DIR is not set")
    return Path(p)


def list_episodes() -> list[str]:
    root = data_dir()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / f"{d.name}_snippets.json").exists())


def list_snippets(episode: str) -> list[dict]:
    root = data_dir() / episode
    meta = root / f"{episode}_snippets.json"
    if not meta.exists():
        return []
    with open(meta) as f:
        return json.load(f)


def load_snippet(episode: str, snippet_id: str) -> SnippetInfo:
    """Load a snippet's frame metadata. Does NOT load frame pixels."""
    root = data_dir() / episode / f"snippet_{snippet_id}"
    if not root.is_dir():
        raise FileNotFoundError(f"snippet dir not found: {root}")

    frames_dir = root / "frames_left"
    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"no frame files in {frames_dir}")

    # Peek at first frame to get H/W
    with Image.open(frame_files[0]) as im:
        w, h = im.size

    # Snippet meta from episode snippets.json
    all_meta = list_snippets(episode)
    s = next((m for m in all_meta if m["snippet_id"] == snippet_id), None)
    if s is None:
        raise ValueError(f"snippet {snippet_id} not in {episode}_snippets.json")
    split_size = int(s.get("split_size", 120))

    return SnippetInfo(
        episode=episode,
        snippet_id=snippet_id,
        root=root,
        frames_dir=frames_dir,
        frame_files=frame_files,
        width=w,
        height=h,
        split_size=split_size,
        start_frame=int(frame_files[0].stem.split("_")[1]),
        end_frame=int(frame_files[-1].stem.split("_")[1]),
    )


def read_frame_bytes(snippet: SnippetInfo, frame_idx: int) -> tuple[bytes, str]:
    """Return (webp-encoded bytes, content_type) for a frame index.

    If the native file is .webp, return it directly. Otherwise re-encode.
    """
    if frame_idx < 0 or frame_idx >= len(snippet.frame_files):
        raise IndexError(f"frame_idx {frame_idx} out of range")
    fpath = snippet.frame_files[frame_idx]
    if fpath.suffix.lower() == ".webp":
        return fpath.read_bytes(), "image/webp"
    img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    if not ok:
        raise RuntimeError("WEBP encode failed")
    return buf.tobytes(), "image/webp"


def read_frame_array(snippet: SnippetInfo, frame_idx: int) -> np.ndarray:
    """Read a frame as RGB uint8 numpy array."""
    fpath = snippet.frame_files[frame_idx]
    with Image.open(fpath) as im:
        return np.asarray(im.convert("RGB"))


# ---------- GT (snippet_annotations.json) ----------

def load_gt(snippet: SnippetInfo) -> dict | None:
    """Load snippet_annotations.json if present."""
    p = snippet.root / "snippet_annotations.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def gt_polys_for_frame(gt: dict, frame_num: int) -> dict[str, list[list[float]]]:
    """Return {category_name: [polygon_flat, ...]} for GT at a given absolute frame_num.

    Matches on image_id == frame_num (CRCD convention).
    """
    if gt is None:
        return {}
    cat_by_id = {c["id"]: c["name"] for c in gt.get("categories", [])}
    result: dict[str, list[list[float]]] = {}
    for ann in gt.get("annotations", []):
        if ann.get("image_id") != frame_num:
            continue
        cat = cat_by_id.get(ann["category_id"], f"cat_{ann['category_id']}")
        seg = ann.get("segmentation") or []
        if isinstance(seg, list):
            result.setdefault(cat, []).extend(seg)
    return result


# ---------- Session autosave ----------

def autosave_path(snippet: SnippetInfo) -> Path:
    return snippet.root / "session_autosave.json"


def save_session(snippet: SnippetInfo, state: dict) -> None:
    """Atomic write of session JSON."""
    p = autosave_path(snippet)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, p)


def load_session(snippet: SnippetInfo) -> dict | None:
    p = autosave_path(snippet)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# ---------- Export ----------

def export_coco(snippet: SnippetInfo,
                approved_masks: dict[int, dict[str, np.ndarray]],
                propagated_masks: dict[int, dict[str, np.ndarray]],
                categories: list[str],
                cat_to_objid: dict[str, int],
                concat_tool_instances: bool = False,
                min_area: int = 50) -> Path:
    """Write annotated_masks.json in COCO format.

    approved_masks / propagated_masks: {frame_idx: {category: binary mask (H,W) uint8}}
    concat_tool_instances: if True, merge Tool_1/Tool_2/... into a single 'Tool' class.
    """
    merged: dict[int, dict[str, np.ndarray]] = {}
    for fidx, cats in propagated_masks.items():
        merged[fidx] = dict(cats)
    for fidx, cats in approved_masks.items():
        merged.setdefault(fidx, {}).update(cats)

    cats_out = list(categories)
    cat_to_objid = dict(cat_to_objid)

    if concat_tool_instances:
        # Relabel Tool_* as Tool
        if "Tool" not in cats_out:
            tool_id = max(cat_to_objid.values(), default=0) + 1
            cats_out.append("Tool")
            cat_to_objid["Tool"] = tool_id
        new_merged: dict[int, dict[str, np.ndarray]] = {}
        for fidx, d in merged.items():
            for cat, m in d.items():
                target = "Tool" if cat.startswith("Tool_") or cat == "Tool" else cat
                if fidx not in new_merged:
                    new_merged[fidx] = {}
                if target in new_merged[fidx]:
                    new_merged[fidx][target] = np.maximum(new_merged[fidx][target], m)
                else:
                    new_merged[fidx][target] = m
        merged = new_merged
        cats_out = [c for c in cats_out if not c.startswith("Tool_")]

    images_list: list[dict] = []
    ann_list: list[dict] = []
    ann_id = 1
    for fidx in sorted(merged.keys()):
        if fidx >= len(snippet.frame_files):
            continue
        fpath = snippet.frame_files[fidx]
        frame_num = int(fpath.stem.split("_")[1])
        split_num = frame_num // snippet.split_size
        offset = frame_num % snippet.split_size
        images_list.append({
            "id": frame_num,
            "file_name": f"./split_imgs/split_{split_num}/{offset:05d}.jpg",
            "height": snippet.height,
            "width": snippet.width,
        })
        for cat, mask in merged[fidx].items():
            if int(mask.sum()) == 0:
                continue
            polys = mask_to_polygons(mask, min_area=min_area)
            if not polys:
                continue
            ys, xs = np.where(mask > 0)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())]
            ann_list.append({
                "id": ann_id,
                "image_id": frame_num,
                "category_id": cat_to_objid.get(cat, 0),
                "segmentation": polys,
                "bbox": bbox,
                "area": float(mask.sum()),
                "iscrowd": 0,
            })
            ann_id += 1

    categories_list = [{"id": cat_to_objid[c], "name": c}
                       for c in cats_out if c in cat_to_objid]
    out = {
        "categories": categories_list,
        "images": images_list,
        "annotations": ann_list,
    }
    out_path = snippet.root / "annotated_masks.json"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, out_path)
    return out_path
