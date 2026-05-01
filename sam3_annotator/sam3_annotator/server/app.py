"""FastAPI app: session + frames + masks + propagation + export."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_VERSION = "0.1.0"


logger = logging.getLogger("sam3_annotator")

# Ensure parent repo's scripts are importable (reuse _load_frames_for_tracker etc.)
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]  # sam3facebook/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from . import storage
from .rle import mask_to_rle, rle_to_mask
from .sam3_service import SAM3Service


NO_MODEL = os.environ.get("SAM3_ANNOT_NO_MODEL", "0") == "1"
NO_TEXT = os.environ.get("SAM3_ANNOT_NO_TEXT", "0") == "1"
LIGHTWEIGHT = os.environ.get("SAM3_ANNOT_LIGHTWEIGHT", "0") == "1"


# ---------- Global app state ----------

class AppState:
    def __init__(self):
        self.sam3 = SAM3Service(no_model=NO_MODEL, enable_text=not NO_TEXT,
                                 lightweight=LIGHTWEIGHT)
        self.snippet: Optional[storage.SnippetInfo] = None
        # approved_masks: {frame_idx: {category: uint8 mask}}
        self.approved: dict[int, dict[str, np.ndarray]] = {}
        # propagated_masks: {frame_idx: {category: uint8 mask}}
        self.propagated: dict[int, dict[str, np.ndarray]] = {}
        self.categories: list[str] = []
        # cat_to_objid mirrors sam3_service
        self.cat_to_objid: dict[str, int] = {}
        # Undo/redo: each entry = {op, description, timestamp, restore_items}
        # restore_items is list of {frame_idx, category, kind, rle (None=absent)}
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    def reset_snippet_state(self):
        self.approved = {}
        self.propagated = {}
        self.cat_to_objid = {}
        self.undo_stack = []
        self.redo_stack = []
        # keep categories list (user-configurable)


STATE = AppState()


# ---------- FastAPI app ----------

app = FastAPI(title="SAM3 Annotator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bind 127.0.0.1 via uvicorn; tunnel is the access control
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Static frontend ----------

_FRONT = Path(__file__).parent.parent / "frontend"
_STATIC = _FRONT / "static"
_TPL = _FRONT / "templates"

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    html = (_TPL / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    })


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__,
                 "path": request.url.path},
    )


# ---------- Health / version ----------

@app.get("/api/version")
def api_version():
    return {"version": APP_VERSION, "name": "sam3_annotator"}


@app.get("/api/health")
def api_health():
    """Liveness + GPU + data sanity check."""
    import torch
    cuda_ok = bool(torch.cuda.is_available())
    gpu = None
    if cuda_ok:
        try:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "vram_total_mb": int(total / 1e6),
                "vram_free_mb": int(free / 1e6),
                "vram_pct_used": round(100 * (1 - free / total), 1),
            }
        except Exception as e:
            gpu = {"error": str(e)}
    data_dir = Path(os.environ.get("SAM3_ANNOT_DATA_DIR", ""))
    n_eps = 0
    if data_dir.is_dir():
        n_eps = sum(1 for d in data_dir.iterdir()
                    if d.is_dir() and (d / f"{d.name}_snippets.json").exists())
    return {
        "ok": True,
        "version": APP_VERSION,
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "?"),
        "cuda_available": cuda_ok,
        "gpu": gpu,
        "no_model": NO_MODEL,
        "no_text": NO_TEXT,
        "lightweight": LIGHTWEIGHT,
        "model_loaded": STATE.sam3.predictor is not None,
        "image_model_loaded": STATE.sam3._image_processor is not None,
        "data_dir": str(data_dir),
        "episodes_available": n_eps,
        "active_snippet": (
            None if STATE.snippet is None
            else f"{STATE.snippet.episode}/{STATE.snippet.snippet_id}"
        ),
        "approved_anchor_count": sum(len(v) for v in STATE.approved.values()),
        "propagated_frame_count": len(STATE.propagated),
    }


_STARTED_AT = time.time()


# ---------- Undo/redo helpers ----------

UNDO_LIMIT = 50


def _bucket_for(kind: str) -> dict[int, dict[str, np.ndarray]]:
    if kind == "approved":
        return STATE.approved
    if kind == "propagated":
        return STATE.propagated
    raise ValueError(f"unknown kind: {kind}")


def _snapshot_mask(frame_idx: int, category: str, kind: str) -> dict:
    """Capture current state of a single (frame, category, kind) for undo."""
    bucket = _bucket_for(kind)
    cur = bucket.get(frame_idx, {}).get(category)
    return {
        "frame_idx": int(frame_idx),
        "category": category,
        "kind": kind,
        "rle": mask_to_rle(cur) if cur is not None else None,
    }


def _push_undo(op: str, description: str, restore_items: list[dict]) -> None:
    """Record a reversible operation. Each restore_item is the PRIOR state."""
    STATE.undo_stack.append({
        "op": op,
        "description": description,
        "timestamp": time.time(),
        "restore_items": restore_items,
    })
    if len(STATE.undo_stack) > UNDO_LIMIT:
        STATE.undo_stack.pop(0)
    # New action invalidates redo
    STATE.redo_stack = []


def _apply_restore(items: list[dict]) -> list[dict]:
    """Apply restore_items, returning the inverse for redo."""
    inverse: list[dict] = []
    for it in items:
        bucket = _bucket_for(it["kind"])
        # Capture current state for redo
        cur = bucket.get(it["frame_idx"], {}).get(it["category"])
        inverse.append({
            "frame_idx": it["frame_idx"],
            "category": it["category"],
            "kind": it["kind"],
            "rle": mask_to_rle(cur) if cur is not None else None,
        })
        # Apply: restore mask or delete it
        if it["rle"] is None:
            if it["frame_idx"] in bucket and it["category"] in bucket[it["frame_idx"]]:
                del bucket[it["frame_idx"]][it["category"]]
                if not bucket[it["frame_idx"]]:
                    del bucket[it["frame_idx"]]
        else:
            mask = rle_to_mask(it["rle"])
            bucket.setdefault(it["frame_idx"], {})[it["category"]] = mask
    return inverse


@app.get("/api/history")
def api_history():
    """Return undo + redo stack summaries."""
    def summarize(stack: list[dict]) -> list[dict]:
        return [
            {"op": e["op"], "description": e["description"],
             "timestamp": e["timestamp"], "items": len(e["restore_items"])}
            for e in stack
        ]
    return {
        "undo": summarize(STATE.undo_stack),
        "redo": summarize(STATE.redo_stack),
    }


@app.post("/api/undo")
def api_undo():
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if not STATE.undo_stack:
        raise HTTPException(404, "nothing to undo")
    entry = STATE.undo_stack.pop()
    inverse = _apply_restore(entry["restore_items"])
    STATE.redo_stack.append({
        "op": "redo:" + entry["op"],
        "description": entry["description"],
        "timestamp": time.time(),
        "restore_items": inverse,
    })
    if len(STATE.redo_stack) > UNDO_LIMIT:
        STATE.redo_stack.pop(0)
    _autosave()
    return {"ok": True, "description": entry["description"],
            "restored": len(entry["restore_items"]),
            "undo_remaining": len(STATE.undo_stack),
            "redo_available": len(STATE.redo_stack)}


@app.post("/api/redo")
def api_redo():
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if not STATE.redo_stack:
        raise HTTPException(404, "nothing to redo")
    entry = STATE.redo_stack.pop()
    inverse = _apply_restore(entry["restore_items"])
    STATE.undo_stack.append({
        "op": entry["op"].replace("redo:", "", 1),
        "description": entry["description"],
        "timestamp": time.time(),
        "restore_items": inverse,
    })
    if len(STATE.undo_stack) > UNDO_LIMIT:
        STATE.undo_stack.pop(0)
    _autosave()
    return {"ok": True, "description": entry["description"],
            "restored": len(entry["restore_items"]),
            "undo_available": len(STATE.undo_stack),
            "redo_remaining": len(STATE.redo_stack)}


# ---------- Startup ----------

@app.on_event("startup")
async def startup():
    logger.info("sam3_annotator v%s starting", APP_VERSION)
    logger.info("data_dir=%s no_model=%s no_text=%s",
                os.environ.get("SAM3_ANNOT_DATA_DIR"), NO_MODEL, NO_TEXT)
    if not NO_MODEL:
        try:
            STATE.sam3.load_model()
        except Exception as e:
            logger.exception("model load failed at startup; will retry on first request")
            # Don't crash the server — let /api/health surface the error
        # Eager-load the image model when text is enabled OR when running
        # lightweight (where it also handles click/box). Avoids a slow first
        # request that often 504s through tunnels.
        if not NO_TEXT or LIGHTWEIGHT:
            try:
                STATE.sam3.ensure_image_model()
            except Exception:
                logger.exception("image model load failed at startup; will retry on demand")


# ---------- Session + metadata ----------

@app.get("/api/episodes")
def api_episodes():
    return {"episodes": storage.list_episodes()}


@app.get("/api/episodes/{episode}/snippets")
def api_snippets(episode: str):
    snips = storage.list_snippets(episode)
    # slim down
    return {"snippets": [{"snippet_id": s["snippet_id"],
                          "num_frames": s.get("num_frames"),
                          "split_size": s.get("split_size"),
                          "start_frame": s.get("start_frame"),
                          "end_frame": s.get("end_frame")}
                         for s in snips]}


class OpenSnippetReq(BaseModel):
    episode: str
    snippet_id: str
    categories: list[str] = ["Tool", "Liver", "Gallbladder"]


@app.post("/api/session/open")
def api_session_open(req: OpenSnippetReq):
    try:
        snippet = storage.load_snippet(req.episode, req.snippet_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))

    STATE.snippet = snippet
    STATE.reset_snippet_state()
    STATE.categories = list(req.categories)
    STATE.sam3.open_snippet(snippet)

    # Restore session_autosave if present
    restored = storage.load_session(snippet)
    if restored:
        for fidx_s, cats in restored.get("approved_masks", {}).items():
            fidx = int(fidx_s)
            STATE.approved[fidx] = {}
            for cat, md in cats.items():
                m = np.zeros((snippet.height, snippet.width), dtype=np.uint8)
                for poly in md.get("polygons", []):
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    if len(pts) >= 3:
                        cv2.fillPoly(m, [pts], 1)
                STATE.approved[fidx][cat] = m
                if cat not in STATE.categories:
                    STATE.categories.append(cat)

    return {
        "episode": snippet.episode,
        "snippet_id": snippet.snippet_id,
        "n_frames": len(snippet.frame_files),
        "width": snippet.width,
        "height": snippet.height,
        "split_size": snippet.split_size,  # may be null for cluster snippets
        "frames_dir_name": snippet.frames_dir_name,
        "start_frame": snippet.start_frame,
        "end_frame": snippet.end_frame,
        "categories": STATE.categories,
        "restored_anchors": sum(len(v) for v in STATE.approved.values()),
        "lightweight": LIGHTWEIGHT,
    }


@app.post("/api/session/close")
def api_session_close():
    STATE.sam3.reset_tracker_state()
    STATE.snippet = None
    STATE.reset_snippet_state()
    return {"ok": True}


# ---------- Frame + GT ----------

@app.get("/api/frame/{frame_idx}")
def api_frame(frame_idx: int):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    try:
        buf, ct = storage.read_frame_bytes(STATE.snippet, frame_idx)
    except IndexError:
        raise HTTPException(404, "frame out of range")
    return Response(content=buf, media_type=ct, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/gt/{frame_idx}")
def api_gt(frame_idx: int):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    gt = storage.load_gt(STATE.snippet)
    if gt is None:
        return {"polygons": {}}
    fpath = STATE.snippet.frame_files[frame_idx]
    frame_num = int(fpath.stem.split("_")[1])
    polys = storage.gt_polys_for_frame(gt, frame_num)
    return {"frame_num": frame_num, "polygons": polys}


# ---------- Masks ----------

@app.get("/api/masks/list")
def api_masks_list():
    """Summary of every mask currently in GT + approved + propagated state for this snippet."""
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    snip = STATE.snippet
    items = []
    seen = set()

    # Build frame_num -> frame_idx map once
    num_to_idx: dict[int, int] = {}
    for i, fp in enumerate(snip.frame_files):
        try:
            num_to_idx[int(fp.stem.split("_")[1])] = i
        except (ValueError, IndexError):
            continue

    # GT (read-only original polygons)
    gt = storage.load_gt(snip)
    if gt is not None:
        cat_by_id = {c["id"]: c["name"] for c in gt.get("categories", [])}
        # Aggregate per (frame_num, category)
        agg: dict[tuple[int, str], int] = {}
        for ann in gt.get("annotations", []):
            fn = ann.get("image_id")
            if fn is None:
                continue
            cat = cat_by_id.get(ann.get("category_id"), f"cat_{ann.get('category_id')}")
            seg = ann.get("segmentation") or []
            pts = 0
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list):
                        pts += len(poly) // 2
            agg[(fn, cat)] = agg.get((fn, cat), 0) + pts
        for (fn, cat), pts in agg.items():
            fidx = num_to_idx.get(fn)
            if fidx is None:
                continue
            items.append({
                "frame_idx": int(fidx),
                "frame_num": int(fn),
                "category": cat,
                "kind": "gt",
                "pixels": int(pts),  # polygon-point count proxy
            })

    # Approved (committed anchor masks)
    for fidx, cats in STATE.approved.items():
        for cat, m in cats.items():
            items.append({
                "frame_idx": int(fidx),
                "frame_num": int(snip.frame_files[fidx].stem.split("_")[1]) if fidx < len(snip.frame_files) else None,
                "category": cat,
                "kind": "approved",
                "pixels": int(m.sum()),
            })
            seen.add((fidx, cat))

    # Propagated (tracker output, not yet promoted)
    for fidx, cats in STATE.propagated.items():
        for cat, m in cats.items():
            if (fidx, cat) in seen:
                continue
            items.append({
                "frame_idx": int(fidx),
                "frame_num": int(snip.frame_files[fidx].stem.split("_")[1]) if fidx < len(snip.frame_files) else None,
                "category": cat,
                "kind": "propagated",
                "pixels": int(m.sum()),
            })

    items.sort(key=lambda x: (x["frame_idx"], x["kind"], x["category"]))
    by_kind = {"gt": 0, "approved": 0, "propagated": 0}
    for it in items:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    return {"items": items, "totals": by_kind}


class DeleteMaskReq(BaseModel):
    frame_idx: int
    category: str
    kind: str  # "approved" or "propagated"


@app.post("/api/masks/delete")
def api_masks_delete(req: DeleteMaskReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if req.kind == "gt":
        raise HTTPException(400, "GT is read-only (edit snippet_annotations.json directly)")
    if req.kind not in ("approved", "propagated"):
        raise HTTPException(400, f"unknown kind: {req.kind}")
    bucket = _bucket_for(req.kind)
    if req.frame_idx not in bucket or req.category not in bucket[req.frame_idx]:
        raise HTTPException(404, "mask not found")
    # Capture for undo BEFORE mutation
    snap = _snapshot_mask(req.frame_idx, req.category, req.kind)
    _push_undo("delete", f"delete {req.kind} {req.category} @ f{req.frame_idx}", [snap])
    del bucket[req.frame_idx][req.category]
    if not bucket[req.frame_idx]:
        del bucket[req.frame_idx]
    _autosave()
    return {"ok": True}


class ClearMasksReq(BaseModel):
    kind: str = "all"  # "approved" | "propagated" | "all" (both, GT excluded)
    category: Optional[str] = None
    frame_idx: Optional[int] = None  # if set, only clear at this frame


@app.post("/api/masks/clear")
def api_masks_clear(req: ClearMasksReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")

    # First snapshot everything we're about to delete
    snapshot: list[dict] = []
    for kind, bucket in [("approved", STATE.approved), ("propagated", STATE.propagated)]:
        if req.kind not in (kind, "all"):
            continue
        for fidx, cats in bucket.items():
            if req.frame_idx is not None and fidx != req.frame_idx:
                continue
            for cat in cats.keys():
                if req.category is not None and cat != req.category:
                    continue
                snapshot.append(_snapshot_mask(fidx, cat, kind))

    if not snapshot:
        return {"ok": True, "deleted": 0}

    desc = f"clear {req.kind}"
    if req.category:
        desc += f" {req.category}"
    if req.frame_idx is not None:
        desc += f" @ f{req.frame_idx}"
    desc += f" ({len(snapshot)} mask{'s' if len(snapshot) != 1 else ''})"
    _push_undo("clear", desc, snapshot)

    # Now actually delete
    deleted = 0
    def _strip(bucket: dict[int, dict[str, np.ndarray]]) -> int:
        n = 0
        for fidx in list(bucket.keys()):
            if req.frame_idx is not None and fidx != req.frame_idx:
                continue
            for cat in list(bucket[fidx].keys()):
                if req.category is not None and cat != req.category:
                    continue
                del bucket[fidx][cat]
                n += 1
            if not bucket[fidx]:
                del bucket[fidx]
        return n

    if req.kind in ("approved", "all"):
        deleted += _strip(STATE.approved)
    if req.kind in ("propagated", "all"):
        deleted += _strip(STATE.propagated)
    _autosave()
    return {"ok": True, "deleted": deleted}


@app.get("/api/masks/{frame_idx}")
def api_masks(frame_idx: int):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    approved_rle = {cat: mask_to_rle(m) for cat, m in STATE.approved.get(frame_idx, {}).items()}
    prop_rle = {cat: mask_to_rle(m) for cat, m in STATE.propagated.get(frame_idx, {}).items()}
    return {"approved": approved_rle, "propagated": prop_rle}


class CategoriesReq(BaseModel):
    categories: list[str]


@app.post("/api/categories")
def api_set_categories(req: CategoriesReq):
    for c in req.categories:
        if c not in STATE.categories:
            STATE.categories.append(c)
    return {"categories": STATE.categories}


# ---------- Commit + preview ----------

class CommitReq(BaseModel):
    frame_idx: int
    category: str
    # Painted mask as PNG (base64-encoded). Any non-zero pixel = foreground.
    # Client should downscale to target size OR send at original resolution.
    mask_png_b64: str


def _decode_mask_png(b64: str, target_h: int, target_w: int) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("PNG decode failed")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
    elif img.ndim == 3:
        alpha = img.any(axis=-1).astype(np.uint8) * 255
    else:
        alpha = img
    mask = (alpha > 0).astype(np.uint8)
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask


@app.post("/api/anchor/commit")
def api_anchor_commit(req: CommitReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    snip = STATE.snippet
    try:
        painted = _decode_mask_png(req.mask_png_b64, snip.height, snip.width)
    except Exception as e:
        raise HTTPException(400, f"bad mask: {e}")

    cleaned = STATE.sam3.commit_anchor(req.frame_idx, req.category, painted)
    if cleaned.sum() == 0:
        raise HTTPException(400, "mask empty after preprocessing")

    # Snapshot prior state at (frame, cat, approved) for undo
    prior = _snapshot_mask(req.frame_idx, req.category, "approved")
    _push_undo("commit",
               f"commit {req.category} @ f{req.frame_idx} ({int(cleaned.sum())}px)",
               [prior])

    STATE.approved.setdefault(req.frame_idx, {})[req.category] = cleaned
    STATE.cat_to_objid[req.category] = STATE.sam3.cat_to_objid(req.category)
    _autosave()
    return {
        "ok": True,
        "rle": mask_to_rle(cleaned),
        "pixels": int(cleaned.sum()),
    }


class PreviewReq(BaseModel):
    frame_idx: int
    category: str
    points: list[tuple[float, float, int]] = []
    box: Optional[tuple[float, float, float, float]] = None


class TextPreviewReq(BaseModel):
    frame_idx: int
    category: str
    text: str
    conf_threshold: float = 0.3
    max_results: int = 10


@app.post("/api/preview/text")
def api_preview_text(req: TextPreviewReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if NO_TEXT:
        raise HTTPException(503, "text prompting disabled (--no-text)")
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    try:
        dets = STATE.sam3.text_preview(req.frame_idx, req.text.strip(),
                                       conf_threshold=req.conf_threshold)
    except Exception as e:
        raise HTTPException(500, f"text preview failed: {e}")
    out = []
    for i, d in enumerate(dets[:req.max_results]):
        out.append({
            "idx": i,
            "score": round(d["score"], 4),
            "box": [round(b, 1) for b in d["box"]],
            "rle": mask_to_rle(d["mask"]),
            "pixels": int(d["mask"].sum()),
        })
    return {"detections": out, "n_total": len(dets)}


@app.post("/api/preview/click")
def api_preview_click(req: PreviewReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if not req.points and req.box is None:
        raise HTTPException(400, "need points or box")
    try:
        mask = STATE.sam3.preview_click(
            req.frame_idx, req.category,
            points=req.points or None,
            box=req.box,
        )
    except Exception as e:
        raise HTTPException(500, f"preview failed: {e}")
    return {"rle": mask_to_rle(mask), "pixels": int(mask.sum())}


# ---------- Propagate ----------

class PropagateReq(BaseModel):
    frame_idx: int
    max_frames_per_direction: int = 120


@app.post("/api/propagate")
def api_propagate(req: PropagateReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    if LIGHTWEIGHT:
        raise HTTPException(
            503,
            "propagate is disabled in lightweight mode. Anchors are saved "
            "to session_autosave.json — re-run on a stronger GPU "
            "(--no-lightweight) or use scripts/propagate_from_autosave.py.",
        )
    t0 = time.time()
    obj_results = STATE.sam3.propagate_bidir(
        anchor_frame_idx=req.frame_idx,
        max_frames_per_direction=req.max_frames_per_direction,
        progress=None,
    )
    # Snapshot all propagated entries we're about to overwrite, for undo
    objid_to_cat = STATE.sam3.objid_to_cat
    keys_touching: set[tuple[int, str]] = set()
    for fidx, per_obj in obj_results.items():
        for oid in per_obj:
            cat = objid_to_cat.get(oid)
            if cat is None:
                continue
            if fidx in STATE.approved and cat in STATE.approved[fidx]:
                continue
            keys_touching.add((fidx, cat))
    snapshot = [_snapshot_mask(f, c, "propagated") for f, c in keys_touching]
    if snapshot:
        _push_undo("propagate",
                   f"propagate from f{req.frame_idx} ({len(snapshot)} masks affected)",
                   snapshot)

    # Convert obj_id -> category, merge into STATE.propagated
    touched_frames = 0
    for fidx, per_obj in obj_results.items():
        if fidx not in STATE.propagated:
            STATE.propagated[fidx] = {}
        for oid, m in per_obj.items():
            cat = objid_to_cat.get(oid)
            if cat is None:
                continue
            # Never overwrite an approved mask
            if fidx in STATE.approved and cat in STATE.approved[fidx]:
                continue
            STATE.propagated[fidx][cat] = m
        touched_frames += 1
    _autosave()
    return {
        "ok": True,
        "frames_touched": touched_frames,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ---------- Export ----------

class ExportReq(BaseModel):
    concat_tool_instances: bool = False


@app.post("/api/export")
def api_export(req: ExportReq):
    if STATE.snippet is None:
        raise HTTPException(400, "no snippet open")
    out = storage.export_coco(
        STATE.snippet,
        STATE.approved,
        STATE.propagated,
        STATE.categories,
        STATE.cat_to_objid or {c: i+1 for i, c in enumerate(STATE.categories)},
        concat_tool_instances=req.concat_tool_instances,
    )
    return {"path": str(out), "approved_frames": len(STATE.approved),
            "propagated_frames": len(STATE.propagated)}


# ---------- Autosave helper ----------

def _autosave():
    if STATE.snippet is None:
        return
    from .rle import mask_to_polygons
    approved = {}
    for fidx, cats in STATE.approved.items():
        approved[str(fidx)] = {}
        for cat, m in cats.items():
            approved[str(fidx)][cat] = {
                "polygons": mask_to_polygons(m, min_area=10),
                "area": int(m.sum()),
            }
    propagated = {}
    for fidx, cats in STATE.propagated.items():
        propagated[str(fidx)] = {}
        for cat, m in cats.items():
            propagated[str(fidx)][cat] = {
                "polygons": mask_to_polygons(m, min_area=10),
                "area": int(m.sum()),
            }
    state = {
        "episode": STATE.snippet.episode,
        "snippet_id": STATE.snippet.snippet_id,
        "categories": STATE.categories,
        "cat_to_objid": STATE.cat_to_objid,
        "approved_masks": approved,
        "propagated_masks": propagated,
        "timestamp": time.time(),
    }
    storage.save_session(STATE.snippet, state)


# ---------- WebSocket (propagation progress, heartbeat) ----------

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")
            if t == "ping":
                await ws.send_json({"type": "pong", "t": time.time()})
            elif t == "hello":
                await ws.send_json({"type": "hello",
                                    "snippet": None if STATE.snippet is None else {
                                        "episode": STATE.snippet.episode,
                                        "snippet_id": STATE.snippet.snippet_id,
                                        "n_frames": len(STATE.snippet.frame_files)}})
            else:
                await ws.send_json({"type": "error", "msg": f"unknown type: {t}"})
    except WebSocketDisconnect:
        return
