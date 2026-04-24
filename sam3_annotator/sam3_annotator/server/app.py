"""FastAPI app: session + frames + masks + propagation + export."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure parent repo's scripts are importable (reuse _load_frames_for_tracker etc.)
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]  # sam3facebook/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from . import storage
from .rle import mask_to_rle, rle_to_mask
from .sam3_service import SAM3Service


NO_MODEL = os.environ.get("SAM3_ANNOT_NO_MODEL", "0") == "1"


# ---------- Global app state ----------

class AppState:
    def __init__(self):
        self.sam3 = SAM3Service(no_model=NO_MODEL)
        self.snippet: Optional[storage.SnippetInfo] = None
        # approved_masks: {frame_idx: {category: uint8 mask}}
        self.approved: dict[int, dict[str, np.ndarray]] = {}
        # propagated_masks: {frame_idx: {category: uint8 mask}}
        self.propagated: dict[int, dict[str, np.ndarray]] = {}
        self.categories: list[str] = []
        # cat_to_objid mirrors sam3_service
        self.cat_to_objid: dict[str, int] = {}

    def reset_snippet_state(self):
        self.approved = {}
        self.propagated = {}
        self.cat_to_objid = {}
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
    return HTMLResponse(html)


# ---------- Startup ----------

@app.on_event("startup")
async def startup():
    if not NO_MODEL:
        try:
            STATE.sam3.load_model()
        except Exception as e:
            print(f"[startup] model load failed: {e}", file=sys.stderr)
            # Continue — /session/open will surface errors to the user


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
        "split_size": snippet.split_size,
        "start_frame": snippet.start_frame,
        "end_frame": snippet.end_frame,
        "categories": STATE.categories,
        "restored_anchors": sum(len(v) for v in STATE.approved.values()),
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
    t0 = time.time()
    obj_results = STATE.sam3.propagate_bidir(
        anchor_frame_idx=req.frame_idx,
        max_frames_per_direction=req.max_frames_per_direction,
        progress=None,
    )
    # Convert obj_id -> category, merge into STATE.propagated
    objid_to_cat = STATE.sam3.objid_to_cat
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
