# SAM3 Annotator

Photoshop-style paint annotator for **SAM3 video mask seeding**. Built for medical / surgical video pipelines but generic to any SAM3-trackable footage.

⚙ **Wyrd light theme** — steampunk-parchment aesthetic. Designed for long annotation sessions.

---

## What it does

You annotate one keyframe with a brush (or SAM3 click/box, or a text prompt), commit it as an anchor, and SAM3's tracker bidirectionally propagates the mask through the rest of the video. Iterate: scrub through, fix bad frames, re-propagate.

The output is COCO polygons appended to your existing dataset's annotation file.

## Why use this over a generic labeller

- **SAM3 in the loop** — paint a rough mask, hit commit, get a refined SAM3 mask. Brush is the seed; SAM3 cleans it up.
- **Multi-modal prompts** — brush, eraser, SAM3 click (positive/negative), SAM3 box (two-click), SAM3 text (open-vocab detection with confidence threshold), GT-as-editable (rasterize an existing polygon, edit, recommit).
- **Bidirectional propagation** with user-set N-frames each direction.
- **Mask list panel** — see GT, approved anchors, and propagated masks at a glance per frame or across the whole snippet.
- **Autosave to disk** — survives session timeouts (Colab, transient SSH).
- **Browser-only client** — runs over SSH `-L` or VSCode Remote Tunnel, no installs on your laptop.

## Install

```bash
pip install -e ./sam3_annotator    # this package
pip install -e ./sam3              # SAM3 model (Meta, gated; HF auth required)
```

Requires Python 3.10+, PyTorch with CUDA (recommended) or CPU. SAM3 weights download on first run from `facebook/sam3` (HuggingFace gated).

## Launch

```bash
sam3-annotator --data-dir /path/to/data/Segments --port 7860
# open http://127.0.0.1:7860
```

CLI options:
```
--data-dir PATH        required: directory containing {EP}/{EP}_snippets.json subtrees
--host HOST            bind host, default 127.0.0.1 (recommended)
--port N               default 7860
--no-model             skip SAM3 load (UI dev only)
--no-text              disable text prompting (saves ~3GB VRAM)
--log-level LEVEL      debug|info|warning|error
```

## Expected data layout

```
data/Segments/
  {EP}/                        # e.g. F_3
    {EP}_snippets.json         # snippet metadata (start_frame, end_frame, num_frames, split_size, ...)
    snippet_NNN/
      frames_left/             # required: frame_NNNNNN.{webp,png,jpg}
      snippet_annotations.json # optional: COCO GT (Liver, Gallbladder, ...) — image_id == frame_num
      session_autosave.json    # auto-managed: per-snippet annotation state
      annotated_masks.json     # written on Export
```

If your dataset uses a different layout you'll need to adapt `sam3_annotator/server/storage.py`'s `load_snippet`.

## Workflow

1. Sidebar: **Episode → Snippet → Load Snippet**
2. Pick an active **Category** (Tool / Liver / Gallbladder / your own dynamic categories)
3. Paint with **Brush** (B), or use **SAM3 Click** (S), **SAM3 Box** (Shift+S), or **SAM3 Text** (T)
4. Press **Enter** to **Commit Anchor** — your edit-layer mask becomes a SAM3 anchor
5. Press **Space** to **Propagate** N frames bidirectional (sidebar slider, default 120)
6. Scrub frames — fix any errors by re-painting and re-committing
7. **Export Snippet** writes `annotated_masks.json`. Toggle **Concat Tool_* → Tool** if you used per-instance tool names.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| B | Brush |
| E | Eraser |
| S | SAM Click |
| Shift+S | SAM Box |
| T | SAM Text |
| G | Load GT as editable |
| `[` `]` | Brush size − / + |
| Enter | Commit Anchor |
| Space | Propagate |
| Ctrl+Z | Undo stroke |
| ← → | Prev / Next frame |
| Shift+← → | ±10 frames |
| PgUp / PgDn | Prev / Next keyframe (split-size boundary) |
| Home / End | First / Last frame in snippet |

## Health

`GET /api/health` returns version, GPU info (VRAM, capability), data dir status, model load status, active session, autosave summary. Indicator at the top right of the title bar updates every 10 s.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | liveness + diagnostics |
| GET | `/api/version` | semver |
| GET | `/api/episodes` | list episodes |
| GET | `/api/episodes/{ep}/snippets` | list snippets in episode |
| POST | `/api/session/open` | open snippet, restore autosave |
| POST | `/api/session/close` | tear down GPU state |
| GET | `/api/frame/{idx}` | WEBP frame bytes |
| GET | `/api/gt/{idx}` | GT polygons for this frame |
| GET | `/api/masks/{idx}` | approved + propagated RLE for this frame |
| GET | `/api/masks/list` | summary of GT + approved + propagated for whole snippet |
| POST | `/api/preview/click` | SAM3 click/box single-frame preview |
| POST | `/api/preview/text` | SAM3 image-level text-prompt detection |
| POST | `/api/anchor/commit` | painted-mask PNG → SAM3 `add_new_mask` |
| POST | `/api/propagate` | bidirectional from anchor, N frames each direction |
| POST | `/api/export` | write `annotated_masks.json` |
| WS | `/ws` | heartbeat ping/pong |

## Deployment

- **Local laptop with NVIDIA GPU**: works for everything. GPUs ≥ Pascal (SM 6.0+) recommended; older cards may hit Windows TDR.
- **Linux / cluster + SSH tunnel**: `ssh -L 7860:127.0.0.1:7860 user@host` then open `http://localhost:7860`.
- **Google Colab A100**: see `install_colab.md`. Browser via VSCode Remote Tunnel auto-forwards.
- **Production / shared host**: bind 127.0.0.1 only, expose via SSH/VSCode tunnel. Don't use `0.0.0.0` without authentication in front.

## Testing

```bash
SAM3_ANNOT_TEST_DATA_DIR=/path/to/data/Segments pytest sam3_annotator/tests/
```

Smoke tests run the server in `--no-model` mode and exercise core endpoints.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `RuntimeError: CUDA error: the launch timed out` | Windows GPU TDR — kernel ran past 2 s timeout. Restart server. Run on Linux or stronger GPU. |
| `No module named sam3_annotator` | Wrong Python — use the conda env / venv where you ran `pip install -e .`. |
| Frame loads but GT outline missing | `image_id` in `snippet_annotations.json` doesn't match the frame numbers in `frames_left/`. Regenerate the COCO subset. |
| Text prompt hangs ~20 s on first call | Lazy-loading the image model. Subsequent calls are ~30-100 ms. Use `--no-text` to skip. |
| Browser white screen | Hard refresh (Ctrl+Shift+R). Konva CDN may have failed — check network. |
| Commit returns 500 | Check server log. If "CUDA error", see TDR row. If "mask empty", you didn't paint anything. |

## License

MIT — see `LICENSE`.
