# SAM3 Annotator on Colab A100 — Full Run

Two-phase flow:
- **Colab notebook (browser)**: minimum bootstrap — mount Drive, start VSCode Remote Tunnel
- **VSCode terminal (connected via tunnel)**: everything else — clone, install, launch server

You only touch the Colab notebook for the first 2 cells. The rest is a normal VSCode + bash session over SSH-style tunneling.

---

## Layout

**Drive (persistent, you push data here manually):**
```
/MyDrive/Datasets/CRCD/
├── hf_cache/                  # auto-created on first run
└── To Be Annotated/
    ├── E_3/snippet_001/...    # the 11 GT-ready snippets
    ├── F_3/snippet_001/...
    └── ...
```

**Colab (ephemeral, code only):**
```
/content/sam3/                 # cloned each session from GitHub
```

---

## Phase A — Colab notebook (2 cells)

### Cell A1 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell A2 — Install + start VSCode Remote Tunnel
```python
!curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
!tar -xf vscode_cli.tar.gz
!./code tunnel user login --provider github
!./code tunnel --accept-server-license-terms --name colab-sam3
```

The cell hangs (that's correct — the tunnel must keep running). The output shows a `vscode.dev/tunnel/colab-sam3` URL.

### On your local laptop
1. VSCode (one-time): install the **Remote - Tunnels** extension
2. `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel...` → pick `colab-sam3`
3. VSCode is now editing files on the Colab box. Open a terminal: `Ctrl+\` ` (it's a bash shell on the A100 host)

---

## Phase B — VSCode terminal (everything else)

Run all of this in the VSCode terminal connected to `colab-sam3`.

### B1 — Symlink HuggingFace cache to Drive (persists SAM3 weights across sessions)
```bash
mkdir -p /content/drive/MyDrive/Datasets/CRCD/hf_cache
rm -rf /root/.cache/huggingface
ln -s /content/drive/MyDrive/Datasets/CRCD/hf_cache /root/.cache/huggingface
```

### B2 — Clone latest code
```bash
cd /content
rm -rf sam3
git clone https://github.com/bwright000/sam3.git
cd sam3
```

### B3 — Install
```bash
# Use absolute paths to avoid cwd confusion
pip install -q -e /content/sam3/sam3
pip install -q -e /content/sam3/sam3_annotator
```
(numpy<2 dependency warnings about jax/cupy/rasterio etc. are safe to ignore — we don't use those packages.)

### B4 — HuggingFace auth (only first session — token persists in cached dir)
```bash
ls /root/.cache/huggingface/token >/dev/null 2>&1 \
  && echo "HF token already cached" \
  || huggingface-cli login   # paste token from https://huggingface.co/settings/tokens
                              # token must have read access to facebook/sam3
```

### B5 — Launch annotator server
```bash
cd /content/sam3
DATA_DIR='/content/drive/MyDrive/Datasets/CRCD/To Be Annotated'
test -d "$DATA_DIR" || { echo "MISSING: $DATA_DIR"; exit 1; }

pkill -f sam3_annotator.server 2>/dev/null
sleep 1

nohup python -u -m sam3_annotator.server \
    --data-dir "$DATA_DIR" \
    --host 127.0.0.1 \
    --port 7860 \
    --log-level info \
    > /content/sam3_annot.log 2>&1 &

echo "server pid=$!"
sleep 35
tail -25 /content/sam3_annot.log
```

You should see `Uvicorn running on http://127.0.0.1:7860`. The pid is yours to kill later.

### B6 — Health check
```bash
curl -s http://127.0.0.1:7860/api/health | python -m json.tool | head -20
```

Look for `cuda_available: true`, `model_loaded: true`, `episodes_available: 3` (or however many).

### B7 — Open the UI from your local laptop
VSCode auto-forwards port 7860 (you'll see it in the **Ports** tab at the bottom of VSCode). Click the globe icon next to port 7860, or just open **http://localhost:7860** in your local browser.

---

## Annotation workflow (per snippet × 11)

Targets: **E_3/001-004**, **F_3/001-007**.

For each snippet:
1. Sidebar → Episode → Snippet → **Load Snippet**
2. Pick **Tool** as active category (or add Tool_1, Tool_2 for multi-instance)
3. Navigate to first keyframe — verify Liver (red) + Gallbladder (green) GT outlines render
4. **Paint Tool mask:** `B` (brush) → paint over instruments → `Enter` (Commit Anchor)
5. **Propagate:** `Space` (default 120 frames bidirectional)
6. Scrub frames — fix bad frames by re-painting + Commit, then Propagate from corrected frame
7. **Optional GT cleanup:** `G` rasterizes GT polygon onto edit layer — brush/eraser to refine — `Enter` to commit
8. **Export Snippet** button — writes `annotated_masks.json` to the snippet dir on Drive

Per snippet: ~3-7 keyframes × 30s + propagation runs + scrub. ~15-30 min/snippet, ~4-5h total for 11 snippets.

---

## After all snippets — auto gap-fill pipeline

Detect remaining tool-detection gaps (zero / under-count / undermask),
re-propagate from the closest known-good neighbour frame, and merge the
result back into each snippet's canonical `annotated_masks.json`.

**All run in the same VSCode terminal that's already connected to the
Colab tunnel — no notebook cells, no file shuffling.**

### G1 — Set env paths (one-off per session)
```bash
export DATA_DIR='/content/drive/MyDrive/Datasets/CRCD/To Be Annotated'
export OUT_DIR='/content/sam3/outputs'
test -d "$DATA_DIR" || { echo "MISSING: $DATA_DIR"; }
mkdir -p "$OUT_DIR"
```

### G2 — Stage 1+2: build manifest + extract anchor seed PNGs (local-cheap)
```bash
cd /content/sam3
bash scripts/run_auto_gapfill_pipeline.sh prepare
```

This walks every `*/snippet_* tbd/` under `$DATA_DIR`, classifies gaps
(zero / under / undermask), reports the snippet-level asymmetry score,
and rasterises a binary seed PNG from the nearest healthy neighbour
frame for each gap. Output: `$OUT_DIR/gap_manifest.json` +
`$OUT_DIR/anchors/{EP}/{snippet}/*.png`.

Inspect the printed table — any snippet flagged with `asymmetry < 0.30`
needs a manual paint seed (no clean reference frame exists for SAM3 to
seed from) and won't be improved by Stage G3.

### G3 — Stage 3: SAM3 re-propagation on the A100
```bash
cd /content/sam3
python scripts/propagate_gap_fill.py \
    --manifest "$OUT_DIR/gap_manifest.json" \
    --anchors-dir "$OUT_DIR/anchors" \
    --data-dir "$DATA_DIR" \
    --output-suffix .gapfill
```

Per snippet, loads frames into SAM3's tracker, calls `add_new_mask` at
each anchor frame with the corresponding seed PNG, runs a single
bidirectional propagation, and writes:
- `<snippet>/annotated_masks.gapfill.json`
- `<snippet>/tool_detection_stats.gapfill.json`

Free-rerun targets (e.g. F_3/snippet_001) get the full snippet covered
by the gap-fill output rather than just the gap range.

### G4 — Stage 4+6: merge into canonical + build review queue
```bash
cd /content/sam3
bash scripts/run_auto_gapfill_pipeline.sh finalize
```

Per-frame merge: where the canonical mask was zero / under-count or has
materially smaller area than the gap-fill mask, the gap-fill wins;
otherwise canonical is kept. Output: `<snippet>/annotated_masks.merged.json`,
`<snippet>/tool_detection_stats.merged.json`, plus
`$OUT_DIR/review_queue.json` listing residual frames that still need
manual paint.

This step also runs Stage 5 (`promote_tbd_to_production.py`) in
**`--dry-run` mode** so you can see what would be written into production.

### G5 — Sync Drive back to local laptop, then promote
The merged outputs live alongside each tbd snippet on Drive. Drive
desktop autosyncs to the local laptop. On the laptop (production lives
only there at `c:/Users/benli/sam3facebook/data/Segments/`):

```bash
# In a local terminal:
cd c:/Users/benli/sam3facebook
bash scripts/run_auto_gapfill_pipeline.sh promote
```

Stage 5 injects Tool annotations into each production
`snippet_annotations.json` (slicing F_3/006 staging 387 frames →
production 200 by image_id range), rasterises the polygons into
`semantic_instance/` at pixel value 7 *only where the canvas is
currently background*, preserving Liver(3) / Gallbladder(4) tissue
priority. Backups: `snippet_annotations.json.bak_pre_tool_promote`.

---

## Sync back to local (optional)

If you want updated `snippet_annotations.json` on your local box, copy from Drive — Drive desktop app handles this automatically, or use Robocopy:
```cmd
robocopy "G:\My Drive\Datasets\CRCD\To Be Annotated" "c:\Users\benli\sam3facebook\data\Segments" /E /Z /XO /R:1 /W:5
```

---

## Shutdown

```bash
pkill -f sam3_annotator.server
```

The tunnel keeps running until you stop the Colab notebook cell (Cell A2) or the Colab session itself ends.

---

## Quick re-launch (every subsequent session)

Once Drive + HF cache are set up, the VSCode-terminal sequence collapses to:

```bash
ln -sfn /content/drive/MyDrive/Datasets/CRCD/hf_cache /root/.cache/huggingface
cd /content && rm -rf sam3 && git clone https://github.com/bwright000/sam3.git && cd sam3
pip install -q -e ./sam3 -e ./sam3_annotator
nohup python -u -m sam3_annotator.server \
    --data-dir '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated' \
    --port 7860 > /content/sam3_annot.log 2>&1 &
sleep 35 && tail /content/sam3_annot.log
```

Open `http://localhost:7860` in your local browser.

---

## Gotchas

| Issue | Fix |
|-------|-----|
| Tunnel cell completes too fast | The tunnel must keep running; cell will look hung — that's correct. |
| `huggingface-cli login` is interactive | Paste your token when prompted; once cached on Drive, future sessions skip. |
| `Uvicorn running...` doesn't appear in log | Check `/content/sam3_annot.log` — most likely an import error or missing CUDA. |
| Server hits CUDA error mid-session | A100 doesn't TDR, so this is rare; check `/api/health` for VRAM exhaustion. |
| Colab session ends mid-annotation | Drive autosave persists state; relaunch and reopen the snippet — restored. |
| `git clone` fails | If repo is private, use `git clone https://<PAT>@github.com/bwright000/sam3.git` with a GitHub Personal Access Token. |
| Hard-refresh needed in browser | Cache-bust query string is on, but if CSS looks stale: Ctrl+Shift+R. |
