# SAM3 Annotator

Photoshop-style paint annotator for SAM3 video mask seeding. Designed for the CRCD surgical video pipeline but portable to any SAM3 video workflow.

## Features
- Sidebar: episode → snippet picker
- Central frame canvas with layered overlays (GT polygons, approved masks, propagated masks, edit layer)
- Toolbox: brush, eraser, polygon, lasso, SAM3-click, SAM3-box, vertex drag
- Painted mask → SAM3 `add_new_mask` with specular-highlight preprocessing
- Bidirectional chunked propagation with user-configurable N frames
- Stroke-level undo/redo (command pattern)
- Keyboard shortcuts (B/E/P/L/S/V/G/Enter/Space/Ctrl+Z/Ctrl+Y/`[``]`)
- Autosave to disk (survives Colab session timeouts)
- Review mode — play propagated masks as video
- Export to COCO polygons → merges into existing `snippet_annotations.json`

## Install (Colab A100)

```python
# Cell 1 — mount Drive + install
from google.colab import drive
drive.mount('/content/drive')
!git clone <sam3facebook repo> /content/sam3facebook
%cd /content/sam3facebook
!pip install -e ./sam3
!pip install -e ./sam3_annotator
```

```python
# Cell 2 — launch server
%cd /content/sam3facebook
!python -m sam3_annotator.server \
    --data-dir /content/drive/MyDrive/crcd/data/Segments \
    --port 7860 --host 127.0.0.1
# VSCode Remote Tunnel auto-forwards 7860 → open http://localhost:7860 locally
```

## Usage

1. Select episode + snippet from sidebar → Load Snippet
2. Paint tool mask with brush, or use SAM3 Click / SAM3 Box for quick preview
3. Commit Anchor → mask registered as SAM3 seed
4. Propagate N Frames (default 120, bidirectional)
5. Scrub frames; fix errors by re-painting + committing on a corrected frame
6. Export Snippet → writes `annotated_masks.json` + merges into `snippet_annotations.json`

## Keyboard

| Key | Action |
|-----|--------|
| B | Brush |
| E | Eraser |
| P | Polygon |
| L | Lasso |
| S | SAM3 Click (positive) |
| Shift+S | SAM3 Click (negative) |
| V | Vertex drag |
| G | Load GT as editable |
| `[` `]` | Brush size down/up |
| Enter | Commit Anchor |
| Space | Propagate N frames |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| ← → | Prev / Next frame |
| Shift+← → | ±10 frames |
