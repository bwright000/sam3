# SAM3 Annotator on Colab A100 — Full Run

Browser-based paint annotator with SAM3 in the loop, served from a Colab A100 GPU and accessed from your local laptop via VSCode Remote Tunnel.

## Layout

**Drive (persistent):**
```
/MyDrive/Datasets/CRCD/
├── hf_cache/                 # auto-created on first run; persists SAM3 weights
└── To Be Annotated/          # the 11 GT-ready snippets, in-place edits
    ├── E_3/
    │   ├── E_3_snippets.json
    │   ├── snippet_001/
    │   │   ├── frames_left/
    │   │   ├── frames_right/
    │   │   ├── snippet_annotations.json
    │   │   ├── session_autosave.json   # auto-managed; survives session timeout
    │   │   ├── annotated_masks.json    # written on Export, in-place
    │   │   └── overlays/
    │   ├── snippet_002/...
    │   └── ...
    ├── F_3/...
    └── C_1/...
```

**Colab (ephemeral, code only):**
```
/content/sam3/                # cloned fresh each session from GitHub
├── sam3/
└── sam3_annotator/
```

---

## Per-session run

### Cell 1 — Mount Drive + symlink HF cache
```python
from google.colab import drive
drive.mount('/content/drive')

import os
HF_CACHE = '/content/drive/MyDrive/Datasets/CRCD/hf_cache'
os.makedirs(HF_CACHE, exist_ok=True)
!rm -rf /root/.cache/huggingface
!ln -s $HF_CACHE /root/.cache/huggingface
print('Drive mounted. HF cache symlinked → persists SAM3 weights across sessions.')
```

### Cell 2 — Clone code + install
```python
%cd /content
!rm -rf /content/sam3
!git clone https://github.com/bwright000/sam3.git
%cd /content/sam3
!pip install -q -e ./sam3
!pip install -q -e ./sam3_annotator
```

### Cell 3 — HuggingFace auth (only if HF cache is empty)
```python
# First session: log in once. Token is saved to ~/.huggingface (now on Drive via symlink)
# so subsequent sessions skip this cell.
import os
if not os.path.exists('/root/.cache/huggingface/token'):
    from huggingface_hub import login
    login(token='hf_YOUR_TOKEN_HERE')   # https://huggingface.co/settings/tokens
                                         # Must have access to facebook/sam3
else:
    print('HF token found in cache — skip')
```

### Cell 4 — Launch annotator server
```python
import subprocess, time
DATA_DIR = '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated'
assert __import__('os').path.isdir(DATA_DIR), f'data dir not found: {DATA_DIR}'

!pkill -f sam3_annotator.server 2>/dev/null
time.sleep(1)

proc = subprocess.Popen(
    ['python', '-u', '-m', 'sam3_annotator.server',
     '--data-dir', DATA_DIR,
     '--host', '127.0.0.1',
     '--port', '7860',
     '--log-level', 'info'],
    stdout=open('/content/sam3_annot.log', 'w'),
    stderr=subprocess.STDOUT,
)
print(f'server pid={proc.pid}')
print('waiting ~30s for SAM3 video model load...')
time.sleep(35)
!tail -25 /content/sam3_annot.log
```

You should see `Uvicorn running on http://127.0.0.1:7860`. If it errors, check the log.

### Cell 5 — Health check
```python
import requests
r = requests.get('http://127.0.0.1:7860/api/health').json()
print('CUDA:', r['cuda_available'], '| GPU:', r['gpu']['name'])
print('VRAM free:', r['gpu']['vram_free_mb'], 'MB /', r['gpu']['vram_total_mb'], 'MB')
print('Model loaded:', r['model_loaded'])
print('Episodes available:', r['episodes_available'])
```

### Cell 6 — Start VSCode Remote Tunnel
```python
import subprocess
subprocess.Popen(['curl', '-Lk',
                  'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64',
                  '-o', 'vscode_cli.tar.gz'],
                 cwd='/content').wait()
!tar -xf /content/vscode_cli.tar.gz -C /content
!nohup /content/code tunnel --accept-server-license-terms --name colab-sam3 > /content/tunnel.log 2>&1 &
import time; time.sleep(8)
!tail -30 /content/tunnel.log
```

A device-code prompt appears in the log. Visit https://github.com/login/device, enter the code, authorize.

### On your local laptop
1. Install the **Remote - Tunnels** VSCode extension (one-time)
2. `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel...` → pick `colab-sam3`
3. VSCode auto-forwards port 7860
4. Open **http://localhost:7860** in your local browser

---

## Annotation workflow (per snippet × 11 snippets)

Targets: **E_3/001-004**, **F_3/001-007** (skipping E_3/005 and C_1 unannotated).

For each snippet:
1. Sidebar → Episode → Snippet → **Load Snippet**
2. Pick **Tool** as active category (or add Tool_1, Tool_2 for multi-instance)
3. Navigate to the first keyframe (frame 0). GT outlines for Liver (red) + Gallbladder (green) should render
4. **Paint a Tool mask:** press `B` (brush), paint over visible instruments, press `Enter` to **Commit Anchor**
5. Press `Space` to **Propagate** N frames bidirectional (default 120)
6. Scrub through frames — fix any wrong frames by re-painting + Commit, then Propagate again
7. **Refine GT** (optional): press `G` to load GT polygon as raster on edit layer, brush/eraser to clean, `Enter` to commit
8. **Export Snippet** → writes `annotated_masks.json` to the snippet dir on Drive

Per-snippet effort: ~3-7 keyframes at ~30s each + propagation runs + scrub. Estimate **15-30 min/snippet**, ~4-5 hours total for 11 snippets.

---

## Phase 4 — Merge into snippet_annotations.json

After all 11 snippets exported:

### Cell 7 — Merge tool masks
```python
%cd /content/sam3
# merge_tool_masks.py lives in scripts/ at the repo root, NOT inside sam3_annotator/
!python scripts/merge_tool_masks.py \
    --segments-dir '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated'
```

The script appends the Tool category from `annotated_masks.json` into each `snippet_annotations.json`, with `.bak_pre_tool` backups.

---

## Phase 5 — Sync back to local (optional, after annotation pass)

If you want the annotated `snippet_annotations.json` files locally:

**Windows Robocopy:**
```cmd
robocopy "G:\My Drive\Datasets\CRCD\To Be Annotated" "c:\Users\benli\sam3facebook\data\Segments" /E /Z /XO /R:1 /W:5
```
(`/XO` = exclude older — only copies files newer in source)

Or just leave them on Drive and have the SLAM pipeline read from there.

---

## Phase 6 — Shutdown

### Cell 8 — Clean up
```python
!pkill -f sam3_annotator.server
!pkill -f 'code tunnel'
print('clean shutdown')
```

---

## Quick-relaunch (after first session)

Once HF cache + token are on Drive, every subsequent session is one cell:

```python
from google.colab import drive; drive.mount('/content/drive')
!ln -sfn /content/drive/MyDrive/Datasets/CRCD/hf_cache /root/.cache/huggingface
%cd /content
!rm -rf sam3 && git clone https://github.com/bwright000/sam3.git
%cd sam3
!pip install -q -e ./sam3 -e ./sam3_annotator

import subprocess, time
subprocess.Popen(['python', '-u', '-m', 'sam3_annotator.server',
                  '--data-dir', '/content/drive/MyDrive/Datasets/CRCD/To Be Annotated',
                  '--port', '7860'],
                 stdout=open('/content/sam3_annot.log','w'), stderr=subprocess.STDOUT)
time.sleep(35)
!tail /content/sam3_annot.log
!tar -xf /content/vscode_cli.tar.gz -C /content 2>/dev/null || \
  (curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /content/vscode_cli.tar.gz && tar -xf /content/vscode_cli.tar.gz -C /content)
!nohup /content/code tunnel --accept-server-license-terms --name colab-sam3 > /content/tunnel.log 2>&1 &
time.sleep(8)
!tail /content/tunnel.log
```

---

## Gotchas

| Issue | Fix |
|-------|-----|
| Colab session times out (12h hard, ~90min idle) | Drive autosave persists; relaunch + reload snippet, state restores |
| HF auth re-prompt every session | Cache on Drive (Cell 1 symlink) keeps token across sessions |
| Tunnel link not working | Re-run Cell 6; check `tunnel.log` for the device-code URL |
| Server log shows CUDA error | A100 doesn't TDR; check VRAM via Cell 5 health endpoint |
| `git clone` fails | Repo private? Use `git clone https://<TOKEN>@github.com/bwright000/sam3.git` with a GitHub PAT |
| Master annotation files needed for regen | Not on Drive yet (only Segments). Run [scripts/regen_snippet_annotations.py](../scripts/regen_snippet_annotations.py) locally before syncing if regen needed. |
| Browser shows blank | Hard-refresh (Ctrl+Shift+R) — version-busted query string forces fresh CSS/JS |
