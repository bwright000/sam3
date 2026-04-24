# Colab setup for SAM3 Annotator

Run these cells in order in a **fresh Colab with A100 runtime**.

## Cell 1 — Mount Drive + clone repo + install
```python
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess
os.chdir('/content')
if not os.path.isdir('/content/sam3facebook'):
    !git clone https://github.com/<YOU>/sam3facebook.git

%cd /content/sam3facebook
# SAM3 model (same as current workflow)
!pip install -q -e ./sam3
# Annotator tool
!pip install -q -e ./sam3_annotator
```

## Cell 2 — Launch server (background)
```python
import subprocess, os, time

DATA_DIR = '/content/drive/MyDrive/crcd/data/Segments'  # adjust to your Drive path
assert os.path.isdir(DATA_DIR), f"Segments dir not found at {DATA_DIR}"

# Kill any previous instance
!pkill -f 'sam3_annotator.server' 2>/dev/null || true
time.sleep(1)

# Launch detached
proc = subprocess.Popen([
    'python', '-m', 'sam3_annotator.server',
    '--data-dir', DATA_DIR,
    '--host', '127.0.0.1',
    '--port', '7860',
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(f"Started sam3_annotator server, pid={proc.pid}")
```

## Cell 3 — Connect via VSCode Remote Tunnel
From your local VSCode, use the **Remote-Tunnels** extension to connect to this Colab instance. Once connected, VSCode auto-forwards port 7860. Open `http://localhost:7860` in your local browser.

Alternative (SSH): use `ngrok` or `cloudflared` to expose port 7860 — not recommended for sensitive surgical data.

## Sync flow

**Before a session:**
- Push latest `data/Segments/` to Drive:
  ```
  rsync -avz data/Segments/ <drive>/crcd/data/Segments/
  ```

**After a session:**
- Pull updated `session_autosave.json`, `annotated_masks.json`, `snippet_annotations.json` back to local from Drive.

## Notes

- Colab A100 sessions max 12h. Autosave writes to Drive per-commit, so you can pick up where you left off.
- SAM3 model checkpoint downloads to `/root/.cache/huggingface/` on first launch. Ephemeral — ~20s reload on new Colab session. To persist, symlink `/root/.cache/huggingface` to a Drive folder.
- GPU OOM risk on long snippets (600+ frames at 1008×1008). Tracker uses fp16; if you hit OOM, open one snippet at a time and close between sessions.
