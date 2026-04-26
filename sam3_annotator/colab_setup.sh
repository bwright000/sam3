#!/bin/bash
# colab_setup.sh — once-per-session bootstrap for SAM3 annotator on Colab.
#
# Run AFTER:
#   1. drive.mount('/content/drive') in the Colab notebook
#   2. VSCode Remote Tunnel started + connected from local laptop
#
# This script:
#   - symlinks HF cache to Drive (persists SAM3 weights across sessions)
#   - clones latest code
#   - installs packages
#   - copies snippet data to local SSD via tarball if available, else direct rsync
#   - launches the annotator server in background
#
# Usage:
#   bash /content/sam3/sam3_annotator/colab_setup.sh

set -e

DRIVE_ROOT="/content/drive/MyDrive/Datasets/CRCD"
SRC_DIR="$DRIVE_ROOT/To Be Annotated"
DST_DIR="/content/data/To Be Annotated"
HF_CACHE="$DRIVE_ROOT/hf_cache"
TAR_FILE="$DRIVE_ROOT/snippets.tar"   # optional pre-staged tarball
REPO_URL="https://github.com/bwright000/sam3.git"
REPO_DIR="/content/sam3"
PORT=7860

echo "=== [1/5] HF cache symlink ==="
mkdir -p "$HF_CACHE"
rm -rf /root/.cache/huggingface
ln -s "$HF_CACHE" /root/.cache/huggingface
echo "  /root/.cache/huggingface -> $HF_CACHE"

echo
echo "=== [2/5] Clone latest code ==="
cd /content
rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"

echo
echo "=== [3/5] Install packages ==="
pip install -q -e "$REPO_DIR/sam3"
pip install -q -e "$REPO_DIR/sam3_annotator"
echo "  done"

echo
echo "=== [4/5] Stage snippet data to local SSD ==="
DST_PARENT="$(dirname "$DST_DIR")"
mkdir -p "$DST_PARENT"

if [ -d "$DST_DIR" ] && [ "$(ls -A "$DST_DIR" 2>/dev/null)" ]; then
    echo "  local copy already present at $DST_DIR — skipping copy"
    echo "  (delete it manually if you want a fresh copy: rm -rf '$DST_DIR')"
elif [ -f "$TAR_FILE" ]; then
    echo "  tarball found at $TAR_FILE — fast path"
    cd "$DST_PARENT"
    time tar -xf "$TAR_FILE"
elif [ -d "$SRC_DIR" ]; then
    echo "  no tarball — using cp -r (fast for many small files over Drive FUSE)"
    echo "  source size: $(du -sh "$SRC_DIR" | cut -f1)"
    time cp -r "$SRC_DIR" "$DST_DIR"
else
    echo "  ERROR: neither $TAR_FILE nor $SRC_DIR exists"
    exit 1
fi
echo "  staged: $(du -sh "$DST_DIR" | cut -f1)"

echo
echo "=== [5/5] Launch server ==="
pkill -f sam3_annotator.server 2>/dev/null || true
sleep 2

cd "$REPO_DIR"
nohup python -u -m sam3_annotator.server \
    --data-dir "$DST_DIR" \
    --host 127.0.0.1 \
    --port $PORT \
    --log-level info \
    > /content/sam3_annot.log 2>&1 &

echo "  server pid=$!"
echo "  waiting ~35s for SAM3 model load..."
sleep 35

echo
echo "=== Server tail ==="
tail -20 /content/sam3_annot.log

echo
echo "=== Health check ==="
curl -s http://127.0.0.1:$PORT/api/health | python -m json.tool | grep -E "ok|version|cuda|gpu|loaded|episodes" || true

echo
echo "Open http://localhost:$PORT in your local browser (port auto-forwarded by VSCode tunnel)."
echo "Sync edits back to Drive at end of session: bash $REPO_DIR/sam3_annotator/colab_sync_back.sh"
