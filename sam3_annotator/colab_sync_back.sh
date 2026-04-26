#!/bin/bash
# colab_sync_back.sh — push edits from local Colab SSD back to Drive.
#
# Run before stopping a Colab session, OR periodically during a long session
# (autosave still works on local; this just mirrors changes to Drive).
#
# Usage:
#   bash /content/sam3/sam3_annotator/colab_sync_back.sh

set -e

SRC='/content/data/To Be Annotated'
DST='/content/drive/MyDrive/Datasets/CRCD/To Be Annotated'

if [ ! -d "$SRC" ]; then
    echo "ERROR: local source not found at $SRC"
    echo "Did you run colab_setup.sh first?"
    exit 1
fi

if [ ! -d "$DST" ]; then
    echo "ERROR: Drive target not found at $DST"
    echo "Is Drive mounted?"
    exit 1
fi

echo "=== Local → Drive sync ==="
echo "  source: $SRC ($(du -sh "$SRC" | cut -f1))"
echo "  target: $DST"

# Push only changed/new files. Skip frames_left/right (they are read-only inputs).
# Sync: snippet_annotations.json, session_autosave.json, annotated_masks.json,
#       overlays/, plus any new files the annotator may write.
time rsync -av --update \
    --exclude 'frames_left/' \
    --exclude 'frames_right/' \
    --exclude '*.bak*' \
    --exclude '*.tmp' \
    "$SRC/" "$DST/"

echo
echo "Sync complete. Safe to stop the Colab session."
