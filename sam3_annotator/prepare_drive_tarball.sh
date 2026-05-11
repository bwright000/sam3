#!/bin/bash
# prepare_drive_tarball.sh — once-only on local laptop. Creates a single tarball
# of the snippet data and pushes to Drive. Future Colab sessions extract the
# tarball in seconds (vs minutes of FUSE-rsync for thousands of webp files).
#
# Usage (Linux/WSL/Git Bash):
#   bash prepare_drive_tarball.sh /local/path/to/data/Segments /drive/mount/Datasets/CRCD
#
# Or just adapt the paths below for your local environment.

set -e

LOCAL_SRC="${1:-F:/Datasets/CRCD}"
DRIVE_DST="${2:-G:/My Drive/Datasets/CRCD}"
TAR_NAME="snippets.tar"

if [ ! -d "$LOCAL_SRC" ]; then
    echo "ERROR: local source not found at $LOCAL_SRC"
    exit 1
fi

mkdir -p "$DRIVE_DST/To Be Annotated"

echo "Local source: $LOCAL_SRC"
echo "  size: $(du -sh "$LOCAL_SRC" 2>/dev/null | cut -f1)"

echo
echo "Step 1 — copy individual snippets to Drive 'To Be Annotated/' (slow, sets up baseline)"
# This is for the annotator path layout. Tarball below is a faster reload format.
rsync -a --info=progress2 "$LOCAL_SRC/" "$DRIVE_DST/To Be Annotated/"

echo
echo "Step 2 — pack the tarball for fast Colab reload"
TAR_TMP="/tmp/$TAR_NAME"
cd "$(dirname "$LOCAL_SRC")"
tar -cf "$TAR_TMP" "$(basename "$LOCAL_SRC")"
echo "  tarball: $(du -sh "$TAR_TMP" | cut -f1)"

echo
echo "Step 3 — push tarball to Drive (single file = fast)"
time cp "$TAR_TMP" "$DRIVE_DST/$TAR_NAME"
rm "$TAR_TMP"

echo
echo "Done."
echo "  Drive folder: $DRIVE_DST/To Be Annotated/"
echo "  Drive tarball: $DRIVE_DST/$TAR_NAME"
echo
echo "Future Colab sessions will use the tarball automatically (colab_setup.sh detects it)."
