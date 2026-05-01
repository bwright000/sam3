#!/usr/bin/env bash
# Auto-gap-fill orchestrator.
#
#   Stages 1, 2, 4, 5, 6 are LOCAL.
#   Stage 3 (propagate_gap_fill.py) needs Colab A100 — run separately.
#
# Typical local + colab flow:
#
#   # 1) On laptop — build manifest + extract anchor seed PNGs (purely local).
#   bash scripts/run_auto_gapfill_pipeline.sh prepare
#
#   # 2) On Colab A100 — run SAM3 propagation on each seed
#   #    (mount Drive, copy the manifest + anchors over, then):
#   python scripts/propagate_gap_fill.py \
#       --manifest /content/outputs/gap_manifest.json \
#       --anchors-dir /content/outputs/anchors \
#       --data-dir '/content/data/To Be Annotated'
#
#   # 3) Back on laptop — merge gapfill into canonical, promote to production,
#   #    build manual-review queue.
#   bash scripts/run_auto_gapfill_pipeline.sh finalize
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-$REPO/data/Segments/data-20260427T144719Z-3-002/data/To Be Annotated}"
OUT_DIR="${OUT_DIR:-$REPO/outputs}"
PROD_ROOT="${PROD_ROOT:-$REPO/data/Segments}"
MANIFEST="$OUT_DIR/gap_manifest.json"
ANCHORS_DIR="$OUT_DIR/anchors"
REVIEW_QUEUE="$OUT_DIR/review_queue.json"

mkdir -p "$OUT_DIR"

case "${1:-}" in
  prepare)
    echo "=== Stage 1: build gap manifest ==="
    python "$SCRIPT_DIR/build_gap_manifest.py" \
        --data-dir "$DATA_DIR" \
        --out "$MANIFEST" \
        --full-rerun F_3/snippet_001
    echo
    echo "=== Stage 2: extract anchor masks ==="
    python "$SCRIPT_DIR/extract_anchor_masks.py" \
        --manifest "$MANIFEST" \
        --anchors-dir "$ANCHORS_DIR"
    echo
    echo "Now copy these to Colab and run propagate_gap_fill.py:"
    echo "  $MANIFEST"
    echo "  $ANCHORS_DIR/"
    ;;

  finalize)
    echo "=== Stage 4: merge gapfill into canonical ==="
    python "$SCRIPT_DIR/merge_gapfill_into_canonical.py" \
        --manifest "$MANIFEST" \
        --gapfill-suffix .gapfill \
        --merged-suffix .merged
    echo
    echo "=== Stage 5 (dry-run first): promote to production ==="
    python "$SCRIPT_DIR/promote_tbd_to_production.py" \
        --manifest "$MANIFEST" \
        --production-root "$PROD_ROOT" \
        --merged-suffix .merged \
        --dry-run
    echo
    echo "(Re-run without --dry-run to apply.)"
    echo
    echo "=== Stage 6: build residual review queue ==="
    python "$SCRIPT_DIR/build_residual_review_queue.py" \
        --manifest "$MANIFEST" \
        --stats-suffix .merged \
        --out "$REVIEW_QUEUE"
    ;;

  promote)
    echo "=== Stage 5 (apply): promote to production ==="
    python "$SCRIPT_DIR/promote_tbd_to_production.py" \
        --manifest "$MANIFEST" \
        --production-root "$PROD_ROOT" \
        --merged-suffix .merged
    ;;

  *)
    echo "usage: $0 {prepare|finalize|promote}"
    exit 1
    ;;
esac
