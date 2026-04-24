"""Uvicorn entrypoint for the annotator server."""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="SAM3 Annotator server")
    ap.add_argument("--data-dir", required=True, help="Path to data/Segments directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-model", action="store_true",
                    help="Skip SAM3 model loading (UI testing only)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    # Export config via env-like globals before importing the app
    import os
    os.environ["SAM3_ANNOT_DATA_DIR"] = str(data_dir)
    os.environ["SAM3_ANNOT_NO_MODEL"] = "1" if args.no_model else "0"

    import uvicorn
    uvicorn.run(
        "sam3_annotator.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        ws_max_size=32 * 1024 * 1024,  # 32MB for mask payloads
        log_level="info",
    )


if __name__ == "__main__":
    main()
