"""Uvicorn entrypoint for the annotator server."""

import argparse
import logging
import os
import sys
from pathlib import Path


def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt, datefmt="%H:%M:%S")
    # Quiet uvicorn access spam unless DEBUG
    if level.upper() != "DEBUG":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser(
        description="SAM3 Annotator — paint-style mask seeding for SAM3 video propagation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True,
                    help="Path to data/Segments directory containing {EP}/snippet_NNN/ subdirs")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind host. Use 127.0.0.1 in shared environments and rely on SSH/tunnel for access.")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-model", action="store_true",
                    help="Skip SAM3 model loading (UI dev / smoke testing only)")
    ap.add_argument("--no-text", action="store_true",
                    help="Disable text prompting (saves ~3GB VRAM)")
    ap.add_argument("--log-level", default="info",
                    choices=["debug", "info", "warning", "error"],
                    help="Logging verbosity")
    ap.add_argument("--workers", type=int, default=1,
                    help="Uvicorn workers. Keep at 1 — model state is in-process.")
    args = ap.parse_args()

    _setup_logging(args.log_level)
    log = logging.getLogger("sam3_annotator.startup")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        log.error("data dir not found: %s", data_dir)
        sys.exit(1)
    if not data_dir.is_dir():
        log.error("data path is not a directory: %s", data_dir)
        sys.exit(1)

    # Quick sanity: count episodes
    eps = [d.name for d in data_dir.iterdir()
           if d.is_dir() and (d / f"{d.name}_snippets.json").exists()]
    if not eps:
        log.warning("no episodes found in %s — nothing to annotate. "
                    "Expected {EP}/{EP}_snippets.json under that dir.", data_dir)
    else:
        log.info("found %d episodes: %s", len(eps), ", ".join(eps))

    if args.host == "0.0.0.0":
        log.warning("binding 0.0.0.0 — anyone on the network can connect. "
                    "Use 127.0.0.1 + SSH/VSCode tunnel for shared hosts.")

    os.environ["SAM3_ANNOT_DATA_DIR"] = str(data_dir)
    os.environ["SAM3_ANNOT_NO_MODEL"] = "1" if args.no_model else "0"
    os.environ["SAM3_ANNOT_NO_TEXT"] = "1" if args.no_text else "0"

    import uvicorn
    uvicorn.run(
        "sam3_annotator.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        ws_max_size=32 * 1024 * 1024,  # 32MB for mask payloads
        log_level=args.log_level,
        access_log=(args.log_level == "debug"),
    )


if __name__ == "__main__":
    main()
