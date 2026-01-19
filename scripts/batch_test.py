#!/usr/bin/env python3
"""Small-scale batch test for SAM3: run text prompts on images and video (5 frames).

Usage examples:
  python scripts/batch_test.py --images assets/images/test_image.jpg,assets/images/truck.jpg,... 
  python scripts/batch_test.py --image-folder /path/to/five_images_folder

Notes:
  - This script expects a working SAM3 installation and (optionally) model
    checkpoints available via HuggingFace. Set `--no-download` to avoid
    attempting to download checkpoints (may fail if no local checkpoint).
  - For video tests the script will create a temporary JPEG-folder from the
    provided images and start a `Sam3VideoPredictor` session on it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images",
        type=str,
        default=None,
        help="Comma-separated list of image paths (provides up to 5 images).",
    )
    p.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Folder containing images; first 5 images will be used.",
    )
    p.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated text prompts to run for each image (default simple list).",
    )
    p.add_argument(
        "--out-dir", type=str, default="batch_test_out", help="Output folder"
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not attempt to download checkpoints from HF (may fail if no local ckpt).",
    )
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def gather_images(args) -> List[Path]:
    imgs: List[Path] = []
    if args.images:
        for p in args.images.split(","):
            p = p.strip()
            if p:
                imgs.append(Path(p))
    if args.image_folder:
        folder = Path(args.image_folder)
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    imgs.append(p)
    # fallback: use assets sample images if none provided
    if len(imgs) == 0:
        sample_dir = Path(__file__).resolve().parents[1] / "assets" / "images"
        for name in ["test_image.jpg", "truck.jpg", "groceries.jpg"]:
            p = sample_dir / name
            if p.exists():
                imgs.append(p)

    return imgs[:5]


def run_image_batch(image_paths: List[Path], prompts: List[str], out_dir: Path, device: str, no_download: bool):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading image model on device={device} (no_download={no_download})...")
    model = build_sam3_image_model(device=device, load_from_HF=(not no_download))
    processor = Sam3Processor(model, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing image {img_path}")
        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)
        img_out = {"path": str(img_path), "results": []}
        for prompt in prompts:
            print(f"  Prompt: {prompt}")
            s = processor.set_text_prompt(prompt, state.copy())
            boxes = s.get("boxes")
            scores = s.get("scores")
            masks = s.get("masks_logits")

            # Move to cpu and convert
            boxes_cpu = boxes.detach().cpu().numpy().tolist() if boxes is not None else []
            scores_cpu = scores.detach().cpu().numpy().tolist() if scores is not None else []

            # Save masks as .npy
            mask_fname = out_dir / f"{img_path.stem}_{prompt.replace(' ', '_')}_masks.npy"
            if masks is not None:
                masks_cpu = masks.detach().cpu().numpy()
                try:
                    np_save_path = str(mask_fname)
                    import numpy as _np

                    _np.save(np_save_path, masks_cpu)
                except Exception as e:
                    print(f"    Warning: cannot save masks: {e}")
            else:
                masks_cpu = []

            result = {"prompt": prompt, "boxes": boxes_cpu, "scores": scores_cpu, "masks_file": str(mask_fname) if len(masks_cpu) else None}
            img_out["results"].append(result)

        # write JSON summary
        json_path = out_dir / f"{img_path.stem}_results.json"
        with open(json_path, "w") as f:
            json.dump(img_out, f, indent=2)


def run_video_test(image_paths: List[Path], prompts: List[str], out_dir: Path, device: str, no_download: bool):
    # For video testing we create a temporary JPEG frames folder and use the video predictor
    from sam3.model_builder import build_sam3_video_predictor

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_video_frames_"))
    for i, p in enumerate(image_paths):
        dst = tmp_dir / f"{i:05d}.jpg"
        shutil.copy(p, dst)

    print(f"Starting video predictor on frames folder {tmp_dir} (device={device})")
    # build predictor (this will load the model and may download ckpt)
    predictor = build_sam3_video_predictor(gpus_to_use=[0] if device.startswith("cuda") else [0])

    sess = predictor.handle_request({"type": "start_session", "resource_path": str(tmp_dir)})
    session_id = sess["session_id"]
    print(f"Started session {session_id}")

    # Add a single text prompt on frame 0, then propagate
    for prompt in prompts:
        resp = predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
        print(f"Added prompt '{prompt}' -> resp keys: {list(resp.keys())}")

    # Propagate and save outputs
    stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "both", "start_frame_index": 0})
    for out in stream:
        idx = out["frame_index"]
        outputs = out["outputs"]
        savep = out_dir / f"video_frame_{idx:05d}_outputs.json"
        with open(savep, "w") as f:
            json.dump(outputs, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    predictor.close_session(session_id)
    shutil.rmtree(tmp_dir)


def main():
    args = parse_args()
    imgs = gather_images(args)
    if len(imgs) == 0:
        print("No images found. Provide --images or --image-folder with at least one image.")
        return

    prompts = ["a person", "a vehicle"] if not args.prompts else [p.strip() for p in args.prompts.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run image batch
    run_image_batch(imgs, prompts, out_dir, args.device, args.no_download)

    # Run video test (use same images as frames)
    run_video_test(imgs, prompts, out_dir, args.device, args.no_download)


if __name__ == "__main__":
    main()
