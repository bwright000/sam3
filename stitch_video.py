"""
Simple script to stitch frames into a video.
Uses OpenCV (cv2) to create video from image sequence.

Usage:
    # Stitch a single split directory
    python stitch_video.py --input data/split_imgs/split_0 --output split_0.mp4 --fps 30

    # Stitch ALL splits together into one video
    python stitch_video.py --input data/split_imgs --output full_video.mp4 --fps 30 --all-splits
"""

import argparse
import cv2
import re
from pathlib import Path


def natural_sort_key(path: Path) -> tuple:
    """Sort key for natural ordering (split_1, split_2, ... split_10, split_11)."""
    name = path.name
    # Extract numbers from the name
    parts = re.split(r'(\d+)', name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def stitch_frames_to_video(input_dir: str, output_path: str, fps: int = 30):
    """
    Stitch frames from a directory into a video.

    Args:
        input_dir: Directory containing numbered frame images (00000.jpg, 00001.jpg, etc.)
        output_path: Output video file path
        fps: Frames per second for the output video
    """
    input_path = Path(input_dir)

    # Find all jpg frames and sort them
    frames = sorted(input_path.glob("*.jpg"))

    if not frames:
        print(f"No .jpg files found in {input_dir}")
        return

    print(f"Found {len(frames)} frames in {input_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"Could not read first frame: {frames[0]}")
        return

    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    # Create video writer
    # Use mp4v codec for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Could not open video writer for {output_path}")
        return

    # Write frames
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_path}")

        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            print(f"  Processed {i + 1}/{len(frames)} frames")

    out.release()
    print(f"\nVideo saved to: {output_path}")
    print(f"Duration: {len(frames) / fps:.2f} seconds at {fps} FPS")


def stitch_all_splits(input_dir: str, output_path: str, fps: int = 30):
    """
    Stitch all split directories together into one video.

    Args:
        input_dir: Parent directory containing split_0, split_1, etc.
        output_path: Output video file path
        fps: Frames per second for the output video
    """
    input_path = Path(input_dir)

    # Find all split directories
    split_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("split_")]

    if not split_dirs:
        print(f"No split directories found in {input_dir}")
        return

    # Sort naturally (split_0, split_1, ..., split_9, split_10, ...)
    split_dirs = sorted(split_dirs, key=natural_sort_key)

    print(f"Found {len(split_dirs)} split directories")
    print(f"Order: {', '.join(d.name for d in split_dirs[:5])}{'...' if len(split_dirs) > 5 else ''}")

    # Collect all frames in order
    all_frames = []
    for split_dir in split_dirs:
        frames = sorted(split_dir.glob("*.jpg"))
        if frames:
            all_frames.extend(frames)
            print(f"  {split_dir.name}: {len(frames)} frames")

    if not all_frames:
        print("No frames found in any split directory")
        return

    print(f"\nTotal frames: {len(all_frames)}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(all_frames[0]))
    if first_frame is None:
        print(f"Could not read first frame: {all_frames[0]}")
        return

    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Could not open video writer for {output_path}")
        return

    # Write frames
    print(f"\nStitching video at {fps} FPS...")
    for i, frame_path in enumerate(all_frames):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_path}")

        # Progress indicator
        if (i + 1) % 100 == 0 or i == len(all_frames) - 1:
            print(f"  Processed {i + 1}/{len(all_frames)} frames ({100*(i+1)/len(all_frames):.1f}%)")

    out.release()

    duration = len(all_frames) / fps
    print(f"\nVideo saved to: {output_path}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes) at {fps} FPS")


def main():
    parser = argparse.ArgumentParser(description="Stitch image frames into a video")
    parser.add_argument(
        "--input", "-i",
        default="data/split_imgs",
        help="Input directory containing frame images or split directories"
    )
    parser.add_argument(
        "--output", "-o",
        default="output_video.mp4",
        help="Output video file path"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Stitch all split_* directories together into one video"
    )

    args = parser.parse_args()

    if args.all_splits:
        stitch_all_splits(args.input, args.output, args.fps)
    else:
        stitch_frames_to_video(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
