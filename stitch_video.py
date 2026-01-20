"""
Simple script to stitch frames into a video.
Uses OpenCV (cv2) to create video from image sequence.

Usage:
    python stitch_video.py --input data/split_imgs/split_0 --output split_0.mp4 --fps 30
"""

import argparse
import cv2
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description="Stitch image frames into a video")
    parser.add_argument(
        "--input", "-i",
        default="data/split_imgs/split_0",
        help="Input directory containing frame images"
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

    args = parser.parse_args()
    stitch_frames_to_video(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
