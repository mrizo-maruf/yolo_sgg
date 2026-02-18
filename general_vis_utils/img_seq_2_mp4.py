#!/usr/bin/env python3
"""
Convert an image sequence to an MP4 video.
Supports JPG/JPEG/PNG files (including mixed extensions).

Example:
    python img_seq_2_mp4.py /path/to/frames --fps 10 --output general_vis_utils/output/video.mp4
"""
import cv2
from pathlib import Path
import argparse
import re
from typing import List


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def collect_images(folder: Path, pattern: str = "*") -> List[Path]:
    files = [
        file_path
        for file_path in folder.glob(pattern)
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=natural_sort_key)


def make_video_from_images(
    image_folder: str,
    output_path: str = "output.mp4",
    fps: int = 30,
    pattern: str = "*"
):
    """
    Create an MP4 video from a sequence of images.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Output video file path
        fps: Frames per second for output video
        pattern: Glob pattern to pre-filter files (default: *)
    """
    folder = Path(image_folder)
    if not folder.exists():
        print(f"Error: Folder '{image_folder}' does not exist.")
        return False
    
    # Get all supported images matching pattern and sort them naturally
    image_files = collect_images(folder, pattern)
    
    if len(image_files) == 0:
        print(
            f"Error: No supported images (.jpg/.jpeg/.png) found matching pattern '{pattern}' in '{image_folder}'"
        )
        return False
    
    print(f"Found {len(image_files)} images")
    print(f"First image: {image_files[0].name}")
    print(f"Last image: {image_files[-1].name}")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return False
    
    height, width = first_img.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    print(f"Output FPS: {fps}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{output_path}'")
        return False
    
    # Write each image to video
    print(f"Writing video to '{output_path}'...")
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path.name}, skipping...")
            continue
        
        # Ensure image has correct dimensions
        if img.shape[:2] != (height, width):
            print(f"Warning: Image {img_path.name} has different dimensions, resizing...")
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"  Progress: {i + 1}/{len(image_files)} frames", end='\r')
    
    print()  # New line after progress
    out.release()
    
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Video created successfully: {output_path} ({size_mb:.2f} MB)")
        return True
    else:
        print("Error: Video file was not created")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create an MP4 video from an image sequence (jpg/jpeg/png)."
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to the folder containing image sequence",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.mp4",
        help="Output MP4 file path (default: output.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate (default: 30)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Optional glob pattern to pre-filter files (default: *)",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        print("Error: --fps must be a positive integer")
        return 1

    success = make_video_from_images(
        image_folder=args.image_folder,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
