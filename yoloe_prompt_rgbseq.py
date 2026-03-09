#!/usr/bin/env python3
"""
Run YOLOE with prompt classes on an RGB image sequence and save annotated video.

Example:
  python yoloe_prompt_rgbseq.py \
    --model yoloe-11l-seg-pf.pt \
    --rgb-dir /path/to/RGB \
    --output-video /path/to/output/annotated.mp4 \
    --prompts "blue mug,banana,mustard bottle" \
    --fps 30 --conf 0.3 --iou 0.5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLOE


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def collect_rgb_images(folder: Path) -> List[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=natural_sort_key)


def parse_prompts(prompt_text: str) -> List[str]:
    prompts = [item.strip() for item in prompt_text.split(",")]
    prompts = [item for item in prompts if item]
    return prompts


def run_sequence(
    model_path: str,
    rgb_dir: str,
    output_video: str,
    prompts: List[str],
    fps: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    tracker: str,
) -> None:
    rgb_folder = Path(rgb_dir)
    if not rgb_folder.exists() or not rgb_folder.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist or is not a folder: {rgb_dir}")

    frame_paths = collect_rgb_images(rgb_folder)
    if not frame_paths:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_video}")

    model = YOLOE(model_path)
    if prompts:
        model.set_classes(prompts)

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Prompts ({len(prompts)}): {prompts}")
    print(f"[INFO] Frames: {len(frame_paths)}")
    print(f"[INFO] Output: {output_video}")

    try:
        for idx, frame_path in enumerate(frame_paths, start=1):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                print(f"[WARN] Skipping unreadable frame: {frame_path}")
                continue

            results = model.track(
                source=[frame_bgr],
                persist=True,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                tracker=tracker,
                device=device,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                masks=True,
                probs=False,
            )

            if annotated.shape[:2] != (height, width):
                annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_LINEAR)

            writer.write(annotated)

            if idx % 25 == 0 or idx == len(frame_paths):
                print(f"[INFO] Processed {idx}/{len(frame_paths)} frames", end="\r")
    finally:
        writer.release()

    print()
    print(f"[OK] Saved video: {output_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLOE with prompt classes on RGB sequence and save annotated video."
    )
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. '0' or 'cpu'")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model_path = "yoloe-11l-seg-pf.pt"
    rgb_dir = "/home/yehia/rizo/THUD_Robot/Synthetic_Scenes/House/static/Capture_1/RGB"
    output_video = "/home/yehia/rizo/code/yolo_sgg/general_vis_utils/close_vocab_house_c_1.mp4"
    fps = 30
    prompts = [
        "Chair",
        "Bench",
        "Table",
        "Sofa",
        "bookshelf",
        "Fridge",
        "Desk",
        "Bed"
    ]
    prompts = []

    run_sequence(
        model_path=model_path,
        rgb_dir=rgb_dir,
        output_video=output_video,
        prompts=prompts,
        fps=fps,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
