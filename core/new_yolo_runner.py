"""
YOLO tracking stream — thin generator over YOLOE.

Yields YoloFrameResult per frame. No 3-D, no masks, no depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional

from data_loaders.base import DatasetLoader


@dataclass(slots=True)
class YoloFrameResult:
    """Raw output from the 2-D YOLO tracker for one frame."""
    frame_idx: int
    result: object          # ultralytics result
    rgb_path: str


def run_yolo_tracking_stream(
    loader: DatasetLoader,
    model_path: str = "yoloe-11l-seg.pt",
    conf: float = 0.25,
    iou: float = 0.5,
    verbose: bool = False,
    persistent: bool = True,
    agnostic_nms: bool = True,
    class_names_to_track: Optional[List[str]] = None,
    tracker_cfg: str = "botsort.yaml",
    device: str = "0",
) -> Generator[YoloFrameResult, None, None]:
    """Iterate over all loader frames, yield YOLO tracking results.

    The model is loaded once; frames are read through ``loader.get_rgb()``.
    """
    from ultralytics import YOLOE as YOLOEModel

    model = YOLOEModel(model_path)
    if class_names_to_track:
        model.set_classes(class_names_to_track)
        print(f"[YOLO] Tracking only classes: {class_names_to_track}")

    n_frames = loader.get_num_frames()
    for idx in range(n_frames):
        rgb, rgb_path = loader.get_rgb(idx)
        print(f"[YOLO] Processing frame {idx}/{n_frames} ({rgb_path})")
        if rgb is None:
            continue

        out = model.track(
            source=[rgb],
            tracker=tracker_cfg,
            device=device,
            conf=conf,
            verbose=verbose,
            persist=persistent,
            agnostic_nms=agnostic_nms,
        )
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        yield YoloFrameResult(frame_idx=idx, result=res, rgb_path=rgb_path)
