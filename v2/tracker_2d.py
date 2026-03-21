"""
2-D YOLO tracker — thin generator over YOLOE.

Yields ``(yolo_result, rgb_path, frame_idx)`` per frame.
No 3-D, no masks, no depth — just raw YOLO output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import cv2

from v2.loaders.base import DatasetLoader


@dataclass(slots=True)
class YoloFrameResult:
    result: object        # ultralytics result
    rgb_path: str
    frame_idx: int


def run_2d_tracker(
    loader: DatasetLoader,
    model_path: str = "yoloe-11l-seg.pt",
    conf: float = 0.25,
    iou: float = 0.5,
    verbose: bool = False,
    persistent: bool = True,
    agnostic_nms: bool = True,
    class_names: Optional[List[str]] = None,
    tracker_cfg: str = "botsort.yaml",
    device: str = "0",
) -> Generator[YoloFrameResult, None, None]:
    """Iterate over all frames via the loader, yield YOLO results.

    The model is loaded once; frames are read through ``loader.get_rgb()``.
    """
    from ultralytics import YOLOE as YOLOEModel

    model = YOLOEModel(model_path)
    if class_names:
        model.set_classes(class_names)

    n_frames = loader.get_num_frames()
    for idx in range(n_frames):
        rgb, rgb_path = loader.get_rgb(idx)
        if rgb is None:
            continue

        # YOLOE expects RGB
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb_input = rgb  # already RGB from loader
        else:
            rgb_input = rgb

        out = model.track(
            source=[rgb_input],
            tracker=tracker_cfg,
            device=device,
            conf=conf,
            verbose=verbose,
            persist=persistent,
            agnostic_nms=agnostic_nms,
        )
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        yield YoloFrameResult(result=res, rgb_path=rgb_path, frame_idx=idx)
