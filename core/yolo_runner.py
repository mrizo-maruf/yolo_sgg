"""
Unified YOLO tracking stream generator.

Works on a list of RGB image paths (any format: PNG, JPG, etc.)
so that every dataset loader feeds the same interface.
"""
from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import cv2

import YOLOE.utils as yutils
from ultralytics import YOLOE as YOLOEModel


def run_yolo_tracking_stream(
    rgb_paths: List[str],
    depth_paths: List[str],
    model_path: str = "yoloe-11l-seg-pf.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    verbose: bool = False,
    persistent: bool = True,
    agnostic_nms: bool = True,
    class_names_to_track: Optional[List[str]] = None,
) -> Generator[Tuple, None, None]:
    """Yield ``(yolo_result, rgb_path, depth_path)`` for each frame.

    Parameters
    ----------
    rgb_paths : list[str]
        Ordered list of RGB image paths.
    depth_paths : list[str]
        Corresponding depth image paths (same length & order).
    model_path : str
        Path to the YOLOE model weights.
    conf / iou : float
        Confidence and IoU thresholds for YOLO.
    verbose : bool
        If True, print detailed YOLO tracking information.
    persistent : bool
        If True, keep the tracker state between frames.
    agnostic_nms : bool
        If True, perform class-agnostic NMS.
    class_names_to_track : list[str] | None
        If provided, ``model.set_classes(...)`` is called to restrict
        detections to these classes (used e.g. by THUD Real Scenes).
    """

    model = YOLOEModel(model_path)

    if class_names_to_track:
        model.set_classes(class_names_to_track)
        print(f"[YOLO] Tracking only classes: {class_names_to_track}")

    for idx, rgb_p in enumerate(rgb_paths):
        rgb = cv2.imread(rgb_p)
        if rgb is None:
            print(f"[WARN] Could not read image: {rgb_p}")
            continue
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        out = model.track(
            source=[rgb_input],
            tracker=yutils.TRACKER_CFG,
            device=yutils.DEVICE,
            conf=conf,
            verbose=verbose,
            persist=persistent,
            agnostic_nms=agnostic_nms,
        )
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        dp = depth_paths[idx] if idx < len(depth_paths) else ""
        yield res, rgb_p, dp
