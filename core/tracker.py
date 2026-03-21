"""
Core tracking loop — shared by ``run.py`` and ``bench.py``.

The function :func:`run_tracking` is a **generator** that processes each
frame and yields a :class:`TrackedFrame` dataclass.  Callers decide what
to do with each frame (SSG edge prediction, GT matching, visualisation,
…) without duplicating the YOLO → mask → 3-D tracking pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set

import numpy as np
import torch
from omegaconf import OmegaConf

import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry

from core.helpers import apply_class_filter, extract_yolo_ids
from core.yolo_runner import run_yolo_tracking_stream


# ---------------------------------------------------------------------------
# Per-frame output
# ---------------------------------------------------------------------------

@dataclass
class TrackedFrame:
    """Everything produced by the core tracker for a single frame."""
    frame_idx: int
    rgb_path: str
    depth_path: str

    # Tracking outputs
    frame_objs: List[Dict]
    scene_graph: Any               # nx.MultiDiGraph (or None when not using SSG)
    masks_clean: List[np.ndarray]
    track_ids: np.ndarray
    class_names: Optional[List[str]]

    # Scene data
    depth_m: np.ndarray
    T_w_c: Optional[np.ndarray]

    # Raw YOLO result (useful for timing, extra attributes)
    yolo_result: Any = None

    # Timing information (ms)
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core tracking generator
# ---------------------------------------------------------------------------

def run_tracking(
    rgb_paths: List[str],
    depth_paths: List[str],
    depth_cache: Dict[str, np.ndarray],
    poses: Optional[List[np.ndarray]],
    cfg: OmegaConf,
    object_registry: Optional[GlobalObjectRegistry] = None,
    class_names_to_track: Optional[List[str]] = None,
    point_extractor: Optional[Callable] = None,
) -> Generator[TrackedFrame, None, None]:
    """Core tracking generator.

    Parameters
    ----------
    rgb_paths, depth_paths : list[str]
        Ordered image paths (must be same length).
    depth_cache : dict[str, np.ndarray]
        Pre-loaded depth maps keyed by path.
    poses : list[np.ndarray] | None
        Camera-to-world 4×4 poses (one per frame, or *None*).
    cfg : OmegaConf
        Merged configuration (default + dataset overrides + CLI).
    object_registry : GlobalObjectRegistry | None
        If *None* a new one is created from *cfg*.
    class_names_to_track : list[str] | None
        Passed to :func:`run_yolo_tracking_stream` (``model.set_classes``).

    Yields
    ------
    TrackedFrame
    """
    # --- Build registry if not supplied --------------------------------------
    if object_registry is None:
        object_registry = GlobalObjectRegistry(
            overlap_threshold=float(cfg.tracking_overlap_threshold),
            distance_threshold=float(cfg.tracking_distance_threshold),
            max_points_per_object=int(cfg.max_accumulated_points),
            inactive_frames_limit=int(cfg.tracking_inactive_limit),
            volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
            reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
        )
    
    # --- YOLO tracking stream ------------------------------------------------
    results_stream = run_yolo_tracking_stream(
        rgb_paths,
        depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
        verbose=bool(cfg.verbose),
        persistent=bool(cfg.persistent),
        agnostic_nms=bool(cfg.agnostic_nms),
        class_names_to_track=class_names_to_track,
        tracker_cfg=cfg.get("tracker_cfg"),
        device=cfg.get("device"),
    )

    skip_set: Set[str] = set(c.lower() for c in cfg.get("skip_classes", []))
    _substring_skip: Set[str] = set(c.lower() for c in cfg.get("_substring_skip", []))


    frame_idx = 0
    for yolo_res, rgb_path, depth_path in results_stream:
        timings: Dict[str, float] = {}

        # YOLO inference time (from the result object)
        if hasattr(yolo_res, "speed") and isinstance(yolo_res.speed, dict):
            timings["yolo"] = yolo_res.speed.get("inference", 0.0)

        # --- Depth -----------------------------------------------------------
        depth_m = depth_cache.get(depth_path)
        if depth_m is None:
            frame_idx += 1
            continue

        # --- Camera pose -----------------------------------------------------
        T_w_c = None
        if poses is not None and len(poses) > 0:
            T_w_c = poses[min(frame_idx, len(poses) - 1)]

        # --- Mask preprocessing ----------------------------------------------
        t0 = time.perf_counter()
        _, masks_clean = yutils.preprocess_mask(
            yolo_res=yolo_res,
            index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=bool(cfg.fast_mask),
        )
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        # --- Track IDs & class names -----------------------------------------
        track_ids, class_names = extract_yolo_ids(yolo_res, masks_clean)

        # --- Class filter ----------------------------------------------------
        masks_clean, track_ids, class_names = apply_class_filter(
            masks_clean, track_ids, class_names, skip_set, _substring_skip,
        )

        # --- 3-D objects with global tracking --------------------------------
        t0 = time.perf_counter()
        frame_objs, current_graph = yutils.create_3d_objects_with_tracking(
            track_ids,
            masks_clean,
            int(cfg.max_points_per_obj),
            depth_m,
            T_w_c,
            frame_idx,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            object_registry=object_registry,
            class_names=class_names,
            point_extractor=point_extractor,
        )
        timings["create_3d"] = (time.perf_counter() - t0) * 1000

        yield TrackedFrame(
            frame_idx=frame_idx,
            rgb_path=rgb_path,
            depth_path=depth_path,
            frame_objs=frame_objs,
            scene_graph=current_graph,
            masks_clean=masks_clean,
            track_ids=track_ids,
            class_names=class_names,
            depth_m=depth_m,
            T_w_c=T_w_c,
            yolo_result=yolo_res,
            timings=timings,
        )

        frame_idx += 1
