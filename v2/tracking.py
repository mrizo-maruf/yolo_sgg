"""
Main tracking + scene-graph generator.

``run_tracking_ssg`` is the single generator the runner consumes:

    for frame in run_tracking_ssg(loader, registry, cfg):
        merge(frame.local_graph, global_graph)
"""
from __future__ import annotations

import time
from typing import Generator, List, Optional, Set

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from .geometry import cam_to_world, compute_bbox, extract_points_from_mask
from .loaders.base import DatasetLoader
from .mask_utils import extract_ids_and_classes, filter_by_class, preprocess_masks
from .object_registry import GlobalObjectRegistry
from .ssg.edges import predict_edges
from .tracker_2d import YoloFrameResult, run_2d_tracker
from .types import CameraIntrinsics, TrackedFrame, TrackedObject


def run_tracking_ssg(
    loader: DatasetLoader,
    registry: GlobalObjectRegistry,
    cfg: DictConfig,
) -> Generator[TrackedFrame, None, None]:
    """Core pipeline generator.

    1. Run YOLOE 2-D tracker over loader frames.
    2. Per frame: preprocess masks → extract 3-D points → track globally.
    3. Build local scene graph (edge prediction).
    4. Yield ``TrackedFrame``.
    """
    intrinsics = loader.get_intrinsics()

    # Skip-class sets
    skip_exact: Set[str] = set(c.lower() for c in cfg.get("skip_classes", []))
    skip_sub: Set[str] = set(c.lower() for c in cfg.get("_substring_skip", []))

    # 2-D tracker config
    yolo_stream = run_2d_tracker(
        loader,
        model_path=cfg.get("yolo_model", "yoloe-11l-seg.pt"),
        conf=float(cfg.get("conf", 0.25)),
        iou=float(cfg.get("iou", 0.5)),
        verbose=bool(cfg.get("verbose", False)),
        persistent=bool(cfg.get("persistent", True)),
        agnostic_nms=bool(cfg.get("agnostic_nms", True)),
        class_names=list(cfg.get("class_names_to_track", [])) or None,
        tracker_cfg=cfg.get("tracker_cfg", "botsort.yaml"),
        device=str(cfg.get("device", "0")),
    )

    max_pts = int(cfg.get("max_points_per_obj", 2000))
    o3_nb = int(cfg.get("o3_nb_neighbors", 50))
    o3_std = float(cfg.get("o3std_ratio", 0.1))

    for yf in yolo_stream:
        timings: dict = {}
        res = yf.result
        idx = yf.frame_idx
        rgb_path = yf.rgb_path

        # YOLO inference time
        if hasattr(res, "speed") and isinstance(res.speed, dict):
            timings["yolo_ms"] = res.speed.get("inference", 0.0)

        # --- Depth ---
        depth_m = loader.get_depth(idx)
        if depth_m is None:
            continue

        # --- Pose ---
        T_w_c = loader.get_pose(idx)

        # --- Mask preprocessing ---
        t0 = time.perf_counter()
        masks = preprocess_masks(res, kernel_size=int(cfg.get("kernel_size", 9)))
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # --- IDs & class names ---
        track_ids, class_names = extract_ids_and_classes(res, len(masks))

        # --- Class filter ---
        masks, track_ids, class_names = filter_by_class(
            masks, track_ids, class_names, skip_exact, skip_sub,
        )

        # --- 3-D point extraction + global tracking ---
        t0 = time.perf_counter()
        # Use depth provider's custom unproject if available
        dp = loader.depth_provider
        unproject_fn = dp.unproject if dp is not None else None
        detections = _build_detections(
            masks, track_ids, class_names,
            depth_m, T_w_c, intrinsics, idx,
            max_pts, o3_nb, o3_std,
            unproject_fn=unproject_fn,
        )
        frame_objs = registry.process_frame(idx, detections)

        # Reprojection-visible objects
        detected_gids = {o.global_id for o in frame_objs}
        extra = registry.get_reprojection_visible(
            T_w_c, intrinsics, detected_gids,
        )
        frame_objs.extend(extra)
        timings["tracking_3d_ms"] = (time.perf_counter() - t0) * 1000

        # --- Local scene graph ---
        t0 = time.perf_counter()
        local_graph = nx.MultiDiGraph()
        for obj in frame_objs:
            local_graph.add_node(obj.global_id, data=obj)

        predict_edges(local_graph, frame_objs, T_w_c, depth_m, intrinsics)
        timings["edges_ms"] = (time.perf_counter() - t0) * 1000

        yield TrackedFrame(
            frame_idx=idx,
            rgb_path=rgb_path,
            depth_path="",
            objects=frame_objs,
            local_graph=local_graph,
            T_w_c=T_w_c,
            masks=masks,
            track_ids=track_ids,
            class_names=class_names,
            depth_m=depth_m,
            timings=timings,
        )


# ---------------------------------------------------------------------------
# Internal: build detection dicts for the registry
# ---------------------------------------------------------------------------

def _build_detections(
    masks: list,
    track_ids: np.ndarray,
    class_names: Optional[List[str]],
    depth_m: np.ndarray,
    T_w_c: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    frame_idx: int,
    max_pts: int,
    o3_nb: int,
    o3_std: float,
    unproject_fn=None,
) -> List[dict]:
    detections: List[dict] = []
    for i, (tid, mask) in enumerate(zip(track_ids, masks)):
        if mask is None:
            continue

        pts_cam = extract_points_from_mask(
            depth_m, mask, intrinsics,
            max_points=max_pts,
            o3_nb_neighbors=o3_nb, o3_std_ratio=o3_std,
            seed=int(tid),
            unproject_fn=unproject_fn,
        )
        if pts_cam.shape[0] == 0:
            continue

        pts_world = cam_to_world(pts_cam, T_w_c) if T_w_c is not None else pts_cam
        bbox = compute_bbox(pts_world, fast=True)

        cls = class_names[i] if class_names and i < len(class_names) else None
        detections.append({
            "yolo_track_id": int(tid),
            "points": pts_world,
            "bbox_3d": bbox,
            "class_name": cls,
            "mask": mask,
        })
    return detections
