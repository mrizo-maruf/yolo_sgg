"""Core tracking generator.

``run_tracking`` is a generator that processes each frame and yields a
``TrackedFrame``.  Callers decide what to do with each frame (graph
building, visualisation, metrics) without duplicating the pipeline.

Pipeline per frame:
    YOLO 2-D track → mask preprocessing → 3-D point extraction
        → 4-level global object matching → registry update → yield TrackedFrame

Global matching priority (ported from the proven old tracker):
    1. YOLO track-ID mapping (verified spatially, with distance fallback)
    2. Previous-frame temporal continuity (best overlap)
    3. Global registry lookup (re-observation after occlusion)
    4. New object creation
"""
from __future__ import annotations

import time
from typing import Generator, List, Optional, Set

from omegaconf import DictConfig

try:
    import torch
except Exception:
    torch = None

from core.geometry import clean_pcd, compute_bbox
from core.mask_utils import extract_ids_and_classes, filter_by_class, preprocess_masks
from core.new_yolo_runner import run_yolo_tracking_stream
from core.object_registry import GlobalObjectRegistry
from core.tracker_utils import (
    gpu_mem_mb,
    match_prev_frame,
    match_registry_reobservation,
    match_yolo_track_id,
)
from core.types import TrackedFrame, TrackedObject
from data_loaders.base import DatasetLoader


def run_tracking(
    loader: DatasetLoader,
    cfg: DictConfig,
    object_registry: Optional[GlobalObjectRegistry] = None,
) -> Generator[TrackedFrame, None, None]:
    """Core tracking generator.

    Parameters
    ----------
    loader : DatasetLoader
        Provides RGB, depth, pose, intrinsics.
    cfg : DictConfig
        Merged configuration.
    object_registry : GlobalObjectRegistry | None
        If None, a new one is created from cfg.

    Yields
    ------
    TrackedFrame
    """
    if object_registry is None:
        object_registry = _build_default_registry(cfg)

    intrinsics = loader.get_intrinsics()
    class_names_to_track = _resolve_open_vocab_classes(loader, cfg)
    skip_exact, skip_sub = _build_skip_sets(cfg)

    yolo_stream = run_yolo_tracking_stream(
        loader,
        model_path=cfg.get("yolo_model", "yoloe-11l-seg.pt"),
        conf=float(cfg.get("conf", 0.25)),
        iou=float(cfg.get("iou", 0.5)),
        verbose=bool(cfg.get("verbose", False)),
        persistent=bool(cfg.get("persistent", True)),
        agnostic_nms=bool(cfg.get("agnostic_nms", True)),
        class_names_to_track=class_names_to_track,
        tracker_cfg=cfg.get("tracker_cfg", "botsort.yaml"),
        device=str(cfg.get("device", "0")),
    )

    max_pts = int(cfg.get("max_points_per_obj", 2000))
    sample_ratio = float(cfg.get("sample_ratio", 0.5))
    o3_nb = int(cfg.get("o3_nb_neighbors", 50))
    o3_std = float(cfg.get("o3std_ratio", 0.1))
    kernel_size = int(cfg.get("kernel_size", 9))
    cuda_available = bool(torch is not None and torch.cuda.is_available())

    overlap_th = object_registry.overlap_threshold
    dist_th = object_registry.distance_threshold

    for yf in yolo_stream:
        timings: dict[str, float] = {}
        res = yf.result
        idx = yf.frame_idx
        rgb_path = yf.rgb_path

        _record_yolo_timings(yf, res, timings, cuda_available)

        T_w_c = loader.get_pose(idx)

        t0 = time.perf_counter()
        masks = preprocess_masks(res, kernel_size=kernel_size)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000
        _record_gpu(timings, "gpu_after_preprocess_mb", cuda_available)

        track_ids, class_names = extract_ids_and_classes(res, len(masks))
        masks, track_ids, class_names = filter_by_class(
            masks, track_ids, class_names, skip_exact, skip_sub,
        )

        t_tracking_3d = time.perf_counter()

        t0 = time.perf_counter()
        depth_m = loader.get_depth(idx)
        timings["depth_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        raw_pcds = loader.get_masked_pcds_from_depth(
            idx,
            depth_m,
            masks,
            max_points=max_pts,
            sample_ratio=sample_ratio,
        )
        timings["pcd_extract_ms"] = (time.perf_counter() - t0) * 1000
        _record_gpu(timings, "gpu_after_pcd_mb", cuda_available)

        object_registry.begin_frame(idx)
        matched_gids: Set[int] = set()
        frame_objs: List[TrackedObject] = []

        t0 = time.perf_counter()
        for i, (tid, mask, pts_world) in enumerate(zip(track_ids, masks, raw_pcds)):
            tracked_obj = _track_one_detection(
                i=i,
                tid=int(tid),
                mask=mask,
                pts_world=pts_world,
                class_names=class_names,
                object_registry=object_registry,
                matched_gids=matched_gids,
                overlap_th=overlap_th,
                dist_th=dist_th,
                o3_nb=o3_nb,
                o3_std=o3_std,
                frame_idx=idx,
            )
            if tracked_obj is not None:
                frame_objs.append(tracked_obj)
        timings["track_update_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        extra = object_registry.get_reprojection_visible(
            T_w_c, intrinsics, matched_gids,
        )
        visible_gids = set(matched_gids)
        if extra:
            frame_objs.extend(extra)
            visible_gids.update(o.global_id for o in extra)
        object_registry.end_frame(visible_gids)
        timings["reprojection_ms"] = (time.perf_counter() - t0) * 1000

        timings["tracking_3d_ms"] = (time.perf_counter() - t_tracking_3d) * 1000
        _record_gpu(timings, "gpu_after_tracking_3d_mb", cuda_available)

        yield TrackedFrame(
            frame_idx=idx,
            rgb_path=rgb_path,
            objects=frame_objs,
            masks=masks,
            track_ids=track_ids,
            class_names=class_names,
            depth_m=depth_m,
            T_w_c=T_w_c,
            timings=timings,
        )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _build_default_registry(cfg: DictConfig) -> GlobalObjectRegistry:
    return GlobalObjectRegistry(
        overlap_threshold=float(cfg.get("tracking_overlap_threshold", 0.1)),
        distance_threshold=float(cfg.get("tracking_distance_threshold", 1.0)),
        max_points=int(cfg.get("max_accumulated_points", 10000)),
        inactive_limit=int(cfg.get("tracking_inactive_limit", 0)),
        volume_ratio_threshold=float(cfg.get("tracking_volume_ratio_threshold", 0.1)),
        visibility_threshold=float(cfg.get("reprojection_visibility_threshold", 0.2)),
        merge_iou_threshold=float(cfg.get("merge_iou_threshold", 0.5)),
        merge_containment_threshold=float(cfg.get("merge_containment_threshold", 0.7)),
    )


def _resolve_open_vocab_classes(
    loader: DatasetLoader, cfg: DictConfig,
) -> Optional[List[str]]:
    """Return scene-specific open-vocab class list, or None for closed-set."""
    if not cfg.get("is_open_vocabulary", False):
        return None
    scene_name = loader.get_scene_name()
    scene_key = f"{scene_name}_class_names"
    classes = list(cfg.get(scene_key, [])) or None
    if classes is None:
        classes = list(cfg.get("class_names_to_track", [])) or None
    return classes


def _build_skip_sets(cfg: DictConfig) -> tuple[Set[str], Set[str]]:
    skip_exact: Set[str] = {c.lower() for c in cfg.get("skip_classes", [])}
    skip_sub: Set[str] = {c.lower() for c in cfg.get("_substring_skip", [])}
    return skip_exact, skip_sub


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _record_yolo_timings(yf, res, timings: dict, cuda_available: bool) -> None:
    """Populate yolo_ms / yolo_inference_ms / gpu_after_yolo_mb keys."""
    if getattr(yf, "yolo_wall_ms", 0.0) > 0.0:
        timings["yolo_ms"] = float(yf.yolo_wall_ms)
    if hasattr(res, "speed") and isinstance(res.speed, dict):
        timings["yolo_inference_ms"] = float(res.speed.get("inference", 0.0))
        if "yolo_ms" not in timings:
            timings["yolo_ms"] = timings["yolo_inference_ms"]
    _record_gpu(timings, "gpu_after_yolo_mb", cuda_available)


def _record_gpu(timings: dict, key: str, cuda_available: bool) -> None:
    mem = gpu_mem_mb(cuda_available)
    if mem is not None:
        timings[key] = mem


def _track_one_detection(
    *,
    i: int,
    tid: int,
    mask,
    pts_world,
    class_names,
    object_registry,
    matched_gids: Set[int],
    overlap_th: float,
    dist_th: float,
    o3_nb: int,
    o3_std: float,
    frame_idx: int,
) -> Optional[TrackedObject]:
    """Clean one detection's points, match against the registry, and
    update/register the object. Returns the per-frame TrackedObject
    view or None if the detection had no usable points.
    """
    if pts_world.shape[0] == 0:
        return None
    pts_world = clean_pcd(
        pts_world,
        o3_nb_neighbors=o3_nb,
        o3_std_ratio=o3_std,
        cluster_eps=0.2,
        cluster_min_samples=10,
        cluster_min_fraction=0.6,
    )
    if pts_world.shape[0] == 0:
        return None

    bbox = compute_bbox(pts_world, fast=True)
    cls = class_names[i] if class_names and i < len(class_names) else None

    gid = _match_detection_to_global_id(
        bbox=bbox,
        tid=tid,
        object_registry=object_registry,
        matched_gids=matched_gids,
        overlap_th=overlap_th,
        dist_th=dist_th,
    )

    if gid is None:
        gid = object_registry.new_id()
        object_registry.register_new(gid, pts_world, bbox, cls, mask, tid, frame_idx)
    else:
        object_registry.update_object(gid, pts_world, bbox, cls, mask, tid, frame_idx)

    matched_gids.add(gid)
    reg_obj = object_registry.objects[gid]
    return TrackedObject(
        global_id=gid,
        yolo_id=tid,
        class_name=reg_obj.get("class_name"),
        bbox_3d=reg_obj.get("bbox_3d"),
        mask=mask,
        first_seen=reg_obj["first_seen_frame"],
        last_seen=frame_idx,
        observation_count=reg_obj.get("observation_count", 0),
    )


def _match_detection_to_global_id(
    *,
    bbox,
    tid: int,
    object_registry,
    matched_gids: Set[int],
    overlap_th: float,
    dist_th: float,
) -> Optional[int]:
    """Run the 4-level matching cascade and return a gid (or None)."""
    gid = match_yolo_track_id(
        bbox, tid, object_registry, matched_gids, overlap_th, dist_th,
    )
    if gid is not None:
        return gid

    gid = match_prev_frame(
        bbox, object_registry, matched_gids, overlap_th, dist_th,
    )
    if gid is not None:
        return gid

    return match_registry_reobservation(
        bbox, object_registry, matched_gids, overlap_th, dist_th,
    )
