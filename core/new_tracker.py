
"""
Core tracking generator.

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
from typing import Generator, List, Optional, Set, Tuple

import numpy as np
from omegaconf import DictConfig
try:
    import torch
except Exception:
    torch = None

from core.mask_utils import extract_ids_and_classes, filter_by_class, preprocess_masks
from core.geometry import aabb_containment, aabb_iou, clean_pcd, compute_bbox
from core.object_registry import GlobalObjectRegistry
from core.new_yolo_runner import run_yolo_tracking_stream
from core.types import BBox3D, TrackedFrame, TrackedObject
from data_loaders.base import DatasetLoader


# ---------------------------------------------------------------------------
# Overlap scoring (mirrors the old _compute_overlap_score faithfully)
# ---------------------------------------------------------------------------

def _compute_overlap_score(
    det_bbox: Optional[BBox3D],
    reg_bbox: Optional[BBox3D],
    distance_threshold: float,
    overlap_threshold: float,
) -> Tuple[float, float, bool]:
    """Compute matching score between a detection and a registry object.

    Returns
    -------
    effective_score : float
        max(aabb_iou, aabb_containment).  0.0 when rejected.
    centroid_distance : float
        Euclidean distance between bbox centres.
    is_candidate : bool
        True when the score passes the overlap or containment threshold.
    """
    if det_bbox is None or reg_bbox is None:
        return 0.0, float("inf"), False

    dist = float(np.linalg.norm(det_bbox.center - reg_bbox.center))
    if dist > distance_threshold:
        return 0.0, dist, False

    # Compute containment early — needed to handle partial re-observation
    # (e.g. partial view of a sofa matching the full sofa in the registry).
    iou = aabb_iou(det_bbox, reg_bbox)
    containment = aabb_containment(det_bbox, reg_bbox)

    # Size-similarity check — reject if ≥2 extents wildly differ.
    # BUT skip when one bbox is largely contained in the other, because
    # that indicates a partial re-observation of an existing object.
    if containment < 0.3:
        ext_a = det_bbox.obb_extent
        ext_b = reg_bbox.obb_extent
        ratios = np.minimum(ext_a, ext_b) / (np.maximum(ext_a, ext_b) + 1e-6)
        if np.sum(ratios < 0.15) >= 2:
            return 0.0, dist, False

    effective_score = max(iou, containment)
    is_candidate = iou >= overlap_threshold or containment >= 0.3

    return effective_score, dist, is_candidate


def _gpu_mem_mb(cuda_available: bool) -> Optional[float]:
    """Return current allocated CUDA memory in MB."""
    if not cuda_available:
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.memory_allocated() / (1024 ** 2))


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
    # --- Build registry if not supplied ---
    if object_registry is None:
        object_registry = GlobalObjectRegistry(
            overlap_threshold=float(cfg.get("tracking_overlap_threshold", 0.1)),
            distance_threshold=float(cfg.get("tracking_distance_threshold", 1.0)),
            max_points=int(cfg.get("max_accumulated_points", 10000)),
            inactive_limit=int(cfg.get("tracking_inactive_limit", 0)),
            volume_ratio_threshold=float(cfg.get("tracking_volume_ratio_threshold", 0.1)),
            visibility_threshold=float(cfg.get("reprojection_visibility_threshold", 0.2)),
            merge_iou_threshold=float(cfg.get("merge_iou_threshold", 0.5)),
            merge_containment_threshold=float(cfg.get("merge_containment_threshold", 0.7)),
        )

    intrinsics = loader.get_intrinsics()

    # --- Resolve class names for open-vocabulary mode ---
    class_names_to_track = None
    if cfg.get("is_open_vocabulary", False):
        scene_name = loader.get_scene_name()
        # Look for scene-specific class list in config
        scene_classes_key = f"{scene_name}_class_names"
        class_names_to_track = list(cfg.get(scene_classes_key, [])) or None
        if class_names_to_track is None:
            class_names_to_track = list(cfg.get("class_names_to_track", [])) or None

    # --- Skip-class sets ---
    skip_exact: Set[str] = set(c.lower() for c in cfg.get("skip_classes", []))
    skip_sub: Set[str] = set(c.lower() for c in cfg.get("_substring_skip", []))

    # --- YOLO tracking stream ---
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
    cuda_available = bool(torch is not None and torch.cuda.is_available())

    # --- Per-frame loop ---
    for yf in yolo_stream:
        timings: dict[str, float] = {}
        res = yf.result
        idx = yf.frame_idx
        rgb_path = yf.rgb_path

        # YOLO inference time
        if hasattr(res, "speed") and isinstance(res.speed, dict):
            timings["yolo_ms"] = res.speed.get("inference", 0.0)
        gpu_after_yolo = _gpu_mem_mb(cuda_available)
        if gpu_after_yolo is not None:
            timings["gpu_after_yolo_mb"] = gpu_after_yolo

        # --- Pose ---
        T_w_c = loader.get_pose(idx)

        # --- Mask preprocessing ---
        t0 = time.perf_counter()
        masks = preprocess_masks(res, kernel_size=int(cfg.get("kernel_size", 9)))
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000
        gpu_after_preprocess = _gpu_mem_mb(cuda_available)
        if gpu_after_preprocess is not None:
            timings["gpu_after_preprocess_mb"] = gpu_after_preprocess

        # --- IDs & class names ---
        track_ids, class_names = extract_ids_and_classes(res, len(masks))

        # --- Class filter ---
        masks, track_ids, class_names = filter_by_class(
            masks, track_ids, class_names, skip_exact, skip_sub,
        )

        # --- 3-D point extraction + global tracking ---
        t_tracking_3d = time.perf_counter()

        # Depth fetch and mask→PCD extraction are timed separately to
        # better expose where 3-D latency is spent.
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
        gpu_after_pcd = _gpu_mem_mb(cuda_available)
        if gpu_after_pcd is not None:
            timings["gpu_after_pcd_mb"] = gpu_after_pcd

        # -- Per-frame state --
        object_registry.begin_frame(idx)
        matched_gids: Set[int] = set()
        frame_objs: List[TrackedObject] = []

        overlap_th = object_registry.overlap_threshold
        dist_th = object_registry.distance_threshold

        t0 = time.perf_counter()
        for i, (tid, mask, pts_world) in enumerate(
            zip(track_ids, masks, raw_pcds)
        ):
            if pts_world.shape[0] == 0:
                continue
            pts_world = clean_pcd(
                pts_world, o3_nb_neighbors=o3_nb, o3_std_ratio=o3_std,
                cluster_eps=0.2, cluster_min_samples=10,
                cluster_min_fraction=0.6,
            )
            if pts_world.shape[0] == 0:
                continue
            bbox = compute_bbox(pts_world, fast=True)
            cls = class_names[i] if class_names and i < len(class_names) else None

            # ═══════════════════════════════════════════════════════════
            # 4-LEVEL GLOBAL MATCHING
            # ═══════════════════════════════════════════════════════════
            gid: Optional[int] = None

            # LEVEL 1: YOLO track-ID mapping (verified spatially)
            if tid >= 0 and tid in object_registry.yolo_to_global:
                candidate = object_registry.yolo_to_global[tid]
                if candidate not in matched_gids and candidate in object_registry.objects:
                    existing = object_registry.objects[candidate]
                    score, cdist, is_match = _compute_overlap_score(
                        bbox, existing.get("bbox_3d"), dist_th, overlap_th,
                    )
                    # Accept if overlap passes OR objects are close enough
                    if is_match or cdist < dist_th * 0.5:
                        gid = candidate

            # LEVEL 2: Previous-frame temporal continuity
            if gid is None:
                best_gid, best_score = None, 0.0
                for cand_gid, prev_obj in object_registry.prev_frame.items():
                    if cand_gid in matched_gids:
                        continue
                    score, cdist, is_cand = _compute_overlap_score(
                        bbox, prev_obj.get("bbox_3d"), dist_th, overlap_th,
                    )
                    if is_cand and score > best_score:
                        best_score = score
                        best_gid = cand_gid
                if best_gid is not None:
                    gid = best_gid

            # LEVEL 3: Global registry (re-observation after occlusion)
            # Use a relaxed distance threshold proportional to the existing
            # object size so that partial views of large objects (sofa, table)
            # whose centroid is offset can still match.
            if gid is None:
                best_gid, best_score = None, 0.0
                for cand_gid, obj in object_registry.objects.items():
                    if cand_gid in matched_gids:
                        continue
                    existing_bbox = obj.get("bbox_3d")
                    level3_dist = dist_th
                    if existing_bbox is not None:
                        max_extent = float(np.max(existing_bbox.obb_extent))
                        level3_dist = max(dist_th, max_extent * 0.75)
                    score, cdist, is_cand = _compute_overlap_score(
                        bbox, existing_bbox, level3_dist, overlap_th,
                    )
                    if is_cand and score > best_score:
                        best_score = score
                        best_gid = cand_gid
                if best_gid is not None:
                    gid = best_gid

            # LEVEL 4: No match — create new object
            if gid is None:
                gid = object_registry.new_id()
                object_registry.register_new(
                    gid, pts_world, bbox, cls, mask, int(tid), idx,
                )
            else:
                object_registry.update_object(
                    gid, pts_world, bbox, cls, mask, int(tid), idx,
                )

            matched_gids.add(gid)
            reg_obj = object_registry.objects[gid]

            frame_objs.append(TrackedObject(
                global_id=gid,
                yolo_id=int(tid),
                class_name=reg_obj.get("class_name"),
                bbox_3d=reg_obj.get("bbox_3d"),
                mask=mask,
                first_seen=reg_obj["first_seen_frame"],
                last_seen=idx,
                observation_count=reg_obj.get("observation_count", 0),
            ))
        timings["track_update_ms"] = (time.perf_counter() - t0) * 1000

        # Reprojection-visible objects
        t0 = time.perf_counter()
        detected_gids = matched_gids
        extra = object_registry.get_reprojection_visible(
            T_w_c, intrinsics, detected_gids,
        )
        visible_gids = set(matched_gids)
        if extra:
            frame_objs.extend(extra)
            visible_gids.update(o.global_id for o in extra)
        object_registry.end_frame(visible_gids)
        timings["reprojection_ms"] = (time.perf_counter() - t0) * 1000

        # Post-frame merge: collapse duplicate objects whose bboxes now
        # significantly overlap (e.g. a partial re-observation that was
        # initially registered as a new object and later grew to match
        # the original).
        # merged = object_registry.merge_overlapping_objects()
        # if merged:
        #     # Remap frame_objs so downstream consumers see the survivor IDs
        #     survivor_map = {absorbed: survivor for absorbed, survivor in merged}
        #     for obj in frame_objs:
        #         if obj.global_id in survivor_map:
        #             obj.global_id = survivor_map[obj.global_id]

        timings["tracking_3d_ms"] = (time.perf_counter() - t_tracking_3d) * 1000
        gpu_after_tracking_3d = _gpu_mem_mb(cuda_available)
        if gpu_after_tracking_3d is not None:
            timings["gpu_after_tracking_3d_mb"] = gpu_after_tracking_3d

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
