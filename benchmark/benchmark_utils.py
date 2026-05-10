"""
benchmark_utils.py
==================
Utility functions shared by the benchmark runner.

Covers:
  - Timing / GPU-memory constants and helpers
  - Match-mode resolution
  - Geometry helpers (3-D bbox → 2-D projection, mask → bbox)
  - GT / Pred instance builders
  - Loader-kwargs builder
  - Frame visualisation
  - 3-D artifact saving
  - Performance dict / printing helpers
  - Multi-scene aggregate helpers
  - Cross-scene comparison chart
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None

from core.types import CameraIntrinsics, TrackedFrame
from metrics.tracking_metrics import GTInstance, PredInstance
from benchmark.visualization import (
    draw_boxes_with_labels,
    draw_masks_with_labels,
    visualize_3d_bboxes,
    visualize_matching,
    visualize_matching_boxes,
)


# ═══════════════════════════════════════════════════════════════════════════
# Timing / GPU constants
# ═══════════════════════════════════════════════════════════════════════════

TIMING_KEY_MAP = {
    "yolo_ms": "yolo",
    "yolo_inference_ms": "yolo_inference",
    "preprocess_ms": "preprocess",
    "depth_ms": "depth",
    "pcd_extract_ms": "pcd_extract",
    "track_update_ms": "track_update",
    "reprojection_ms": "reprojection",
    "tracking_3d_ms": "tracking_3d",
    "graph_ms": "graph",
    "graph_build_nodes_ms": "graph_build_nodes",
    "graph_merge_ms": "graph_merge",
    "graph_predict_basic_ms": "graph_predict_basic",
    "graph_predict_baseline_ms": "graph_predict_baseline",
    "graph_predict_vlsat_ms": "graph_predict_vlsat",
}

GPU_KEY_MAP = {
    "gpu_after_yolo_mb": "after_yolo",
    "gpu_after_preprocess_mb": "after_preprocess",
    "gpu_after_pcd_mb": "after_pcd",
    "gpu_after_tracking_3d_mb": "after_tracking_3d",
    "gpu_after_graph_mb": "after_graph",
}


def cuda_mem_mb() -> Optional[float]:
    if _torch is None or not _torch.cuda.is_available():
        return None
    _torch.cuda.synchronize()
    return float(_torch.cuda.memory_allocated() / (1024 ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_match_mode(cfg) -> str:
    mode = str(cfg.get("match_mode", "mask2d")).strip().lower()
    valid = {"mask2d", "bbox2d", "bbox3d"}
    if mode not in valid:
        raise ValueError(f"Invalid match_mode '{mode}'. Choose one of: {sorted(valid)}")
    return mode


def similarity_label(match_mode: str) -> str:
    return {
        "mask2d": "mask IoU",
        "bbox2d": "2D bbox IoU",
        "bbox3d": "3D AABB IoU",
    }.get(match_mode, "IoU")


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def bbox3d_to_xyzxyz(bbox) -> Optional[Tuple[float, ...]]:
    if bbox is None:
        return None
    mn = getattr(bbox, "aabb_min", None)
    mx = getattr(bbox, "aabb_max", None)
    if mn is None or mx is None:
        return None
    return (
        float(mn[0]), float(mn[1]), float(mn[2]),
        float(mx[0]), float(mx[1]), float(mx[2]),
    )


def corners_from_xyzxyz(b: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    xn, yn, zn, xx, yx, zx = b
    return np.array([
        [xn, yn, zn], [xn, yn, zx], [xn, yx, zn], [xn, yx, zx],
        [xx, yn, zn], [xx, yn, zx], [xx, yx, zn], [xx, yx, zx],
    ], dtype=np.float64)


def project_xyzxyz_to_2d(
    bbox_xyzxyz: Optional[Tuple[float, ...]],
    T_w_c: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    min_depth: float = 1e-3,
) -> Optional[Tuple[float, float, float, float]]:
    if bbox_xyzxyz is None or T_w_c is None:
        return None

    try:
        T_c_w = np.linalg.inv(T_w_c)
    except np.linalg.LinAlgError:
        return None

    corners_w = corners_from_xyzxyz(bbox_xyzxyz)
    R = T_c_w[:3, :3]
    t = T_c_w[:3, 3]
    corners_c = (corners_w @ R.T) + t

    valid = corners_c[:, 2] > min_depth
    if not np.any(valid):
        return None

    z = corners_c[valid, 2]
    u = corners_c[valid, 0] / z * intrinsics.fx + intrinsics.cx
    v = corners_c[valid, 1] / z * intrinsics.fy + intrinsics.cy

    x1 = float(np.clip(np.min(u), 0.0, intrinsics.width - 1))
    y1 = float(np.clip(np.min(v), 0.0, intrinsics.height - 1))
    x2 = float(np.clip(np.max(u), 0.0, intrinsics.width - 1))
    y2 = float(np.clip(np.max(v), 0.0, intrinsics.height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_from_mask(mask: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
    if mask is None:
        return None
    m = mask.astype(bool)
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


# ═══════════════════════════════════════════════════════════════════════════
# Instance builders
# ═══════════════════════════════════════════════════════════════════════════

def prepare_gt_instances(
    gt_instances: List[GTInstance],
    tf: TrackedFrame,
    intrinsics: CameraIntrinsics,
    match_mode: str,
) -> List[GTInstance]:
    prepared: List[GTInstance] = []
    for gt in gt_instances:
        bbox_xyxy = gt.bbox_xyxy
        if bbox_xyxy is None and gt.bbox_xyzxyz is not None and match_mode in {"bbox2d", "bbox3d"}:
            bbox_xyxy = project_xyzxyz_to_2d(gt.bbox_xyzxyz, tf.T_w_c, intrinsics)
        if bbox_xyxy is None and gt.mask is not None and match_mode == "bbox2d":
            bbox_xyxy = bbox_from_mask(gt.mask)
        prepared.append(GTInstance(
            track_id=gt.track_id,
            class_name=gt.class_name,
            mask=gt.mask,
            bbox_xyxy=bbox_xyxy,
            bbox_xyzxyz=gt.bbox_xyzxyz,
        ))
    return prepared


def build_pred_instances(
    tf: TrackedFrame,
    intrinsics: CameraIntrinsics,
    match_mode: str,
    include_reprojected_masks: bool = False,
) -> List[PredInstance]:
    """Convert ``TrackedFrame`` outputs to a ``PredInstance`` list."""
    preds: List[PredInstance] = []
    for obj in tf.objects:
        # Reprojection-only masks are stale; skip them for mask2d by default.
        if (
            match_mode == "mask2d"
            and not include_reprojected_masks
            and int(obj.yolo_id) < 0
        ):
            continue

        mask = obj.mask
        if mask is None and tf.track_ids is not None and tf.masks:
            yolo_tid = obj.yolo_id
            if yolo_tid is not None and yolo_tid >= 0:
                idxs = np.where(tf.track_ids == yolo_tid)[0]
                if len(idxs) > 0 and idxs[0] < len(tf.masks):
                    mask = tf.masks[idxs[0]]

        bbox_xyzxyz = bbox3d_to_xyzxyz(obj.bbox_3d)
        bbox_xyxy = project_xyzxyz_to_2d(bbox_xyzxyz, tf.T_w_c, intrinsics)
        if bbox_xyxy is None and mask is not None:
            bbox_xyxy = bbox_from_mask(mask)

        preds.append(PredInstance(
            pred_id=obj.global_id,
            class_name=obj.class_name,
            mask=mask,
            bbox_xyxy=bbox_xyxy,
            bbox_xyzxyz=bbox_xyzxyz,
        ))
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Loader kwargs builder
# ═══════════════════════════════════════════════════════════════════════════

def build_loader_kwargs(dataset_name: str, cfg) -> dict:
    """Build dataset-specific keyword arguments for loader construction."""
    kwargs: dict = {}

    skip_labels = cfg.get("loader_skip_labels")
    if skip_labels:
        kwargs["skip_labels"] = set(str(s).lower() for s in skip_labels)

    if dataset_name == "isaacsim":
        kwargs["image_width"] = int(cfg.get("image_width", 1280))
        kwargs["image_height"] = int(cfg.get("image_height", 720))
        kwargs["focal_length"] = float(cfg.get("focal_length", 50))
        kwargs["horizontal_aperture"] = float(cfg.get("horizontal_aperture", 80))
        kwargs["vertical_aperture"] = float(cfg.get("vertical_aperture", 45))
    elif dataset_name == "thud_synthetic":
        kwargs["depth_scale"] = float(cfg.get("depth_scale", 1000.0))
    elif dataset_name == "coda":
        kwargs["max_depth"] = float(cfg.get("max_depth", 80.0))
    elif dataset_name == "scanetpp":
        kwargs["fx"] = float(cfg.get("fx", 692.52))
        kwargs["fy"] = float(cfg.get("fy", 693.83))
        kwargs["cx"] = float(cfg.get("cx", 459.76))
        kwargs["cy"] = float(cfg.get("cy", 344.76))
        kwargs["image_width"] = int(cfg.get("image_width", 920))
        kwargs["image_height"] = int(cfg.get("image_height", 690))

    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def visualize_frame(
    rgb_path: str,
    gt_instances: List[GTInstance],
    pred_instances: List[PredInstance],
    mapping: Dict,
    ious: Dict,
    frame_idx: int,
    vis_cfg: dict,
    vis_save: Optional[str],
    match_mode: str,
) -> None:
    """Render 2-D matching + tracking panels."""
    import cv2 as cv

    rgb = cv.imread(rgb_path)
    if rgb is None:
        return

    save_match = None
    save_2d = None
    save_mode_dir = vis_save
    if vis_save:
        save_mode_dir = os.path.join(vis_save, match_mode)
        os.makedirs(save_mode_dir, exist_ok=True)
        save_match = os.path.join(save_mode_dir, f"matching_{frame_idx:06d}.png")
        save_2d = os.path.join(save_mode_dir, f"tracking_2d_{frame_idx:06d}.png")

    show = vis_cfg.get("show_windows", True)

    if match_mode == "mask2d":
        if vis_cfg.get("show_matching", True):
            visualize_matching(
                rgb=rgb,
                gt_masks=[g.mask for g in gt_instances],
                gt_ids=[g.track_id for g in gt_instances],
                gt_labels=[g.class_name for g in gt_instances],
                pred_masks=[p.mask for p in pred_instances],
                pred_ids=[p.pred_id for p in pred_instances],
                pred_labels=[p.class_name or "" for p in pred_instances],
                mapping=mapping,
                ious=ious,
                frame_idx=frame_idx,
                save_path=save_match,
                show=show,
            )

        if vis_cfg.get("show_2d", True):
            labels = [f"G:{p.pred_id} {p.class_name or ''}" for p in pred_instances]
            overlay = draw_masks_with_labels(
                rgb,
                [p.mask for p in pred_instances],
                [p.pred_id for p in pred_instances],
                labels,
                title=f"Frame {frame_idx}",
            )
            if show:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(14, 8))
                plt.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
                plt.title(f"Tracking – Frame {frame_idx}")
                plt.axis("off")
                plt.tight_layout()
                if save_2d:
                    plt.savefig(save_2d, dpi=150, bbox_inches="tight")
                plt.show()
            elif save_2d:
                cv.imwrite(save_2d, overlay)
        return

    # bbox2d / bbox3d visualizations
    gt_boxes = [g.bbox_xyxy for g in gt_instances]
    pred_boxes = [p.bbox_xyxy for p in pred_instances]

    if vis_cfg.get("show_matching", True):
        visualize_matching_boxes(
            rgb=rgb,
            gt_boxes=gt_boxes,
            gt_ids=[g.track_id for g in gt_instances],
            gt_labels=[g.class_name for g in gt_instances],
            pred_boxes=pred_boxes,
            pred_ids=[p.pred_id for p in pred_instances],
            pred_labels=[p.class_name or "" for p in pred_instances],
            mapping=mapping,
            ious=ious,
            frame_idx=frame_idx,
            save_path=save_match,
            show=show,
        )

    if vis_cfg.get("show_2d", True):
        labels = [f"G:{p.pred_id} {p.class_name or ''}" for p in pred_instances]
        overlay = draw_boxes_with_labels(
            rgb=rgb,
            boxes=pred_boxes,
            ids=[p.pred_id for p in pred_instances],
            labels=labels,
            title=f"Frame {frame_idx}",
        )
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 8))
            plt.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
            plt.title(f"Tracking ({match_mode}) – Frame {frame_idx}")
            plt.axis("off")
            plt.tight_layout()
            if save_2d:
                plt.savefig(save_2d, dpi=150, bbox_inches="tight")
            plt.show()
        elif save_2d:
            cv.imwrite(save_2d, overlay)

    if match_mode == "bbox3d" and save_mode_dir:
        save_3d_frame_artifacts(
            out_dir=save_mode_dir,
            frame_idx=frame_idx,
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            mapping=mapping,
            ious=ious,
        )

    if match_mode == "bbox3d" and show and vis_cfg.get("show_3d", False):
        gt_bboxes = [
            {"track_id": g.track_id, "aabb": list(g.bbox_xyzxyz)}
            for g in gt_instances
            if g.bbox_xyzxyz is not None
        ]
        pred_bboxes = [
            {"track_id": p.pred_id, "aabb": list(p.bbox_xyzxyz)}
            for p in pred_instances
            if p.bbox_xyzxyz is not None
        ]
        if gt_bboxes or pred_bboxes:
            visualize_3d_bboxes(
                gt_bboxes=gt_bboxes,
                pred_bboxes=pred_bboxes,
                frame_idx=frame_idx,
                window_title=f"3D Matching – Frame {frame_idx}",
            )


def save_3d_frame_artifacts(
    out_dir: str,
    frame_idx: int,
    gt_instances: List[GTInstance],
    pred_instances: List[PredInstance],
    mapping: Dict[int, int],
    ious: Dict[int, float],
) -> None:
    art_dir = Path(out_dir) / "bbox3d_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_idx": int(frame_idx),
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "ious": {str(k): float(v) for k, v in ious.items()},
        "gt": [
            {
                "track_id": int(g.track_id),
                "class_name": str(g.class_name),
                "bbox_xyzxyz": list(g.bbox_xyzxyz) if g.bbox_xyzxyz is not None else None,
                "bbox_xyxy": list(g.bbox_xyxy) if g.bbox_xyxy is not None else None,
            }
            for g in gt_instances
        ],
        "pred": [
            {
                "pred_id": int(p.pred_id),
                "class_name": str(p.class_name) if p.class_name is not None else "",
                "bbox_xyzxyz": list(p.bbox_xyzxyz) if p.bbox_xyzxyz is not None else None,
                "bbox_xyxy": list(p.bbox_xyxy) if p.bbox_xyxy is not None else None,
            }
            for p in pred_instances
        ],
    }
    out_path = art_dir / f"frame_{frame_idx:06d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Performance helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_perf_dict(
    timings_agg: Dict[str, List[float]],
    gpu_usage: Dict[str, List[float]],
) -> Dict:
    perf: Dict = {}
    timing_stages = {stage: vals for stage, vals in timings_agg.items() if vals}
    perf["timing_mean_ms"] = {s: float(np.mean(v)) for s, v in timing_stages.items()}
    perf["timing_std_ms"] = {s: float(np.std(v)) for s, v in timing_stages.items()}
    gpu_stages = {k: v for k, v in gpu_usage.items() if v}
    perf["gpu_mean_mb"] = {k: float(np.mean(v)) for k, v in gpu_stages.items()}
    perf["gpu_max_mb"] = {k: float(np.max(v)) for k, v in gpu_stages.items()}

    total_keys = ["yolo", "preprocess", "depth", "pcd_extract",
                  "track_update", "reprojection", "tracking_3d", "graph"]
    total_vals = []
    for key in total_keys:
        if key in timing_stages:
            arr = np.array(timings_agg[key])
            if len(total_vals) == 0:
                total_vals = arr
            else:
                n = min(len(total_vals), len(arr))
                total_vals = total_vals[:n] + arr[:n]
    if len(total_vals) > 0:
        perf["total_avg_ms"] = float(np.mean(total_vals))
        perf["fps"] = round(1000.0 / perf["total_avg_ms"], 2) if perf["total_avg_ms"] > 0 else 0.0
    return perf


def print_perf_summary(
    timings_agg: Dict[str, List[float]],
    gpu_usage: Dict[str, List[float]],
    cuda_available: bool,
    title: str = "PERFORMANCE",
) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    stage_order = [
        "yolo", "yolo_inference", "preprocess", "depth", "pcd_extract",
        "track_update", "reprojection", "tracking_3d", "graph",
        "graph_build_nodes", "graph_merge", "graph_predict_basic",
        "graph_predict_baseline", "graph_predict_vlsat",
    ]
    print(f"  {'Stage':<30} {'mean (ms)':>10} {'std (ms)':>10} {'frames':>8}")
    print(sep)
    total_per_frame = np.zeros(0)
    core_keys = {"yolo", "preprocess", "depth", "pcd_extract",
                 "track_update", "reprojection", "tracking_3d", "graph"}
    for stage in stage_order:
        vals = timings_agg.get(stage, [])
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {stage:<30} {np.mean(arr):>10.1f} {np.std(arr):>10.1f} {len(arr):>8}")
        if stage in core_keys:
            if len(total_per_frame) == 0:
                total_per_frame = arr.copy()
            else:
                n = min(len(total_per_frame), len(arr))
                total_per_frame = total_per_frame[:n] + arr[:n]
    if total_per_frame.size > 0:
        avg_total = float(np.mean(total_per_frame))
        print(sep)
        print(f"  {'TOTAL (core stages)':<30} {avg_total:>10.1f} {'':>10} {'':>8}")
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        print(f"  {'FPS':<30} {fps:>10.2f}")
    if cuda_available:
        gpu_stages = {k: v for k, v in gpu_usage.items() if v}
        if gpu_stages:
            print(sep)
            print(f"  {'GPU Memory Stage':<30} {'mean (MB)':>10} {'max (MB)':>10}")
            print(sep)
            for k, v in gpu_stages.items():
                arr = np.array(v)
                print(f"  {k:<30} {np.mean(arr):>10.1f} {np.max(arr):>10.1f}")
    print(sep)


def print_perf_aggregate(perf_per_scene: Dict[str, Dict]) -> None:
    if not perf_per_scene:
        return
    sep = "═" * 70
    print(f"\n{sep}")
    print("  PERFORMANCE AGGREGATE  (all scenes)")
    print(sep)

    all_stages: List[str] = []
    for pd in perf_per_scene.values():
        for s in pd.get("timing_mean_ms", {}):
            if s not in all_stages:
                all_stages.append(s)

    stage_order = [
        "yolo", "yolo_inference", "preprocess", "depth", "pcd_extract",
        "track_update", "reprojection", "tracking_3d", "graph",
        "graph_build_nodes", "graph_merge", "graph_predict_basic",
        "graph_predict_baseline", "graph_predict_vlsat",
    ]
    ordered = [s for s in stage_order if s in all_stages]
    ordered += [s for s in all_stages if s not in ordered]

    print(f"  {'Stage':<30} {'mean (ms)':>10} {'std (ms)':>10} {'#scenes':>8}")
    print("─" * 70)
    for stage in ordered:
        vals = [
            pd["timing_mean_ms"][stage]
            for pd in perf_per_scene.values()
            if stage in pd.get("timing_mean_ms", {})
        ]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {stage:<30} {np.mean(arr):>10.1f} {np.std(arr):>10.1f} {len(arr):>8}")

    fps_vals = [
        pd["fps"] for pd in perf_per_scene.values() if "fps" in pd and pd["fps"] > 0
    ]
    if fps_vals:
        print("─" * 70)
        print(f"  {'FPS (avg across scenes)':<30} {np.mean(fps_vals):>10.2f} "
              f"{np.std(fps_vals):>10.2f} {len(fps_vals):>8}")

    all_gpu: List[str] = []
    for pd in perf_per_scene.values():
        for k in pd.get("gpu_mean_mb", {}):
            if k not in all_gpu:
                all_gpu.append(k)
    if all_gpu:
        print("─" * 70)
        print(f"  {'GPU Memory Stage':<30} {'mean (MB)':>10} {'max (MB)':>10} {'#scenes':>8}")
        print("─" * 70)
        for k in all_gpu:
            means = [pd["gpu_mean_mb"][k] for pd in perf_per_scene.values() if k in pd.get("gpu_mean_mb", {})]
            maxes = [pd["gpu_max_mb"][k] for pd in perf_per_scene.values() if k in pd.get("gpu_max_mb", {})]
            if means:
                print(f"  {k:<30} {np.mean(means):>10.1f} {np.mean(maxes):>10.1f} {len(means):>8}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate / reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_aggregate(overall: Dict, keys: List[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AGGREGATE  (all scenes)")
    print(sep)
    for k in keys:
        if k in overall:
            m = overall[k]
            print(f"  {k:25s}  {m['mean']:.4f} ± {m['std']:.4f}  "
                  f"[{m['min']:.4f} – {m['max']:.4f}]")
    print(sep)


def build_overall_plot_results(all_results: Dict[str, Dict]) -> Dict:
    """Merge per-scene results into a single dict for ``plot_results``."""
    if not all_results:
        return {}

    scalar_keys = [
        "T_mIoU", "T_mIoU_std", "T_SR", "ID_consistency",
        "MOTA", "MOTP",
        "MOTA_FN_ratio", "MOTA_FP_ratio", "MOTA_IDSW_ratio",
        "ID_switches_total",
        "frames_processed", "unique_gt_objects",
        "total_gt_instances", "total_pred_instances",
        "total_matches", "total_false_positives", "total_false_negatives",
    ]
    merged: Dict = {}
    for k in scalar_keys:
        vals = [r[k] for r in all_results.values() if k in r]
        if vals:
            merged[k] = float(np.mean(vals))

    # Per-object T-mIoU pooled across scenes
    per_obj: Dict[str, float] = {}
    for scene_name, res in all_results.items():
        for obj_id, val in res.get("T_mIoU_per_object", {}).items():
            per_obj[f"{scene_name}/{obj_id}"] = val
    merged["T_mIoU_per_object"] = per_obj
    if per_obj:
        merged["T_mIoU"] = float(np.mean(list(per_obj.values())))
        merged["T_mIoU_std"] = float(np.std(list(per_obj.values())))

    # Per-class (weighted average)
    class_ious: Dict[str, List[float]] = defaultdict(list)
    class_counts: Dict[str, int] = defaultdict(int)
    for res in all_results.values():
        for cls, m in res.get("per_class_metrics", {}).items():
            cnt = m.get("count", 1)
            class_ious[cls].extend([m["T_mIoU"]] * cnt)
            class_counts[cls] += cnt
    per_class: Dict[str, Dict] = {}
    for cls in class_ious:
        vals = class_ious[cls]
        per_class[cls] = {
            "T_mIoU": float(np.mean(vals)),
            "T_mIoU_std": float(np.std(vals)),
            "count": class_counts[cls],
        }
    merged["per_class_metrics"] = per_class

    return merged


def plot_cross_scene(
    all_results: Dict[str, Dict],
    agg_keys: List[str],
    out: Path,
) -> None:
    """Generate a grouped bar chart comparing all scenes side by side."""
    import matplotlib.pyplot as plt

    scene_names = list(all_results.keys())
    n_scenes = len(scene_names)
    n_metrics = len(agg_keys)

    fig, ax = plt.subplots(figsize=(max(10, n_scenes * 2), 6))
    x = np.arange(n_scenes)
    width = 0.8 / n_metrics

    for i, key in enumerate(agg_keys):
        vals = [all_results[s].get(key, 0.0) for s in scene_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=key, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )
        avg = float(np.mean(vals))
        color = bars[0].get_facecolor()
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            n_scenes - 0.5, avg + 0.02, f"avg {key}: {avg:.2f}",
            fontsize=7, color=color, ha="right", va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scene_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Scene Metrics Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(min(0, ax.get_ylim()[0] - 0.05), 1.15)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    plot_dir = out / "benchmark_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "cross_scene_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {plot_dir / 'cross_scene_comparison.png'}")
