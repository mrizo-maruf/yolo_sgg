#!/usr/bin/env python3
"""
Tracking Benchmark Runner
=========================

Modes
-----
1. **Single-scene debug**  (default)

       python benchmark_tracking.py /path/to/scene_1 --vis

   Runs the tracking pipeline on one scene, matches predictions to GT per
   frame, computes metrics and (optionally) shows GT-vs-pred visualisations.

2. **Multi-scene benchmark**

       python benchmark_tracking.py /path/to/dataset --multi

   Iterates over every sub-directory that contains ``rgb/`` and ``depth/``
   and produces per-scene + aggregated metrics.

Metrics (dataset-agnostic, computed in ``metrics.tracking_metrics``)
--------------------------------------------------------------------
- T-mIoU, T-SR, ID Switches, MOTA, MOTP, per-class breakdown.

The benchmark itself only depends on the dataset through a thin
``SceneLoader`` protocol (see ``_load_scene``).  Swapping to a different
dataset means providing a loader that exposes the same interface as
``IsaacSimSceneLoader``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path so `YOLOE`, `metrics`, etc. are importable
# even when the script is invoked as  `python benchmark/benchmark_tracking.py`.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# -- project imports (framework-specific) ----------------------------------
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry
from isaacsim_utils.isaac_sim_loader import IsaacSimSceneLoader, GTObject
from Pi3.utils import process_depth_model

# -- decoupled metrics & vis -----------------------------------------------
from metrics.tracking_metrics import (
    FrameRecord,
    GTInstance,
    MetricsAccumulator,
    PredInstance,
    match_greedy,
    print_summary,
    save_metrics,
)
from benchmark.visualization import (
    draw_masks_with_labels,
    plot_results,
    visualize_matching,
)


# ═══════════════════════════════════════════════════════════════════════════
# Default configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # --- YOLO ---
    "yolo_model": "yoloe-11l-seg-pf.pt",
    "depth_model": "yyfz233/Pi3X",  # set None if you want to use original depth
    "conf": 0.25,
    "iou": 0.5,
    # --- Mask pre-processing ---
    "kernel_size": 11,
    "alpha": 0.7,
    "fast_mask": True,
    # --- Point cloud ---
    "max_points_per_obj": 2000,
    "max_accumulated_points": 10000,
    "o3_nb_neighbors": 50,
    "o3std_ratio": 0.1,
    # --- Tracking registry ---
    "tracking_overlap_threshold": 0.3,
    "tracking_distance_threshold": 0.5,
    "tracking_inactive_limit": 0,
    "tracking_volume_ratio_threshold": 0.1,
    "reprojection_visibility_threshold": 0.2,
    # --- Matching ---
    "iou_threshold": 0.3,
    # --- Classes to ignore (large structural / background) ---
    "skip_classes": [
        "wall", "floor", "ceiling", "roof", "carpet", "mat", 'ground', 'workspace', 'workplace',
        "stairway", "stairs", "elevator",
        "room", "kitchen", "bathroom", "bedroom", "living room",
        "dining room", "office", "hallway", "corridor", "lobby",
        "garage", "basement", "attic",
        "sky", "ground", "grass", "field", "lawn",
        "building", "house", "warehouse",
        "road", "street", "sidewalk", "parking",
    ],
    # --- Visualisation ---
    "visualization": {
        "enabled": False,
        "interval": 10,        # show every N frames
        "show_matching": True,  # 3-panel GT/pred/combined
        "show_2d": True,        # 2-D mask overlay
        "show_windows": True,
        "save_dir": None,       # if set, PNGs are saved there
    },
}

ISAACSIM_SKIP_LABELS = {'wall', 'floor', 'ground', 'ceiling', 'background'}

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _should_skip_class(name: str, skip_set: Set[str]) -> bool:
    """Return True if *name* matches any skip pattern."""
    if name is None:
        return False
    low = name.lower()
    if low in skip_set:
        return True
    # catch common substrings that appear in YOLO class names
    for kw in ("room", "shot", "carpet", "yard", "floor", "mat", "resort"):
        if kw in low:
            return True
    return False


def _gt_objects_to_instances(gt_objects: List[GTObject]) -> List[GTInstance]:
    """Convert Isaac-Sim GT objects to the dataset-agnostic ``GTInstance``."""
    return [
        GTInstance(
            track_id=g.track_id,
            class_name=g.class_name,
            mask=g.mask,
            bbox_xyxy=g.bbox2d_xyxy,
        )
        for g in gt_objects
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Single-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_scene(scene_path: str, cfg: OmegaConf) -> Dict:
    """Run tracking + evaluation on a single scene. Returns metrics dict."""

    scene_dir = Path(scene_path)
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK  –  {scene_dir.name}")
    print(f"{'=' * 60}")

    # --- load GT data -------------------------------------------------------
    loader = IsaacSimSceneLoader(str(scene_dir), load_rgb=True, skip_labels=ISAACSIM_SKIP_LABELS)
    frame_indices = loader.frame_indices
    n_frames = len(frame_indices)
    traj = loader.traj_data
    print(f"Frames: {n_frames}  |  Trajectory entries: {len(traj) if traj else 0}")

    # --- initialise tracking pipeline ----------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    rgb_dir = str(loader.rgb_dir)
    depth_dir = str(loader.depth_dir)
    
    # Process depth with Pi3X model if configured
    temp_cfg = OmegaConf.create({
        'rgb_dir': rgb_dir,
        'depth_dir': depth_dir,
        'traj_path': str(scene_dir / "traj.txt"),
        'depth_model': cfg.get('depth_model', None)
    })
    temp_cfg = process_depth_model(temp_cfg)
    
    # Update depth_dir if it was changed by depth model processing
    if temp_cfg.depth_dir != depth_dir:
        depth_dir = temp_cfg.depth_dir
        print(f"Using processed depth from: {depth_dir}")
    
    depth_paths = yutils.list_png_paths(depth_dir)

    # cache depth maps
    depth_cache: Dict[str, np.ndarray] = {}
    for dp in depth_paths:
        depth_cache[dp] = yutils.load_depth_as_meters(dp)

    poses = yutils.load_camera_poses(str(loader.scene_dir / "traj.txt")) if (loader.scene_dir / "traj.txt").exists() else None

    results_stream = yutils.track_objects_in_video_stream(
        rgb_dir, depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
    )

    # --- vis setup -----------------------------------------------------------
    vis_cfg = OmegaConf.to_container(cfg.visualization, resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 10)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    skip_set = set(c.lower() for c in cfg.get("skip_classes", []))

    # --- metrics accumulator -------------------------------------------------
    acc = MetricsAccumulator()

    # --- main loop -----------------------------------------------------------
    frame_idx = 0
    for yolo_res, rgb_path, depth_path in tqdm(results_stream, total=n_frames, desc="Processing"):
        # GT for this frame
        fd = loader.get_frame_data(frame_indices[frame_idx] if frame_idx < len(frame_indices) else frame_indices[-1])
        gt_objects = fd.gt_objects
        gt_instances = _gt_objects_to_instances(gt_objects)

        # Depth
        depth_m = depth_cache.get(depth_path)
        if depth_m is None:
            frame_idx += 1
            continue

        # Camera pose
        T_w_c = poses[min(frame_idx, len(poses) - 1)] if poses else None

        # YOLO masks
        _, masks_clean = yutils.preprocess_mask(
            yolo_res=yolo_res,
            index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=True,
        )

        # Track IDs & class names
        track_ids, class_names = _extract_yolo_ids(yolo_res, masks_clean)

        # Filter skip classes
        masks_clean, track_ids, class_names = _apply_class_filter(
            masks_clean, track_ids, class_names, skip_set,
        )

        # Build 3-D objects with global tracking
        frame_objs, _ = yutils.create_3d_objects_with_tracking(
            track_ids, masks_clean,
            int(cfg.max_points_per_obj), depth_m, T_w_c, frame_idx,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            object_registry=object_registry,
            class_names=class_names,
        )

        # Build PredInstance list
        pred_instances = _build_pred_instances(frame_objs, track_ids, masks_clean)

        # Match GT ↔ pred
        mapping, ious = match_greedy(
            gt_instances, pred_instances,
            iou_threshold=float(cfg.iou_threshold),
        )

        # Feed accumulator
        rec = FrameRecord(
            frame_idx=frame_idx,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        acc.add_frame(rec)

        # --- optional visualisation ------------------------------------------
        if vis_on and (frame_idx % vis_interval == 0 or frame_idx == 0):
            _visualize_frame(
                rgb_path, gt_instances, pred_instances,
                mapping, ious, frame_idx, vis_cfg, vis_save,
            )

        frame_idx += 1

    # --- final metrics -------------------------------------------------------
    metrics = acc.compute()
    print_summary(metrics, title=f"BENCHMARK – {scene_dir.name}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_dataset(dataset_path: str, cfg: OmegaConf, output_dir: Optional[str] = None) -> Dict:
    """Iterate over all sub-scenes in *dataset_path* and aggregate metrics."""
    root = Path(dataset_path)
    scenes = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "rgb").exists() and (d / "depth").exists()
    )
    if not scenes:
        print(f"No valid scenes found under {root}")
        return {}

    print(f"Found {len(scenes)} scenes under {root}")

    all_results: Dict[str, Dict] = {}
    agg_keys = [
        "T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP",
    ]
    agg: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in scenes:
        try:
            res = benchmark_scene(str(scene_dir), cfg)
            all_results[scene_dir.name] = res
            for k in agg_keys:
                if k in res:
                    agg[k].append(res[k])
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {scene_dir.name}: {exc}")
            traceback.print_exc()

    # aggregate
    overall: Dict[str, Dict] = {}
    for k in agg_keys:
        vals = agg.get(k, [])
        if vals:
            overall[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    _print_aggregate(overall, agg_keys)

    out = Path(output_dir) if output_dir else root
    combined = {"per_scene": all_results, "overall": overall, "num_scenes": len(all_results)}
    save_metrics(combined, out, scene_name="all_scenes_aggregate")
    for name, res in all_results.items():
        save_metrics(res, out, scene_name=name)
        # per-scene charts
        scene_plot_dir = out / name / "benchmark_plots"
        plot_results(res, output_dir=str(scene_plot_dir))

    # cross-scene comparison chart
    if len(all_results) > 1:
        _plot_cross_scene(all_results, agg_keys, out)

    return combined


def _print_aggregate(overall: Dict, keys: List[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AGGREGATE  (all scenes)")
    print(sep)
    for k in keys:
        if k in overall:
            m = overall[k]
            print(f"  {k:25s}  {m['mean']:.4f} ± {m['std']:.4f}  [{m['min']:.4f} – {m['max']:.4f}]")
    print(sep)


def _plot_cross_scene(all_results: Dict[str, Dict], agg_keys: List[str], out: Path) -> None:
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
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)
        # average line for this metric
        avg = float(np.mean(vals))
        color = bars[0].get_facecolor()
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(n_scenes - 0.5, avg + 0.02, f"avg {key}: {avg:.2f}",
                fontsize=7, color=color, ha="right", va="bottom")

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
    print(f"[plots] Saved {plot_dir / 'cross_scene_comparison.png'}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_yolo_ids(yolo_res, masks_clean):
    """Pull track IDs and class names from a YOLO result object."""
    track_ids = None
    class_names = None
    if hasattr(yolo_res, "boxes") and yolo_res.boxes is not None:
        if getattr(yolo_res.boxes, "id", None) is not None:
            try:
                track_ids = yolo_res.boxes.id.detach().cpu().numpy().astype(np.int64)
            except Exception:
                pass
        if getattr(yolo_res.boxes, "cls", None) is not None and hasattr(yolo_res, "names"):
            try:
                cls_ids = yolo_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                class_names = [yolo_res.names[int(c)] for c in cls_ids]
            except Exception:
                pass
    n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
    if track_ids is None:
        track_ids = np.arange(n, dtype=np.int64)
    return track_ids, class_names


def _apply_class_filter(masks_clean, track_ids, class_names, skip_set):
    """Remove detections whose class name is in *skip_set*."""
    if not skip_set or class_names is None:
        return masks_clean, track_ids, class_names
    keep = [i for i, c in enumerate(class_names) if not _should_skip_class(c, skip_set)]
    if len(keep) == len(class_names):
        return masks_clean, track_ids, class_names
    masks_clean = [masks_clean[i] for i in keep] if masks_clean else []
    track_ids = track_ids[keep] if track_ids is not None else None
    class_names = [class_names[i] for i in keep]
    return masks_clean, track_ids, class_names


def _build_pred_instances(frame_objs, track_ids, masks_clean) -> List[PredInstance]:
    """Convert pipeline output dicts → ``PredInstance`` list."""
    preds: List[PredInstance] = []
    for obj in frame_objs:
        mask = None
        yolo_tid = obj.get("yolo_track_id", -1)
        if yolo_tid >= 0 and track_ids is not None:
            idxs = np.where(track_ids == yolo_tid)[0]
            if len(idxs) > 0 and masks_clean and idxs[0] < len(masks_clean):
                mask = masks_clean[idxs[0]]
        preds.append(PredInstance(
            pred_id=obj["global_id"],
            class_name=obj.get("class_name"),
            mask=mask,
        ))
    return preds


def _visualize_frame(
    rgb_path, gt_instances, pred_instances,
    mapping, ious, frame_idx, vis_cfg, vis_save,
):
    """Optionally render 2-D tracking + matching panels."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return

    save_match = None
    save_2d = None
    if vis_save:
        save_match = os.path.join(vis_save, f"matching_{frame_idx:06d}.png")
        save_2d = os.path.join(vis_save, f"tracking_2d_{frame_idx:06d}.png")

    show = vis_cfg.get("show_windows", True)

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
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"Tracking – Frame {frame_idx}")
            plt.axis("off")
            plt.tight_layout()
            if save_2d:
                plt.savefig(save_2d, dpi=150, bbox_inches="tight")
            plt.show()
        elif save_2d:
            cv2.imwrite(save_2d, overlay)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tracking benchmark runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Single scene with visualisation every 5 frames
  python benchmark_tracking.py /data/scene_1 --vis --vis-interval 5

  # Full dataset benchmark (finds all sub-dirs with rgb/ + depth/)
  python benchmark_tracking.py /data/IsaacSim_Dataset --multi

  # Save visualisations without displaying
  python benchmark_tracking.py /data/scene_1 --vis --vis-save ./debug_vis --no-show
""",
    )
    p.add_argument("path", help="Path to a single scene dir or dataset root (with --multi).")
    p.add_argument("--multi", action="store_true", help="Benchmark all scenes under PATH.")
    p.add_argument("--output", type=str, default=None, help="Output directory for metrics.")
    # visualisation flags
    p.add_argument("--vis", action="store_true", help="Enable debug visualisation.")
    p.add_argument("--vis-interval", type=int, default=10, help="Visualise every N frames.")
    p.add_argument("--vis-save", type=str, default=None, help="Dir to save vis PNGs.")
    p.add_argument("--no-show", action="store_true", help="Don't pop up windows (only save).")
    # YOLO overrides
    p.add_argument("--model", type=str, default=None, help="YOLO model path.")
    p.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold.")
    p.add_argument("--iou-thresh", type=float, default=None, help="Matching IoU threshold.")
    # Depth model
    p.add_argument("--depth-model", type=str, default=None, help="Depth model (e.g., 'yyfz233/Pi3X' or 'none' for original).")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = OmegaConf.create(DEFAULT_CFG)

    # apply CLI overrides
    if args.vis:
        cfg.visualization.enabled = True
    if args.vis_interval:
        cfg.visualization.interval = args.vis_interval
    if args.vis_save:
        cfg.visualization.save_dir = args.vis_save
    if args.no_show:
        cfg.visualization.show_windows = False
    if args.model:
        cfg.yolo_model = args.model
    if args.conf is not None:
        cfg.conf = args.conf
    if args.iou_thresh is not None:
        cfg.iou_threshold = args.iou_thresh
    if args.depth_model is not None:
        if args.depth_model.lower() == 'none':
            cfg.depth_model = None
        else:
            cfg.depth_model = args.depth_model

    path = Path(args.path)
    if not path.exists():
        print(f"Path does not exist: {path}")
        return 1

    if args.multi:
        results = benchmark_dataset(str(path), cfg, output_dir=args.output)
    else:
        results = benchmark_scene(str(path), cfg)
        out_dir = args.output or str(path)
        save_metrics(results, out_dir, scene_name=path.name)
        plot_results(results, output_dir=os.path.join(out_dir, "benchmark_plots"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
