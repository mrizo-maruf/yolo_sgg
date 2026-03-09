#!/usr/bin/env python3
"""
bench.py — Run tracking benchmarks on any supported dataset.

Usage
-----
  # Single IsaacSim scene
  python bench.py /path/to/scene_1

  # All IsaacSim scenes under a root
  python bench.py /path/to/IsaacSim_Dataset --multi

  # Single THUD synthetic scene
  python bench.py /path/to/Gym/static/Capture_1 --dataset thud_synthetic

  # All THUD synthetic captures
  python bench.py /path/to/THUD_Robot --dataset thud_synthetic --multi

  # THUD Real scenes
  python bench.py /path/to/THUD_Robot --dataset thud_real --multi

  # With visualisation
  python bench.py /path/to/scene --vis --vis-interval 5

  # Custom config
  python bench.py /path/to/scene --config configs/isaacsim.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry

from core.tracker import run_tracking
from core.helpers import build_pred_instances
from loaders import get_loader
from loaders.base import DatasetLoader

from metrics.tracking_metrics import (
    FrameRecord,
    MetricsAccumulator,
    match_greedy,
    print_summary,
    save_metrics,
)
from benchmark.visualization import (
    draw_masks_with_labels,
    plot_results,
    visualize_matching,
)

import cv2


# ═══════════════════════════════════════════════════════════════════════════
# Override YOLOE.utils camera intrinsics
# ═══════════════════════════════════════════════════════════════════════════

def _override_intrinsics(K, img_h, img_w):
    yutils.fx = float(K[0, 0])
    yutils.fy = float(K[1, 1])
    yutils.cx = float(K[0, 2])
    yutils.cy = float(K[1, 2])
    yutils.IMAGE_WIDTH = img_w
    yutils.IMAGE_HEIGHT = img_h


# ═══════════════════════════════════════════════════════════════════════════
# Single-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_scene(loader: DatasetLoader, cfg: OmegaConf) -> Dict:
    """Run tracking + evaluation on a single scene. Returns metrics dict."""

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK – {loader.scene_label}")
    print(f"{'=' * 60}")

    frame_indices = loader.get_frame_indices()
    n_frames = len(frame_indices)
    print(f"Frames: {n_frames}")

    # --- Camera intrinsics ---------------------------------------------------
    intrinsics = loader.get_camera_intrinsics()
    if intrinsics is not None:
        K, img_h, img_w = intrinsics
        _override_intrinsics(K, img_h, img_w)
        print(f"Intrinsics: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
              f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  image={img_w}x{img_h}")

    # --- Prepare data --------------------------------------------------------
    rgb_paths = loader.get_rgb_paths()
    depth_paths = loader.get_depth_paths()
    depth_cache = loader.build_depth_cache()
    poses = loader.get_all_poses()

    # Track classes (for THUD Real, etc.)
    class_names_to_track = list(cfg.get("track_classes", [])) or None

    # --- Vis setup -----------------------------------------------------------
    vis_cfg = OmegaConf.to_container(cfg.visualization, resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 10)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    # --- Object registry -----------------------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    # --- Metrics accumulator -------------------------------------------------
    acc = MetricsAccumulator()

    # --- Core tracking loop --------------------------------------------------
    frame_counter = 0
    tracker = run_tracking(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        depth_cache=depth_cache,
        poses=poses,
        cfg=cfg,
        object_registry=object_registry,
        class_names_to_track=class_names_to_track,
    )

    for tf in tqdm(tracker, total=n_frames, desc=f"[{loader.scene_label}]"):
        # GT for this frame
        fidx = frame_indices[frame_counter] if frame_counter < n_frames else frame_indices[-1]
        gt_instances = loader.get_gt_instances(fidx)
        if gt_instances is None:
            gt_instances = []

        # Build PredInstance list
        pred_instances = build_pred_instances(tf.frame_objs, tf.track_ids, tf.masks_clean)

        # Match GT ↔ pred
        mapping, ious = match_greedy(
            gt_instances,
            pred_instances,
            iou_threshold=float(cfg.iou_threshold),
        )

        # Feed accumulator
        rec = FrameRecord(
            frame_idx=frame_counter,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        acc.add_frame(rec)

        # --- Optional visualisation ------------------------------------------
        if vis_on and (frame_counter % vis_interval == 0 or frame_counter == 0):
            _visualize_frame(
                tf.rgb_path,
                gt_instances,
                pred_instances,
                mapping,
                ious,
                frame_counter,
                vis_cfg,
                vis_save,
            )

        frame_counter += 1

    # --- Final metrics -------------------------------------------------------
    metrics = acc.compute()
    print_summary(metrics, title=f"BENCHMARK – {loader.scene_label}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_dataset(
    root: str,
    dataset_name: str,
    cfg: OmegaConf,
    output_dir: Optional[str] = None,
) -> Dict:
    """Iterate over all scenes under *root* and aggregate metrics."""

    LoaderCls = get_loader(dataset_name)

    discover_kwargs = {}
    if dataset_name == "thud_synthetic":
        discover_kwargs["scene_type"] = cfg.get("scene_type", "static")

    scene_dirs = LoaderCls.discover_scenes(root, **discover_kwargs)

    if not scene_dirs:
        print(f"No valid scenes found under {root}")
        return {}

    print(f"Found {len(scene_dirs)} scenes under {root}")

    all_results: Dict[str, Dict] = {}
    agg_keys = ["T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP"]
    agg: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in scene_dirs:
        try:
            loader = _build_loader(dataset_name, scene_dir, cfg)
            key = loader.scene_label
            res = benchmark_scene(loader, cfg)
            all_results[key] = res
            for k in agg_keys:
                if k in res:
                    agg[k].append(res[k])
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {scene_dir}: {exc}")
            traceback.print_exc()

    # --- Aggregate -----------------------------------------------------------
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

    # --- Save ----------------------------------------------------------------
    out = Path(output_dir) if output_dir else Path(root)
    combined = {
        "per_scene": all_results,
        "overall": overall,
        "num_scenes": len(all_results),
    }
    save_metrics(combined, out, scene_name=f"{dataset_name}_all_scenes_aggregate")
    for name, res in all_results.items():
        safe_name = name.replace("/", "_")
        save_metrics(res, out, scene_name=safe_name)
        scene_plot_dir = out / safe_name / "benchmark_plots"
        plot_results(res, output_dir=str(scene_plot_dir))

    if len(all_results) > 1:
        _plot_cross_scene(all_results, agg_keys, out)

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build loader from config
# ═══════════════════════════════════════════════════════════════════════════

def _build_loader(dataset_name: str, scene_dir: str, cfg: OmegaConf) -> DatasetLoader:
    """Instantiate a loader with the right kwargs for the dataset."""
    LoaderCls = get_loader(dataset_name)

    kwargs = {}
    skip_labels = cfg.get("loader_skip_labels")
    if skip_labels:
        kwargs["skip_labels"] = set(skip_labels)
    if dataset_name in ("thud_synthetic", "thud_real"):
        kwargs["depth_scale"] = float(cfg.get("depth_scale", 1000.0))
        kwargs["depth_max_m"] = float(cfg.get("depth_max_m", 100.0))
    if dataset_name == "thud_real":
        kwargs["tracking_distance"] = float(cfg.get("real_tracking_distance", 0.3))

    return LoaderCls(scene_dir, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation & display helpers
# ═══════════════════════════════════════════════════════════════════════════

def _visualize_frame(
    rgb_path, gt_instances, pred_instances,
    mapping, ious, frame_idx, vis_cfg, vis_save,
):
    """Render 2-D tracking + matching panels."""
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
        if save_2d:
            cv2.imwrite(save_2d, overlay)
        if show:
            cv2.imshow("Tracking 2D", overlay)
            cv2.waitKey(1)


def _print_aggregate(overall: Dict, keys: List[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AGGREGATE  (all scenes)")
    print(sep)
    for k in keys:
        if k in overall:
            m = overall[k]
            print(
                f"  {k:25s}  {m['mean']:.4f} ± {m['std']:.4f}  "
                f"[{m['min']:.4f} – {m['max']:.4f}]"
            )
    print(sep)


def _plot_cross_scene(all_results: Dict[str, Dict], agg_keys: List[str], out: Path) -> None:
    """Generate a grouped bar chart comparing all scenes side by side."""
    import matplotlib
    matplotlib.use("Agg")
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
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        avg = float(np.mean(vals))
        color = bars[0].get_facecolor()
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            n_scenes - 0.5, avg + 0.02,
            f"avg {key}: {avg:.2f}",
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
    print(f"[plots] Saved {plot_dir / 'cross_scene_comparison.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tracking benchmark runner — works with any supported dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Single IsaacSim scene with visualisation
  python bench.py /data/scene_1 --vis --vis-interval 5

  # Full IsaacSim dataset benchmark
  python bench.py /data/IsaacSim_Dataset --multi

  # THUD Synthetic (all static captures)
  python bench.py /data/THUD_Robot --dataset thud_synthetic --multi

  # THUD Synthetic (dynamic captures)
  python bench.py /data/THUD_Robot --dataset thud_synthetic --multi --scene-type dynamic

  # THUD Real scenes
  python bench.py /data/THUD_Robot --dataset thud_real --multi

  # Save visualisations without displaying
  python bench.py /data/scene_1 --vis --vis-save ./debug_vis --no-show
""",
    )
    p.add_argument("path", help="Path to a single scene dir or dataset root (with --multi).")
    p.add_argument("--dataset", type=str, default="isaacsim",
                   choices=["isaacsim", "thud_synthetic", "thud_real"],
                   help="Dataset type (default: isaacsim).")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (merged on top of default).")
    p.add_argument("--multi", action="store_true",
                   help="Benchmark all scenes under PATH.")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory for metrics.")
    # THUD-specific
    p.add_argument("--scene-type", type=str, default="static",
                   choices=["static", "dynamic"],
                   help="THUD scene type for multi-mode (default: static).")
    # Visualisation
    p.add_argument("--vis", action="store_true", help="Enable debug visualisation.")
    p.add_argument("--vis-interval", type=int, default=10, help="Visualise every N frames.")
    p.add_argument("--vis-save", type=str, default=None, help="Dir to save vis PNGs.")
    p.add_argument("--no-show", action="store_true", help="Don't pop up windows (only save).")
    # YOLO overrides
    p.add_argument("--model", type=str, default=None, help="YOLO model path.")
    p.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold.")
    p.add_argument("--iou-thresh", type=float, default=None, help="Matching IoU threshold.")
    # Depth
    p.add_argument("--depth-scale", type=float, default=None, help="Raw depth / this = metres.")
    p.add_argument("--depth-max", type=float, default=None, help="Clamp depth above (metres).")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Load config ---------------------------------------------------------
    default_cfg = OmegaConf.load(Path(__file__).parent / "configs" / "default.yaml")

    # Layer dataset-specific YAML
    ds_yaml = Path(__file__).parent / "configs" / f"{args.dataset}.yaml"
    if ds_yaml.exists():
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml))
    else:
        cfg = default_cfg

    # Layer user-specified YAML
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))

    # --- CLI overrides -------------------------------------------------------
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
    if args.depth_scale is not None:
        cfg.depth_scale = args.depth_scale
    if args.depth_max is not None:
        cfg.depth_max_m = args.depth_max
    if args.scene_type:
        cfg.scene_type = args.scene_type

    dataset_name = cfg.get("dataset", args.dataset)

    path = Path(args.path)
    if not path.exists():
        print(f"Path does not exist: {path}")
        return 1

    # --- Run -----------------------------------------------------------------
    if args.multi:
        results = benchmark_dataset(
            str(path), dataset_name, cfg, output_dir=args.output,
        )
    else:
        loader = _build_loader(dataset_name, str(path), cfg)
        results = benchmark_scene(loader, cfg)
        out_dir = args.output or str(path)
        save_metrics(results, out_dir, scene_name=path.name)
        plot_results(results, output_dir=os.path.join(out_dir, "benchmark_plots"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
