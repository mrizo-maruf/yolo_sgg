#!/usr/bin/env python3
"""
Generic Tracking Benchmark
===========================

Dataset-agnostic benchmark runner.  Works with any dataset that has a
``DatasetLoader`` in ``data_loaders/`` with ``get_gt_instances()`` support.

Modes
-----
1. **Single-scene**

       python -m benchmark.benchmark --dataset isaacsim --scene_path /path/to/scene_1

2. **Multi-scene**

       python -m benchmark.benchmark --dataset isaacsim --scene_path /path/to/dataset --multi

Supported datasets: isaacsim, thud_synthetic, coda, scanepp
(as registered in ``data_loaders.registry``).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.new_tracker import run_tracking
from core.object_registry import GlobalObjectRegistry
from core.types import TrackedFrame, TrackedObject
from data_loaders import get_loader
from depth_providers.factory import PROVIDER_CHOICES, build_depth_provider
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
# Single-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_scene(
    scene_path: str,
    dataset_name: str,
    cfg: OmegaConf,
) -> Dict:
    """Run tracking + evaluation on a single scene.  Returns metrics dict."""

    # --- Build loader --------------------------------------------------------
    LoaderCls = get_loader(dataset_name)

    # Depth provider (from config)
    dp_type = str(cfg.get("depth_provider", "gt"))
    depth_provider = build_depth_provider(dp_type, dataset_name, scene_path, cfg)

    loader_kwargs = _build_loader_kwargs(dataset_name, cfg)
    loader = LoaderCls(scene_path, depth_provider=depth_provider, **loader_kwargs)

    n_frames = loader.get_num_frames()
    intrinsics = loader.get_intrinsics()

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK  –  {loader.scene_label}  (dataset: {dataset_name})")
    print(f"{'=' * 60}")
    print(f"Frames: {n_frames}")
    print(f"Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}  "
          f"image={intrinsics.width}x{intrinsics.height}")

    # Quick check: does this loader support GT?
    test_gt = loader.get_gt_instances(0)
    if test_gt is None:
        print("[WARN] Loader returned no GT for frame 0 – metrics will be empty.")

    # --- Object registry -----------------------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.get("tracking_overlap_threshold", 0.1)),
        distance_threshold=float(cfg.get("tracking_distance_threshold", 1.0)),
        max_points=int(cfg.get("max_accumulated_points", 10000)),
        inactive_limit=int(cfg.get("tracking_inactive_limit", 0)),
        volume_ratio_threshold=float(cfg.get("tracking_volume_ratio_threshold", 0.1)),
        visibility_threshold=float(cfg.get("reprojection_visibility_threshold", 0.2)),
    )

    # --- Vis setup -----------------------------------------------------------
    vis_cfg = OmegaConf.to_container(cfg.get("visualization", {}), resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 5)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    # --- Metrics accumulator -------------------------------------------------
    acc = MetricsAccumulator()

    # --- Core tracking loop --------------------------------------------------
    for tf in tqdm(
        run_tracking(loader=loader, cfg=cfg, object_registry=object_registry),
        total=n_frames,
        desc=f"[{loader.scene_label}]",
    ):
        # GT for this frame
        gt_instances = loader.get_gt_instances(tf.frame_idx)
        if gt_instances is None:
            gt_instances = []

        # Build PredInstance list from TrackedFrame
        pred_instances = _build_pred_instances(tf)

        # Match GT ↔ pred
        mapping, ious = match_greedy(
            gt_instances,
            pred_instances,
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
        )

        # Accumulate
        rec = FrameRecord(
            frame_idx=tf.frame_idx,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        acc.add_frame(rec)

        # --- optional visualisation ------------------------------------------
        if vis_on and (tf.frame_idx % vis_interval == 0 or tf.frame_idx == 0):
            _visualize_frame(
                tf.rgb_path,
                gt_instances,
                pred_instances,
                mapping,
                ious,
                tf.frame_idx,
                vis_cfg,
                vis_save,
            )

    # --- Final metrics -------------------------------------------------------
    metrics = acc.compute()
    print_summary(metrics, title=f"BENCHMARK – {loader.scene_label}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_dataset(
    root_path: str,
    dataset_name: str,
    cfg: OmegaConf,
    output_dir: Optional[str] = None,
) -> Dict:
    """Iterate over all scenes under *root_path* and aggregate metrics."""
    LoaderCls = get_loader(dataset_name)
    scenes = LoaderCls.discover_scenes(root_path)

    if not scenes:
        print(f"No scenes found under {root_path} for dataset '{dataset_name}'")
        return {}

    print(f"Found {len(scenes)} scenes under {root_path}")

    agg_keys = ["T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP"]
    all_results: Dict[str, Dict] = {}
    agg: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in scenes:
        scene_label = Path(scene_dir).name
        try:
            res = benchmark_scene(scene_dir, dataset_name, cfg)
            all_results[scene_label] = res
            for k in agg_keys:
                if k in res:
                    agg[k].append(res[k])
        except Exception as exc:
            print(f"\n[ERROR] {scene_label}: {exc}")
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

    # --- Save results --------------------------------------------------------
    if output_dir is None:
        output_dir = cfg.get("benchmark_metrics_path", "results/benchmark_metrics")
    out_root = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / "all_scenes_aggregate"

    combined = {
        "dataset": dataset_name,
        "per_scene": all_results,
        "overall": overall,
        "num_scenes": len(all_results),
    }
    save_metrics(combined, out, scene_name="all_scenes_aggregate")

    for name, res in all_results.items():
        save_metrics(res, out, scene_name=name)
        scene_plot_dir = out / name / "benchmark_plots"
        plot_results(res, output_dir=str(scene_plot_dir))

    # Overall aggregate plots
    overall_for_plot = _build_overall_plot_results(all_results)
    plot_results(overall_for_plot, output_dir=str(out / "overall_benchmark_plots"))

    # Cross-scene comparison chart
    if len(all_results) > 1:
        _plot_cross_scene(all_results, agg_keys, out)

    # Save config for reference
    cfg_out = out_root / "benchmark_config.yaml"
    with open(cfg_out, "w") as f:
        OmegaConf.save(cfg, f)

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_pred_instances(tf: TrackedFrame) -> List[PredInstance]:
    """Convert ``TrackedFrame`` outputs to ``PredInstance`` list.

    Each ``TrackedObject`` in ``tf.objects`` has a ``global_id``,
    ``class_name``, and ``mask``.  We pair these with the per-detection
    masks from the YOLO output to get full-resolution masks.
    """
    preds: List[PredInstance] = []
    for obj in tf.objects:
        # Use the tracked object's own mask if available
        mask = obj.mask

        # Fall back: try to find in tf.masks via yolo_id matching
        if mask is None and tf.track_ids is not None and tf.masks:
            yolo_tid = obj.yolo_id
            if yolo_tid is not None and yolo_tid >= 0:
                idxs = np.where(tf.track_ids == yolo_tid)[0]
                if len(idxs) > 0 and idxs[0] < len(tf.masks):
                    mask = tf.masks[idxs[0]]

        preds.append(PredInstance(
            pred_id=obj.global_id,
            class_name=obj.class_name,
            mask=mask,
        ))
    return preds


def _build_loader_kwargs(dataset_name: str, cfg) -> dict:
    """Build dataset-specific keyword arguments for loader construction."""
    kwargs: dict = {}

    # Skip labels from config
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
    elif dataset_name == "scanepp":
        kwargs["max_depth"] = float(cfg.get("max_depth", 10.0))

    return kwargs


def _visualize_frame(
    rgb_path: str,
    gt_instances: List[GTInstance],
    pred_instances: List[PredInstance],
    mapping: Dict,
    ious: Dict,
    frame_idx: int,
    vis_cfg: dict,
    vis_save: Optional[str],
) -> None:
    """Render 2-D matching + tracking panels."""
    import cv2 as cv

    rgb = cv.imread(rgb_path)
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
            plt.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
            plt.title(f"Tracking – Frame {frame_idx}")
            plt.axis("off")
            plt.tight_layout()
            if save_2d:
                plt.savefig(save_2d, dpi=150, bbox_inches="tight")
            plt.show()
        elif save_2d:
            cv.imwrite(save_2d, overlay)


def _print_aggregate(overall: Dict, keys: List[str]) -> None:
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


def _build_overall_plot_results(all_results: Dict[str, Dict]) -> Dict:
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


def _plot_cross_scene(
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic tracking benchmark runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Single IsaacSim scene with visualisation
  python -m benchmark.benchmark \\
      --dataset isaacsim --scene_path /data/scene_1 --vis

  # All THUD synthetic scenes
  python -m benchmark.benchmark \\
      --dataset thud_synthetic --scene_path /data/THUD_Robot --multi

  # Save metrics to custom dir
  python -m benchmark.benchmark \\
      --dataset isaacsim --scene_path /data/scene_1 \\
      --output_dir ./my_results
""",
    )
    p.add_argument(
        "--dataset", type=str, default="isaacsim",
        choices=["isaacsim", "thud_synthetic", "coda", "scanepp"],
        help="Dataset type (default: isaacsim).",
    )
    p.add_argument(
        "--scene_path", type=str, required=True,
        help="Path to a single scene dir, or dataset root with --multi.",
    )
    p.add_argument("--multi", action="store_true",
                   help="Benchmark all scenes under scene_path.")

    # Depth provider
    p.add_argument("--depth_provider", type=str, default=None,
                   choices=list(PROVIDER_CHOICES),
                   help="Depth provider type (overrides config). "
                        "Default from config: 'gt'.")

    # Model overrides
    p.add_argument("--yolo_model", type=str, default=None,
                   help="Path to YOLOE model weights.")
    p.add_argument("--is_open_vocab", action="store_true", default=None,
                   help="Enable open-vocabulary mode.")

    # Visualisation
    p.add_argument("--vis", action="store_true",
                   help="Enable debug visualisation.")
    p.add_argument("--vis_interval", type=int, default=None,
                   help="Visualise every N frames.")
    p.add_argument("--vis_save", type=str, default=None,
                   help="Dir to save visualisation PNGs.")
    p.add_argument("--no_show", action="store_true",
                   help="Don't display windows (only save).")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Dir to save metrics/plots (overrides config).")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    # --- Load config (default + dataset yaml) --------------------------------
    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    default_cfg = OmegaConf.load(cfg_dir / "core_tracking.yaml")

    ds_yaml = cfg_dir / f"{args.dataset}.yaml"
    if ds_yaml.exists():
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml))
    else:
        cfg = default_cfg

    # --- CLI overrides -------------------------------------------------------
    if args.yolo_model:
        cfg.yolo_model = args.yolo_model
    if args.is_open_vocab is not None:
        cfg.is_open_vocabulary = args.is_open_vocab
    if args.depth_provider:
        cfg.depth_provider = args.depth_provider
    if args.vis:
        cfg.visualization = cfg.get("visualization", {})
        cfg.visualization.enabled = True
    if args.vis_interval is not None:
        cfg.visualization = cfg.get("visualization", {})
        cfg.visualization.interval = args.vis_interval
    if args.no_show:
        cfg.visualization = cfg.get("visualization", {})
        cfg.visualization.show_windows = False
    if args.vis_save:
        cfg.visualization = cfg.get("visualization", {})
        cfg.visualization.save_dir = args.vis_save

    # --- Run -----------------------------------------------------------------
    scene_path = str(Path(args.scene_path).resolve())
    if not Path(scene_path).exists():
        print(f"Path does not exist: {scene_path}")
        return 1

    if args.multi:
        output_dir = args.output_dir or cfg.get(
            "benchmark_metrics_path", "results/benchmark_metrics",
        )
        benchmark_dataset(scene_path, args.dataset, cfg, output_dir=output_dir)
    else:
        results = benchmark_scene(scene_path, args.dataset, cfg)

        output_dir = args.output_dir or cfg.get(
            "benchmark_metrics_path", "results/benchmark_metrics",
        )
        out = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name = Path(scene_path).name
        save_metrics(results, out, scene_name=scene_name)
        plot_results(results, output_dir=str(out / "benchmark_plots"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
