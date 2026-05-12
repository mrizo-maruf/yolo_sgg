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

Supported datasets: isaacsim, thud_synthetic, coda, scanetpp
(as registered in ``data_loaders.registry``).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import torch as _torch
except ImportError:
    _torch = None

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.new_tracker import (
    build_default_registry,
    resolve_open_vocab_classes,
    run_tracking,
)
from core.run_info import print_run_banner
from core.types import CameraIntrinsics
from data_loaders import get_loader
from depth_providers.factory import PROVIDER_CHOICES, build_depth_provider
from depth_providers.pi3_online import apply_pi3_online_transform_from_cfg
from metrics.tracking_metrics import (
    FrameRecord,
    MetricsAccumulator,
    match_greedy,
    match_hungarian,
    print_summary,
    save_metrics,
)
from benchmark.visualization import plot_results
from benchmark.rerun_debug_vis import BenchmarkDebugVisualizer
from benchmark.benchmark_utils import (
    TIMING_KEY_MAP,
    GPU_KEY_MAP,
    resolve_match_mode,
    similarity_label,
    prepare_gt_instances,
    build_pred_instances,
    build_loader_kwargs,
    visualize_frame,
    build_perf_dict,
    print_perf_summary,
    print_perf_aggregate,
    print_aggregate,
    build_overall_plot_results,
    plot_cross_scene,
)


# ═══════════════════════════════════════════════════════════════════════════
# Local helpers
# ═══════════════════════════════════════════════════════════════════════════

_RERUN_LEGEND_TERMINAL = """\
[bench] Rerun debug viewer — panel legend
─────────────────────────────────────────────────────────────────────
  Same blueprint as new_run.py: 3-D world view + Semantic
  Segmentation panel + RGB + 2-D boxes panel. The benchmark adds GT
  overlays on top of the standard view (additive, not replacing).

  3-D world view  (world3d/...)
    point clouds   accumulated per-object, colour = hash(global_id)
    camera frustum yellow trajectory, RGB thumbnail on the image plane
    pred bbox      green  = visible this frame  (in camera frustum)
                   red    = registry object NOT currently visible

    GT bbox        blue   = GT matched to a prediction (true positive)
                   orange = GT MISSED — false negative
                            (tracker didn't find this object)

  Semantic Seg panel  (seg_view)
    mask overlay per detection, colour = hash(yolo_id);
    label = class#yolo_id  (raw upstream YOLO output)

  RGB + 2-D Boxes panel  (rgb_view)
    reprojected 3-D bboxes drawn on the RGB image;
    label = class#global_id  (post-cascade tracking ids)

  How to read it together
    • green  pred + blue   GT overlap → true positive, visible
    • red    pred (no GT)                → reprojection-only carry-over
    • orange GT alone (no pred nearby)   → false negative (missed)
    • Pred without a nearby GT and no overlap → false positive
─────────────────────────────────────────────────────────────────────"""


def _load_rgb_image(path: str) -> Optional[np.ndarray]:
    """Read an RGB image from disk for the Rerun debug viewer. Returns None
    on failure so the loop keeps going."""
    try:
        import cv2
    except ImportError:
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════
# Single-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_scene(
    scene_path: str,
    dataset_name: str,
    cfg: OmegaConf,
    rerun_debug: bool = False,
) -> Dict:
    """Run tracking + evaluation on a single scene.  Returns metrics dict.

    Parameters
    ----------
    rerun_debug
        If ``True``, open a Rerun debug viewer with four 2-D panels per frame
        (GT / Predictions / YOLO 2-D / GT-Pred matching). Single-scene only;
        the caller is responsible for refusing this in ``--multi`` mode.
    """

    # --- Build loader --------------------------------------------------------
    LoaderCls = get_loader(dataset_name)

    dp_type = str(cfg.get("depth_provider", "gt"))
    depth_provider = build_depth_provider(dp_type, dataset_name, scene_path, cfg)

    if dp_type == "pi3_online":
        apply_pi3_online_transform_from_cfg(depth_provider, scene_path, cfg)

    loader = LoaderCls(scene_path, depth_provider=depth_provider,
                       **build_loader_kwargs(dataset_name, cfg))

    n_frames = loader.get_num_frames()
    intrinsics = loader.get_intrinsics()
    match_mode = resolve_match_mode(cfg)

    extras = {
        "Benchmark": [
            f"match_mode:        {match_mode} ({similarity_label(match_mode)})",
            f"iou_threshold:     {cfg.get('iou_threshold', 0.3)}",
            f"include_reproj:    {bool(cfg.get('benchmark_include_reprojected_masks', True))}",
        ],
    }
    print_run_banner(
        title="BENCHMARK",
        dataset_name=dataset_name,
        scene_label=loader.scene_label,
        scene_path=scene_path,
        n_frames=n_frames,
        intrinsics=intrinsics,
        dp_type=dp_type,
        cfg=cfg,
        classes=resolve_open_vocab_classes(loader, cfg),
        extras=extras,
    )

    # Quick check: does this loader support GT?
    test_gt = loader.get_gt_instances(0)
    if test_gt is None:
        print("[WARN] Loader returned no GT for frame 0 – metrics will be empty.")
    elif match_mode == "bbox3d" and not any(g.bbox_xyzxyz is not None for g in test_gt):
        raise ValueError(
            "match_mode=bbox3d requires GT 3D AABBs, "
            "but loader.get_gt_instances() returned none."
        )

    # --- Setup ---------------------------------------------------------------
    object_registry = build_default_registry(cfg)

    vis_cfg = OmegaConf.to_container(cfg.get("visualization", {}), resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 5)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    acc = MetricsAccumulator()
    cuda_available = _torch is not None and _torch.cuda.is_available()
    timings_agg: Dict[str, List[float]] = {k: [] for k in TIMING_KEY_MAP.values()}
    gpu_usage: Dict[str, List[float]] = {k: [] for k in GPU_KEY_MAP.values()}

    # --- Optional Rerun debug viewer (single-scene only) --------------------
    # Uses the SAME RerunVisualizer as new_run.py — same blueprint, same
    # data flow (full object_registry, not just tf.objects). Adds a
    # benchmark-specific GT overlay on top so matched/missed GT bboxes are
    # visible against the predicted bboxes.
    rerun_vis = None
    if rerun_debug:
        try:
            from rerun_utils import RerunVisualizer, _build_axis_remap_matrix

            axis_remap = None
            if dataset_name == "isaacsim":
                axis_remap = _build_axis_remap_matrix(swap_yz=True, flip_y=True)
                print("[Rerun] Applying Isaac axis remap (RFU -> RDF).")
            elif dataset_name == "scanetpp":
                axis_remap = _build_axis_remap_matrix(swap_yz=True, flip_y=True)
                print("[Rerun] Applying ScanNet++ axis remap (Z-up -> RDF).")

            _voxel = float(cfg.get("registry_voxel_size", 0.0))
            _default_radius = (_voxel / 2.0) if _voxel > 0.0 else 0.008
            rerun_vis = RerunVisualizer(
                recording_id=f"bench_{loader.scene_label}",
                axis_remap=axis_remap,
                point_radius=float(cfg.get("rerun_point_radius", _default_radius)),
            )
            rerun_vis.init(
                img_w=intrinsics.width, img_h=intrinsics.height,
                fx=intrinsics.fx, fy=intrinsics.fy,
                cx=intrinsics.cx, cy=intrinsics.cy,
            )
            print(f"[bench] Rerun debug viewer ready for {loader.scene_label}")
            print(_RERUN_LEGEND_TERMINAL)
        except Exception as exc:
            print(f"[bench] Could not start Rerun debug viewer: {exc}")
            rerun_vis = None

    # --- Background online depth feeder --------------------------------------
    # Online providers emit depth in chunks; without a pre-feeder thread the tracking
    # loop deadlocks on get_depth(0).  The feeder pushes RGBs ahead so
    # chunks are ready when needed.
    _pi3_feeder = None
    if dp_type in ("pi3_online", "dav3_online"):
        def _pi3_feed_worker():
            for fidx in range(n_frames):
                loader.get_rgb(fidx)
            if hasattr(depth_provider, "drain"):
                depth_provider.drain()

        _pi3_feeder = threading.Thread(
            target=_pi3_feed_worker, daemon=True, name="depth-feeder",
        )
        _pi3_feeder.start()
        print(f"[Depth] Background depth feeder started ({n_frames} frames)")

    # --- Core tracking loop --------------------------------------------------
    for tf in tqdm(
        run_tracking(loader=loader, cfg=cfg, object_registry=object_registry),
        total=n_frames,
        desc=f"[{loader.scene_label}]",
    ):
        gt_instances = loader.get_gt_instances(tf.frame_idx) or []
        gt_instances = prepare_gt_instances(gt_instances, tf, intrinsics, match_mode)

        pred_instances = build_pred_instances(
            tf,
            intrinsics=intrinsics,
            match_mode=match_mode,
            include_reprojected_masks=bool(
                cfg.get("benchmark_include_reprojected_masks", False)
            ),
        )

        # mapping, ious = match_greedy(
        #     gt_instances,
        #     pred_instances,
        #     iou_threshold=float(cfg.get("iou_threshold", 0.3)),
        #     match_mode=match_mode,
        # )
        mapping, ious = match_hungarian(
            gt_instances,
            pred_instances,
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            match_mode=match_mode,
        )

        acc.add_frame(FrameRecord(
            frame_idx=tf.frame_idx,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        ))

        for src, dst in TIMING_KEY_MAP.items():
            if src in tf.timings:
                timings_agg[dst].append(float(tf.timings[src]))
        for src, dst in GPU_KEY_MAP.items():
            if src in tf.timings:
                gpu_usage[dst].append(float(tf.timings[src]))

        # --- Rerun debug viewer (single-scene) ------------------------------
        # Same logic as new_run.py: log_frame consumes the full
        # object_registry (not just tf.objects) so the 3-D world view shows
        # accumulated point clouds, per-object bboxes (green=visible /
        # red=invisible), camera frustum, and the per-track-id segmentation
        # masks on the 2-D panels. The benchmark-specific overlay then adds
        # GT bboxes coloured by match status (blue=matched, orange=missed).
        if rerun_vis is not None:
            import numpy as np
            try:
                rerun_vis.log_frame(
                    frame_idx=tf.frame_idx,
                    object_registry=object_registry,
                    persistent_graph=None,
                    T_w_c=tf.T_w_c,
                    rgb_path=tf.rgb_path,
                    masks_clean=tf.masks,
                    track_ids=(tf.track_ids if tf.track_ids is not None
                               else np.array([], dtype=int)),
                    class_names=tf.class_names,
                    vis_edges=False,
                )
                rerun_vis.log_benchmark_overlay(
                    frame_idx=tf.frame_idx,
                    gt_instances=gt_instances,
                    mapping=mapping,
                )
            except Exception as exc:
                print(f"[bench] Rerun log_frame failed at f={tf.frame_idx}: {exc}")

        if vis_on and (tf.frame_idx % vis_interval == 0 or tf.frame_idx == 0):
            visualize_frame(
                tf.rgb_path, gt_instances, pred_instances,
                mapping, ious, tf.frame_idx,
                vis_cfg, vis_save, match_mode=match_mode,
            )

    # --- Cleanup -------------------------------------------------------------
    if hasattr(depth_provider, "close"):
        try:
            depth_provider.close()
        except Exception:
            pass
    if _pi3_feeder is not None:
        _pi3_feeder.join(timeout=5.0)

    # --- Metrics -------------------------------------------------------------
    metrics = acc.compute()
    metrics["match_mode"] = match_mode
    metrics["similarity"] = similarity_label(match_mode)

    print_perf_summary(timings_agg, gpu_usage, cuda_available,
                       title=f"PERFORMANCE – {loader.scene_label}")
    metrics["perf"] = build_perf_dict(timings_agg, gpu_usage)

    print_summary(metrics, title=f"BENCHMARK – {loader.scene_label} [{match_mode}]")
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

        # Force reclamation between scenes to prevent GPU OOM.
        gc.collect()
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()

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

    print_aggregate(overall, agg_keys)

    perf_per_scene = {
        name: res["perf"] for name, res in all_results.items() if "perf" in res
    }
    print_perf_aggregate(perf_per_scene)

    # --- Save results --------------------------------------------------------
    if output_dir is None:
        output_dir = cfg.get("benchmark_metrics_path", "results/benchmark_metrics")
    out_root = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / "all_scenes_aggregate"

    combined = {
        "dataset": dataset_name,
        "match_mode": resolve_match_mode(cfg),
        "per_scene": all_results,
        "overall": overall,
        "num_scenes": len(all_results),
        "perf_per_scene": perf_per_scene,
    }
    save_metrics(combined, out, scene_name="all_scenes_aggregate")

    for name, res in all_results.items():
        save_metrics(res, out, scene_name=name)
        plot_results(res, output_dir=str(out / name / "benchmark_plots"))

    overall_for_plot = build_overall_plot_results(all_results)
    plot_results(overall_for_plot, output_dir=str(out / "overall_benchmark_plots"))

    if len(all_results) > 1:
        plot_cross_scene(all_results, agg_keys, out)

    cfg_out = out_root / "benchmark_config.yaml"
    with open(cfg_out, "w") as f:
        OmegaConf.save(cfg, f)

    return combined


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
        choices=["isaacsim", "thud_synthetic", "coda", "scanetpp"],
        help="Dataset type (default: isaacsim).",
    )
    p.add_argument(
        "--scene_path", type=str, required=True,
        help="Path to a single scene dir, or dataset root with --multi.",
    )
    p.add_argument("--multi", action="store_true",
                   help="Benchmark all scenes under scene_path.")
    p.add_argument(
        "--rerun", action="store_true",
        help="Open Rerun debug visualisation (single-scene only). "
             "Logs four 2-D views per frame: GT, Predictions, YOLO 2-D, and "
             "GT/Pred match. Disabled with --multi.",
    )

    # Depth provider
    p.add_argument("--depth_provider", type=str, default=None,
                   choices=list(PROVIDER_CHOICES),
                   help="Depth provider type (overrides config). Default from config: 'gt'.")
    p.add_argument(
        "--match_mode", type=str, default=None,
        choices=["mask2d", "bbox2d", "bbox3d"],
        help="Matching mode for the run.",
    )

    # Model overrides
    p.add_argument("--yolo_model", type=str, default=None,
                   help="Path to YOLOE model weights.")
    p.add_argument("--is_open_vocab", action="store_true", default=None,
                   help="Enable open-vocabulary mode.")

    # Visualisation
    p.add_argument("--vis", action="store_true", help="Enable debug visualisation.")
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

    # --- Load config ---------------------------------------------------------
    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    default_cfg = OmegaConf.load(cfg_dir / "core_tracking.yaml")

    ds_yaml = cfg_dir / f"{args.dataset}.yaml"
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml)) if ds_yaml.exists() else default_cfg

    # --- CLI overrides -------------------------------------------------------
    if args.yolo_model:
        cfg.yolo_model = args.yolo_model
    if args.is_open_vocab is not None:
        cfg.is_open_vocabulary = args.is_open_vocab
    if args.depth_provider:
        cfg.depth_provider = args.depth_provider
    if args.match_mode:
        cfg.match_mode = args.match_mode
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

    if args.rerun and args.multi:
        print(
            "[benchmark] --rerun is single-scene only; ignoring it for --multi.",
            file=sys.stderr,
        )
        args.rerun = False

    output_dir = args.output_dir or cfg.get(
        "benchmark_metrics_path", "results/benchmark_metrics",
    )

    if args.multi:
        benchmark_dataset(scene_path, args.dataset, cfg, output_dir=output_dir)
    else:
        results = benchmark_scene(
            scene_path, args.dataset, cfg, rerun_debug=args.rerun,
        )
        out = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_metrics(results, out, scene_name=Path(scene_path).name)
        plot_results(results, output_dir=str(out / "benchmark_plots"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
