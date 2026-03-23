
#!/usr/bin/env python3
"""
new_run.py — Run the YOLO tracking pipeline on any scene.

Usage
-----
  # IsaacSim (default)
  python new_run.py --scene_path /path/to/scene_1

  # THUD Synthetic
  python new_run.py --dataset thud_synthetic --scene_path /path/to/Office/static/Capture_1

  # CODa
  python new_run.py --dataset coda --scene_path /path/to/coda_scene

  # ScanNet++
  python new_run.py --dataset scanepp --scene_path /path/to/scannetpp_scene

  # Open vocabulary with custom YOLO model
  python new_run.py --scene_path /path/to/scene --is_open_vocab --yolo_model yoloe-11l-seg-pf.pt

  # With visualization flags
  python new_run.py --scene_path /path/to/scene --rerun --vis_graph --save_graph
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.new_tracker import run_tracking
from core.object_registry import GlobalObjectRegistry
from core.scene_graph import SceneGraph
from data_loaders import get_loader
from depth_providers.factory import PROVIDER_CHOICES, build_depth_provider


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = _build_parser().parse_args()

    # --- Load config (default -> dataset yaml -> CLI overrides) ---
    cfg_dir = Path(__file__).parent / "configs"
    default_cfg = OmegaConf.load(cfg_dir / "core_tracking.yaml")

    ds_yaml = cfg_dir / f"{args.dataset}.yaml"
    if ds_yaml.exists():
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml))
    else:
        cfg = default_cfg

    # CLI overrides
    if args.yolo_model:
        cfg.yolo_model = args.yolo_model
    if args.is_open_vocab is not None:
        cfg.is_open_vocabulary = args.is_open_vocab
    if args.rerun:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.rerun = True
    if args.vis_graph:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.vis_graph = True
    if args.save_graph:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.save_graph = True
    if args.save_global_graph:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.save_global_graph = True

    dataset_name = args.dataset

    # --- Build loader ---
    scene_path = str(Path(args.scene_path).resolve())
    LoaderCls = get_loader(dataset_name)

    # Depth provider (from CLI or config)
    dp_type = args.depth_provider or str(cfg.get("depth_provider", "gt"))
    depth_provider = build_depth_provider(dp_type, dataset_name, scene_path, cfg)

    loader_kwargs = _build_loader_kwargs(dataset_name, cfg)
    loader = LoaderCls(scene_path, depth_provider=depth_provider, **loader_kwargs)

    print(f"\n{'=' * 60}")
    print(f"  RUN — {loader.scene_label}  (dataset: {dataset_name})")
    print(f"{'=' * 60}")

    n_frames = loader.get_num_frames()
    print(f"Frames: {n_frames}")

    intrinsics = loader.get_intrinsics()
    print(f"Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}  "
          f"image={intrinsics.width}x{intrinsics.height}")

    # --- Object registry ---
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.get("tracking_overlap_threshold", 0.1)),
        distance_threshold=float(cfg.get("tracking_distance_threshold", 1.0)),
        max_points=int(cfg.get("max_accumulated_points", 10000)),
        inactive_limit=int(cfg.get("tracking_inactive_limit", 0)),
        volume_ratio_threshold=float(cfg.get("tracking_volume_ratio_threshold", 0.1)),
        visibility_threshold=float(cfg.get("reprojection_visibility_threshold", 0.2)),
    )

    # SceneGraph
    scene_graph = SceneGraph(cfg.ssg)

    # --- Timings ---
    timings_agg = {"yolo": [], "preprocess": [], "tracking_3d": []}

    # --- Core tracking loop ---
    for tf in run_tracking(
        loader=loader,
        cfg=cfg,
        object_registry=object_registry,
    ):
        # Collect timings
        if "yolo_ms" in tf.timings:
            timings_agg["yolo"].append(tf.timings["yolo_ms"])
        if "preprocess_ms" in tf.timings:
            timings_agg["preprocess"].append(tf.timings["preprocess_ms"])
        if "tracking_3d_ms" in tf.timings:
            timings_agg["tracking_3d"].append(tf.timings["tracking_3d_ms"])

        n_objs = len(tf.objects)
        print(f"  Frame {tf.frame_idx}: {n_objs} objects tracked", end="\r")

        # Scene graph
        local_graph = scene_graph.generate_graph(
            frame_objects=tf.objects,
            object_registry=object_registry,
            T_w_c=tf.T_w_c,
            depth_m=tf.depth_m,
            intrinsics=intrinsics,
        )
        tf.local_graph = local_graph

        if cfg.ssg.save_local_graph:
            scene_graph.save_local_graph(scene_name=loader.scene_label)

    if cfg.ssg.save_global_graph:
        scene_graph.save_global_graph(scene_name=loader.scene_label)

    print()  # newline after \r

    # --- Summary ---
    _print_summary(object_registry, timings_agg)

    # --- Save global graph (placeholder) ---
    ssg_cfg = OmegaConf.to_container(cfg.get("ssg", {}), resolve=True)
    if ssg_cfg.get("save_graph", False) or ssg_cfg.get("save_global_graph", False):
        save_dir = Path(ssg_cfg.get("save_graph_dir", "results/scene_graphs"))
        _save_objects_json(object_registry, save_dir, dataset_name)

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Loader kwargs based on dataset type
# ═══════════════════════════════════════════════════════════════════════════

def _build_loader_kwargs(dataset_name: str, cfg) -> dict:
    kwargs = {}
    skip_labels = cfg.get("loader_skip_labels")
    if skip_labels:
        kwargs["skip_labels"] = set(skip_labels)

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


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(object_registry: GlobalObjectRegistry, timings: dict):
    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)
    all_objs = object_registry.get_all_objects()
    print(f"Total unique objects tracked: {len(all_objs)}")
    for gid, obj in all_objs.items():
        class_str = f" ({obj.get('class_name')})" if obj.get("class_name") else ""
        pts = obj.get("points_accumulated")
        n_pts = len(pts) if pts is not None else 0
        print(f"  Object {gid}{class_str}: "
              f"seen in {obj.get('observation_count', 0)} frames, "
              f"first: {obj.get('first_seen_frame')}, "
              f"last: {obj.get('last_seen_frame')}, "
              f"points: {n_pts}")

    print("\n" + "-" * 60)
    print("PERFORMANCE (ms)")
    print("-" * 60)
    for k, vals in timings.items():
        if vals:
            print(f"  {k:20s} {np.mean(vals):8.2f} ± {np.std(vals):.2f}")
    total_avg = sum(np.mean(v) for v in timings.values() if v)
    print(f"  {'Total per frame':20s} {total_avg:8.2f}")
    n_frames = max(len(v) for v in timings.values()) if timings else 0
    print(f"\nTotal frames processed: {n_frames}")
    print("=" * 60)


def _save_objects_json(
    object_registry: GlobalObjectRegistry,
    save_dir: Path,
    dataset_name: str,
):
    """Save tracked objects as JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    all_objs = object_registry.get_all_objects()

    nodes = {}
    for gid, obj in all_objs.items():
        bbox = obj.get("bbox_3d")
        bbox_dict = bbox.to_dict() if bbox is not None else None
        nodes[int(gid)] = {
            "track_id": int(gid),
            "class_name": obj.get("class_name"),
            "bbox_3d": bbox_dict,
            "observation_count": obj.get("observation_count", 0),
            "first_seen": obj.get("first_seen_frame"),
            "last_seen": obj.get("last_seen_frame"),
        }

    output = {
        "dataset": dataset_name,
        "num_objects": len(nodes),
        "nodes": nodes,
    }
    out_path = save_dir / "tracked_objects.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTracked objects saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run YOLO tracking pipeline on any scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="isaacsim",
                   choices=["isaacsim", "thud_synthetic", "coda", "scanepp"],
                   help="Dataset type (default: isaacsim).")
    p.add_argument("--scene_path", type=str, required=True,
                   help="Path to the scene directory.")
    p.add_argument("--depth_provider", type=str, default=None,
                   choices=list(PROVIDER_CHOICES),
                   help="Depth provider type (overrides config). "
                        "Default from config: 'gt'.")
    p.add_argument("--yolo_model", type=str, default=None,
                   help="Path to YOLOE model weights.")
    p.add_argument("--is_open_vocab", action="store_true", default=None,
                   help="Enable open-vocabulary mode.")
    p.add_argument("--rerun", action="store_true",
                   help="Enable Rerun 3D/2D visualisation (placeholder).")
    p.add_argument("--vis_graph", action="store_true",
                   help="Visualize scene graph per frame (placeholder).")
    p.add_argument("--save_graph", action="store_true",
                   help="Save scene graph as JSON.")
    p.add_argument("--save_global_graph", action="store_true",
                   help="Save global scene graph as JSON.")
    return p


if __name__ == "__main__":
    sys.exit(main())
