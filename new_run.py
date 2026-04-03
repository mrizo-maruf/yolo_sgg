
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
  python new_run.py --dataset scanetpp --scene_path /path/to/scannetpp_scene

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
try:
    import torch
except Exception:
    torch = None

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.new_tracker import run_tracking
from core.object_registry import GlobalObjectRegistry
from core.scene_graph import SceneGraph
from data_loaders import get_loader
from depth_providers.factory import PROVIDER_CHOICES, build_depth_provider


def _cuda_mem_mb(cuda_available: bool) -> float | None:
    if not cuda_available:
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.memory_allocated() / (1024 ** 2))


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
    if args.vis_edge:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.vis_edge = True
    if args.print_resources:
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.print_resource_usage = True
    if args.edges is not None:
        try:
            selected_edges = _parse_edges_cli(args.edges)
        except ValueError as exc:
            raise SystemExit(f"--edges: {exc}") from exc
        cfg.ssg = cfg.get("ssg", {})
        cfg.ssg.basic_edges = "sv" in selected_edges
        cfg.ssg.baseline_edges = "bs" in selected_edges
        cfg.ssg.vlsat_edges = "vlsat" in selected_edges

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
        merge_iou_threshold=float(cfg.get("merge_iou_threshold", 0.5)),
        merge_containment_threshold=float(cfg.get("merge_containment_threshold", 0.7)),
    )

    # SceneGraph
    scene_graph = SceneGraph(cfg.ssg)

    # --- Rerun visualizer (optional) ---
    ssg_cfg = OmegaConf.to_container(cfg.get("ssg", {}), resolve=True)
    print_resource_usage = bool(ssg_cfg.get("print_resource_usage", False))
    rerun_vis = None
    if ssg_cfg.get("rerun", False):
        from rerun_utils import RerunVisualizer, _build_axis_remap_matrix

        apply_isaac_axis_fix = bool(ssg_cfg.get("isaac_axis_fix", dataset_name == "isaacsim"))
        axis_remap = None
        if apply_isaac_axis_fix and dataset_name == "isaacsim":
            # Isaac/world is typically RFU (Z-up), while this Rerun setup uses RDF.
            # Remap RFU -> RDF: (x, y, z) -> (x, -z, y).
            axis_remap = _build_axis_remap_matrix(swap_yz=True, flip_y=True)
            print("[Rerun] Applying Isaac axis remap (RFU -> RDF).")
        elif dataset_name == "scanetpp":
            # ScanNet++ world is Z-up; Rerun view is RDF.
            # Remap: (x, y, z) -> (x, -z, y) to fix 90° X-axis inversion.
            axis_remap = _build_axis_remap_matrix(swap_yz=True, flip_y=True)
            print("[Rerun] Applying ScanNet++ axis remap (Z-up -> RDF).")

        rerun_vis = RerunVisualizer(
            recording_id=f"yolo_ssg_{dataset_name}",
            axis_remap=axis_remap,
        )
        rerun_vis.init(
            img_w=intrinsics.width,
            img_h=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
        )

    # --- Timings ---
    timings_agg = {
        "yolo": [],
        "preprocess": [],
        "depth": [],
        "pcd_extract": [],
        "track_update": [],
        "reprojection": [],
        "tracking_3d": [],
        "graph": [],
    }
    gpu_usage = {
        "after_yolo": [],
        "after_preprocess": [],
        "after_pcd": [],
        "after_tracking_3d": [],
        "after_graph": [],
    }
    cuda_available = bool(torch is not None and torch.cuda.is_available())

    # --- Core tracking loop ---
    for tf in run_tracking(
        loader=loader,
        cfg=cfg,
        object_registry=object_registry,
    ):
        n_objs = len(tf.objects)
        if not print_resource_usage:
            print(f"  Frame {tf.frame_idx}: {n_objs} objects tracked", end="\r")

        # Scene graph
        t0 = time.perf_counter()
        local_graph = scene_graph.generate_graph(
            frame_objects=tf.objects,
            object_registry=object_registry,
            T_w_c=tf.T_w_c,
            depth_m=tf.depth_m,
            intrinsics=intrinsics,
        )
        tf.timings["graph_ms"] = (time.perf_counter() - t0) * 1000
        gpu_after_graph = _cuda_mem_mb(cuda_available)
        if gpu_after_graph is not None:
            tf.timings["gpu_after_graph_mb"] = gpu_after_graph
        tf.local_graph = local_graph

        # Collect timings
        key_map = {
            "yolo_ms": "yolo",
            "preprocess_ms": "preprocess",
            "depth_ms": "depth",
            "pcd_extract_ms": "pcd_extract",
            "track_update_ms": "track_update",
            "reprojection_ms": "reprojection",
            "tracking_3d_ms": "tracking_3d",
            "graph_ms": "graph",
        }
        for src, dst in key_map.items():
            if src in tf.timings:
                timings_agg[dst].append(tf.timings[src])

        gpu_map = {
            "gpu_after_yolo_mb": "after_yolo",
            "gpu_after_preprocess_mb": "after_preprocess",
            "gpu_after_pcd_mb": "after_pcd",
            "gpu_after_tracking_3d_mb": "after_tracking_3d",
            "gpu_after_graph_mb": "after_graph",
        }
        for src, dst in gpu_map.items():
            if src in tf.timings:
                gpu_usage[dst].append(tf.timings[src])

        if print_resource_usage:
            print("=" * 50)
            print(
                f"[new_run] Frame {tf.frame_idx}: Latency (ms) - "
                f"yolo: {tf.timings.get('yolo_ms', 0.0):.2f}, "
                f"preprocess: {tf.timings.get('preprocess_ms', 0.0):.2f}, "
                f"depth: {tf.timings.get('depth_ms', 0.0):.2f}, "
                f"pcd: {tf.timings.get('pcd_extract_ms', 0.0):.2f}, "
                f"track_update: {tf.timings.get('track_update_ms', 0.0):.2f}, "
                f"reproj: {tf.timings.get('reprojection_ms', 0.0):.2f}, "
                f"tracking_3d: {tf.timings.get('tracking_3d_ms', 0.0):.2f}, "
                f"graph: {tf.timings.get('graph_ms', 0.0):.2f}"
            )
            if cuda_available and gpu_usage["after_yolo"]:
                print(
                    f"[new_run] Frame {tf.frame_idx}: GPU mem (MB) - "
                    f"after_yolo: {tf.timings.get('gpu_after_yolo_mb', 0.0):.1f}, "
                    f"after_tracking_3d: {tf.timings.get('gpu_after_tracking_3d_mb', 0.0):.1f}, "
                    f"after_graph: {tf.timings.get('gpu_after_graph_mb', 0.0):.1f}"
                )
            print("=" * 50)

        if cfg.ssg.save_local_graph:
            scene_graph.save_local_graph(scene_name=loader.scene_label)

        # --- Rerun visualisation ---
        if rerun_vis is not None:
            rerun_vis.log_frame(
                frame_idx=tf.frame_idx,
                object_registry=object_registry,
                persistent_graph=scene_graph.global_graph,
                T_w_c=tf.T_w_c,
                rgb_path=tf.rgb_path,
                masks_clean=tf.masks,
                track_ids=tf.track_ids if tf.track_ids is not None else np.array([], dtype=int),
                class_names=tf.class_names,
                vis_edges=ssg_cfg.get("vis_edge", False),
            )

    if cfg.ssg.save_global_graph:
        scene_graph.save_global_graph(scene_name=loader.scene_label)

    print()  # newline after \r

    # --- Summary ---
    _print_summary(object_registry, timings_agg, gpu_usage, cuda_available)

    # --- Save global graph ---
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
    elif dataset_name == "scanetpp":
        kwargs["fx"] = float(cfg.get("fx", 692.52))
        kwargs["fy"] = float(cfg.get("fy", 693.83))
        kwargs["cx"] = float(cfg.get("cx", 459.76))
        kwargs["cy"] = float(cfg.get("cy", 344.76))
        kwargs["image_width"] = int(cfg.get("image_width", 920))
        kwargs["image_height"] = int(cfg.get("image_height", 690))

    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(
    object_registry: GlobalObjectRegistry,
    timings: dict,
    gpu_usage: dict,
    cuda_available: bool,
):
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
    main_stages = ("yolo", "preprocess", "tracking_3d", "graph")
    breakdown_stages = ("depth", "pcd_extract", "track_update", "reprojection")

    for k in main_stages:
        vals = timings.get(k, [])
        if vals:
            print(f"  {k:20s} {np.mean(vals):8.2f} ± {np.std(vals):.2f}")

    if any(timings.get(k) for k in breakdown_stages):
        print("\nTracking 3D Breakdown (ms):")
        for k in breakdown_stages:
            vals = timings.get(k, [])
            if vals:
                print(f"  {k:20s} {np.mean(vals):8.2f} ± {np.std(vals):.2f}")

    total_avg = sum(np.mean(timings[k]) for k in main_stages if timings.get(k))
    print(f"  {'Total per frame':20s} {total_avg:8.2f}")

    if cuda_available and gpu_usage.get("after_yolo"):
        print("\nGPU Memory Usage Averages (MB):")
        for key in ("after_yolo", "after_tracking_3d", "after_graph"):
            vals = gpu_usage.get(key, [])
            if vals:
                print(f"  {key:20s} {np.mean(vals):8.1f} ± {np.std(vals):.1f}")
    elif cuda_available:
        print("\nGPU Memory: CUDA available but no measurements recorded")
    else:
        print("\nGPU Memory: CUDA not available")

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


def _parse_edges_cli(raw: str) -> set[str]:
    """Parse CLI edge selection string into canonical keys: {sv, bs, vlsat}."""
    tokens: list[str] = []
    for chunk in str(raw).split(","):
        for part in chunk.split():
            p = part.strip().lower()
            if p:
                tokens.append(p)

    if not tokens:
        raise ValueError("empty edge selection")

    alias = {
        "all": "all",
        "*": "all",
        "sv": "sv",
        "sceneverse": "sv",
        "scenverse": "sv",
        "scene-verse": "sv",
        "basic": "sv",
        "bs": "bs",
        "baseline": "bs",
        "vlsat": "vlsat",
        "vl-sat": "vlsat",
    }

    selected: set[str] = set()
    for t in tokens:
        if t not in alias:
            allowed = "all, bs, sv, vlsat (comma-separated)"
            raise ValueError(f"Unknown edge selector '{t}'. Allowed: {allowed}")
        canon = alias[t]
        if canon == "all":
            return {"sv", "bs", "vlsat"}
        selected.add(canon)
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run YOLO tracking pipeline on any scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="isaacsim",
                   choices=["isaacsim", "thud_synthetic", "coda", "scanetpp"],
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
                   help="Enable Rerun 3D/2D visualisation.")
    p.add_argument("--vis_edge", action="store_true",
                   help="Show scene-graph edges in the Rerun 3D view.")
    p.add_argument("--vis_graph", action="store_true",
                   help="Visualize scene graph per frame.")
    p.add_argument("--print_resources", action="store_true",
                   help="Print per-frame latency and GPU resource usage.")
    p.add_argument("--save_graph", action="store_true",
                   help="Save scene graph as JSON.")
    p.add_argument("--save_global_graph", action="store_true",
                   help="Save global scene graph as JSON.")
    p.add_argument("--edges", type=str, default=None,
                   help="Edge predictors to run: all | bs | sv | vlsat, or comma-separated combos "
                        "(e.g. bs,sv).")
    return p


if __name__ == "__main__":
    sys.exit(main())
