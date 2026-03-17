#!/usr/bin/env python3
"""
run.py — Run the YOLO-SSG tracking + scene-graph pipeline on any sequence.

Usage
-----
  # IsaacSim scene (default dataset)
  python run.py /path/to/scene_1

  # THUD Synthetic scene
  python run.py /path/to/Gym/static/Capture_1 --dataset thud_synthetic

  # THUD Real scene
  python run.py /path/to/Real_Scenes/10L/static/Capture_1 --dataset thud_real

  # Override config values from CLI
  python run.py /path/to/scene --conf 0.3 --show-pcds --vis-graph

  # Use custom YAML config
  python run.py /path/to/scene --config configs/isaacsim.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry
from core.tracker import run_tracking
from loaders import get_loader
from ssg.ssg_main import edges

import matplotlib.pyplot as plt
import networkx as nx


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
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Load config (default → dataset yaml → CLI overrides) ----------------
    default_cfg = OmegaConf.load(Path(__file__).parent / "configs" / "core_tracking.yaml")
    if args.dataset:
        ds_yaml = Path(__file__).parent / "configs" / f"{args.dataset}.yaml"
        if ds_yaml.exists():
            cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml))
            print(f"Merged 2 cofs {args.dataset}")
        else:
            cfg = default_cfg
    else:
        cfg = default_cfg

    # CLI overrides
    if args.show_pcds:
        cfg.ssg.show_pcds = True
    if args.vis_graph:
        cfg.ssg.vis_graph = True
    if args.print_resources:
        cfg.ssg.print_resource_usage = True
    if args.print_tracking:
        cfg.ssg.print_tracking_info = True
    if args.save_graph:
        cfg.ssg.save_graph = True

    dataset_name = cfg.get("dataset", args.dataset or "isaacsim")

    # --- Configure globals from config (device, tracker, depth, intrinsics) --
    yutils.configure_globals(cfg)

    # --- Build loader --------------------------------------------------------
    scene_path = str(Path(args.path).resolve())
    scene_path = "/home/yehia/Downloads/kg_nav_IsaacSimData/kg_nav_IsaacSimData/scene_2"
    LoaderCls = get_loader(dataset_name)

    loader_kwargs = {}
    skip_labels = cfg.get("loader_skip_labels")

    # based on scene type, get the appropreate dataset config values, like if thud, then thud config

    if skip_labels:
        loader_kwargs["skip_labels"] = set(skip_labels)
    if dataset_name in ("thud_synthetic", "thud_real"):
        loader_kwargs["depth_scale"] = float(cfg.get("depth_scale", 1000.0))
        loader_kwargs["depth_max_m"] = float(cfg.get("depth_max_m", 100.0))
    if dataset_name == "thud_real":
        loader_kwargs["tracking_distance"] = float(cfg.get("real_tracking_distance", 0.3))
    if dataset_name == "isaacsim":
        loader_kwargs["image_width"] = int(cfg.get("image_width", 1280))
        loader_kwargs["image_height"] = int(cfg.get("image_height", 720))
        loader_kwargs["focal_length"] = float(cfg.get("focal_length", 50))
        loader_kwargs["horizontal_aperture"] = float(cfg.get("horizontal_aperture", 80))
        loader_kwargs["vertical_aperture"] = float(cfg.get("vertical_aperture", 45))

    loader = LoaderCls(scene_path, **loader_kwargs)
    print(f"\n{'=' * 60}")
    print(f"  RUN – {loader.scene_label}  (dataset: {dataset_name})")
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

    # Track classes (for THUD Real)
    class_names_to_track = list(cfg.get("scene_0_class_names", [])) or None

    # --- Object registry -----------------------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    # --- SSG state -----------------------------------------------------------
    persistent_graph = nx.MultiDiGraph()
    timings = {"yolo": [], "preprocess": [], "create_3d": [], "edges": [], "merge": []}
    gpu_usage = {"yolo": [], "edges": []}
    cuda_available = torch.cuda.is_available()
    ssg_cfg = OmegaConf.to_container(cfg.get("ssg", {}), resolve=True)

    # --- Core tracking loop --------------------------------------------------
    for tf in run_tracking(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        depth_cache=depth_cache,
        poses=poses,
        cfg=cfg,
        object_registry=object_registry,
        class_names_to_track=class_names_to_track,
    ):
        # Collect timings from core tracker
        for k in ("yolo", "preprocess", "create_3d"):
            if k in tf.timings:
                timings[k].append(tf.timings[k])

        # GPU usage after YOLO
        if cuda_available:
            torch.cuda.synchronize()
            gpu_usage["yolo"].append(torch.cuda.memory_allocated() / (1024**2))

        # Print tracking info
        if ssg_cfg.get("print_tracking_info", False):
            summary = object_registry.get_tracking_summary()
            yolo_detected = sum(1 for obj in tf.frame_objs if obj.get("match_source") != "reprojection")
            reproj_visible = sum(1 for obj in tf.frame_objs if obj.get("match_source") == "reprojection")
            print(f"  [Track] Frame {tf.frame_idx}: YOLO={yolo_detected}, "
                  f"Reproj={reproj_visible}, Total={len(tf.frame_objs)}, "
                  f"Registry={summary['total_objects']}")

        # --- Edge prediction (SSG) -------------------------------------------
        t0 = time.perf_counter()
        edges(tf.scene_graph, tf.frame_objs, tf.T_w_c, tf.depth_m)
        timings["edges"].append((time.perf_counter() - t0) * 1000)

        if cuda_available:
            torch.cuda.synchronize()
            gpu_usage["edges"].append(torch.cuda.memory_allocated() / (1024**2))

        # --- Visualize reconstruction ----------------------------------------
        if ssg_cfg.get("show_pcds", False):
            yutils.visualize_reconstruction(
                object_registry=object_registry,
                frame_index=tf.frame_idx,
                show_visible_only=False,
                show_aabb=True,
            )

        # --- Merge graphs ----------------------------------------------------
        t0 = time.perf_counter()
        persistent_graph = yutils.merge_scene_graphs(persistent_graph, tf.scene_graph)
        timings["merge"].append((time.perf_counter() - t0) * 1000)

        # --- Visualize graph -------------------------------------------------
        if ssg_cfg.get("vis_graph", False):
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            yutils.draw_labeled_multigraph(persistent_graph, ax=axes[0])
            axes[0].set_title(f"Persistent Graph frame: {tf.frame_idx}")
            yutils.draw_labeled_multigraph(tf.scene_graph, ax=axes[1])
            axes[1].set_title(f"Current Graph frame: {tf.frame_idx}")
            plt.tight_layout()
            plt.show()

        if ssg_cfg.get("print_resource_usage", False):
            print("=" * 50)
            print(f"[run] Frame {tf.frame_idx}: Latency (ms) - "
                  f"preprocess: {timings['preprocess'][-1]:.2f}, "
                  f"create_3d: {timings['create_3d'][-1]:.2f}, "
                  f"edges: {timings['edges'][-1]:.2f}, "
                  f"yolo: {timings['yolo'][-1]:.2f}")
            if cuda_available and gpu_usage["yolo"]:
                print(f"[run] Frame {tf.frame_idx}: GPU mem (MB) - "
                      f"yolo: {gpu_usage['yolo'][-1]:.1f}, "
                      f"edges: {gpu_usage['edges'][-1]:.1f}")
            print("=" * 50)

    # --- Summary -------------------------------------------------------------
    _print_summary(object_registry, timings, gpu_usage, cuda_available)

    # --- Save graph as JSON --------------------------------------------------
    if ssg_cfg.get("save_graph", False):
        save_dir = Path(ssg_cfg.get("save_graph_dir", "results/scene_graphs"))
        _save_graph_json(persistent_graph, object_registry, save_dir, dataset_name)

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(object_registry, timings, gpu_usage, cuda_available):
    print("\n" + "=" * 60)
    print("TRACKING & RECONSTRUCTION SUMMARY")
    print("=" * 60)
    all_objs = object_registry.get_all_objects()
    print(f"Total unique objects tracked: {len(all_objs)}")
    for gid, obj in all_objs.items():
        class_str = f" ({obj.get('class_name')})" if obj.get("class_name") else ""
        visible_str = "VISIBLE" if obj.get("visible_current_frame") else "not visible"
        print(f"  Object {gid}{class_str}: {visible_str}, "
              f"seen in {obj['observation_count']} frames, "
              f"first: {obj['first_seen_frame']}, last: {obj['last_seen_frame']}, "
              f"points: {len(obj['points_accumulated'])}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    if timings["preprocess"]:
        print(f"\nLatency Averages (ms):")
        for k in ("preprocess", "create_3d", "edges", "yolo", "merge"):
            vals = timings.get(k, [])
            if vals:
                print(f"  {k:20s} {np.mean(vals):8.2f} ± {np.std(vals):.2f}")
        total_avg = sum(np.mean(timings[k]) for k in timings if timings[k])
        print(f"  {'Total per frame':20s} {total_avg:8.2f}")

    if cuda_available and gpu_usage.get("yolo"):
        print(f"\nGPU Memory Usage Averages (MB):")
        print(f"  After YOLO:  {np.mean(gpu_usage['yolo']):.1f} ± {np.std(gpu_usage['yolo']):.1f}")
        print(f"  After Edges: {np.mean(gpu_usage['edges']):.1f} ± {np.std(gpu_usage['edges']):.1f}")

    n_frames = len(timings.get("preprocess", []))
    print(f"\nTotal frames processed: {n_frames}")
    print("=" * 60)


def _save_graph_json(graph, object_registry, save_dir: Path, dataset_name: str):
    """Serialize the persistent scene graph to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)

    all_objs = object_registry.get_all_objects()

    def _bbox_to_serializable(bbox_3d):
        if bbox_3d is None:
            return None
        result = {}
        aabb = bbox_3d.get("aabb")
        if aabb is not None:
            result["aabb"] = {
                "min": np.asarray(aabb["min"]).tolist() if aabb.get("min") is not None else None,
                "max": np.asarray(aabb["max"]).tolist() if aabb.get("max") is not None else None,
            }
        obb = bbox_3d.get("obb")
        if obb is not None:
            result["obb"] = {
                "center": np.asarray(obb["center"]).tolist() if obb.get("center") is not None else None,
                "extent": np.asarray(obb["extent"]).tolist() if obb.get("extent") is not None else None,
            }
        return result

    # Build nodes dict keyed by global (track) id
    nodes = {}
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        obj = all_objs.get(node_id, {})
        # Prefer registry data, fall back to graph node data
        bbox_3d = obj.get("bbox_3d") or (node_data.get("data", {}).get("bbox_3d") if isinstance(node_data.get("data"), dict) else None)
        class_name = obj.get("class_name") or (node_data.get("data", {}).get("class_name") if isinstance(node_data.get("data"), dict) else None)

        # Collect outgoing edges
        edges_list = []
        for _, target, _, edge_data in graph.out_edges(node_id, keys=True, data=True):
            edges_list.append({
                "target_id": int(target),
                "relation_type": edge_data.get("label", edge_data.get("label_class", "")),
            })

        nodes[int(node_id)] = {
            "track_id": int(node_id),
            "class_name": class_name,
            "bbox_3d": _bbox_to_serializable(bbox_3d),
            "edges": edges_list,
        }

    output = {
        "dataset": dataset_name,
        "num_objects": len(nodes),
        "nodes": nodes,
    }

    out_path = save_dir / "scene_graph.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScene graph saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run YOLO-SSG tracking + scene-graph pipeline on any sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  python run.py /data/scene_1
  python run.py /data/Gym/static/Capture_1 --dataset thud_synthetic
  python run.py /data/scene_1 --config configs/isaacsim.yaml --show-pcds
""",
    )
    p.add_argument("--path", default="/home/yehia/Downloads/kg_nav_IsaacSimData/kg_nav_IsaacSimData/scene_0", help="Path to the scene directory.")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["isaacsim", "thud_synthetic", "thud_real", "code", "any_scene"],
                   help="Dataset type (default: 'isaacsim').")
    # SSG visualisation
    p.add_argument("--show-pcds", action="store_true", help="Show point cloud reconstruction.")
    p.add_argument("--vis-graph", action="store_true", help="Show scene graph per frame.")
    p.add_argument("--print-resources", action="store_true", help="Print per-frame resource usage.")
    p.add_argument("--print-tracking", action="store_true", help="Print per-frame tracking info.")
    p.add_argument("--save-graph", action="store_true", help="Save final scene graph as JSON.")
    return p


if __name__ == "__main__":
    sys.exit(main())
