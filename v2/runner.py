#!/usr/bin/env python3
"""
v2/runner.py — Run the v2 tracking + scene-graph pipeline.

Usage
-----
  python -m v2.runner /path/to/scene_1
  python -m v2.runner /path/to/scene_1 --dataset isaacsim
  python -m v2.runner /path/to/scene_1 --dataset thud_synthetic
  python -m v2.runner --camera 0 --depth-provider pi3_online

  # Override config values
  python -m v2.runner /path/to/scene --conf 0.3 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from omegaconf import OmegaConf

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v2.depth_providers import (
    IsaacSimDepthProvider,
    Pi3OfflineDepthProvider,
    THUDSyntheticDepthProvider,
)
from v2.depth_providers.pi3_online import Pi3OnlineDepthProvider
from v2.loaders import get_loader
from v2.loaders.camera import CameraLoader
from v2.object_registry import GlobalObjectRegistry
from v2.ssg.graph_merge import merge_local_to_global
from v2.tracking import run_tracking_ssg
from v2.types import CameraIntrinsics


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

_CFG_DIR = Path(__file__).parent / "cfg"


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Merge default → dataset → CLI overrides."""
    default = OmegaConf.load(_CFG_DIR / "default.yaml")

    dataset = args.dataset or "isaacsim"
    ds_yaml = _CFG_DIR / f"{dataset}.yaml"
    ds_cfg = OmegaConf.load(ds_yaml) if ds_yaml.exists() else OmegaConf.create()

    cfg = OmegaConf.merge(default, ds_cfg)

    # CLI overrides
    if args.conf is not None:
        cfg.conf = args.conf
    if args.device is not None:
        cfg.device = args.device
    if args.depth_provider is not None:
        cfg.depth_provider = args.depth_provider
    cfg["dataset"] = dataset
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Depth provider factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_depth_provider(cfg, scene_dir: Optional[str]):
    kind = cfg.get("depth_provider", "gt")
    dataset = cfg.get("dataset", "isaacsim")

    if kind == "gt":
        if dataset == "isaacsim" and scene_dir:
            return IsaacSimDepthProvider(
                str(Path(scene_dir) / "depth"),
                png_max_value=int(cfg.get("png_max_value", 65535)),
                max_depth=float(cfg.get("max_depth", 10.0)),
                min_depth=float(cfg.get("min_depth", 0.01)),
            )
        elif dataset == "thud_synthetic" and scene_dir:
            return THUDSyntheticDepthProvider(
                str(Path(scene_dir) / "Depth"),
            )
        return None

    if kind == "pi3_offline":
        if scene_dir is None:
            raise ValueError("pi3_offline requires a scene directory")
        return Pi3OfflineDepthProvider(str(Path(scene_dir) / "rgb"))

    if kind == "pi3_online":
        return Pi3OnlineDepthProvider()

    raise ValueError(f"Unknown depth_provider: {kind}")


# ═══════════════════════════════════════════════════════════════════════════
# Loader factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_loader(cfg, args):
    """Construct the appropriate loader + depth provider."""
    dataset = cfg.get("dataset", "isaacsim")

    if args.camera is not None:
        dp = _build_depth_provider(cfg, None)
        intr = None
        if cfg.get("image_width"):
            intr = CameraIntrinsics.from_physical(
                cfg.get("focal_length", 50),
                cfg.get("horizontal_aperture", 80),
                cfg.get("vertical_aperture", 45),
                int(cfg.image_width), int(cfg.image_height),
            )
        return CameraLoader(
            camera_id=args.camera,
            intrinsics=intr,
            depth_provider=dp,
            max_frames=int(args.max_frames or 1000),
        )

    scene_dir = str(Path(args.path).resolve())
    dp = _build_depth_provider(cfg, scene_dir)

    LoaderCls = get_loader(dataset)
    # Build constructor kwargs from config
    kwargs: dict = {}
    if dp is not None:
        kwargs["depth_provider"] = dp
    skip = cfg.get("loader_skip_labels")
    if skip:
        kwargs["skip_labels"] = set(skip)
    if dataset == "isaacsim":
        for k in ("image_width", "image_height", "focal_length",
                   "horizontal_aperture", "vertical_aperture"):
            if cfg.get(k) is not None:
                kwargs[k] = cfg[k]
    if dataset == "thud_synthetic":
        if cfg.get("depth_scale"):
            kwargs["depth_scale"] = float(cfg.depth_scale)

    return LoaderCls(scene_dir, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _load_config(args)

    loader = _build_loader(cfg, args)
    print(f"\n{'=' * 60}")
    print(f"  v2 RUN — {loader.scene_label}  (dataset: {cfg.dataset})")
    print(f"  frames: {loader.get_num_frames()}")
    print(f"{'=' * 60}")

    # Warm up depth provider
    dp = loader.depth_provider
    if dp is not None:
        dp.warmup()

    # Object registry
    registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points=int(cfg.max_accumulated_points),
        inactive_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    # Global scene graph
    global_graph = nx.MultiDiGraph()

    # Timing accumulators
    timings_acc: dict = {}

    # --- Main loop ---
    for frame in run_tracking_ssg(loader, registry, cfg):
        # Merge local → global graph
        merge_local_to_global(global_graph, frame.local_graph, frame.objects, frame.frame_idx)

        # Accumulate timings
        for k, v in frame.timings.items():
            timings_acc.setdefault(k, []).append(v)

        # Progress
        if (frame.frame_idx + 1) % 20 == 0:
            n_edges = global_graph.number_of_edges()
            n_nodes = global_graph.number_of_nodes()
            print(f"  frame {frame.frame_idx + 1:>4d}  |  "
                  f"objects: {len(frame.objects):>3d}  |  "
                  f"global nodes: {n_nodes}  edges: {n_edges}")

    # --- Summary ---
    _print_summary(registry, timings_acc, global_graph)

    # --- Save graph ---
    ssg_cfg = OmegaConf.to_container(cfg.get("ssg", {}), resolve=True)
    if ssg_cfg.get("save_graph"):
        _save_graph(global_graph, registry, ssg_cfg.get("save_graph_dir", "results/scene_graphs_v2"),
                    loader.scene_label)

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(registry: GlobalObjectRegistry, timings: dict, graph: nx.MultiDiGraph):
    all_objs = registry.get_all_objects()
    print(f"\n{'=' * 60}")
    print("TRACKING & SCENE-GRAPH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique objects: {len(all_objs)}")
    for gid, obj in all_objs.items():
        cls = obj.get('class_name', '?')
        seen = obj.get('observation_count', 0)
        print(f"  [{gid:>3d}] {cls:<25s}  seen {seen}x")

    print(f"\nGlobal graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if timings:
        print(f"\nPerformance (avg ms per frame):")
        for k, vals in timings.items():
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {k:<20s}: {avg:>8.2f} ms")
    print(f"{'=' * 60}")


def _save_graph(graph: nx.MultiDiGraph, registry: GlobalObjectRegistry,
                save_dir: str, scene_label: str):
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes = {}
    for gid, obj in registry.get_all_objects().items():
        bbox = obj.get("bbox_3d")
        nodes[str(gid)] = {
            "class_name": obj.get("class_name"),
            "observation_count": obj.get("observation_count", 0),
            "bbox_3d": bbox.to_dict() if bbox else None,
        }

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({
            "src": u, "tgt": v,
            "label": data.get("label", ""),
            "label_class": data.get("label_class", ""),
            "weight": data.get("weight", 1),
        })

    payload = {"scene": scene_label, "nodes": nodes, "edges": edges}

    def _default(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    fp = out / f"{scene_label}_graph.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    print(f"Graph saved → {fp}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2 tracking + scene-graph pipeline")
    p.add_argument("path", nargs="?", default=None, help="Scene directory")
    p.add_argument("--dataset", default=None,
                   choices=["isaacsim", "thud_synthetic", "camera"],
                   help="Dataset type")
    p.add_argument("--camera", type=int, default=None,
                   help="Camera device ID for live capture")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames for camera mode")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--depth-provider", default=None,
                   choices=["gt", "pi3_offline", "pi3_online"])
    p.add_argument("--save-graph", action="store_true")
    return p


if __name__ == "__main__":
    sys.exit(main())
