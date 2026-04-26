#!/usr/bin/env python3
"""
run_per_frame_graphs.py — Run the YOLO tracking pipeline and save a
per-frame scene graph JSON for every frame.  No global graph is built
or maintained; each frame is independent.

Output structure
----------------
  <output_dir>/
    frame_000000.json
    frame_000001.json
    ...

Each JSON has the same schema as save_graph_json:
  {
    "frame_idx": 5,
    "frame_key": 50,          // dataset frame number (e.g. 50 for ScanNet++)
    "num_objects": 4,
    "num_edges": 3,
    "nodes": {
      "<id>": {
        "track_id": ...,
        "class_name": ...,
        "bbox_3d": {...},
        "edges": [{"target_id": ..., "label": ..., "label_class": ...}, ...]
      }
    }
  }

Usage
-----
  python run_per_frame_graphs.py --dataset scanetpp \
      --scene_path /path/to/scene \
      --depth_provider pi3_online \
      --output_dir results/per_frame_graphs/scene_name

  python run_per_frame_graphs.py --dataset isaacsim \
      --scene_path /path/to/scene \
      --output_dir results/per_frame_graphs/scene_name \
      --edges bs,sv
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

try:
    import torch
except Exception:
    torch = None

import networkx as nx

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.new_tracker import run_tracking
from core.object_registry import GlobalObjectRegistry
from core.types import CameraIntrinsics, TrackedObject
from data_loaders import get_loader
from depth_providers.factory import PROVIDER_CHOICES, build_depth_provider
from scene_graph.graph_utils import bbox3d_to_dict, save_graph_json
from scene_graph.predictors import (
    BaselineEdgePredictor,
    BasicEdgePredictor,
    EdgePredictor,
    VLSATEdgePredictor,
)


# ---------------------------------------------------------------------------
# Lightweight per-frame graph builder (no global accumulation)
# ---------------------------------------------------------------------------

class PerFrameGraphBuilder:
    """Build a fresh local scene graph for every frame — no global state."""

    def __init__(self, cfg) -> None:
        self._edge_predictors: List[EdgePredictor] = []

        if cfg.get("basic_edges", True):
            self._edge_predictors.append(BasicEdgePredictor())
            print("[SceneGraph] BasicEdgePredictor enabled.")
        if cfg.get("baseline_edges", True):
            self._edge_predictors.append(BaselineEdgePredictor())
            print("[SceneGraph] BaselineEdgePredictor enabled.")
        if cfg.get("vlsat_edges", False):
            self._edge_predictors.append(VLSATEdgePredictor(cfg))
            print("[SceneGraph] VLSATEdgePredictor enabled.")

    def build(
        self,
        frame_objects: List[TrackedObject],
        object_registry,
        T_w_c: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> nx.MultiDiGraph:
        """Return a per-frame scene graph (nodes + predicted edges)."""
        g = nx.MultiDiGraph()
        for obj in frame_objects:
            gid = obj.global_id
            g.add_node(gid, data={
                "global_id": gid,
                "track_id": gid,
                "class_name": obj.class_name,
                "bbox_3d": bbox3d_to_dict(obj.bbox_3d),
                "visible_current_frame": True,
            })

        for pred in self._edge_predictors:
            pred.predict(
                g,
                object_registry=object_registry,
                T_w_c=T_w_c,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )

        return g


def _graph_to_dict(
    graph: nx.MultiDiGraph,
    frame_idx: int,
    frame_key: int,
) -> dict:
    """Serialise a per-frame graph to a plain dict (mirrors save_graph_json)."""
    nodes: Dict[int, dict] = {}
    for nid, ndata in graph.nodes(data=True):
        d = ndata.get("data", {})
        edges_out = []
        for _, tgt, _, edata in graph.out_edges(nid, keys=True, data=True):
            entry = {
                "target_id": int(tgt),
                "label": edata.get("label", ""),
                "label_class": edata.get("label_class", ""),
            }
            if edata.get("label_subclass"):
                entry["label_subclass"] = edata["label_subclass"]
            edges_out.append(entry)
        nodes[int(nid)] = {
            "track_id": int(nid),
            "class_name": d.get("class_name"),
            "bbox_3d": d.get("bbox_3d"),
            "edges": edges_out,
        }

    return {
        "frame_idx": frame_idx,
        "frame_key": frame_key,
        "num_objects": len(nodes),
        "num_edges": graph.number_of_edges(),
        "nodes": nodes,
    }


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Shared helpers (lifted from new_run.py)
# ---------------------------------------------------------------------------

def _apply_pi3_online_transform(depth_provider, scene_path: str, cfg) -> None:
    transform_raw = cfg.get(
        "pi3_online_transform_path",
        cfg.get("pi3_offline_transform_path"),
    )
    if transform_raw is None:
        return
    tp = Path(transform_raw)
    if not tp.is_absolute():
        tp = Path(scene_path) / tp
    require = bool(cfg.get("pi3_online_require_transform", False))
    if not tp.exists():
        if require:
            raise FileNotFoundError(f"Pi3 alignment transform not found: {tp}")
        return
    from depth_providers.pi3_online import _load_sim3_matrix
    sim3 = _load_sim3_matrix(str(tp), require=True)
    depth_provider.set_sim3_transform(sim3)
    print(f"[Pi3] Sim(3) alignment loaded from {tp}")


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


def _parse_edges_cli(raw: str) -> set:
    tokens = [p.strip().lower() for chunk in raw.split(",") for p in chunk.split() if p.strip()]
    if not tokens:
        raise ValueError("empty edge selection")
    alias = {
        "all": "all", "*": "all",
        "sv": "sv", "sceneverse": "sv", "basic": "sv",
        "bs": "bs", "baseline": "bs",
        "vlsat": "vlsat", "vl-sat": "vlsat",
    }
    selected: set = set()
    for t in tokens:
        if t not in alias:
            raise ValueError(f"Unknown edge selector '{t}'. Allowed: all, bs, sv, vlsat")
        canon = alias[t]
        if canon == "all":
            return {"sv", "bs", "vlsat"}
        selected.add(canon)
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    # --- Config ---
    cfg_dir = Path(__file__).parent / "configs"
    cfg = OmegaConf.load(cfg_dir / "core_tracking.yaml")
    ds_yaml = cfg_dir / f"{args.dataset}.yaml"
    if ds_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ds_yaml))

    if args.yolo_model:
        cfg.yolo_model = args.yolo_model
    if args.is_open_vocab is not None:
        cfg.is_open_vocabulary = args.is_open_vocab

    # Edge predictor selection
    cfg.ssg = cfg.get("ssg", {})
    if args.edges is not None:
        try:
            selected = _parse_edges_cli(args.edges)
        except ValueError as exc:
            raise SystemExit(f"--edges: {exc}") from exc
        cfg.ssg.basic_edges = "sv" in selected
        cfg.ssg.baseline_edges = "bs" in selected
        cfg.ssg.vlsat_edges = "vlsat" in selected

    dataset_name = args.dataset
    scene_path = str(Path(args.scene_path).resolve())

    # --- Output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Depth provider ---
    dp_type = args.depth_provider or str(cfg.get("depth_provider", "gt"))
    depth_provider = build_depth_provider(dp_type, dataset_name, scene_path, cfg)

    if dp_type == "pi3_online":
        _apply_pi3_online_transform(depth_provider, scene_path, cfg)

    # --- Loader ---
    LoaderCls = get_loader(dataset_name)
    loader = LoaderCls(
        scene_path,
        depth_provider=depth_provider,
        **_build_loader_kwargs(dataset_name, cfg),
    )
    frame_numbers = getattr(loader, "_frame_numbers", None)

    n_frames = loader.get_num_frames()
    intrinsics = loader.get_intrinsics()

    print(f"\n{'=' * 60}")
    print(f"  PER-FRAME GRAPHS — {loader.scene_label}  (dataset: {dataset_name})")
    print(f"{'=' * 60}")
    print(f"Frames: {n_frames}")
    print(f"Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}  "
          f"image={intrinsics.width}x{intrinsics.height}")
    print(f"Output dir: {output_dir}")

    # --- Pi3 background feeder ---
    _pi3_feeder = None
    if dp_type == "pi3_online" and hasattr(depth_provider, "feed_frame"):
        def _feed_worker():
            for idx in range(n_frames):
                loader.get_rgb(idx)
            if hasattr(depth_provider, "drain"):
                depth_provider.drain()

        _pi3_feeder = threading.Thread(
            target=_feed_worker, daemon=True, name="pi3-feeder",
        )
        _pi3_feeder.start()
        print(f"[Pi3] Background depth feeder started ({n_frames} frames)")

    # --- Object registry (still needed for 3D tracking state) ---
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

    # --- Graph builder (no global state) ---
    graph_builder = PerFrameGraphBuilder(cfg.ssg)

    # --- Main loop ---
    t_start = time.perf_counter()
    saved = 0

    for tf in run_tracking(loader=loader, cfg=cfg, object_registry=object_registry):
        # Resolve the dataset frame key (e.g. 50 for ScanNet++ sequential idx=5)
        if isinstance(frame_numbers, list) and 0 <= tf.frame_idx < len(frame_numbers):
            frame_key = int(frame_numbers[tf.frame_idx])
        else:
            frame_key = int(tf.frame_idx)

        graph = graph_builder.build(
            frame_objects=tf.objects,
            object_registry=object_registry,
            T_w_c=tf.T_w_c,
            depth_m=tf.depth_m,
            intrinsics=intrinsics,
        )

        payload = _graph_to_dict(graph, frame_idx=tf.frame_idx, frame_key=frame_key)

        out_path = output_dir / f"frame_{tf.frame_idx:06d}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

        saved += 1
        print(
            f"  Frame {tf.frame_idx:4d} (key={frame_key:6d}): "
            f"{len(tf.objects):3d} objects, "
            f"{graph.number_of_edges():3d} edges  → {out_path.name}",
            end="\r",
        )

    if _pi3_feeder is not None:
        _pi3_feeder.join(timeout=300)

    elapsed = time.perf_counter() - t_start
    print(f"\n\nDone. {saved} graphs saved to {output_dir}  ({elapsed:.1f}s total)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Save a per-frame scene graph JSON for every frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="isaacsim",
                   choices=["isaacsim", "thud_synthetic", "coda", "scanetpp"],
                   help="Dataset type (default: isaacsim).")
    p.add_argument("--scene_path", type=str, required=True,
                   help="Path to the scene directory.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where per-frame JSON files will be written.")
    p.add_argument("--depth_provider", type=str, default=None,
                   choices=list(PROVIDER_CHOICES),
                   help="Depth provider type (overrides config, default: 'gt').")
    p.add_argument("--yolo_model", type=str, default=None,
                   help="Path to YOLOE model weights.")
    p.add_argument("--is_open_vocab", action="store_true", default=None,
                   help="Enable open-vocabulary mode.")
    p.add_argument("--edges", type=str, default=None,
                   help="Edge predictors: all | bs | sv | vlsat, comma-separated.")
    return p


if __name__ == "__main__":
    sys.exit(main())
