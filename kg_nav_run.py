#!/usr/bin/env python3
"""
kg_nav_run.py — Tracking-only loop, compute scene graph once at the end
from ALL objects in the global registry, then save.

Usage
-----
  python kg_nav_run.py --dataset isaacsim --save-graph
  python kg_nav_run.py --dataset isaacsim --show-pcds --vis-graph
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

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry, compute_3d_bboxes
from core.tracker import run_tracking
from loaders import get_loader
from ssg.ssg_main import edges

import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# VL-SAT edge predictor paths
_VLSAT_DIR = Path(__file__).resolve().parent / "vl_sat_model"
_VLSAT_CONFIG = _VLSAT_DIR / "config" / "mmgnet.json"
_VLSAT_CKPT = _VLSAT_DIR / "3dssg_best_ckpt"
_VLSAT_RELS = _VLSAT_DIR / "config" / "relationships.txt"


def _init_vlsat_predictor():
    """Lazily import and initialise the VL-SAT EdgePredictor.

    Puts ``vl_sat_model/`` at the **very front** of *sys.path* so that
    bare ``from utils import …`` and ``from data_processing import …``
    inside VL-SAT code resolve to the correct sub-packages instead of
    colliding with ``yolo_sgg/utils.py``.
    """
    vlsat_dir = str(_VLSAT_DIR)

    # Guarantee vlsat_dir is at sys.path[0] — remove first, then insert.
    while vlsat_dir in sys.path:
        sys.path.remove(vlsat_dir)
    sys.path.insert(0, vlsat_dir)

    # If 'utils' was already imported (e.g. yolo_sgg/utils.py), drop it
    # from the module cache so the VL-SAT utils *package* is used instead.
    if "utils" in sys.modules:
        _old_utils = sys.modules.pop("utils")
        # Also drop any sub-modules cached under the old utils
        for key in list(sys.modules):
            if key.startswith("utils."):
                sys.modules.pop(key, None)

    from vl_sat_interface import EdgePredictor          # noqa: E402

    predictor = EdgePredictor(
        config_path=str(_VLSAT_CONFIG),
        ckpt_path=str(_VLSAT_CKPT),
        relationships_list=str(_VLSAT_RELS),
    )
    return predictor


def _override_intrinsics(K, img_h, img_w):
    yutils.fx = float(K[0, 0])
    yutils.fy = float(K[1, 1])
    yutils.cx = float(K[0, 2])
    yutils.cy = float(K[1, 2])
    yutils.IMAGE_WIDTH = img_w
    yutils.IMAGE_HEIGHT = img_h


def _build_graph_from_registry(object_registry):
    """Build a nx.MultiDiGraph with one node per tracked object from the registry."""
    graph = nx.MultiDiGraph()
    for gid, obj in object_registry.get_all_objects().items():
        lightweight = {
            'global_id': gid,
            'track_id': gid,
            'class_name': obj.get('class_name'),
            'bbox_3d': obj.get('bbox_3d'),
            'visible_current_frame': True,
        }
        graph.add_node(gid, data=lightweight)
    return graph


def _egoview_project(pos_world: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """Project a world-frame 3D position into the camera's local frame.

    Returns (1, 3) array where:
      X = camera-right, Y = camera-up, Z = camera-forward (into scene).
    """
    T_c_w = np.linalg.inv(T_w_c)
    R = T_c_w[:3, :3]
    t = T_c_w[:3, 3]
    pos_cam = (R @ np.asarray(pos_world).reshape(3)) + t
    return pos_cam.reshape(1, 3)


def _get_obj_center(obj: dict) -> np.ndarray | None:
    """Extract the 3D centre from an object's bbox."""
    bbox = obj.get('bbox_3d')
    if bbox is None:
        return None
    obb = bbox.get('obb')
    if obb is not None and obb.get('center') is not None:
        return np.asarray(obb['center'], dtype=float)
    aabb = bbox.get('aabb')
    if aabb is not None and aabb.get('min') is not None and aabb.get('max') is not None:
        return (np.asarray(aabb['min'], dtype=float) + np.asarray(aabb['max'], dtype=float)) / 2.0
    return None


def edges_bs(all_objs: dict, T_w_c: np.ndarray) -> dict:
    """Baseline edge predictor: egocentric spatial relations.

    For every ordered pair (anchor, target) computes which of
    {left, right, front, back, above, below} hold in camera frame.

    Returns:
        dict  node_id -> list of {"target_id": int, "relation_type": str}
    """
    # Pre-compute camera-frame centres for every object
    cam_centres: dict[int, np.ndarray] = {}
    for gid, obj in all_objs.items():
        c = _get_obj_center(obj)
        if c is not None:
            cam_centres[gid] = _egoview_project(c, T_w_c)

    bs_edges: dict[int, list] = {gid: [] for gid in all_objs}

    for (gid_a, pos_a), (gid_b, pos_b) in combinations(cam_centres.items(), 2):
        # a -> b direction ("b is ___ of a")
        def _relations(src_pos, tgt_pos):
            rels = []
            if tgt_pos[0, 0] < src_pos[0, 0]:
                rels.append("left")
            if tgt_pos[0, 0] > src_pos[0, 0]:
                rels.append("right")
            if tgt_pos[0, 2] < src_pos[0, 2]:
                rels.append("front")
            if tgt_pos[0, 2] > src_pos[0, 2]:
                rels.append("back")
            if tgt_pos[0, 1] < src_pos[0, 1]:
                rels.append("above")
            if tgt_pos[0, 1] > src_pos[0, 1]:
                rels.append("below")
            return rels

        for rel in _relations(pos_a, pos_b):
            bs_edges[gid_a].append({"target_id": int(gid_b), "relation_type": rel})
        for rel in _relations(pos_b, pos_a):
            bs_edges[gid_b].append({"target_id": int(gid_a), "relation_type": rel})

    return bs_edges


def edges_vlsat(
    all_objs: dict,
    vlsat_predictor,
    relation_threshold: float = 0.35,
    max_relations_per_pair: int = 3,
    force_single_label: bool = False,
    ensure_pairwise_edges: bool = True,
    min_points_per_object: int = 10,
    debug_scores: bool = False,
) -> dict:
    """VL-SAT neural edge predictor.

    Extracts accumulated point clouds from the registry, converts the
    axis convention from our world frame (Z-up) to the 3RScan / 3DSSG
    convention (Y-up) used during VL-SAT training, then runs the
    pre-trained MMGNet model.

    Args:
        all_objs:  dict from ``object_registry.get_all_objects()``.
        vlsat_predictor:  initialised ``EdgePredictor`` instance.

    Decoding:
        MMGNet is commonly trained with ``multi_rel_outputs=true`` (multi-label),
        so each edge can have multiple valid relations. In that case we keep all
        relations with score >= ``relation_threshold`` (optionally capped by
        ``max_relations_per_pair``). If no relation passes threshold, we skip
        the edge ("none").

        If ``force_single_label=True`` (or model is single-label), we use
        ``argmax`` decoding for one relation per directed pair.

    Returns:
        dict  node_id -> list of {"target_id": int, "relation_type": str}
    """
    # Collect objects with enough accumulated points
    valid_ids: list[int] = []
    point_clouds: list[np.ndarray] = []
    skipped_for_points = 0

    for gid, obj in all_objs.items():
        pts = obj.get('points_accumulated')
        if pts is not None and len(pts) >= min_points_per_object:
            valid_ids.append(gid)
            # Axis convention: our pipeline stores world-frame points
            # with Z-up (IsaacSim).  3RScan / VL-SAT uses Y-up.
            # Swap:  (X, Y, Z)_ours  →  (X, Z, -Y)_3rscan
            pts_conv = pts[:, [0, 2, 1]].copy()   # Y ↔ Z
            pts_conv[:, 2] = -pts_conv[:, 2]       # negate new-Z (was Y)
            point_clouds.append(pts_conv)
        else:
            skipped_for_points += 1

    vlsat_edges: dict[int, list] = {gid: [] for gid in all_objs}

    if len(valid_ids) < 2:
        print(
            "[edges_vlsat] Need ≥2 objects with enough points "
            f"(min_points_per_object={min_points_per_object}), "
            f"valid={len(valid_ids)}, skipped={skipped_for_points}. Skipping."
        )
        return vlsat_edges

    num_points = vlsat_predictor.config.dataset.num_points
    model_multi_rel = bool(getattr(vlsat_predictor.config.MODEL, "multi_rel_outputs", False))
    decode_multi_rel = model_multi_rel and not force_single_label

    if max_relations_per_pair <= 0:
        max_relations_per_pair = 1

    obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = \
        vlsat_predictor.preprocess_poinclouds(point_clouds, num_points)

    predicted_relations = vlsat_predictor.predict_relations(
        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids,
    )

    relation_hist: dict[str, int] = {}

    # Convert to per-node edge dict
    for k in range(predicted_relations.shape[0]):
        src_idx = edge_indices[0][k].item()
        tgt_idx = edge_indices[1][k].item()
        src_gid = valid_ids[src_idx]
        tgt_gid = valid_ids[tgt_idx]
        rel_scores = predicted_relations[k]

        if decode_multi_rel:
            candidate_rel_ids = torch.where(rel_scores >= relation_threshold)[0].tolist()
            if not candidate_rel_ids:
                if ensure_pairwise_edges:
                    candidate_rel_ids = [int(torch.argmax(rel_scores).item())]
                else:
                    continue
            candidate_rel_ids = sorted(
                candidate_rel_ids,
                key=lambda rid: float(rel_scores[rid]),
                reverse=True,
            )[:max_relations_per_pair]
        else:
            candidate_rel_ids = [int(torch.argmax(rel_scores).item())]

        for rel_id in candidate_rel_ids:
            rel_name = vlsat_predictor.rel_id_to_rel_name.get(rel_id, "none")
            if rel_name == "none":
                continue
            vlsat_edges[src_gid].append({
                "target_id": int(tgt_gid),
                "relation_type": rel_name,
            })
            relation_hist[rel_name] = relation_hist.get(rel_name, 0) + 1

    if debug_scores:
        top_hist = sorted(relation_hist.items(), key=lambda x: x[1], reverse=True)[:10]
        print(
            "[edges_vlsat] decode="
            f"{'multi-label' if decode_multi_rel else 'argmax'} "
            f"(threshold={relation_threshold:.2f}, max_per_pair={max_relations_per_pair}, "
            f"ensure_pairwise_edges={ensure_pairwise_edges}, "
            f"min_points_per_object={min_points_per_object}, "
            f"valid_objs={len(valid_ids)}, skipped_objs={skipped_for_points}) "
            f"top relations={top_hist}"
        )

    return vlsat_edges


def _save_graph_json(graph, object_registry, save_dir: Path, dataset_name: str,
                     bs_edges: dict | None = None,
                     vlsat_edges: dict | None = None):
    """Serialize the scene graph to JSON."""
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

    nodes = {}
    for node_id in graph.nodes:
        obj = all_objs.get(node_id, {})
        node_data = graph.nodes[node_id]
        bbox_3d = obj.get("bbox_3d") or (node_data.get("data", {}).get("bbox_3d") if isinstance(node_data.get("data"), dict) else None)
        class_name = obj.get("class_name") or (node_data.get("data", {}).get("class_name") if isinstance(node_data.get("data"), dict) else None)

        edges_list = []
        for _, target, _, edge_data in graph.out_edges(node_id, keys=True, data=True):
            edges_list.append({
                "target_id": int(target),
                "relation_type": edge_data.get("label", edge_data.get("label_class", "")),
            })

        node_bs = []
        if bs_edges and node_id in bs_edges:
            node_bs = bs_edges[node_id]

        node_vlsat = []
        if vlsat_edges and node_id in vlsat_edges:
            node_vlsat = vlsat_edges[node_id]

        nodes[int(node_id)] = {
            "track_id": int(node_id),
            "class_name": class_name,
            "bbox_3d": _bbox_to_serializable(bbox_3d),
            "edges_sv": edges_list,
            "edges_bs": node_bs,
            "edges_vlsat": node_vlsat,
        }

    output = {
        "dataset": dataset_name,
        "num_objects": len(nodes),
        "nodes": nodes,
    }

    out_path = save_dir / f"{scene_name}_graph.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScene graph saved to {out_path}\n")


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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Config --------------------------------------------------------------
    default_cfg = OmegaConf.load(Path(__file__).parent / "configs" / "core_tracking.yaml")
    if args.dataset:
        ds_yaml = Path(__file__).parent / "configs" / f"{args.dataset}.yaml"
        if ds_yaml.exists():
            cfg = OmegaConf.merge(default_cfg, OmegaConf.load(ds_yaml))
            print(f"Merged config: {args.dataset}")
        else:
            cfg = default_cfg
    else:
        cfg = default_cfg

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
    if args.edges is not None:
        try:
            selected_edges = _parse_edges_cli(args.edges)
        except ValueError as exc:
            raise SystemExit(f"--edges: {exc}") from exc
        cfg.ssg.basic_edges = "sv" in selected_edges
        cfg.ssg.baseline_edges = "bs" in selected_edges
        cfg.ssg.vlsat_edges = "vlsat" in selected_edges
        # Backward-compat key used in this script:
        cfg.ssg.use_vlsat = cfg.ssg.vlsat_edges

    if args.vlsat_rel_thr is not None:
        cfg.ssg.vlsat_relation_threshold = float(args.vlsat_rel_thr)
    if args.vlsat_max_rels is not None:
        cfg.ssg.vlsat_max_relations_per_pair = int(args.vlsat_max_rels)
    if args.vlsat_force_single_label:
        cfg.ssg.vlsat_force_single_label = True
    if args.vlsat_debug_scores:
        cfg.ssg.vlsat_debug_scores = True

    dataset_name = cfg.get("dataset", args.dataset or "isaacsim")
    yutils.configure_globals(cfg)

    # --- Loader --------------------------------------------------------------
    scene_path = str(Path(args.path).resolve())
    LoaderCls = get_loader(dataset_name)

    loader_kwargs = {}
    skip_labels = cfg.get("loader_skip_labels")
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
    print(f"  KG-NAV RUN – {loader.scene_label}  (dataset: {dataset_name})")
    print(f"{'=' * 60}")

    frame_indices = loader.get_frame_indices()
    print(f"Frames: {len(frame_indices)}")

    intrinsics = loader.get_camera_intrinsics()
    if intrinsics is not None:
        K, img_h, img_w = intrinsics
        _override_intrinsics(K, img_h, img_w)
        print(f"Intrinsics: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
              f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  image={img_w}x{img_h}")

    rgb_paths = loader.get_rgb_paths()
    depth_paths = loader.get_depth_paths()
    depth_cache = loader.build_depth_cache()
    poses = loader.get_all_poses()
    class_names_to_track = list(cfg.get("scene_0_class_names", [])) or None

    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    timings = {"yolo": [], "preprocess": [], "create_3d": []}
    ssg_cfg = OmegaConf.to_container(cfg.get("ssg", {}), resolve=True)

    # ── Tracking loop (NO per-frame edge computation) ────────────────────────
    last_T_w_c = None
    last_depth_m = None

    for tf in run_tracking(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        depth_cache=depth_cache,
        poses=poses,
        cfg=cfg,
        object_registry=object_registry,
        class_names_to_track=class_names_to_track,
    ):
        for k in ("yolo", "preprocess", "create_3d"):
            if k in tf.timings:
                timings[k].append(tf.timings[k])

        # Keep last pose & depth for final graph computation
        last_T_w_c = tf.T_w_c
        last_depth_m = tf.depth_m

        if ssg_cfg.get("print_tracking_info", False):
            summary = object_registry.get_tracking_summary()
            yolo_det = sum(1 for o in tf.frame_objs if o.get("match_source") != "reprojection")
            reproj = sum(1 for o in tf.frame_objs if o.get("match_source") == "reprojection")
            print(f"  [Track] Frame {tf.frame_idx}: YOLO={yolo_det}, "
                  f"Reproj={reproj}, Total={len(tf.frame_objs)}, "
                  f"Registry={summary['total_objects']}")

        if ssg_cfg.get("show_pcds", False):
            yutils.visualize_reconstruction(
                object_registry=object_registry,
                frame_index=tf.frame_idx,
                show_visible_only=False,
                show_aabb=True,
            )

        if ssg_cfg.get("print_resource_usage", False):
            print(f"[kg_nav] Frame {tf.frame_idx}: "
                  f"preprocess={timings['preprocess'][-1]:.2f}ms, "
                  f"create_3d={timings['create_3d'][-1]:.2f}ms, "
                  f"yolo={timings['yolo'][-1]:.2f}ms")

    # ── Build graph from ALL registry objects & compute edges ONCE ───────────
    print("\n--- Computing scene graph from global registry ---")

    graph = _build_graph_from_registry(object_registry)
    all_objs = object_registry.get_all_objects()

    # Build frame_objs list for edges() (lightweight dicts)
    frame_objs = []
    for gid, obj in all_objs.items():
        frame_objs.append({
            'global_id': gid,
            'track_id': gid,
            'class_name': obj.get('class_name'),
            'bbox_3d': obj.get('bbox_3d'),
            'visible_current_frame': True,
        })

    sv_time = 0.0
    if ssg_cfg.get("basic_edges", True):
        t0 = time.perf_counter()
        edges(graph, frame_objs, last_T_w_c, last_depth_m)
        sv_time = (time.perf_counter() - t0) * 1000
    else:
        print("SceneVerse edges disabled.")

    bs_edge_map: dict | None = None
    bs_time = 0.0
    if ssg_cfg.get("baseline_edges", True):
        t0 = time.perf_counter()
        bs_edge_map = edges_bs(all_objs, last_T_w_c)
        bs_time = (time.perf_counter() - t0) * 1000
    else:
        print("Baseline edges disabled.")

    # --- VL-SAT neural edges (optional) -------------------------------------
    vlsat_edge_map: dict | None = None
    vlsat_time = 0.0
    use_vlsat = bool(ssg_cfg.get("use_vlsat", ssg_cfg.get("vlsat_edges", False)))
    if use_vlsat:
        print("Initialising VL-SAT edge predictor …")
        vlsat_pred = _init_vlsat_predictor()
        t0 = time.perf_counter()
        vlsat_edge_map = edges_vlsat(
            all_objs,
            vlsat_pred,
            relation_threshold=float(ssg_cfg.get("vlsat_relation_threshold", 0.35)),
            max_relations_per_pair=int(ssg_cfg.get("vlsat_max_relations_per_pair", 3)),
            force_single_label=bool(ssg_cfg.get("vlsat_force_single_label", False)),
            ensure_pairwise_edges=bool(ssg_cfg.get("vlsat_ensure_pairwise_edges", True)),
            min_points_per_object=int(ssg_cfg.get("vlsat_min_points_per_object", 10)),
            debug_scores=bool(ssg_cfg.get("vlsat_debug_scores", False)),
        )
        vlsat_time = (time.perf_counter() - t0) * 1000
        n_vlsat = sum(len(v) for v in vlsat_edge_map.values())
        print(f"  VL-SAT edges: {n_vlsat}  ({vlsat_time:.2f}ms)")

    timing_parts_parts = []
    if ssg_cfg.get("basic_edges", True):
        timing_parts_parts.append(f"SV={sv_time:.2f}ms")
    if ssg_cfg.get("baseline_edges", True):
        timing_parts_parts.append(f"BS={bs_time:.2f}ms")
    if vlsat_edge_map is not None:
        timing_parts_parts.append(f"VLSAT={vlsat_time:.2f}ms")

    edge_count_parts = []
    if ssg_cfg.get("basic_edges", True):
        edge_count_parts.append(f"SV edges: {graph.number_of_edges()}")
    if bs_edge_map is not None:
        edge_count_parts.append(f"BS edges: {sum(len(v) for v in bs_edge_map.values())}")
    if vlsat_edge_map is not None:
        edge_count_parts.append(f"VLSAT edges: {sum(len(v) for v in vlsat_edge_map.values())}")

    timing_parts = "  ".join(timing_parts_parts) if timing_parts_parts else "none"
    edge_counts = "  |  ".join(edge_count_parts) if edge_count_parts else "none"
    print(
        f"Edge computation: {timing_parts}  |  "
        f"Nodes: {graph.number_of_nodes()}  |  {edge_counts}"
    )

    # ── Visualize ────────────────────────────────────────────────────────────
    if ssg_cfg.get("vis_graph", False):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        yutils.draw_labeled_multigraph(graph, ax=ax)
        ax.set_title("Final Scene Graph (all registry objects)")
        plt.tight_layout()
        plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)
    for gid, obj in all_objs.items():
        cls = f" ({obj.get('class_name')})" if obj.get("class_name") else ""
        print(f"  Object {gid}{cls}: "
              f"seen {obj['observation_count']} frames, "
              f"first: {obj['first_seen_frame']}, last: {obj['last_seen_frame']}, "
              f"points: {len(obj['points_accumulated'])}")
    print(f"Total unique objects: {len(all_objs)}")

    if timings["preprocess"]:
        print(f"\nLatency Averages (ms):")
        for k in ("preprocess", "create_3d", "yolo"):
            vals = timings.get(k, [])
            if vals:
                print(f"  {k:20s} {np.mean(vals):8.2f} ± {np.std(vals):.2f}")
    print("=" * 60)

    # ── Save ─────────────────────────────────────────────────────────────────
    if ssg_cfg.get("save_graph", False):
        save_dir = Path(ssg_cfg.get("save_graph_dir", "results/scene_graphs"))
        _save_graph_json(graph, object_registry, save_dir, dataset_name,
                         bs_edges=bs_edge_map,
                         vlsat_edges=vlsat_edge_map)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KG-Nav: Track objects, build scene graph once at the end.",
    )
    
    p.add_argument("--path", default=f"/home/yehia/Downloads/kg_nav_2_IsaacSimData/{scene_name}",
                   help="Path to the scene directory.")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["isaacsim", "thud_synthetic", "thud_real", "code", "any_scene"],
                   help="Dataset type.")
    p.add_argument("--show-pcds", action="store_true", help="Show point cloud reconstruction.")
    p.add_argument("--vis-graph", action="store_true", help="Show final scene graph.")
    p.add_argument("--print-resources", action="store_true", help="Print per-frame resource usage.")
    p.add_argument("--print-tracking", action="store_true", help="Print per-frame tracking info.")
    p.add_argument("--save-graph", action="store_true", help="Save final scene graph as JSON.")
    p.add_argument("--edges", type=str, default=None,
                   help="Edge predictors to run: all | bs | sv | vlsat, or comma-separated combos "
                        "(e.g. bs,sv).")
    p.add_argument("--vlsat-rel-thr", type=float, default=None,
                   help="VL-SAT relation threshold for multi-label decoding (default from config).")
    p.add_argument("--vlsat-max-rels", type=int, default=None,
                   help="Max VL-SAT relations to keep per directed pair (default from config).")
    p.add_argument("--vlsat-force-single-label", action="store_true",
                   help="Force old argmax decoding for VL-SAT (one relation per pair).")
    p.add_argument("--vlsat-debug-scores", action="store_true",
                   help="Print VL-SAT relation histogram summary.")
    return p

scene_name = "scene_8"
if __name__ == "__main__":
    sys.exit(main())
