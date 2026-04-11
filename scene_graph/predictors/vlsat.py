"""VL-SAT neural edge predictor."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from .base import EdgePredictor


class VLSATEdgePredictor(EdgePredictor):
    """VL-SAT (MMGNet) neural relationship predictor.

    Lazily initialises the model on first call.
    """

    def __init__(self, cfg: Optional[DictConfig] = None) -> None:
        self._predictor = None
        cfg = cfg or {}
        self._relation_threshold = float(cfg.get("vlsat_relation_threshold", 0.35))
        self._max_relations_per_pair = max(1, int(cfg.get("vlsat_max_relations_per_pair", 3)))
        self._force_single_label = bool(cfg.get("vlsat_force_single_label", False))
        self._ensure_pairwise_edges = bool(cfg.get("vlsat_ensure_pairwise_edges", True))
        self._min_points_per_object = int(cfg.get("vlsat_min_points_per_object", 10))
        self._debug_scores = bool(cfg.get("vlsat_debug_scores", False))

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if object_registry is None:
            return
        all_objs = object_registry.get_all_objects()
        if len(all_objs) < 2:
            return

        valid_ids, point_clouds, skipped_for_points = self._collect_valid_pointclouds(
            graph, all_objs,
        )

        if len(valid_ids) < 2:
            if self._debug_scores:
                print(
                    "[VLSAT] Need ≥2 objects with enough points "
                    f"(min_points_per_object={self._min_points_per_object}), "
                    f"valid={len(valid_ids)}, skipped={skipped_for_points}. Skipping."
                )
            return

        if self._predictor is None:
            self._predictor = _init_vlsat()

        import torch

        model_multi_rel = bool(getattr(self._predictor.config.MODEL, "multi_rel_outputs", False))
        decode_multi_rel = model_multi_rel and not self._force_single_label

        num_points = self._predictor.config.dataset.num_points
        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = \
            self._predictor.preprocess_poinclouds(point_clouds, num_points)
        predicted = self._predictor.predict_relations(
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids,
        )

        relation_hist = self._add_predicted_edges(
            graph, predicted, edge_indices, valid_ids, decode_multi_rel, torch,
        )

        if self._debug_scores:
            top_hist = sorted(relation_hist.items(), key=lambda x: x[1], reverse=True)[:10]
            print(
                "[VLSAT] decode="
                f"{'multi-label' if decode_multi_rel else 'argmax'} "
                f"(threshold={self._relation_threshold:.2f}, "
                f"max_per_pair={self._max_relations_per_pair}, "
                f"ensure_pairwise_edges={self._ensure_pairwise_edges}, "
                f"min_points_per_object={self._min_points_per_object}, "
                f"valid_objs={len(valid_ids)}, skipped_objs={skipped_for_points}) "
                f"top relations={top_hist}"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_valid_pointclouds(self, graph, all_objs):
        """Collect registry objects with enough accumulated points, converted
        from world frame (Z-up) to 3RScan/VL-SAT (Y-up)."""
        valid_ids: List[int] = []
        point_clouds: List[np.ndarray] = []
        skipped = 0
        for gid, obj in all_objs.items():
            if gid not in graph.nodes:
                continue
            pts = obj.get("points_accumulated")
            if pts is not None and len(pts) >= self._min_points_per_object:
                valid_ids.append(gid)
                pts_conv = pts[:, [0, 2, 1]].copy()
                pts_conv[:, 2] = -pts_conv[:, 2]
                point_clouds.append(pts_conv)
            else:
                skipped += 1
        return valid_ids, point_clouds, skipped

    def _add_predicted_edges(
        self,
        graph,
        predicted,
        edge_indices,
        valid_ids,
        decode_multi_rel,
        torch_mod,
    ) -> Dict[str, int]:
        """Walk predicted relation scores and add edges to ``graph``."""
        relation_hist: Dict[str, int] = {}
        for k in range(predicted.shape[0]):
            src_idx = edge_indices[0][k].item()
            tgt_idx = edge_indices[1][k].item()
            src_gid = valid_ids[src_idx]
            tgt_gid = valid_ids[tgt_idx]
            rel_scores = predicted[k]

            rel_ids = self._select_relation_ids(rel_scores, decode_multi_rel, torch_mod)
            if rel_ids is None:
                continue

            for rel_id in rel_ids:
                rel_name = self._predictor.rel_id_to_rel_name.get(rel_id, "none")
                if rel_name == "none":
                    continue
                graph.add_edge(src_gid, tgt_gid, label=rel_name, label_class="vlsat")
                relation_hist[rel_name] = relation_hist.get(rel_name, 0) + 1
        return relation_hist

    def _select_relation_ids(self, rel_scores, decode_multi_rel, torch_mod):
        """Pick which relation ids should become edges for one pair, or
        return ``None`` to skip the pair entirely."""
        if not decode_multi_rel:
            return [int(torch_mod.argmax(rel_scores).item())]

        rel_ids = torch_mod.where(rel_scores >= self._relation_threshold)[0].tolist()
        if not rel_ids:
            if not self._ensure_pairwise_edges:
                return None
            rel_ids = [int(torch_mod.argmax(rel_scores).item())]
        return sorted(
            rel_ids,
            key=lambda rid: float(rel_scores[rid]),
            reverse=True,
        )[:self._max_relations_per_pair]


def _init_vlsat():
    """Lazily import and init the VL-SAT EdgePredictor.

    ``vl_sat_model/`` is pushed to the front of ``sys.path`` so that bare
    ``from src.utils.config import …`` / ``from utils import …`` statements
    inside VL-SAT code resolve against its local sub-packages instead of
    the project-level ``utils.py`` module.
    """
    vlsat_dir = str(Path(__file__).resolve().parents[1] / "vl_sat_model")
    while vlsat_dir in sys.path:
        sys.path.remove(vlsat_dir)
    sys.path.insert(0, vlsat_dir)

    # Evict any cached top-level ``utils`` (e.g. project ``utils.py``) so
    # the VL-SAT ``utils`` sub-package is what gets imported next.
    if "utils" in sys.modules:
        sys.modules.pop("utils", None)
        for key in list(sys.modules):
            if key.startswith("utils."):
                sys.modules.pop(key, None)

    from vl_sat_interface import EdgePredictor as _VLSatEdgePredictor  # noqa: E402

    cfg_dir = Path(vlsat_dir)
    return _VLSatEdgePredictor(
        config_path=str(cfg_dir / "config" / "mmgnet.json"),
        ckpt_path=str(cfg_dir / "3dssg_best_ckpt"),
        relationships_list=str(cfg_dir / "config" / "relationships.txt"),
    )
