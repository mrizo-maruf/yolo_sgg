from __future__ import annotations

import os
import sys
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class VLSATEdgePredictorService:
    """
    One-time initialized VL-SAT inference service.
    Accepts object bounding boxes, builds synthetic parallelepiped point clouds,
    runs VL-SAT once per scene, and exposes directed pair relation names.
    """

    def __init__(
        self,
        model_root: str,
        config_path: str,
        ckpt_path: str,
        relationships_path: str,
        num_points: int = 512,
        seed: int = 123,
    ):
        self.model_root = model_root
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.relationships_path = relationships_path
        self.num_points = max(32, int(num_points))
        self.rng = np.random.default_rng(seed)

        self._ensure_import_paths()

        from src.utils.config import Config  # type: ignore
        from src.model.model import MMGNet  # type: ignore
        from src.utils import op_utils  # type: ignore

        self._op_utils = op_utils

        self.config = Config(self.config_path)
        self.config.exp = self.ckpt_path
        self.config.MODE = "eval"

        if torch.cuda.is_available() and len(self.config.GPU) > 0:
            self.config.DEVICE = torch.device("cuda")
        else:
            self.config.DEVICE = torch.device("cpu")

        self.model = MMGNet(self.config)
        loaded = self.model.load(best=True)
        if not loaded:
            raise RuntimeError(
                f"VL-SAT checkpoints were not loaded from '{self.ckpt_path}'. "
                "Please provide a valid vl_sat_ckpt_path containing *_best.pth files."
            )
        self.model.model.eval()

        with open(self.relationships_path, "r", encoding="utf-8") as f:
            rel_lines = [line.strip() for line in f.readlines() if line.strip()]

        # The model is trained with multi_rel_outputs=true which removes the
        # first entry ("none") from the relation list.  Model class 0 therefore
        # corresponds to the *second* line (e.g. "supported by").  We store
        # the mapping with a +1 offset so that 0 is always reserved for
        # "none / not-computed" in our pipeline.
        non_none_names = rel_lines[1:]  # skip "none"
        self.rel_id_to_rel_name: Dict[int, str] = {
            0: "none",
            **{i + 1: name for i, name in enumerate(non_none_names)}
        }
        self.num_model_classes = len(non_none_names)  # 26

    def _ensure_import_paths(self) -> None:
        candidates = [
            self.model_root,
            os.path.join(self.model_root, "src"),
        ]
        for path in candidates:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _sample_bbox_parallelepiped(self, center: np.ndarray, extent: np.ndarray, n: int) -> np.ndarray:
        cx, cy, cz = center.tolist()
        ex, ey, ez = np.maximum(extent, 1e-4).tolist()
        hx, hy, hz = ex / 2.0, ey / 2.0, ez / 2.0

        face_ids = self.rng.integers(0, 6, size=n)
        u = self.rng.uniform(-1.0, 1.0, size=n)
        v = self.rng.uniform(-1.0, 1.0, size=n)

        pts = np.zeros((n, 3), dtype=np.float32)
        for i, face in enumerate(face_ids):
            if face == 0:  # +x
                pts[i] = [cx + hx, cy + u[i] * hy, cz + v[i] * hz]
            elif face == 1:  # -x
                pts[i] = [cx - hx, cy + u[i] * hy, cz + v[i] * hz]
            elif face == 2:  # +y
                pts[i] = [cx + u[i] * hx, cy + hy, cz + v[i] * hz]
            elif face == 3:  # -y
                pts[i] = [cx + u[i] * hx, cy - hy, cz + v[i] * hz]
            elif face == 4:  # +z
                pts[i] = [cx + u[i] * hx, cy + v[i] * hy, cz + hz]
            else:  # -z
                pts[i] = [cx + u[i] * hx, cy + v[i] * hy, cz - hz]
        return pts

    def _preprocess_pointclouds(
        self, points: List[np.ndarray], num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(points) > 1, "Number of objects should be at least 2"

        edge_indices = list(product(list(range(len(points))), list(range(len(points)))))
        edge_indices = [i for i in edge_indices if i[0] != i[1]]

        num_objects = len(points)
        dim_point = points[0].shape[-1]

        obj_points = torch.zeros([num_objects, num_points, dim_point], dtype=torch.float32)
        descriptor = torch.zeros([num_objects, 11], dtype=torch.float32)
        obj_2d_feats = np.zeros([num_objects, 512], dtype=np.float32)

        for i, pcd in enumerate(points):
            if pcd.shape[0] == 0:
                pcd = np.zeros((1, dim_point), dtype=np.float32)
            choice = np.random.choice(len(pcd), num_points, replace=True)
            pcd = pcd[choice, :]
            descriptor[i] = self._op_utils.gen_descriptor(torch.from_numpy(pcd))
            pcd_t = torch.from_numpy(pcd.astype(np.float32))
            pcd_t -= torch.mean(pcd_t, dim=0, keepdim=True)
            obj_points[i] = pcd_t

        edge_indices = torch.tensor(edge_indices, dtype=torch.long).permute(1, 0)
        obj_2d_feats = torch.from_numpy(obj_2d_feats)
        obj_points = obj_points.permute(0, 2, 1)
        batch_ids = torch.zeros((num_objects, 1), dtype=torch.float32)
        return obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids

    @torch.no_grad()
    def predict_pair_relations_from_bboxes(
        self,
        centers_xyz: np.ndarray,
        extents_xyz: np.ndarray,
    ) -> Dict[Tuple[int, int], str]:
        """
        Returns directed pair relation map: (src_idx, dst_idx) -> relation_name.
        """
        num_obj = centers_xyz.shape[0]
        if num_obj < 2:
            return {}

        points: List[np.ndarray] = []
        for i in range(num_obj):
            points.append(self._sample_bbox_parallelepiped(centers_xyz[i], extents_xyz[i], self.num_points))

        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = self._preprocess_pointclouds(
            points, self.num_points
        )

        device = self.config.DEVICE
        obj_points = obj_points.to(device)
        obj_2d_feats = obj_2d_feats.to(device)
        edge_indices = edge_indices.to(device)
        descriptor = descriptor.to(device)
        batch_ids = batch_ids.to(device)

        model_output = self.model.model(
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids=batch_ids
        )
        # forward() returns (rel_cls_3d, edge_feat_3d, edge_feat_2d)
        rel_logits = model_output[0] if isinstance(model_output, tuple) else model_output

        rel_ids = torch.argmax(rel_logits, dim=1).detach().cpu().numpy()
        edge_np = edge_indices.detach().cpu().numpy()

        out: Dict[Tuple[int, int], str] = {}
        for k in range(rel_ids.shape[0]):
            src_i = int(edge_np[0, k])
            dst_i = int(edge_np[1, k])
            # +1 because model classes are 0-based without "none";
            # our mapping reserves 0 for "none".
            mapped_id = int(rel_ids[k]) + 1
            out[(src_i, dst_i)] = self.rel_id_to_rel_name.get(mapped_id, "none")
        return out

    @torch.no_grad()
    def predict_pair_relation_ids_from_bboxes(
        self,
        centers_xyz: np.ndarray,
        extents_xyz: np.ndarray,
    ) -> Dict[Tuple[int, int], int]:
        """
        Returns directed pair relation-id map: (src_idx, dst_idx) -> rel_id (0..num_rel-1).
        """
        num_obj = centers_xyz.shape[0]
        if num_obj < 2:
            return {}

        points: List[np.ndarray] = []
        for i in range(num_obj):
            points.append(self._sample_bbox_parallelepiped(centers_xyz[i], extents_xyz[i], self.num_points))

        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = self._preprocess_pointclouds(
            points, self.num_points
        )

        device = self.config.DEVICE
        obj_points = obj_points.to(device)
        obj_2d_feats = obj_2d_feats.to(device)
        edge_indices = edge_indices.to(device)
        descriptor = descriptor.to(device)
        batch_ids = batch_ids.to(device)

        model_output = self.model.model(
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids=batch_ids
        )
        # forward() returns (rel_cls_3d, edge_feat_3d, edge_feat_2d)
        rel_logits = model_output[0] if isinstance(model_output, tuple) else model_output

        rel_ids = torch.argmax(rel_logits, dim=1).detach().cpu().numpy()
        edge_np = edge_indices.detach().cpu().numpy()

        out: Dict[Tuple[int, int], int] = {}
        for k in range(rel_ids.shape[0]):
            src_i = int(edge_np[0, k])
            dst_i = int(edge_np[1, k])
            # +1: model outputs 0-25 (trained without "none" class);
            # our convention reserves 0 for "none / not-computed".
            out[(src_i, dst_i)] = int(rel_ids[k]) + 1
        return out
