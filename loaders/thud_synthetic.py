"""
THUD Synthetic dataset loader adapter.

Wraps ``thud_utils.thud_synthetic_loader.THUDSyntheticLoader``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import YOLOE.utils as yutils
from thud_utils.thud_synthetic_loader import (
    THUDSyntheticLoader,
    GTObject,
    discover_thud_synthetic_scenes,
)
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader


_DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}


def _load_thud_depth_raw(depth_path: str) -> np.ndarray:
    """Load a THUD synthetic uint16 depth PNG as raw float32.

    THUD synthetic depth uses a non-standard encoding that requires
    the custom scale/offset formulas in the point extractor.
    Do NOT divide by 1000 — that destroys information needed by the
    THUD depth→PCD pipeline.
    """
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    return d.astype(np.float32)


class THUDSyntheticDatasetLoader(DatasetLoader):
    """Adapter for THUD Synthetic Scenes."""

    def __init__(
        self,
        scene_dir: str,
        skip_labels: Optional[Set[str]] = None,
        depth_scale: float = 1000.0,
        depth_max_m: float = 100.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS
        self._depth_scale = depth_scale
        self._depth_max_m = depth_max_m
        self._point_extractor = None

        self._loader = THUDSyntheticLoader(
            str(self._scene_dir),
            load_rgb=True,
            load_depth=True,
            skip_labels=self._skip_labels,
            verbose=False,
        )

        # Read intrinsics from first frame
        first_fd = self._loader.get_frame_data(self._loader.frame_indices[0])
        self._K = first_fd.camera_intrinsic
        if first_fd.rgb is not None:
            self._img_h, self._img_w = first_fd.rgb.shape[:2]
        else:
            self._img_h, self._img_w = 530, 730

    # -- metadata ----------------------------------------------------------

    @property
    def scene_label(self) -> str:
        try:
            parts = self._scene_dir.parts
            idx = next(
                i for i, p in enumerate(parts)
                if p in ("Gym", "House", "Office", "Supermarket_1", "Supermarket_2")
            )
            return "/".join(parts[idx:])
        except StopIteration:
            return self._scene_dir.name

    def get_frame_indices(self) -> List[int]:
        return list(self._loader.frame_indices)

    # -- paths -------------------------------------------------------------

    def get_rgb_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.frame_indices:
            rp = self._loader.rgb_dir / f"rgb_{fidx}.png"
            if rp.exists():
                paths.append(str(rp))
        return paths

    def get_depth_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.frame_indices:
            dp = self._loader.depth_dir / f"depth_{fidx}.png"
            if dp.exists():
                paths.append(str(dp))
        return paths

    # -- depth -------------------------------------------------------------

    def load_depth(self, path: str) -> np.ndarray:
        return _load_thud_depth_raw(path)

    def make_point_extractor(self):
        """Return THUD-specific depth→3-D-points extractor.

        Uses the custom scale / offset formulas from the THUD
        ``Depth_to_pointcloud.py`` pipeline.  The returned points are
        in the THUD camera frame; ``cam_to_world`` is still applied by
        the tracking pipeline afterwards.
        """
        if self._point_extractor is not None:
            return self._point_extractor

        K = self._K
        if K is None:
            return None

        _fx = float(K[0, 0])
        _fy = float(K[1, 1])
        _cx = float(K[0, 2])
        _cy = float(K[1, 2])

        def _extractor(
            depth_raw: np.ndarray,
            mask: np.ndarray,
            frame_idx: int,
            max_points: int,
            o3_nb_neighbors: int,
            o3std_ratio: float,
            track_id: int,
        ) -> np.ndarray:
            H, W = depth_raw.shape[:2]
            depth_f = depth_raw.astype(np.float32)

            if mask.dtype != bool:
                mask_bool = mask.squeeze() > 0
            else:
                mask_bool = mask.squeeze()

            valid = mask_bool & (depth_f > 0)
            if not np.any(valid):
                return np.zeros((0, 3), dtype=np.float32)

            # Pixel grids — v flipped as in the THUD pipeline
            u_grid = np.arange(W, dtype=np.float32)
            v_flip = np.arange(H - 1, -1, -1, dtype=np.float32)
            u_grid, v_flip = np.meshgrid(u_grid, v_flip)

            # Raw unprojection (THUD Depth_to_pointcloud.py formula)
            Xr = (u_grid - _cx) * depth_f / _fx
            Zr = (v_flip - _cy) * depth_f / _fy

            # THUD scale / offset + mm → m
            X = Xr / 2.5 / 1000.0
            Y = (depth_f / 6.5 + 200.0) / 1000.0
            Z = (Zr / 2.0 + 300.0) / 1000.0

            pts = np.stack([X[valid], Y[valid], Z[valid]], axis=-1).astype(np.float32)

            # Sub-sample
            rng = np.random.default_rng(track_id)
            n = len(pts)
            if max_points and n > max_points:
                idx = rng.choice(n, size=max_points, replace=False)
                pts = pts[idx]
            elif n > 6:
                idx = rng.choice(n, size=max(1, int(n * 0.5)), replace=False)
                pts = pts[idx]

            # Statistical outlier removal
            if pts.shape[0] > 3:
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    _, ind = pcd.remove_statistical_outlier(
                        nb_neighbors=o3_nb_neighbors, std_ratio=o3std_ratio,
                    )
                    pcd = pcd.select_by_index(ind)
                    pts = np.asarray(pcd.points).astype(np.float32)
                except Exception:
                    pass

            return pts

        self._point_extractor = _extractor
        return self._point_extractor

    # -- camera ------------------------------------------------------------

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        if self._K is not None:
            return self._K, self._img_h, self._img_w
        return None

    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fd = self._loader.get_frame_data(frame_idx)
        return fd.cam_transform_4x4

    def get_all_poses(self) -> Optional[List[np.ndarray]]:
        poses = []
        for fidx in self._loader.frame_indices:
            fd = self._loader.get_frame_data(fidx)
            poses.append(fd.cam_transform_4x4)
        if all(p is None for p in poses):
            return None
        return poses

    # -- ground truth ------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        fd = self._loader.get_frame_data(frame_idx)
        return [
            GTInstance(
                track_id=g.track_id,
                class_name=g.class_name,
                mask=g.mask,
                bbox_xyxy=g.bbox2d_xyxy,
            )
            for g in fd.gt_objects
        ]

    # -- multi-scene -------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        scene_type = kwargs.get("scene_type", "static")
        return discover_thud_synthetic_scenes(root, scene_type=scene_type)
