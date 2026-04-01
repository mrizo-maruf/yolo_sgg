"""
CODA (UT Campus Object Dataset) Data Loader

Implements the DatasetLoader protocol for the CODA dataset.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import cv2
import yaml
from tqdm import tqdm
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base class (included here for self‑containedness)
# ---------------------------------------------------------------------------
class DatasetLoader(ABC):
    """Protocol that every dataset loader must implement."""

    @property
    @abstractmethod
    def scene_label(self) -> str:
        """Human-readable label for the scene."""
        ...

    @abstractmethod
    def get_frame_indices(self) -> List[int]:
        """Ordered list of frame indices."""
        ...

    @abstractmethod
    def get_rgb_paths(self) -> List[str]:
        """Absolute paths to RGB images, one per frame index."""
        ...

    @abstractmethod
    def get_depth_paths(self) -> List[str]:
        """Absolute paths to depth images, one per frame index."""
        ...

    @abstractmethod
    def load_depth(self, path: str) -> np.ndarray:
        """Load a single depth image and return float32 metres (H, W)."""
        ...

    def build_depth_cache(self) -> Dict[str, np.ndarray]:
        """Pre-load all depth maps into a dict keyed by path."""
        return {p: self.load_depth(p) for p in self.get_depth_paths()}

    @abstractmethod
    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return (K_3x3, img_height, img_width) or None."""
        ...

    @abstractmethod
    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return the 4×4 camera-to-world transform for frame_idx."""
        ...

    def get_all_poses(self) -> Optional[List[np.ndarray]]:
        indices = self.get_frame_indices()
        poses = [self.get_camera_pose(i) for i in indices]
        if all(p is None for p in poses):
            return None
        return poses

    def get_gt_instances(self, frame_idx: int):
        """Return a list of GTInstance for frame_idx, or None."""
        return None

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        """Return paths to all valid scene directories under root."""
        return []


# ---------------------------------------------------------------------------
# Simple GT instance class (compatible with metrics.tracking_metrics.GTInstance)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GTInstance:
    """Ground truth object for one frame."""
    track_id: str
    class_name: str
    mask: Optional[np.ndarray] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    box_3d_center_xyz: Optional[Tuple[float, float, float]] = None
    box_3d_size_xyz: Optional[Tuple[float, float, float]] = None
    box_3d_rotation_xyzw: Optional[Tuple[float, float, float, float]] = None


# ---------------------------------------------------------------------------
# CODA Loader
# ---------------------------------------------------------------------------
class CodaLoader(DatasetLoader):
    """
    Data loader for CODA (UT Campus Object Dataset).

    Parameters
    ----------
    root_dir : str
        Path to the CODa root directory.
    sequence_id : str
        Sequence identifier (e.g., "0").
    load_rgb : bool
        If True, load RGB images.
    load_depth : bool
        If True, compute depth maps from LiDAR point clouds (cached).
    skip_labels : set[str] | None
        Class names (case‑insensitive) to exclude entirely.
    require_3d : bool
        If True, skip objects without a 3D bounding box.
    max_depth : float
        Maximum depth to keep when building depth maps (meters).
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        root_dir: str,
        sequence_id: str = "0",
        load_rgb: bool = True,
        load_depth: bool = True,
        skip_labels: Optional[Set[str]] = None,
        require_3d: bool = True,
        max_depth: float = 80.0,
        verbose: bool = True,
    ) -> None:
        self._root = Path(root_dir)
        self._seq = sequence_id
        self._load_rgb = load_rgb
        self._load_depth = load_depth
        self._skip_labels = {s.lower() for s in (skip_labels or set())}
        self._require_3d = require_3d
        self._max_depth = max_depth
        self._verbose = verbose

        self._rgb_dir = self._root / "2d_rect" / "cam0" / self._seq
        self._lidar_dir = self._root / "3d_comp" / "os1" / self._seq
        self._bbox_dir = self._root / "3d_bbox" / "os1" / self._seq
        self._calib_dir = self._root / "calibrations" / self._seq

        for d in [self._rgb_dir, self._lidar_dir, self._bbox_dir, self._calib_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Missing required directory: {d}")

        self._load_calibrations()
        self._frame_indices = self._discover_frames()
        if self._verbose:
            print(f"[CodaLoader] Sequence {self._seq}: {len(self._frame_indices)} frames found")
            if self._frame_indices:
                print(f"  Frame numbers: {self._frame_indices[:10]} ...")

        self._depth_cache: Dict[int, np.ndarray] = {}
        if self._load_depth:
            if self._verbose:
                print("[CodaLoader] Pre‑computing depth maps (this may take a while)...")
            for fid in tqdm(self._frame_indices, desc="Building depth", disable=not self._verbose):
                self._depth_cache[fid] = self._build_depth_map(fid)

    def _load_calibrations(self) -> None:
        intr_path = self._calib_dir / "calib_cam0_intrinsics.yaml"
        with open(intr_path, 'r') as f:
            cam_intr = yaml.safe_load(f)
        K_data = cam_intr['camera_matrix']['data']
        self._K = np.array(K_data).reshape(3, 3)

        ext_path = self._calib_dir / "calib_os1_to_cam0.yaml"
        with open(ext_path, 'r') as f:
            ext = yaml.safe_load(f)
        T_data = ext['extrinsic_matrix']['data']
        self._T_lidar_to_cam = np.array(T_data).reshape(4, 4)

        self._image_width = int(cam_intr.get('image_width', 1224))
        self._image_height = int(cam_intr.get('image_height', 1024))
        if self._verbose:
            print(f"[CodaLoader] Calibration loaded. Image size: {self._image_width}x{self._image_height}")

    def _discover_frames(self) -> List[int]:
        frame_set = set()
        for p in self._rgb_dir.glob("2d_rect_cam0_*.png"):
            parts = p.stem.split('_')
            if len(parts) >= 5:
                frame_num = int(parts[-1])
                lidar_file = self._lidar_dir / f"3d_comp_os1_{self._seq}_{frame_num}.bin"
                if lidar_file.exists():
                    frame_set.add(frame_num)
        return sorted(frame_set)

    def _build_depth_map(self, frame_num: int) -> np.ndarray:
        lidar_file = self._lidar_dir / f"3d_comp_os1_{self._seq}_{frame_num}.bin"
        if not lidar_file.exists():
            return np.zeros((self._image_height, self._image_width), dtype=np.float32)

        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        xyz_lidar = points[:, :3]
        ones = np.ones((len(xyz_lidar), 1))
        xyz_lidar_h = np.hstack([xyz_lidar, ones])
        xyz_cam_h = (self._T_lidar_to_cam @ xyz_lidar_h.T).T
        xyz_cam = xyz_cam_h[:, :3]

        fx, fy = self._K[0,0], self._K[1,1]
        cx, cy = self._K[0,2], self._K[1,2]
        z = xyz_cam[:, 2]
        u = (fx * xyz_cam[:, 0] / z + cx).astype(int)
        v = (fy * xyz_cam[:, 1] / z + cy).astype(int)

        mask = (z > 0) & (u >= 0) & (u < self._image_width) & (v >= 0) & (v < self._image_height)
        if self._max_depth > 0:
            mask &= (z <= self._max_depth)
        u = u[mask]
        v = v[mask]
        z = z[mask]

        depth_map = np.full((self._image_height, self._image_width), np.inf, dtype=np.float32)
        order = np.argsort(z)
        for i in order:
            depth_map[v[i], u[i]] = z[i]
        depth_map[np.isinf(depth_map)] = 0.0
        return depth_map

    def _load_annotations(self, frame_num: int) -> List[GTInstance]:
        json_path = self._bbox_dir / f"3d_bbox_os1_{self._seq}_{frame_num}.json"
        if not json_path.exists():
            return []
        with open(json_path, 'r') as f:
            data = json.load(f)

        objects = data.get('3dbbox', data.get('root', {}).get('3dbbox', []))
        if not objects:
            return []

        gt_list = []
        for obj in objects:
            required = ('cX', 'cY', 'cZ', 'l', 'w', 'h', 'r', 'p', 'y', 'instanceId', 'classId')
            if not all(k in obj for k in required):
                continue
            class_name = obj['classId']
            if class_name.lower() in self._skip_labels:
                continue

            instance_id = str(obj['instanceId'])

            cx, cy, cz = obj['cX'], obj['cY'], obj['cZ']
            length, width, height = obj['l'], obj['w'], obj['h']
            yaw = obj['y']

            # 2D projection
            dx, dy, dz = length/2, width/2, height/2
            corners_local = np.array([
                [-dx, -dy, -dz], [ dx, -dy, -dz],
                [ dx,  dy, -dz], [-dx,  dy, -dz],
                [-dx, -dy,  dz], [ dx, -dy,  dz],
                [ dx,  dy,  dz], [-dx,  dy,  dz],
            ])
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw),  np.cos(yaw), 0],
                          [0, 0, 1]])
            corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])

            corners_lidar_h = np.hstack([corners_world, np.ones((8,1))])
            corners_cam_h = (self._T_lidar_to_cam @ corners_lidar_h.T).T
            corners_cam = corners_cam_h[:, :3]
            fx, fy = self._K[0,0], self._K[1,1]
            cx_p, cy_p = self._K[0,2], self._K[1,2]
            z_c = corners_cam[:, 2]
            u = (fx * corners_cam[:, 0] / z_c + cx_p).astype(int)
            v = (fy * corners_cam[:, 1] / z_c + cy_p).astype(int)
            u = np.clip(u, 0, self._image_width-1)
            v = np.clip(v, 0, self._image_height-1)
            x1, y1 = u.min(), v.min()
            x2, y2 = u.max(), v.max()
            bbox2d = (float(x1), float(y1), float(x2), float(y2)) if (x2 > x1 and y2 > y1) else None

            # 3D AABB (world)
            aabb_min = corners_world.min(axis=0)
            aabb_max = corners_world.max(axis=0)
            aabb = (float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]),
                    float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2]))

            center_xyz = (cx, cy, cz)
            size_xyz = (length, width, height)
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            rot_xyzw = (0.0, 0.0, qz, qw)

            gt_list.append(GTInstance(
                track_id=instance_id,
                class_name=class_name,
                bbox_xyxy=bbox2d,
                box_3d_center_xyz=center_xyz,
                box_3d_size_xyz=size_xyz,
                box_3d_rotation_xyzw=rot_xyzw,
                mask=None,
            ))
        return gt_list

    # ------------------------------------------------------------------
    # DatasetLoader interface
    # ------------------------------------------------------------------
    @property
    def scene_label(self) -> str:
        return f"CODA_Seq{self._seq}"

    def get_frame_indices(self) -> List[int]:
        return self._frame_indices.copy()

    def get_rgb_paths(self) -> List[str]:
        paths = []
        for fid in self._frame_indices:
            p = self._rgb_dir / f"2d_rect_cam0_{self._seq}_{fid}.png"
            paths.append(str(p))
        return paths

    def get_depth_paths(self) -> List[str]:
        # Depth maps are generated on the fly, no external files.
        return []

    def load_depth(self, path: str) -> np.ndarray:
        # Not used because depth maps are cached; return empty array.
        return np.zeros((self._image_height, self._image_width), dtype=np.float32)

    def build_depth_cache(self) -> Dict[str, np.ndarray]:
        cache = {}
        for fid in self._frame_indices:
            rgb_path = self._rgb_dir / f"2d_rect_cam0_{self._seq}_{fid}.png"
            if fid in self._depth_cache:
                cache[str(rgb_path)] = self._depth_cache[fid]
        return cache

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        return (self._K.copy(), self._image_height, self._image_width)

    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        # CODA does not provide usable camera poses in the same frame as depth.
        return None

    def get_gt_instances(self, frame_idx: int):
        # frame_idx here is the CODA frame number (original).
        return self._load_annotations(frame_idx)

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_path = Path(root)
        seq_dirs = sorted((root_path / "2d_rect" / "cam0").glob("[0-9]*"))
        return [str(d) for d in seq_dirs if d.is_dir()]

