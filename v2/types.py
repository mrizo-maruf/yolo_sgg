"""
Core data types for the v2 pipeline.

All lightweight, serializable dataclasses — no heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        return cls(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=width, height=height,
        )

    @classmethod
    def from_physical(
        cls, focal_length: float, h_aperture: float, v_aperture: float,
        width: int, height: int,
    ) -> CameraIntrinsics:
        fx = focal_length / h_aperture * width
        fy = focal_length / v_aperture * height
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)


# ---------------------------------------------------------------------------
# 3-D bounding box
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BBox3D:
    """Lightweight 3-D bounding box (AABB + optional OBB)."""
    aabb_min: np.ndarray          # (3,)
    aabb_max: np.ndarray          # (3,)
    obb_center: np.ndarray        # (3,)
    obb_extent: np.ndarray        # (3,)
    obb_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # (3,3)

    @property
    def center(self) -> np.ndarray:
        return self.obb_center

    @property
    def volume(self) -> float:
        return float(np.prod(np.maximum(self.obb_extent, 1e-9)))

    def to_dict(self) -> Dict:
        return {
            "aabb": {"min": self.aabb_min.tolist(), "max": self.aabb_max.tolist()},
            "obb": {
                "center": self.obb_center.tolist(),
                "extent": self.obb_extent.tolist(),
                "R": self.obb_R.tolist(),
            },
        }


# ---------------------------------------------------------------------------
# Tracked object (per-frame lightweight view)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrackedObject:
    """Per-frame object produced by the 3-D tracker."""
    global_id: int
    yolo_id: int
    class_name: Optional[str]
    bbox_3d: Optional[BBox3D]
    mask: Optional[np.ndarray]           # (H,W) uint8

    # filled in by registry
    first_seen: int = 0
    last_seen: int = 0
    observation_count: int = 0


# ---------------------------------------------------------------------------
# Per-frame output of the tracking+SSG generator
# ---------------------------------------------------------------------------

@dataclass
class TrackedFrame:
    """Everything produced for a single frame."""
    frame_idx: int
    rgb_path: str
    depth_path: str

    objects: List[TrackedObject]
    local_graph: Any                     # nx.MultiDiGraph
    T_w_c: Optional[np.ndarray]          # 4×4 camera-to-world

    masks: List[np.ndarray] = field(default_factory=list)
    track_ids: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    depth_m: Optional[np.ndarray] = None

    timings: Dict[str, float] = field(default_factory=dict)
