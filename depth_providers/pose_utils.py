"""Shared pose-file helpers for depth providers and data loaders."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def load_poses_txt(pose_path: Optional[str]) -> Optional[List[np.ndarray]]:
    """Load a text file of 16-float camera-to-world matrices (one per line).

    Returns None if the path is missing or contains no valid rows.
    """
    if pose_path is None:
        return None
    path = Path(pose_path)
    if not path.exists():
        return None

    poses: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            poses.append(
                np.array(list(map(float, vals)), dtype=np.float32).reshape(4, 4)
            )
    return poses or None


def lookup_pose(
    poses: Optional[List[np.ndarray]],
    frame_idx: int,
    mode: str,
) -> Optional[np.ndarray]:
    """Index into a loaded pose list with ``index`` (0-based) or
    ``frame_number`` (1-based) semantics."""
    if poses is None:
        return None
    if mode == "index":
        idx = frame_idx
    elif mode == "frame_number":
        idx = frame_idx - 1
    else:
        raise ValueError(f"Unknown pose lookup mode: {mode}")
    if 0 <= idx < len(poses):
        return poses[idx]
    return None
