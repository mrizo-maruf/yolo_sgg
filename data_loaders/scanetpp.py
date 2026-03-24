"""ScanNet++ dataset loader.

ScanNet++ directory layout::

    scene/
        images/frame_000000.jpg  ...       (stride-10 numbering)
        gt_depth/frame_000000.png  ...     (uint16, metres = px / 1000)
        masks/frame_000000.jpg.npy  ...    (int16, 0=bg, nonzero=track_id)
        bbox/bboxes000000_info.json  ...   (sequential 0-based index)
        traj.txt                           (one 4×4 per line, 16 floats)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider
from depth_providers.gt_depth import _load_poses_txt
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader

_FRAME_RE = re.compile(r"^frame_(\d+)\.(jpg|png)$")


class ScanNetPPLoader(DatasetLoader):
    """Loader for ScanNet++ scenes with pluggable depth provider."""

    def __init__(
        self,
        scene_dir: str,
        depth_provider: DepthProvider,
        skip_labels: Optional[Set[str]] = None,
        fx: float = 692.52,
        fy: float = 693.83,
        cx: float = 459.76,
        cy: float = 344.76,
        image_width: int = 920,
        image_height: int = 690,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels: Set[str] = (
            {s.lower() for s in skip_labels}
            if skip_labels
            else {"wall", "floor", "ground", "ceiling", "background"}
        )

        self._rgb_dir = self._scene_dir / "images"
        self._mask_dir = self._scene_dir / "masks"
        self._bbox_dir = self._scene_dir / "bbox"

        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)
        self._depth_provider = depth_provider

        # Intrinsics from config (no intrinsic file in ScanNet++)
        self._intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=image_width, height=image_height,
        )

        # Poses from traj.txt (indexed by sequential position)
        traj_path = self._scene_dir / "traj.txt"
        self._poses = _load_poses_txt(str(traj_path)) if traj_path.exists() else None

        self._has_gt = self._bbox_dir.exists() and self._mask_dir.exists()

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        fnum = self._frame_number(frame_idx)
        path = self._rgb_dir / f"frame_{fnum:06d}.jpg"
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None and hasattr(self._depth_provider, "feed_frame"):
            self._depth_provider.feed_frame(fnum, img)
        return img, str(path)

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_number(frame_idx)
        return self._depth_provider.get_depth(fnum)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_number(frame_idx)

        # Prefer depth-provider pose (e.g. Pi3 predicted)
        pred_pose = self._depth_provider.get_pose(fnum)
        if pred_pose is not None:
            return pred_pose

        # Fall back to traj.txt GT poses (indexed by sequential position)
        if self._poses is not None and 0 <= frame_idx < len(self._poses):
            return self._poses[frame_idx]
        return None

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_masked_pcds(
        self,
        frame_idx: int,
        masks,
        max_points: int = 2000,
        sample_ratio: float = 0.5,
    ):
        """Override to pass fnum (actual frame number) to the depth provider."""
        fnum = self._frame_number(frame_idx)
        T_w_c = self.get_pose(frame_idx)
        intrinsics = self.get_intrinsics()
        return self._depth_provider.get_masked_pcds(
            fnum, masks, T_w_c, intrinsics,
            max_points=max_points, sample_ratio=sample_ratio,
        )

    # -- ground truth (benchmarking) ----------------------------------------

    def get_gt_instances(self, frame_idx: int) -> Optional[List[GTInstance]]:
        """Return GT instances for *frame_idx* (0-based).

        Reads bbox JSON (sequentially numbered) + mask .npy file.
        """
        if not self._has_gt:
            return None

        fnum = self._frame_number(frame_idx)

        # Bbox files are sequentially indexed (0, 1, 2, ...)
        bbox_path = self._bbox_dir / f"bboxes{frame_idx:06d}_info.json"
        # Mask files match image naming
        mask_path = self._mask_dir / f"frame_{fnum:06d}.jpg.npy"

        if not bbox_path.exists() or not mask_path.exists():
            return None

        bbox_data = _read_json(bbox_path)
        if bbox_data is None:
            return None

        mask_array = np.load(str(mask_path))  # int16, shape (H, W)

        bbox3d_list = _parse_bbox3d(bbox_data)

        instances: List[GTInstance] = []
        for b3d in bbox3d_list:
            track_id = int(b3d["track_id"])
            class_name = _infer_class_name(b3d)
            cls_lower = class_name.lower()
            if any(skip in cls_lower for skip in self._skip_labels):
                continue

            # Instance mask from .npy (track_id matching)
            mask = mask_array == track_id

            # Skip empty masks
            if not mask.any():
                continue

            # Derive 2D bbox from mask
            ys, xs = np.where(mask)
            bbox_xyxy = (float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max()))

            instances.append(
                GTInstance(
                    track_id=track_id,
                    class_name=class_name,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                )
            )
        return instances

    # -- multi-scene discovery ---------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        return sorted(
            str(d)
            for d in root_p.iterdir()
            if d.is_dir() and (d / "images").exists()
        )

    def _frame_number(self, frame_idx: int) -> int:
        if 0 <= frame_idx < len(self._frame_numbers):
            return self._frame_numbers[frame_idx]
        return frame_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_frame_numbers(rgb_dir: Path) -> List[int]:
    if not rgb_dir.exists():
        return []
    nums: List[int] = []
    for name in os.listdir(str(rgb_dir)):
        m = _FRAME_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_bbox3d(bbox_data: Dict) -> List[Dict[str, Any]]:
    """List of 3D box dicts (must have track_id)."""
    return [
        b
        for b in bbox_data.get("bboxes", {}).get("bbox_3d", {}).get("boxes", [])
        if "track_id" in b
    ]


def _infer_class_name(b3d: Dict[str, Any]) -> str:
    """Extract class name from prim_path (ScanNet++ has no labels).

    prim_path is typically '/Object/N' — not a real class name,
    so we return 'object' unless a meaningful name is present.
    """
    prim_path = b3d.get("prim_path", "")
    if prim_path:
        name = prim_path.rstrip("/").rsplit("/", 1)[-1]
        # If it's just a number, fall back to "object"
        if name and not name.isdigit():
            return name
    return "object"
