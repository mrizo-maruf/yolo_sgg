"""IsaacSim dataset loader."""
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
from depth_providers.pose_utils import load_poses_txt
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader

_FRAME_RE = re.compile(r"^frame(\d+)\.(jpg|png)$")


class IsaacSimLoader(DatasetLoader):
    """Loader for IsaacSim scenes with pluggable depth provider.

    Directory layout::

        scene/
            rgb/frame000001.jpg  ...
            depth/depth000001.png  ...
            bbox/bboxes000001_info.json  ...   (GT annotations)
            seg/semantic000001.png  ...        (GT segmentation)
            seg/semantic000001_info.json  ...  (instance-colour map)
            traj.txt
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: DepthProvider,
        skip_labels: Optional[Set[str]] = None,
        image_width: int = 1280,
        image_height: int = 720,
        focal_length: float = 50.0,
        horizontal_aperture: float = 80.0,
        vertical_aperture: float = 45.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels: Set[str] = (
            {s.lower() for s in skip_labels}
            if skip_labels
            else {"wall", "floor", "ground", "ceiling", "background"}
        )

        self._intrinsics = CameraIntrinsics.from_physical(
            focal_length, horizontal_aperture, vertical_aperture,
            image_width, image_height,
        )

        self._rgb_dir = self._scene_dir / "rgb"
        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)

        self._depth_provider = depth_provider

        traj_path = self._scene_dir / "traj.txt"
        self._poses = load_poses_txt(str(traj_path)) if traj_path.exists() else None

        # GT directories (may not exist for every scene)
        self._bbox_dir = self._scene_dir / "bbox"
        self._seg_dir = self._scene_dir / "seg"
        self._has_gt = self._bbox_dir.exists() and self._seg_dir.exists()

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        fnum = self._frame_number(frame_idx)
        path = self._rgb_dir / f"frame{fnum:06d}.jpg"

        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is not None and hasattr(self._depth_provider, 'feed_frame'):
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

        # Fall back to traj.txt GT poses
        if self._poses is not None:
            if 0 <= frame_idx < len(self._poses):
                return self._poses[frame_idx]
            if 1 <= fnum <= len(self._poses):
                return self._poses[fnum - 1]
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
        """Override to pass fnum (1-based) to the depth provider."""
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

        Reads bbox JSON + segmentation PNG/JSON directly.  Returns
        ``None`` when the scene has no GT annotations.
        """
        if not self._has_gt:
            return None

        fnum = self._frame_number(frame_idx)
        fs = f"{fnum:06d}"

        bbox_path = self._bbox_dir / f"bboxes{fs}_info.json"
        seg_png_path = self._seg_dir / f"semantic{fs}.png"
        seg_info_path = self._seg_dir / f"semantic{fs}_info.json"

        if not bbox_path.exists() or not seg_png_path.exists():
            return None

        bbox_data = _read_json(bbox_path)
        if bbox_data is None:
            return None

        seg_info = _read_json(seg_info_path) if seg_info_path.exists() else {}
        seg_img = cv2.imread(str(seg_png_path), cv2.IMREAD_COLOR)  # BGR
        if seg_img is None:
            return None

        bbox2d_by_id = _parse_bbox2d_tight(bbox_data)
        bbox3d_list = _parse_bbox3d(bbox_data)
        color_by_inst = _parse_instance_color_map(seg_info)

        instances: List[GTInstance] = []
        for b3d in bbox3d_list:
            track_id = int(b3d["track_id"])
            inst_seg_id = int(b3d.get("instance_seg_id", -1))
            bbox_2d_id = int(b3d.get("bbox_2d_id", -1))
            bbox_3d_id = int(b3d.get("bbox_3d_id", -1))

            # Skip objects with incomplete cross-reference IDs
            if bbox_3d_id < 0 or bbox_2d_id < 0 or inst_seg_id < 0:
                continue

            b2d = bbox2d_by_id.get(bbox_2d_id)
            class_name = _infer_class_name(b3d, b2d)
            cls_lower = class_name.lower()
            if any(skip in cls_lower for skip in self._skip_labels):
                continue

            # 2D bbox
            bbox_xyxy: Optional[Tuple[float, ...]] = None
            if b2d is not None:
                xyxy = b2d.get("xyxy")
                if xyxy and len(xyxy) == 4:
                    bbox_xyxy = tuple(float(v) for v in xyxy)

            # Instance mask via colour matching
            color = color_by_inst.get(inst_seg_id)
            mask = (
                np.all(seg_img == np.array(color, dtype=np.uint8), axis=2)
                if color is not None
                else np.zeros(seg_img.shape[:2], dtype=bool)
            )

            instances.append(
                GTInstance(
                    track_id=track_id,
                    class_name=class_name,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    bbox_xyzxyz=_extract_bbox_xyzxyz(b3d),
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
            if d.is_dir() and (d / "rgb").exists()
        )

    def _frame_number(self, frame_idx: int) -> int:
        if 0 <= frame_idx < len(self._frame_numbers):
            return self._frame_numbers[frame_idx]
        return frame_idx

    def provider_frame_key(self, frame_idx: int) -> int:
        return self._frame_number(frame_idx)


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


def _parse_bbox2d_tight(bbox_data: Dict) -> Dict[int, Dict[str, Any]]:
    """bbox_2d_id -> 2D box dict."""
    boxes = (
        bbox_data.get("bboxes", {})
        .get("bbox_2d_tight", {})
        .get("boxes", [])
    )
    out: Dict[int, Dict[str, Any]] = {}
    for b in boxes:
        bid = b.get("bbox_2d_id", b.get("bbox_id"))
        if bid is not None:
            out[int(bid)] = b
    return out


def _parse_bbox3d(bbox_data: Dict) -> List[Dict[str, Any]]:
    """List of 3D box dicts (must have track_id)."""
    return [
        b
        for b in bbox_data.get("bboxes", {}).get("bbox_3d", {}).get("boxes", [])
        if "track_id" in b
    ]


def _parse_instance_color_map(
    seg_info: Dict,
) -> Dict[int, Tuple[int, int, int]]:
    """instance_seg_id -> (B, G, R) colour for mask extraction."""
    out: Dict[int, Tuple[int, int, int]] = {}
    for k, v in seg_info.items():
        if not str(k).isdigit():
            continue
        color = v.get("color_bgr")
        if color and len(color) == 3:
            out[int(k)] = (int(color[0]), int(color[1]), int(color[2]))
    return out


def _infer_class_name(
    b3d: Dict[str, Any], b2d: Optional[Dict[str, Any]]
) -> str:
    """Best-effort class name from 3D label -> 2D label -> fallback."""
    if "label" in b3d and isinstance(b3d["label"], str) and b3d["label"]:
        return b3d["label"]
    if b2d is not None:
        lab = b2d.get("label")
        if isinstance(lab, dict) and lab:
            return str(next(iter(lab.values())))
    return "unknown"


def _extract_bbox_xyzxyz(b3d: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    """Best-effort parse of 3D AABB from a bbox_3d entry."""
    aabb = b3d.get("aabb_xyzmin_xyzmax")
    if isinstance(aabb, (list, tuple)) and len(aabb) == 6:
        return tuple(float(v) for v in aabb)

    aabb = b3d.get("aabb")
    if isinstance(aabb, dict):
        mn = aabb.get("min")
        mx = aabb.get("max")
        if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)) and len(mn) == 3 and len(mx) == 3:
            return (
                float(mn[0]), float(mn[1]), float(mn[2]),
                float(mx[0]), float(mx[1]), float(mx[2]),
            )
    return None
