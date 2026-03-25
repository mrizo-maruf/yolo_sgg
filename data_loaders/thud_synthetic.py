"""THUD Synthetic dataset loader."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader

_DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}

_RGB_RE = re.compile(r"^rgb_(\d+)\.png$")


class THUDSyntheticLoader(DatasetLoader):
    """Loader for THUD Synthetic scenes with pluggable depth provider.

    Frame indexing: RGB/depth files use integer N (rgb_N.png).
    JSON annotations use step where step = N - 2.
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: DepthProvider,
        skip_labels: Optional[Set[str]] = None,
        depth_scale: float = 1000.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS

        self._rgb_dir = self._scene_dir / "RGB"
        self._depth_dir = self._scene_dir / "Depth"
        self._label_dir = self._scene_dir / "Label"

        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)
        self._cam_data: Dict[int, Dict[str, Any]] = _load_camera_data(self._label_dir)
        self._intrinsics = self._resolve_intrinsics()

        self._depth_provider = depth_provider

    @property
    def scene_label(self) -> str:
        parts = self._scene_dir.parts
        try:
            idx = next(
                i for i, p in enumerate(parts)
                if p in ("Gym", "House", "Office", "Supermarket_1", "Supermarket_2")
            )
            return "/".join(parts[idx:])
        except StopIteration:
            return self._scene_dir.name

    def get_scene_name(self) -> str:
        """Return the THUD environment name (Gym, Office, etc.)."""
        parts = self._scene_dir.parts
        for p in parts:
            if p in ("Gym", "House", "Office", "Supermarket_1", "Supermarket_2"):
                return p
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        fnum = self._frame_numbers[frame_idx]
        path = str(self._rgb_dir / f"rgb_{fnum}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None and hasattr(self._depth_provider, 'feed_frame'):
            self._depth_provider.feed_frame(fnum, img)
        return img, path

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_numbers[frame_idx]
        return self._depth_provider.get_depth(fnum)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_numbers[frame_idx]

        # Prefer depth-provider pose (predicted)
        pred_pose = self._depth_provider.get_pose(fnum)
        if pred_pose is not None:
            return pred_pose

        # Fall back to JSON annotation poses
        step = fnum - 2
        cam = self._cam_data.get(step)
        if cam is None:
            return None
        t = cam.get("translation")
        r = cam.get("rotation")
        if t is None or r is None:
            return None
        return _compose_transform_4x4(t, r)

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    # -- ground truth (benchmarking) ----------------------------------------

    def get_gt_instances(self, frame_idx: int) -> Optional[List[GTInstance]]:
        """Return GT instances for *frame_idx* (0-based).

        Reads annotations parsed at init + instance segmentation PNG.
        Returns ``None`` if GT data is unavailable for this frame.
        """
        if frame_idx < 0 or frame_idx >= len(self._frame_numbers):
            return None
        fnum = self._frame_numbers[frame_idx]
        step = fnum - 2

        ann = self._cam_data.get(step)
        if ann is None:
            return None

        bbox2d = ann.get("bbox2d", {})
        bbox3d = ann.get("bbox3d", {})
        inst_colors = ann.get("inst_color", {})
        if not bbox3d and not inst_colors:
            return None

        # Load instance segmentation image
        inst_path = self._label_dir / "Instance" / f"Instance_{fnum}.png"
        if not inst_path.exists():
            return None
        inst_img = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
        if inst_img is None:
            return None

        # Use track_ids from bbox3d (primary), fallback to inst_colors keys
        all_ids = set(bbox3d.keys()) | set(inst_colors.keys())
        skip = self._skip_labels

        instances: List[GTInstance] = []
        for inst_id in sorted(all_ids):
            # Class name from 3D annotation
            b3 = bbox3d.get(inst_id, {})
            class_name = b3.get("label_name", "unknown")
            if class_name.lower() in skip:
                continue

            # 2D bbox: (x, y, w, h) -> (x1, y1, x2, y2)
            bbox_xyxy: Optional[Tuple[float, ...]] = None
            b2 = bbox2d.get(inst_id)
            if b2 is not None:
                x, y, w, h = b2["x"], b2["y"], b2["width"], b2["height"]
                bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))

            # Instance mask
            rgba = inst_colors.get(inst_id)
            mask = _extract_mask(inst_img, rgba) if rgba is not None else None

            instances.append(
                GTInstance(
                    track_id=int(inst_id),
                    class_name=class_name,
                    mask=mask,
                    bbox_xyxy=bbox_xyxy,
                    bbox_xyzxyz=_extract_bbox_xyzxyz(b3),
                )
            )
        return instances

    def _resolve_intrinsics(self) -> CameraIntrinsics:
        for step in sorted(self._cam_data):
            K = self._cam_data[step].get("intrinsic")
            if K is not None:
                if self._frame_numbers:
                    fnum = self._frame_numbers[0]
                    p = self._rgb_dir / f"rgb_{fnum}.png"
                    img = cv2.imread(str(p))
                    if img is not None:
                        h, w = img.shape[:2]
                        return CameraIntrinsics.from_K(K, w, h)
                return CameraIntrinsics.from_K(K, 730, 530)
        return CameraIntrinsics(fx=500.0, fy=500.0, cx=365.0, cy=265.0,
                                width=730, height=530)

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        scenes = []
        for cap_dir in sorted(root_p.rglob("Capture_*")):
            if (cap_dir / "RGB").exists():
                scenes.append(str(cap_dir))
        return scenes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_frame_numbers(rgb_dir: Path) -> List[int]:
    if not rgb_dir.exists():
        return []
    nums: List[int] = []
    for name in os.listdir(str(rgb_dir)):
        m = _RGB_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums


def _load_camera_data(label_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Parse all captures_*.json → step-indexed annotations.

    Each step entry contains:
      intrinsic, translation, rotation  – camera info
      bbox2d   – {instance_id: {x, y, width, height}}
      bbox3d   – {instance_id: {label_name, ...}}
      inst_color – {instance_id: (R, G, B, A)}
    """
    cam: Dict[int, Dict[str, Any]] = {}
    if not label_dir.exists():
        return cam
    for jf in sorted(label_dir.glob("captures_*.json")):
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for capture in data.get("captures", []):
            step = int(capture.get("step", -1))
            if step < 0:
                continue
            d = cam.setdefault(step, {})

            # Camera intrinsics / pose
            sensor = capture.get("sensor", {})
            if "intrinsic" not in d:
                intr = sensor.get("camera_intrinsic")
                if intr:
                    try:
                        arr = np.array(intr, dtype=np.float64)
                        if arr.shape == (3, 3):
                            d["intrinsic"] = arr
                    except Exception:
                        pass
            if "translation" not in d:
                pos = sensor.get("position")
                if pos:
                    d["translation"] = (
                        float(pos.get("x", 0)),
                        float(pos.get("y", 0)),
                        float(pos.get("z", 0)),
                    )
            if "rotation" not in d:
                rot = sensor.get("rotation")
                if rot:
                    d["rotation"] = (
                        float(rot.get("x", 0)),
                        float(rot.get("y", 0)),
                        float(rot.get("z", 0)),
                        float(rot.get("w", 1)),
                    )

            # GT annotations ----------------------------------------------
            for ann in capture.get("annotations", []):
                ann_id = ann.get("annotation_definition", ann.get("id", ""))
                values = ann.get("values", [])

                if "bounding box" == ann_id and "bbox2d" not in d:
                    b2 = {}
                    for v in values:
                        iid = v.get("instance_id")
                        if iid is not None:
                            b2[int(iid)] = v
                    d["bbox2d"] = b2

                elif "bounding box 3D" == ann_id and "bbox3d" not in d:
                    b3 = {}
                    for v in values:
                        iid = v.get("instance_id")
                        if iid is not None:
                            b3[int(iid)] = v
                    d["bbox3d"] = b3

                elif "instance segmentation" == ann_id and "inst_color" not in d:
                    ic = {}
                    for v in values:
                        iid = v.get("instance_id")
                        c = v.get("color")
                        if iid is not None and c:
                            ic[int(iid)] = (
                                int(c.get("r", 0)),
                                int(c.get("g", 0)),
                                int(c.get("b", 0)),
                                int(c.get("a", 255)),
                            )
                    d["inst_color"] = ic
    return cam


def _compose_transform_4x4(
    translation: tuple, rotation: tuple,
) -> np.ndarray:
    """(x,y,z) + (qx,qy,qz,qw) -> 4x4 transform."""
    qx, qy, qz, qw = rotation
    T = np.eye(4, dtype=np.float64)

    # Rotation from quaternion
    T[0, 0] = 1 - 2 * (qy**2 + qz**2)
    T[0, 1] = 2 * (qx * qy - qz * qw)
    T[0, 2] = 2 * (qx * qz + qy * qw)
    T[1, 0] = 2 * (qx * qy + qz * qw)
    T[1, 1] = 1 - 2 * (qx**2 + qz**2)
    T[1, 2] = 2 * (qy * qz - qx * qw)
    T[2, 0] = 2 * (qx * qz - qy * qw)
    T[2, 1] = 2 * (qy * qz + qx * qw)
    T[2, 2] = 1 - 2 * (qx**2 + qy**2)

    T[0, 3] = translation[0]
    T[1, 3] = translation[1]
    T[2, 3] = translation[2]
    return T


def _extract_mask(
    inst_img: np.ndarray, rgba: Tuple[int, int, int, int]
) -> np.ndarray:
    """Extract boolean mask from instance segmentation image.

    *rgba* is (R, G, B, A) from JSON. OpenCV loads images as BGR[A],
    so channels are swapped before comparison.
    """
    r, g, b, a = rgba
    if inst_img.ndim == 2:
        return np.zeros(inst_img.shape[:2], dtype=bool)
    channels = inst_img.shape[2]
    if channels >= 4:
        target = np.array([b, g, r, a], dtype=np.uint8)
        return np.all(inst_img[:, :, :4] == target, axis=2)
    # 3-channel: ignore alpha
    target = np.array([b, g, r], dtype=np.uint8)
    return np.all(inst_img[:, :, :3] == target, axis=2)


def _extract_bbox_xyzxyz(b3d: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    """Best-effort parse of axis-aligned 3D bbox from THUD annotation entry."""
    if not isinstance(b3d, dict):
        return None

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

    center = b3d.get("center")
    size = b3d.get("size")
    if isinstance(center, dict) and isinstance(size, dict):
        cx, cy, cz = float(center.get("x", 0.0)), float(center.get("y", 0.0)), float(center.get("z", 0.0))
        sx, sy, sz = float(size.get("x", 0.0)), float(size.get("y", 0.0)), float(size.get("z", 0.0))
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        return (cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz)

    return None
