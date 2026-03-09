"""
THUD Synthetic Scene Loader
============================
Drop-in replacement for IsaacSimSceneLoader for THUD synthetic scenes.
Uses the same GTObject / FrameData interface so the 3-D tracking benchmark
pipeline works unchanged.

Expected directory layout
-------------------------
scene_dir/                                # e.g. .../Synthetic_Scenes/Gym/static/Capture_1
├── RGB/rgb_<N>.png
├── Depth/depth_<N>.png
└── Label/
    ├── Instance/Instance_<N>.png
    ├── Semantic/segmentation_<N>.png
    ├── captures_000.json
    ├── captures_001.json
    └── ...

File index convention
---------------------
RGB/depth/instance images use the same integer <N>.
JSON annotations use ``step`` where  step = N - 2  (i.e. frame_idx = step + 2).

Usage
-----
    loader = THUDSyntheticLoader(
        "/path/to/Gym/static/Capture_1",
        load_rgb=True,
        load_depth=True,
    )
    for idx in loader.frame_indices:
        fd = loader.get_frame_data(idx)
        for obj in fd.gt_objects:
            print(obj.track_id, obj.class_name, obj.bbox2d_xyxy)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _heading_angle_from_quat(
    qx: float, qy: float, qz: float, qw: float,
) -> float:
    """Extract a single heading angle from a Unity quaternion.

    Mirrors the logic in the THUD ``Export3DAnnotation.py`` script.
    The heading is a yaw angle about the *up* axis (which becomes the
    z-axis after the y↔z swap applied to centres and sizes).
    """
    x_angle = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    val = qw * qy - qz * qx
    val = max(-0.4999, min(0.4999, val))
    if int(x_angle) > 0:
        return math.asin(2.0 * val)
    else:
        return -math.asin(2.0 * val)


def _quat_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float,
) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix (float32)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _compose_transform_4x4(
    translation: Tuple[float, float, float],
    rotation_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    """Translation + quaternion -> 4x4 homogeneous transform (float32)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_to_rotation_matrix(*rotation_xyzw)
    T[:3, 3] = translation
    return T


def _obb_to_aabb(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    rotation_xyzw: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    """Oriented bounding box -> axis-aligned bounding box.

    Returns (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    R = _quat_to_rotation_matrix(*rotation_xyzw)
    half = np.array(size, dtype=np.float32) / 2.0
    signs = np.array(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1],  [1, -1, 1],  [1, 1, -1],  [1, 1, 1],
        ],
        dtype=np.float32,
    )
    corners = (signs * half) @ R.T + np.array(center, dtype=np.float32)
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mn[2]),
            float(mx[0]), float(mx[1]), float(mx[2]))


# ---------------------------------------------------------------------------
# Data structures  (field layout mirrors isaac_sim_loader)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GTObject:
    """Ground-truth object for one frame.

    Core fields match ``IsaacSimSceneLoader.GTObject`` exactly so that
    benchmark / metrics code can consume either loader's objects.

    Extra THUD-specific fields (``box_3d_center_xyz``, ``box_3d_size_xyz``,
    ``box_3d_rotation_xyzw``) preserve oriented-box parameters for future
    3-D IoU computation without information loss.
    """

    track_id: int                 # instance_id — persistent across the sequence
    instance_seg_id: int          # == track_id in THUD synthetic
    bbox_2d_id: int               # == track_id in THUD synthetic
    bbox_3d_id: int               # == track_id in THUD synthetic

    class_name: str
    prim_path: Optional[str]      # always None for THUD

    # 2-D
    bbox2d_xyxy: Optional[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)

    # 3-D (IsaacSim-compatible)
    box_3d_aabb_xyzmin_xyzmax: Tuple[float, float, float, float, float, float]
    box_3d_transform_4x4: np.ndarray  # (4, 4) float32  — pose of the object

    # 3-D extras (OBB parameters, for direct 3-D IoU)
    box_3d_center_xyz: Tuple[float, float, float]
    box_3d_size_xyz: Tuple[float, float, float]
    box_3d_rotation_xyzw: Tuple[float, float, float, float]

    visibility: Optional[float]   # not available in THUD — always None
    occlusion: Optional[float]    # not available in THUD — always None

    mask: np.ndarray              # H x W  bool


@dataclass(frozen=True)
class FrameData:
    """Per-frame container.

    Same layout as ``IsaacSimSceneLoader.FrameData`` plus
    ``camera_intrinsic`` (3x3).
    """

    frame_idx: int
    gt_objects: List[GTObject]

    rgb: Optional[np.ndarray] = None               # H x W x 3  BGR  (cv2 convention)
    depth: Optional[np.ndarray] = None              # H x W      float32
    cam_transform_4x4: Optional[np.ndarray] = None  # 4 x 4      camera-to-world
    camera_intrinsic: Optional[np.ndarray] = None   # 3 x 3      (not in IsaacSim)
    seg: Optional[np.ndarray] = None                # H x W x C  instance-seg image (for viz)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class THUDSyntheticLoader:
    """Loads one THUD synthetic capture (static or dynamic).

    API mirrors ``IsaacSimSceneLoader`` so the benchmark pipeline can use
    either loader interchangeably.

    Parameters
    ----------
    scene_dir : str
        Path to the capture directory, e.g.
        ``/data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1``
    load_rgb : bool
        If True, ``FrameData.rgb`` is populated (BGR, cv2).
    load_depth : bool
        If True, ``FrameData.depth`` is populated (float32).
    skip_labels : set[str] | None
        Class names (case-insensitive) to exclude entirely.
    require_3d : bool
        If True (default), objects without a 3-D bounding box in the JSON
        are silently skipped.
    verbose : bool
        Print per-frame loading info.
    """

    def __init__(
        self,
        scene_dir: str,
        load_rgb: bool = False,
        load_depth: bool = False,
        skip_labels: Optional[Set[str]] = None,
        require_3d: bool = True,
        verbose: bool = True,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene dir not found: {self.scene_dir}")

        self.rgb_dir = self.scene_dir / "RGB"
        self.depth_dir = self.scene_dir / "Depth"
        self.label_dir = self.scene_dir / "Label"
        self.instance_dir = self.label_dir / "Instance"
        self.semantic_dir = self.label_dir / "Semantic"

        for d in [self.rgb_dir, self.label_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Required folder missing: {d}")

        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.skip_labels: Set[str] = {s.lower() for s in (skip_labels or set())}
        self.require_3d = require_3d
        self.verbose = verbose

        # Pre-parse all JSON annotations.
        # _step_ann[step] = {
        #     "bbox2d":  {instance_id: raw_dict, ...},
        #     "bbox3d":  {instance_id: raw_dict, ...},
        #     "inst_color": {instance_id: (r,g,b,a), ...},
        #     "camera_intrinsic": np.ndarray | None,
        #     "cam_translation": (x,y,z) | None,
        #     "cam_rotation":    (qx,qy,qz,qw) | None,
        # }
        self._step_ann: Dict[int, Dict[str, Any]] = self._load_all_annotations()

        # Discover available frame indices from RGB files on disk.
        self.frame_indices: List[int] = self._discover_frames()

        if self.verbose:
            print(
                f"[THUDSyntheticLoader] {self.scene_dir.name}: "
                f"{len(self.frame_indices)} frames, "
                f"{len(self._step_ann)} annotated steps"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame_count(self) -> int:
        return len(self.frame_indices)

    def get_frame_indices(self) -> List[int]:
        return list(self.frame_indices)

    def get_frame_data(self, frame_idx: int) -> FrameData:
        """Build ``FrameData`` for the given frame index.

        Parameters
        ----------
        frame_idx : int
            Frame index (matches the ``<N>`` in ``rgb_<N>.png``).

        Returns
        -------
        FrameData
        """
        step = frame_idx - 2
        ann = self._step_ann.get(step, {})

        # ---- Parse per-object annotations --------------------------------
        bbox3d: Dict[int, dict] = ann.get("bbox3d", {})
        bbox2d: Dict[int, dict] = ann.get("bbox2d", {})
        inst_color: Dict[int, Tuple[int, int, int, int]] = ann.get("inst_color", {})

        # Load instance-segmentation image once (needed for mask extraction).
        instance_img = self._read_image_unchanged(
            self.instance_dir / f"Instance_{frame_idx}.png"
        )

        gt_objects: List[GTObject] = []

        # Union of all instance IDs seen across annotation types.
        all_ids: Set[int] = set(bbox3d.keys()) | set(bbox2d.keys()) | set(inst_color.keys())

        for inst_id in sorted(all_ids):
            b3 = bbox3d.get(inst_id)
            b2 = bbox2d.get(inst_id)

            # Optionally skip objects without 3-D box.
            if self.require_3d and b3 is None:
                continue

            # ---- Class name ----
            class_name = "unknown"
            label_id = -1
            if b3 is not None:
                class_name = str(b3.get("label_name", "unknown"))
                label_id = int(b3.get("label_id", -1))
            elif b2 is not None:
                class_name = str(b2.get("label_name", "unknown"))
                label_id = int(b2.get("label_id", -1))

            if class_name.lower() in self.skip_labels:
                continue

            # ---- 2-D bounding box ----
            bbox2d_xyxy: Optional[Tuple[float, float, float, float]] = None
            if b2 is not None:
                x = float(b2.get("x", 0.0))
                y = float(b2.get("y", 0.0))
                w = float(b2.get("width", 0.0))
                h = float(b2.get("height", 0.0))
                bbox2d_xyxy = (x, y, x + w, y + h)

            # ---- 3-D bounding box ----
            # Matches the coordinate convention used by the THUD
            # ``Export3DAnnotation.py`` + ``Depth_to_pointcloud.py``
            # pipeline.  Both y and z axes are swapped so that the
            # resulting frame is (x-right, y-forward, z-up) and the
            # heading angle rotates about the z (up) axis.
            if b3 is not None:
                t = b3.get("translation", {})
                s = b3.get("size", {})
                r = b3.get("rotation", {})
                center = (
                    float(t.get("x", 0.0)),
                    float(t.get("z", 0.0)),     # swap y↔z (forward)
                    float(t.get("y", 0.0)),     # swap y↔z (up)
                )
                size = (
                    float(s.get("x", 0.0)),
                    float(s.get("z", 0.0)),     # swap y↔z
                    float(s.get("y", 0.0)),     # swap y↔z
                )
                # Heading angle about the z (up) axis — same as Export3DAnnotation
                heading = _heading_angle_from_quat(
                    float(r.get("x", 0.0)),
                    float(r.get("y", 0.0)),
                    float(r.get("z", 0.0)),
                    float(r.get("w", 1.0)),
                )
                # Store as z-axis rotation quaternion so the existing
                # OBB→AABB and vis helpers still work.
                rot = (
                    0.0,
                    0.0,
                    float(np.sin(heading / 2.0)),
                    float(np.cos(heading / 2.0)),
                )
                transform = _compose_transform_4x4(center, rot)
                aabb = _obb_to_aabb(center, size, rot)
            else:
                center = (0.0, 0.0, 0.0)
                size = (0.0, 0.0, 0.0)
                rot = (0.0, 0.0, 0.0, 1.0)
                transform = np.eye(4, dtype=np.float32)
                aabb = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # ---- Instance mask ----
            color_rgba = inst_color.get(inst_id)
            mask = self._extract_mask(instance_img, color_rgba)

            gt_objects.append(
                GTObject(
                    track_id=int(inst_id),
                    instance_seg_id=int(inst_id),
                    bbox_2d_id=int(inst_id),
                    bbox_3d_id=int(inst_id),
                    class_name=class_name,
                    prim_path=None,
                    bbox2d_xyxy=bbox2d_xyxy,
                    box_3d_aabb_xyzmin_xyzmax=aabb,
                    box_3d_transform_4x4=transform,
                    box_3d_center_xyz=center,
                    box_3d_size_xyz=size,
                    box_3d_rotation_xyzw=rot,
                    visibility=None,
                    occlusion=None,
                    mask=mask,
                )
            )

        # ---- Images -------------------------------------------------------
        rgb = (
            self._read_image_bgr(self.rgb_dir / f"rgb_{frame_idx}.png")
            if self.load_rgb
            else None
        )
        depth = (
            self._read_depth(self.depth_dir / f"depth_{frame_idx}.png")
            if self.load_depth
            else None
        )
        seg = self._read_image_unchanged(
            self.semantic_dir / f"segmentation_{frame_idx}.png"
        )

        # ---- Camera pose --------------------------------------------------
        cam_t = ann.get("cam_translation")
        cam_r = ann.get("cam_rotation")
        cam_transform = (
            _compose_transform_4x4(cam_t, cam_r) if cam_t and cam_r else None
        )
        cam_intr = ann.get("camera_intrinsic")  # np.ndarray (3,3) or None

        if self.verbose:
            print(f"  [LOADER] Frame {frame_idx}: {len(gt_objects)} objects")

        return FrameData(
            frame_idx=frame_idx,
            gt_objects=gt_objects,
            rgb=rgb,
            depth=depth,
            cam_transform_4x4=cam_transform,
            camera_intrinsic=cam_intr,
            seg=seg,
        )

    # ------------------------------------------------------------------
    # Annotation parsing
    # ------------------------------------------------------------------

    def _load_all_annotations(self) -> Dict[int, Dict[str, Any]]:
        """Parse every ``captures_*.json`` and index by *step*."""
        step_ann: Dict[int, Dict[str, Any]] = {}

        for jf in sorted(self.label_dir.glob("captures_*.json")):
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for capture in data.get("captures", []):
                step = int(capture.get("step", -1))
                if step < 0:
                    continue

                d = step_ann.setdefault(
                    step,
                    {
                        "bbox2d": {},
                        "bbox3d": {},
                        "inst_color": {},
                        "camera_intrinsic": None,
                        "cam_translation": None,
                        "cam_rotation": None,
                    },
                )

                # ---- Sensor / camera info ----
                sensor = capture.get("sensor", {})

                if d["camera_intrinsic"] is None:
                    intr = sensor.get("camera_intrinsic")
                    if intr:
                        try:
                            arr = np.array(intr, dtype=np.float64)
                            if arr.shape == (3, 3):
                                d["camera_intrinsic"] = arr
                        except Exception:
                            pass

                if d["cam_translation"] is None:
                    t = sensor.get("translation")
                    r = sensor.get("rotation")
                    if t and len(t) == 3:
                        d["cam_translation"] = tuple(float(v) for v in t)
                    if r and len(r) == 4:
                        d["cam_rotation"] = tuple(float(v) for v in r)

                # ---- Per-object annotations ----
                for ann in capture.get("annotations", []):
                    ann_id = str(ann.get("id", ""))
                    values = ann.get("values", [])

                    if ann_id == "bounding box":
                        for v in values:
                            iid = v.get("instance_id")
                            if iid is not None:
                                d["bbox2d"][int(iid)] = v

                    elif ann_id == "bounding box 3D":
                        for v in values:
                            iid = v.get("instance_id")
                            if iid is not None:
                                d["bbox3d"][int(iid)] = v

                    elif ann_id == "instance segmentation":
                        for v in values:
                            iid = v.get("instance_id")
                            c = v.get("color", {})
                            if iid is not None:
                                d["inst_color"][int(iid)] = (
                                    int(c.get("r", 0)),
                                    int(c.get("g", 0)),
                                    int(c.get("b", 0)),
                                    int(c.get("a", 255)),
                                )

        return step_ann

    # ------------------------------------------------------------------
    # Frame discovery
    # ------------------------------------------------------------------

    def _discover_frames(self) -> List[int]:
        """Scan ``RGB/`` for ``rgb_<N>.png`` and return sorted indices."""
        indices: set = set()
        for p in self.rgb_dir.glob("rgb_*.png"):
            m = re.search(r"rgb_(\d+)\.png$", p.name)
            if m:
                indices.add(int(m.group(1)))
        return sorted(indices)

    # ------------------------------------------------------------------
    # Image I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image_bgr(path: Path) -> Optional[np.ndarray]:
        """Read image as BGR uint8 (cv2 convention)."""
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def _read_image_unchanged(path: Path) -> Optional[np.ndarray]:
        """Read image preserving all channels (BGRA for 4-ch PNGs)."""
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img

    @staticmethod
    def _read_depth(path: Path) -> Optional[np.ndarray]:
        """Read depth PNG as float32 (raw values, no unit conversion)."""
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return img.astype(np.float32)

    # ------------------------------------------------------------------
    # Mask extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mask(
        instance_img: Optional[np.ndarray],
        rgba: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Extract a boolean mask from the instance-segmentation image.

        Parameters
        ----------
        instance_img : np.ndarray | None
            Loaded with ``cv2.IMREAD_UNCHANGED`` — could be BGR (3-ch) or
            BGRA (4-ch).  Channel order is always B-G-R-[A] when read by cv2.
        rgba : tuple[int,int,int,int] | None
            Colour from JSON, given as (R, G, B, A).

        Returns
        -------
        np.ndarray
            H x W  bool mask.
        """
        if instance_img is None or rgba is None:
            return np.zeros((0, 0), dtype=bool)

        r, g, b, a = rgba

        if instance_img.ndim < 3:
            return np.zeros(instance_img.shape[:2], dtype=bool)

        # cv2 channel order: B=0, G=1, R=2, [A=3]
        if instance_img.shape[2] >= 4:
            m = (
                (instance_img[:, :, 2] == r)
                & (instance_img[:, :, 1] == g)
                & (instance_img[:, :, 0] == b)
                & (instance_img[:, :, 3] == a)
            )
        else:
            m = (
                (instance_img[:, :, 2] == r)
                & (instance_img[:, :, 1] == g)
                & (instance_img[:, :, 0] == b)
            )
        return m

    # ------------------------------------------------------------------
    # Depth → point cloud  (OpenCV camera frame, matching bbox convention)
    # ------------------------------------------------------------------

    @staticmethod
    def depth_to_pointcloud(
        depth: np.ndarray,
        camera_intrinsic: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        num_sample: int = 40000,
        **_kwargs,
    ) -> np.ndarray:
        """Unproject a depth image into a coloured point cloud.

        Replicates the THUD ``Depth_to_pointcloud.py`` +
        ``ExportPointCloud.py`` pipeline so that the resulting point
        cloud is in the same **(x-right, y-forward, z-up)** frame as
        the 3-D bounding boxes produced by ``get_frame_data``.

        Parameters
        ----------
        depth : np.ndarray
            H × W raw uint16 depth image.
        camera_intrinsic : np.ndarray
            3 × 3 intrinsic matrix from the JSON annotation.
        rgb : np.ndarray | None
            H × W × 3 BGR image (cv2 convention).  If provided the
            returned array has 6 columns (XYZRGB) instead of 3 (XYZ).
        num_sample : int
            Randomly sub-sample to this many points (default 40 000).
            Set 0 to keep all valid points.

        Returns
        -------
        np.ndarray
            (N, 3) or (N, 6) float32 — XYZ [+ RGB 0-1].
        """
        H, W = depth.shape[:2]
        fx = float(camera_intrinsic[0, 0])
        fy = float(camera_intrinsic[1, 1])
        cx = float(camera_intrinsic[0, 2])
        cy = float(camera_intrinsic[1, 2])

        depth_f = depth.astype(np.float32)

        # Pixel grids — v is flipped (top-to-bottom) as in the THUD pipeline
        u = np.arange(W, dtype=np.float32)
        v_flip = np.arange(H - 1, -1, -1, dtype=np.float32)
        u, v_flip = np.meshgrid(u, v_flip)  # both (H, W)

        # Depth_to_pointcloud.py formula (raw, before /1000)
        Xr = (u - cx) * depth_f / fx
        Zr = (v_flip - cy) * depth_f / fy

        # Apply THUD scale/offset then convert mm→m  (ExportPointCloud / 1000)
        X = Xr / 2.5 / 1000.0
        Y = (depth_f / 6.5 + 200.0) / 1000.0   # forward (depth direction)
        Z = (Zr / 2.0 + 300.0) / 1000.0         # up (flipped-v direction)

        valid = depth_f > 0
        pts = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)

        if rgb is not None:
            rgb_f = rgb[:, :, ::-1].astype(np.float32) / 255.0   # BGR → RGB
            colors = rgb_f[valid]
            pts = np.concatenate([pts, colors], axis=-1)

        # Random sub-sampling (matches ExportPointCloud pc_util.random_sampling)
        if num_sample > 0 and len(pts) > num_sample:
            idx = np.random.choice(len(pts), num_sample, replace=False)
            pts = pts[idx]

        return pts


# ---------------------------------------------------------------------------
# Convenience: discover all synthetic static captures
# ---------------------------------------------------------------------------

def discover_thud_synthetic_scenes(
    thud_root: str,
    scene_type: str = "static",
) -> List[str]:
    """Return paths to all ``Capture_*`` directories under synthetic scenes.

    Parameters
    ----------
    thud_root : str
        Root of the THUD dataset, e.g. ``/data/THUD_Robot``
    scene_type : str
        ``"static"`` or ``"dynamic"`` (sub-directory name).
    """
    root = Path(thud_root) / "Synthetic_Scenes"
    if not root.exists():
        return []

    captures: List[str] = []
    for scene_dir in sorted(root.iterdir()):
        type_dir = scene_dir / scene_type
        if not type_dir.is_dir():
            continue
        for cap in sorted(type_dir.iterdir()):
            if cap.is_dir() and cap.name.startswith("Capture_"):
                if (cap / "RGB").exists() and (cap / "Label").exists():
                    captures.append(str(cap))
    return captures


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/yehia/rizo/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_4"
    )

    print(f"Testing THUDSyntheticLoader with: {test_path}")
    print("=" * 70)

    loader = THUDSyntheticLoader(
        test_path,
        load_rgb=True,
        load_depth=True,
        verbose=True,
    )

    print(f"Frame count : {loader.get_frame_count()}")
    indices = loader.get_frame_indices()
    print(f"Frame range : {indices[0]} – {indices[-1]}")

    # Load a few sample frames
    for idx in indices[:3]:
        fd = loader.get_frame_data(idx)
        print(f"\qq-- Frame {fd.frame_idx} ---")
        print(f"  RGB shape     : {fd.rgb.shape if fd.rgb is not None else None}")
        print(f"  Depth shape   : {fd.depth.shape if fd.depth is not None else None}")
        print(f"  Cam transform : {'present' if fd.cam_transform_4x4 is not None else 'None'}")
        print(f"  Cam intrinsic : {'present' if fd.camera_intrinsic is not None else 'None'}")
        print(f"  Seg shape     : {fd.seg.shape if fd.seg is not None else None}")
        print(f"  GT objects    : {len(fd.gt_objects)}")

        for obj in fd.gt_objects[:5]:
            print(
                f"    track_id={obj.track_id:>4d}  "
                f"class={obj.class_name:<20s}  "
                f"bbox2d={obj.bbox2d_xyxy}  "
                f"center={obj.box_3d_center_xyz}  "
                f"size={obj.box_3d_size_xyz}  "
                f"mask_pixels={int(obj.mask.sum()) if obj.mask.size > 0 else 0}"
            )

        if fd.gt_objects:
            obj = fd.gt_objects[0]
            print(f"\n  First object detail:")
            print(f"    transform_4x4:\n{obj.box_3d_transform_4x4}")
            print(f"    AABB: {obj.box_3d_aabb_xyzmin_xyzmax}")
            print(f"    rotation_xyzw: {obj.box_3d_rotation_xyzw}")
