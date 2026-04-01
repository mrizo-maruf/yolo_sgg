"""ScanNet++ dataset loader.

ScanNet++ directory layout::

    scene/
        images/frame_000000.jpg  ...       (stride-10 numbering)
        gt_depth/frame_000000.png  ...     (uint16, metres = px / 1000)
        masks/frame_000000.jpg.npy  ...    (int16, 0=bg, nonzero=track_id)
        bbox/bboxes000000_info.json  ...   (sequential 0-based index)
        traj.txt                           (one 4×4 per line, 16 floats)

        pi3_traj.txt                       (one 4×4 per line, 16 floats)
        pi3_depth/frame_0000.png           (sequential 0-based index)

Can also be run standalone to visualize GT labels::

    python -m data_loaders.scanetpp --scene /path/to/scene                  # all frames
    python -m data_loaders.scanetpp --scene /path/to/scene --frame 10       # single frame
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Ensure project root is importable (needed when run as __main__)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider
from depth_providers.gt_depth import _load_poses_txt
from metrics.tracking_metrics import GTInstance

try:
    from .base import DatasetLoader
except ImportError:
    from data_loaders.base import DatasetLoader

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
        # fnum = self._frame_number(frame_idx)
        return self._depth_provider.get_depth(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:

        # Prefer depth-provider pose (e.g. Pi3 predicted)
        pred_pose = self._depth_provider.get_pose(frame_idx)
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
        # fnum = self._frame_number(frame_idx)
        T_w_c = self.get_pose(frame_idx)
        intrinsics = self.get_intrinsics()
        return self._depth_provider.get_masked_pcds(
            frame_idx, masks, T_w_c, intrinsics,
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


# ---------------------------------------------------------------------------
# Visualization helpers (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

_BBOX_EDGES = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
    [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
]


def _id_to_color(track_id: int) -> Tuple[float, float, float]:
    """Deterministic color from track_id using golden-ratio hashing."""
    import colorsys
    hue = (track_id * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)


def _aabb_corners(aabb) -> np.ndarray:
    """8 corners from [xmin, ymin, zmin, xmax, ymax, zmax]."""
    xn, yn, zn, xx, yx, zx = aabb
    return np.array([
        [xn, yn, zn], [xn, yn, zx], [xn, yx, zn], [xn, yx, zx],
        [xx, yn, zn], [xx, yn, zx], [xx, yx, zn], [xx, yx, zx],
    ])


def _project_3d_bbox_to_2d(aabb, pose, K, img_shape):
    """Project 3D AABB corners to 2D.  Returns (pts_2d, valid_mask) or (None, None)."""
    corners_w = _aabb_corners(aabb)
    T_wc = np.linalg.inv(pose)
    ones = np.ones((8, 1))
    corners_cam = (T_wc @ np.hstack([corners_w, ones]).T).T[:, :3]
    valid = corners_cam[:, 2] > 0.01
    if valid.sum() < 2:
        return None, None
    pts_2d = (K @ corners_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d, valid


def _backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    return np.stack([X, Y, depth], axis=-1)


def _load_poses(traj_path: str) -> List[np.ndarray]:
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def _show_2d(rgb, mask, boxes, pose, K, frame_idx):
    """Show RGB + mask overlay + reprojected 3D bboxes (matplotlib)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection

    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(rgb)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    legend_handles = []
    id_to_col: Dict[int, Tuple[float, float, float]] = {}

    for oid in obj_ids:
        c = _id_to_color(int(oid))
        id_to_col[int(oid)] = c
        m = mask == oid
        overlay[m] = [c[0], c[1], c[2], 0.35]
        ys, xs = np.where(m)
        if len(xs) > 0:
            cx_, cy_ = int(xs.mean()), int(ys.mean())
            ax.text(cx_, cy_, f"#{oid}", color="white", fontsize=7,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=c, alpha=0.7, ec="none"))

    ax.imshow(overlay)

    for box in boxes:
        tid = box["track_id"]
        aabb = box["aabb_xyzmin_xyzmax"]
        c = id_to_col.get(tid, _id_to_color(tid))
        pts_2d, valid = _project_3d_bbox_to_2d(aabb, pose, K, rgb.shape)
        if pts_2d is None:
            continue
        segments = []
        for i0, i1 in _BBOX_EDGES:
            if valid[i0] and valid[i1]:
                x0, y0 = pts_2d[i0]
                x1, y1 = pts_2d[i1]
                if (min(x0, x1) > -w and max(x0, x1) < 2 * w and
                        min(y0, y1) > -h and max(y0, y1) < 2 * h):
                    segments.append([(x0, y0), (x1, y1)])
        if segments:
            lc = LineCollection(segments, colors=[c], linewidths=1.5, alpha=0.9)
            ax.add_collection(lc)
        front_pts = pts_2d[valid]
        if len(front_pts) > 0:
            mx, my = front_pts.mean(axis=0)
            if 0 <= mx < w and 0 <= my < h:
                ax.text(mx, my - 8, f"T{tid}", color=c, fontsize=6,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))
        legend_handles.append(mpatches.Patch(color=c, label=f"Track {tid}"))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f"Frame {frame_idx} — RGB + Masks + Reprojected 3D BBoxes   (close for 3D view)")
    ax.axis("off")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                  ncol=2, framealpha=0.7)
    plt.tight_layout()
    plt.show()


def _show_3d(depth_path, rgb_path, pose, K, boxes, stride=4):
    """Show GT point cloud + 3D bboxes + camera frustum (Open3D)."""
    import open3d as o3d

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pts_cam = _backproject(depth, K)
    pts_sub = pts_cam[::stride, ::stride]
    rgb_sub = rgb[::stride, ::stride]

    pts_flat = pts_sub.reshape(-1, 3)
    cols_flat = rgb_sub.reshape(-1, 3)
    valid = pts_flat[:, 2] > 0
    pts_flat, cols_flat = pts_flat[valid], cols_flat[valid]

    ones = np.ones((pts_flat.shape[0], 1))
    pts_world = (pose @ np.hstack([pts_flat, ones]).T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(cols_flat)

    geometries = [pcd]
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0]))

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    cam_frame.transform(pose)
    geometries.append(cam_frame)

    # camera frustum
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    iw, ih = int(cx * 2), int(cy * 2)
    sc = 0.12
    corners_cam = np.array([
        [0, 0, 0],
        [(0 - cx) / fx * sc, (0 - cy) / fy * sc, sc],
        [(iw - cx) / fx * sc, (0 - cy) / fy * sc, sc],
        [(iw - cx) / fx * sc, (ih - cy) / fy * sc, sc],
        [(0 - cx) / fx * sc, (ih - cy) / fy * sc, sc],
    ])
    ones_c = np.ones((5, 1))
    corners_w = (pose @ np.hstack([corners_cam, ones_c]).T).T[:, :3]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_w)
    frustum.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4],
                                                 [1, 2], [2, 3], [3, 4], [4, 1]])
    frustum.paint_uniform_color([0.2, 0.2, 0.8])
    geometries.append(frustum)

    for box in boxes:
        tid = box["track_id"]
        aabb = box["aabb_xyzmin_xyzmax"]
        c = _id_to_color(tid)
        corners = _aabb_corners(aabb)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(_BBOX_EDGES)
        ls.paint_uniform_color(list(c))
        geometries.append(ls)

    print("  Open3D: GT depth PCD (RGB) + 3D bboxes + camera frustum")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT 3D View — Point Cloud + BBoxes",
        width=1280, height=720,
    )


# ---------------------------------------------------------------------------
# CLI: Visualize GT labels
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize ScanNet++ GT: RGB + masks + 3D bboxes"
    )
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to scene directory")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index (0-based). Omit to iterate all frames.")
    parser.add_argument("--stride", type=int, default=4,
                        help="Pixel stride for 3D point-cloud subsampling")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    args = parser.parse_args()

    scene = Path(args.scene)
    K = np.array([[args.fx, 0, args.cx],
                   [0, args.fy, args.cy],
                   [0, 0, 1]])

    poses = _load_poses(str(scene / "traj.txt"))

    image_files = sorted((scene / "images").glob("frame_*.jpg"))
    depth_files = sorted((scene / "gt_depth").glob("frame_*.png"))
    mask_files = sorted((scene / "masks").glob("frame_*.jpg.npy"))
    bbox_files = sorted((scene / "bbox").glob("bboxes*_info.json"))

    n = min(len(image_files), len(depth_files),
            len(mask_files), len(bbox_files), len(poses))
    print(f"Scene: {scene.name}  ({n} frames)")

    if args.frame is not None:
        frames = [args.frame]
    else:
        frames = list(range(n))

    for i in frames:
        if not (0 <= i < n):
            print(f"Frame {i} out of range [0, {n}), skipping.")
            continue

        print(f"\n── Frame {i}/{n-1}: {image_files[i].name}")

        rgb = cv2.cvtColor(cv2.imread(str(image_files[i])), cv2.COLOR_BGR2RGB)
        mask = np.load(str(mask_files[i]))
        pose = poses[i]

        with open(bbox_files[i]) as f:
            boxes = json.load(f)["bboxes"]["bbox_3d"]["boxes"]

        obj_ids = np.unique(mask)
        print(f"   Objects in mask: {len(obj_ids[obj_ids != 0])},  Boxes: {len(boxes)}")

        _show_2d(rgb, mask, boxes, pose, K, i)
        _show_3d(depth_files[i], image_files[i], pose, K, boxes, stride=args.stride)

    print("\nDone.")
