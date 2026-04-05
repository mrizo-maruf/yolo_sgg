"""Visualize THUD synthetic frames using data_loaders.thud_synthetic.

Mirrors the visualization style of visualize_thud_synthetic.py but driven
by the new data_loaders.thud_synthetic.THUDSyntheticLoader API.

Shows:
1) RGB frame with 2D bounding boxes + class labels (side by side: plain/annotated)
2) Open3D point cloud + axis-aligned 3D bounding boxes  (--show-3d)

Example:
    python thud_utils/visualize_thud_synthetic_new.py \
        --scene /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 \
        --n-frames 5 --show-3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from core.types import CameraIntrinsics
from data_loaders.thud_synthetic import THUDSyntheticLoader
from depth_providers.gt_depth import THUDSyntheticDepthProvider
from metrics.tracking_metrics import GTInstance

try:
    import open3d as o3d

    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


# ---------------------------------------------------------------------------
# Depth → point cloud via THUDSyntheticDepthProvider.unproject
# ---------------------------------------------------------------------------

def _depth_to_pointcloud(
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    provider: THUDSyntheticDepthProvider,
    rgb_rgb: Optional[np.ndarray] = None,
    num_sample: int = 50000,
) -> np.ndarray:
    """Unproject a THUD raw uint16 depth map into a coloured point cloud.

    Delegates the THUD-specific math to
    :meth:`THUDSyntheticDepthProvider.unproject`, yielding points in the
    **(x-right, y-forward, z-up)** frame used by the annotation pipeline.

    Parameters
    ----------
    depth_raw : np.ndarray
        H × W raw uint16 depth (*not* converted to metres).
    intr : CameraIntrinsics
        Pinhole intrinsics.
    provider : THUDSyntheticDepthProvider
        Used for its ``unproject`` method.
    rgb_rgb : np.ndarray | None
        H × W × 3 RGB image.  If provided the returned array has 6 columns
        (XYZRGB 0-1) instead of 3 (XYZ).
    num_sample : int
        Random sub-sample target (0 = keep all valid points).

    Returns
    -------
    np.ndarray
        (N, 3) or (N, 6) float32 — XYZ [+ RGB 0-1].
    """
    depth_f = depth_raw.astype(np.float32)
    valid = depth_f > 0
    vs, us = np.where(valid)
    depths = depth_f[valid]

    pts = provider.unproject(us, vs, depths, intr)  # (N, 3) float32

    if rgb_rgb is not None:
        colors = rgb_rgb.astype(np.float32)[valid] / 255.0
        pts = np.concatenate([pts, colors], axis=-1)

    if num_sample > 0 and len(pts) > num_sample:
        idx = np.random.choice(len(pts), num_sample, replace=False)
        pts = pts[idx]

    return pts


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    if h < 1 / 6:
        r, g, b = c, x, 0.0
    elif h < 2 / 6:
        r, g, b = x, c, 0.0
    elif h < 3 / 6:
        r, g, b = 0.0, c, x
    elif h < 4 / 6:
        r, g, b = 0.0, x, c
    elif h < 5 / 6:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)


def _color_for_track(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.618033988749895) % 1.0
    return _hsv_to_bgr(hue, 0.85, 0.95)


def _color_for_track_float(track_id: int) -> Tuple[float, float, float]:
    b, g, r = _color_for_track(track_id)
    return r / 255.0, g / 255.0, b / 255.0


def _rgb_to_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 2-D visualization
# ---------------------------------------------------------------------------

def _draw_2d_boxes_and_labels(
    image_bgr: np.ndarray,
    instances: Iterable[GTInstance],
) -> np.ndarray:
    """Draw bounding boxes and labels onto *image_bgr* (BGR, in-place copy)."""
    vis = image_bgr.copy()
    for inst in instances:
        if inst.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox_xyxy]
        color = _color_for_track(int(inst.track_id))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{inst.class_name} | id:{inst.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(y1 - th - 6, 0)
        cv2.rectangle(vis, (x1, top), (x1 + tw + 6, top + th + 6), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def visualize_2d_panels(
    frame_idx: int,
    rgb: np.ndarray,
    instances: List[GTInstance],
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Show (or save) a side-by-side panel: plain RGB | RGB + 2-D boxes."""
    if rgb is None:
        print(f"[WARN] Frame {frame_idx}: RGB missing, skipping 2D panels")
        return

    # rgb is in RGB format from the new loader; convert to BGR for cv2
    bgr = _rgb_to_bgr(rgb)
    annotated = _draw_2d_boxes_and_labels(bgr, instances)

    h = bgr.shape[0]
    panel = np.hstack([bgr, annotated])
    title = f"Frame {frame_idx} | Left: RGB | Right: RGB + 2D boxes/classes"
    bar = np.zeros((32, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.vstack([bar, panel])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"frame_{frame_idx:06d}_2d.png"
        cv2.imwrite(str(out), panel)
        print(f"[2D] Saved {out}")

    if show:
        win_name = f"THUD 2D | frame {frame_idx}"
        cv2.imshow(win_name, panel)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win_name)
        if key == 27:
            raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# 3-D visualization helpers
# ---------------------------------------------------------------------------

def _aabb_lineset(
    bbox_xyzxyz: Tuple[float, ...],
    color: Tuple[float, float, float],
):
    """Build an Open3D LineSet wireframe for an axis-aligned bounding box."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox_xyzxyz
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


def visualize_3d(
    frame_idx: int,
    depth_raw: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    provider: THUDSyntheticDepthProvider,
    instances: List[GTInstance],
    pose: Optional[np.ndarray] = None,
    point_sample: int = 50000,
    rgb: Optional[np.ndarray] = None,
) -> None:
    """Display Open3D window with point cloud and AABB wireframes."""
    if not HAS_OPEN3D:
        print("[WARN] open3d is not installed; skipping 3D view")
        return
    if depth_raw is None:
        print(f"[WARN] Frame {frame_idx}: depth missing, skipping 3D view")
        return

    pts = _depth_to_pointcloud(depth_raw, intrinsics, provider, rgb_rgb=rgb, num_sample=point_sample)

    geoms: List = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    if pts.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(pts[:, 3:6], 0.0, 1.0).astype(np.float64))
    geoms.append(pcd)

    box_count = 0
    for inst in instances:
        if inst.bbox_xyzxyz is None:
            continue
        color = _color_for_track_float(int(inst.track_id))
        geoms.append(_aabb_lineset(inst.bbox_xyzxyz, color))
        box_count += 1

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    print(f"[3D] Frame {frame_idx}: points={len(pts)}, boxes={box_count}")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"THUD 3D | frame {frame_idx}",
        width=1400,
        height=900,
    )


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def _select_frames(
    total: int,
    start: Optional[int],
    step: int,
    n_frames: int,
) -> List[int]:
    """Return a list of 0-based frame indices to process."""
    start_idx = start if start is not None else 0
    start_idx = max(0, min(start_idx, total - 1))
    indices = list(range(start_idx, total, max(step, 1)))
    if n_frames > 0:
        indices = indices[:n_frames]
    return indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize THUD synthetic data (new loader) with 2D and 3D views"
    )
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to THUD synthetic Capture_* folder")
    parser.add_argument("--start", type=int, default=None,
                        help="First 0-based frame index to visualize")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame stride")
    parser.add_argument("--n-frames", type=int, default=100,
                        help="Number of frames to process (0 = all)")
    parser.add_argument("--save-2d-dir", type=str, default=None,
                        help="Save 2D rendered panels to this directory")
    parser.add_argument("--no-2d", action="store_true",
                        help="Disable 2D visualization")
    parser.add_argument("--show-3d", action="store_true",
                        help="Enable 3D point cloud + AABB boxes")
    parser.add_argument("--pcd-sample", type=int, default=50000,
                        help="Point cloud sample size (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Path(args.scene)
    if not scene.exists():
        raise FileNotFoundError(f"Scene path does not exist: {scene}")

    depth_provider = THUDSyntheticDepthProvider(depth_dir=str(scene / "Depth"))

    loader = THUDSyntheticLoader(
        scene_dir=str(scene),
        depth_provider=depth_provider,
    )

    total = loader.get_num_frames()
    frames = _select_frames(total, args.start, args.step, args.n_frames)
    print(f"Scene: {scene.name}  |  total frames: {total}  |  selected: {len(frames)}")
    if not frames:
        return

    save_dir = Path(args.save_2d_dir) if args.save_2d_dir else None
    show_2d_windows = save_dir is None and not args.no_2d
    intrinsics = loader.get_intrinsics()

    try:
        for frame_idx in frames:
            rgb, _ = loader.get_rgb(frame_idx)
            instances = loader.get_gt_instances(frame_idx) or []
            print(f"\nFrame {frame_idx}: objects={len(instances)}")

            if not args.no_2d:
                visualize_2d_panels(frame_idx, rgb, instances,
                                    save_dir=save_dir, show=show_2d_windows)

            if args.show_3d:
                depth_raw = loader.get_depth(frame_idx)
                pose = loader.get_pose(frame_idx)
                visualize_3d(
                    frame_idx,
                    depth_raw,
                    intrinsics,
                    depth_provider,
                    instances,
                    pose=pose,
                    point_sample=args.pcd_sample,
                    rgb=rgb,
                )

    except KeyboardInterrupt:
        print("Visualization interrupted by user")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
