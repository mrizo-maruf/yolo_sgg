"""Visualize THUD synthetic frames with 2D and 3D overlays.

Shows:
1) RGB frame
2) Segmentation frame with 2D boxes + class labels
3) Open3D point cloud + 3D bounding boxes

Example:
    python thud_utils/visualize_thud_synthetic.py \
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

from thud_utils.thud_synthetic_loader import THUDSyntheticLoader

try:
    import open3d as o3d

    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


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


def _to_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _draw_2d_boxes_and_labels(image_bgr: np.ndarray, objects: Iterable) -> np.ndarray:
    vis = image_bgr.copy()
    for obj in objects:
        if obj.bbox2d_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in obj.bbox2d_xyxy]
        color = _color_for_track(int(obj.track_id))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.class_name} | id:{obj.track_id}"
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


def visualize_2d_panels(frame_data, save_dir: Optional[Path] = None, show: bool = True) -> None:
    if frame_data.rgb is None:
        print(f"[WARN] Frame {frame_data.frame_idx}: RGB missing, skipping 2D panels")
        return

    rgb = _to_bgr(frame_data.rgb)
    seg = _to_bgr(frame_data.seg)
    if seg is None:
        seg = np.zeros_like(rgb)

    seg_with_boxes = _draw_2d_boxes_and_labels(seg, frame_data.gt_objects)

    h = max(rgb.shape[0], seg_with_boxes.shape[0])
    if rgb.shape[0] != h:
        rgb = cv2.resize(rgb, (rgb.shape[1], h), interpolation=cv2.INTER_NEAREST)
    if seg_with_boxes.shape[0] != h:
        seg_with_boxes = cv2.resize(seg_with_boxes, (seg_with_boxes.shape[1], h), interpolation=cv2.INTER_NEAREST)

    panel = np.hstack([rgb, seg_with_boxes])
    title = f"Frame {frame_data.frame_idx} | Left: RGB | Right: Seg + 2D boxes/classes"
    bar = np.zeros((32, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.vstack([bar, panel])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"frame_{frame_data.frame_idx:06d}_2d.png"
        cv2.imwrite(str(out), panel)
        print(f"[2D] Saved {out}")

    if show:
        win_name = f"THUD 2D | frame {frame_data.frame_idx}"
        cv2.imshow(win_name, panel)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win_name)
        if key == 27:
            raise KeyboardInterrupt


def _quat_to_R(quat: Tuple[float, float, float, float]) -> np.ndarray:
    qx, qy, qz, qw = quat
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _obb_lineset(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    color: Tuple[float, float, float],
):
    obb = o3d.geometry.OrientedBoundingBox(
        center=np.asarray(center, dtype=np.float64),
        R=_quat_to_R(quat),
        extent=np.asarray(size, dtype=np.float64),
    )
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(color)
    return ls


def _obb_corners(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
) -> np.ndarray:
    R = _quat_to_R(quat)
    half = np.asarray(size, dtype=np.float64) / 2.0
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return (signs * half) @ R.T + np.asarray(center, dtype=np.float64)


def _lineset_from_corners(corners: np.ndarray, color: Tuple[float, float, float]):
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


def _transform_points(points_xyz: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    pts = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (T_4x4 @ pts_h.T).T
    return out[:, :3]


def _rotate_corners_about_y(
    corners_xyz: np.ndarray,
    degrees_clockwise: float,
) -> np.ndarray:
    pts = np.asarray(corners_xyz, dtype=np.float64)
    theta = -np.deg2rad(degrees_clockwise)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R_y = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return pts @ R_y.T


def _mean_nn_distance(points_xyz: np.ndarray, query_xyz: np.ndarray) -> Optional[float]:
    if points_xyz.size == 0 or query_xyz.size == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    d2: List[float] = []
    for q in np.asarray(query_xyz, dtype=np.float64):
        k, _, dist2 = tree.search_knn_vector_3d(q, 1)
        if k > 0:
            d2.append(float(dist2[0]))
    if not d2:
        return None
    return float(np.sqrt(np.mean(d2)))


def visualize_3d(
    frame_data,
    point_sample: int = 50000,
    align_mode: str = "raw",
    align_debug: bool = False,
) -> None:
    if not HAS_OPEN3D:
        print("[WARN] open3d is not installed; skipping 3D view")
        return
    if frame_data.depth is None or frame_data.camera_intrinsic is None:
        print(f"[WARN] Frame {frame_data.frame_idx}: missing depth/intrinsics for point cloud")
        return

    pts = THUDSyntheticLoader.depth_to_pointcloud(
        depth=frame_data.depth,
        camera_intrinsic=frame_data.camera_intrinsic,
        rgb=frame_data.rgb,
        num_sample=point_sample,
    )

    geoms: List = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    if pts.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(pts[:, 3:6], 0.0, 1.0).astype(np.float64))
    geoms.append(pcd)

    cam_T = None
    cam_T_inv = None
    if frame_data.cam_transform_4x4 is not None:
        cam_T = np.asarray(frame_data.cam_transform_4x4, dtype=np.float64)
        cam_T_inv = np.linalg.inv(cam_T)

    box_count = 0
    raw_centers: List[np.ndarray] = []
    w2c_centers: List[np.ndarray] = []
    for obj in frame_data.gt_objects:
        if obj.box_3d_size_xyz is None:
            continue

        track_color = _color_for_track_float(int(obj.track_id))
        raw_center = np.asarray(obj.box_3d_center_xyz, dtype=np.float64).reshape(1, 3)
        raw_corners = _obb_corners(
            obj.box_3d_center_xyz,
            obj.box_3d_size_xyz,
            obj.box_3d_rotation_xyzw,
        )

        if cam_T_inv is not None:
            w2c_center = _transform_points(raw_center, cam_T_inv)[0]
            w2c_corners = _transform_points(raw_corners, cam_T_inv)
            w2c_centers.append(w2c_center)
        else:
            w2c_corners = raw_corners

        raw_centers.append(raw_center[0])

        if align_debug:
            geoms.append(_lineset_from_corners(raw_corners, (0.95, 0.75, 0.10)))
            if cam_T_inv is not None:
                geoms.append(_lineset_from_corners(w2c_corners, (1.0, 0.0, 1.0)))
                green_corners = _rotate_corners_about_y(
                    w2c_corners,
                    degrees_clockwise=90.0,
                )
                geoms.append(_lineset_from_corners(green_corners, (0.0, 1.0, 0.0)))
        else:
            if align_mode == "w2c" and cam_T_inv is not None:
                geoms.append(_lineset_from_corners(w2c_corners, track_color))
            elif align_mode == "w2c" and cam_T_inv is None:
                geoms.append(_lineset_from_corners(raw_corners, track_color))
            else:
                geoms.append(_lineset_from_corners(raw_corners, track_color))

        box_count += 1

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    print(f"[3D] Frame {frame_data.frame_idx}: points={len(pts)}, boxes={box_count}")

    if align_debug:
        raw_arr = np.asarray(raw_centers, dtype=np.float64) if raw_centers else np.zeros((0, 3), dtype=np.float64)
        w2c_arr = np.asarray(w2c_centers, dtype=np.float64) if w2c_centers else np.zeros((0, 3), dtype=np.float64)
        pcd_xyz = pts[:, :3].astype(np.float64)

        raw_err = _mean_nn_distance(pcd_xyz, raw_arr)
        w2c_err = _mean_nn_distance(pcd_xyz, w2c_arr)

        raw_s = f"{raw_err:.4f} m" if raw_err is not None else "n/a"
        w2c_s = f"{w2c_err:.4f} m" if w2c_err is not None else "n/a"
        print(f"[ALIGN] mean NN(center->pcd): raw={raw_s}, w2c={w2c_s}")
        if raw_err is not None and w2c_err is not None:
            better = "w2c" if w2c_err < raw_err else "raw"
            print(f"[ALIGN] better mode for this frame: {better}")
        if cam_T_inv is None:
            print("[ALIGN] camera transform missing; cannot evaluate transformed boxes")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=(
            f"THUD 3D | frame {frame_data.frame_idx} | debug(raw=yellow,w2c=magenta)"
            if align_debug
            else f"THUD 3D | frame {frame_data.frame_idx} | mode={align_mode}"
        ),
        width=1400,
        height=900,
    )


def _select_frames(indices: List[int], start: Optional[int], step: int, n_frames: int) -> List[int]:
    if not indices:
        return []
    if start is None:
        start_idx = 0
    else:
        start_idx = 0
        for i, idx in enumerate(indices):
            if idx >= start:
                start_idx = i
                break
    picked = indices[start_idx::max(step, 1)]
    if n_frames > 0:
        picked = picked[:n_frames]
    return picked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize THUD synthetic data with 2D and 3D views")
    parser.add_argument("--scene", type=str, required=True, help="Path to THUD synthetic Capture_* folder")
    parser.add_argument("--start", type=int, default=None, help="First frame index to visualize")
    parser.add_argument("--step", type=int, default=1, help="Frame stride")
    parser.add_argument("--n-frames", type=int, default=100, help="Number of frames to process (0 = all)")
    parser.add_argument("--save-2d-dir", type=str, default=None, help="Save 2D rendered panels to this directory")
    parser.add_argument("--no-2d", action="store_true", help="Disable 2D visualization")
    parser.add_argument("--show-3d", action="store_true", help="Enable 3D point cloud + 3D boxes")
    parser.add_argument("--pcd-sample", type=int, default=50000, help="Point cloud sample size (0 = all)")
    parser.add_argument(
        "--align-mode",
        type=str,
        default="raw",
        choices=["raw", "w2c"],
        help="3D box frame hypothesis: raw annotations or transformed world->camera",
    )
    parser.add_argument(
        "--align-debug",
        action="store_true",
        help="Render both raw and world->camera boxes and print residual metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Path(args.scene)
    if not scene.exists():
        raise FileNotFoundError(f"Scene path does not exist: {scene}")

    loader = THUDSyntheticLoader(
        scene_dir=str(scene),
        load_rgb=True,
        load_depth=args.show_3d,
        verbose=False,
    )

    frames = _select_frames(loader.get_frame_indices(), args.start, args.step, args.n_frames)
    print(f"Selected {len(frames)} frames: {frames}")
    if not frames:
        return

    save_dir = Path(args.save_2d_dir) if args.save_2d_dir else None
    show_2d_windows = save_dir is None and not args.no_2d

    try:
        for frame_idx in frames:
            frame_data = loader.get_frame_data(frame_idx)
            print(f"\nFrame {frame_data.frame_idx}: objects={len(frame_data.gt_objects)}")

            if not args.no_2d:
                visualize_2d_panels(frame_data, save_dir=save_dir, show=show_2d_windows)

            if args.show_3d:
                visualize_3d(
                    frame_data,
                    point_sample=args.pcd_sample,
                    align_mode=args.align_mode,
                    align_debug=args.align_debug,
                )

    except KeyboardInterrupt:
        print("Visualization interrupted by user")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
