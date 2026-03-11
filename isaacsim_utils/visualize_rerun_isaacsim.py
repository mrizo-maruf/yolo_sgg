"""
Visualize Isaac Sim scenes with Rerun.

Logs, per frame:
- Camera pose + camera frame axes + trajectory
- RGB image
- Semantic segmentation image
- RGB-colored point cloud from depth
- 3D GT bounding boxes (from IsaacSimSceneLoader)

Example:
    python isaacsim_utils/visualize_rerun_isaacsim.py \
        /home/yehia/rizo/IsaacSim_Dataset/cabinet_complex
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import List, Optional, Sequence, Set

import cv2
import numpy as np
try:
    import rerun as rr
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'rerun' package is required. Install it with: pip install rerun-sdk"
    ) from exc

try:
    from isaacsim_utils.isaac_sim_loader import IsaacSimSceneLoader
except ImportError:
    from isaac_sim_loader import IsaacSimSceneLoader


DEFAULT_SKIP_LABELS: Set[str] = {
    "wall",
    "floor",
    "ground",
    "ceiling",
    "background",
}

_BOX_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


def _stable_color_u8(track_id: int) -> np.ndarray:
    hue = (int(track_id) * 0.618033988749895) % 1.0
    sat = 0.80
    val = 0.95
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return (255.0 * np.array(rgb, dtype=np.float32)).astype(np.uint8)


def _depth_to_meters(
    depth_raw: np.ndarray,
    png_max_value: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    if max_depth <= min_depth:
        raise ValueError(f"max_depth must be > min_depth, got {max_depth} <= {min_depth}")

    scale = (max_depth - min_depth) / float(png_max_value)
    depth_m = depth_raw.astype(np.float32) * scale + min_depth
    depth_m[(depth_m < min_depth) | (depth_m > max_depth)] = 0.0
    return depth_m


def _depth_rgb_to_pointcloud(
    depth_m: np.ndarray,
    rgb_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float,
    max_depth: float,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))

    depth_s = depth_m[::stride, ::stride]
    rgb_s = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::stride, ::stride]

    h_s, w_s = depth_s.shape
    vv, uu = np.meshgrid(
        np.arange(h_s, dtype=np.float32) * stride,
        np.arange(w_s, dtype=np.float32) * stride,
        indexing="ij",
    )

    valid = (depth_s > min_depth) & (depth_s < max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth_s[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    points = np.stack((x, y, z), axis=1).astype(np.float32)
    colors = rgb_s[valid].astype(np.uint8)

    if max_points > 0 and points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / float(max_points)))
        points = points[::step][:max_points]
        colors = colors[::step][:max_points]

    return points, colors


def _transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    rot = transform_4x4[:3, :3]
    trans = transform_4x4[:3, 3]
    return (points @ rot.T) + trans


def _rgb_to_rgba(colors_rgb: np.ndarray, alpha: float) -> np.ndarray:
    if colors_rgb.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    a = int(np.clip(round(float(alpha) * 255.0), 0, 255))
    alpha_col = np.full((colors_rgb.shape[0], 1), a, dtype=np.uint8)
    return np.concatenate([colors_rgb.astype(np.uint8), alpha_col], axis=1)


def _append_history(
    history_points: np.ndarray,
    history_colors: np.ndarray,
    new_points: np.ndarray,
    new_colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if history_points.size == 0:
        combined_points = new_points.astype(np.float32)
        combined_colors = new_colors.astype(np.uint8)
    else:
        combined_points = np.concatenate([history_points, new_points.astype(np.float32)], axis=0)
        combined_colors = np.concatenate([history_colors, new_colors.astype(np.uint8)], axis=0)

    if max_points > 0 and combined_points.shape[0] > max_points:
        step = int(np.ceil(combined_points.shape[0] / float(max_points)))
        combined_points = combined_points[::step][:max_points]
        combined_colors = combined_colors[::step][:max_points]

    return combined_points, combined_colors


def _build_axis_remap_matrix(
    swap_yz: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> np.ndarray:
    # p' = A @ p for column vectors.
    A = np.eye(3, dtype=np.float32)
    if swap_yz:
        A = A[[0, 2, 1], :]
    if flip_x:
        A[0, :] *= -1.0
    if flip_y:
        A[1, :] *= -1.0
    if flip_z:
        A[2, :] *= -1.0
    return A


def _apply_axis_remap_points(points: np.ndarray, axis_remap: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    return (points @ axis_remap.T).astype(np.float32)


def _apply_axis_remap_transform(transform_4x4: np.ndarray, axis_remap: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = axis_remap @ transform_4x4[:3, :3]
    out[:3, 3] = axis_remap @ transform_4x4[:3, 3]
    return out


def _aabb_to_corners(aabb_xyzmin_xyzmax: Sequence[float]) -> np.ndarray:
    xmin, ymin, zmin, xmax, ymax, zmax = map(float, aabb_xyzmin_xyzmax)
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float32,
    )


def _apply_transform(corners: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
    out = (transform_4x4 @ corners_h.T).T
    return out[:, :3].astype(np.float32)


def _box_corners_world(
    aabb_xyzmin_xyzmax: Sequence[float],
    box_transform_4x4: np.ndarray,
    bbox_frame: str,
    cam_transform_4x4: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    corners = _aabb_to_corners(aabb_xyzmin_xyzmax)

    if bbox_frame == "world":
        return corners
    if bbox_frame == "use_transform":
        return _apply_transform(corners, box_transform_4x4)
    if bbox_frame == "camera":
        if cam_transform_4x4 is None:
            return None
        return _apply_transform(corners, cam_transform_4x4)

    raise ValueError(f"Unsupported bbox_frame: {bbox_frame}")


def _select_frames(
    frame_indices: Sequence[int],
    frame_start: Optional[int],
    frame_end: Optional[int],
    frame_step: int,
) -> List[int]:
    out = []
    for frame_idx in frame_indices:
        if frame_start is not None and frame_idx < frame_start:
            continue
        if frame_end is not None and frame_idx > frame_end:
            continue
        out.append(frame_idx)
    return out[:: max(1, frame_step)]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun visualization for Isaac Sim scenes.")
    parser.add_argument("scene_dir", type=Path, help="Scene directory with rgb/depth/bbox/seg/traj.txt.")

    parser.add_argument("--recording-id", type=str, default="isaacsim_rerun")
    parser.add_argument("--headless", action="store_true", help="Use an existing rerun --serve server.")
    parser.add_argument(
        "--connect",
        type=str,
        default=None,
        help="Rerun gRPC address for headless mode (example: 127.0.0.1:9876).",
    )

    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fx", type=float, default=800.0)
    parser.add_argument("--fy", type=float, default=800.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--image-plane-distance", type=float, default=0.25)

    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--png-max-value", type=float, default=65535.0)
    parser.add_argument("--pc-stride", type=int, default=2)
    parser.add_argument("--pc-max-points", type=int, default=150_000)
    parser.add_argument(
        "--pc-point-radius",
        "--pc-radius",
        dest="pc_point_radius",
        type=float,
        default=0.01,
        help="Point size (radius, in meters) used in rerun point-cloud visualization.",
    )
    parser.add_argument(
        "--pc-history-alpha",
        type=float,
        default=0.20,
        help="Transparency alpha for accumulated previous-frame points (0..1).",
    )
    parser.add_argument(
        "--pc-history-max-points",
        type=int,
        default=500_000,
        help="Max number of accumulated history points kept in memory/logged.",
    )
    parser.add_argument(
        "--disable-pc-history",
        action="store_true",
        help="Disable accumulated previous-frame point-cloud visualization.",
    )
    parser.add_argument(
        "--pointcloud-frame",
        choices=("world", "camera"),
        default="world",
        help="Whether to log the point cloud in world or camera frame.",
    )

    parser.add_argument(
        "--bbox-frame",
        choices=("world", "camera", "use_transform"),
        default="world",
        help="Interpretation of bbox coordinates from GT.",
    )
    parser.add_argument("--max-box-edge", type=float, default=20.0)

    parser.add_argument(
        "--skip-label",
        action="append",
        default=None,
        help="Label to skip. Repeat multiple times. Defaults to structural labels.",
    )
    parser.add_argument(
        "--allow-incomplete-ids",
        action="store_true",
        help="Allow objects with missing 2D/3D/seg IDs.",
    )
    parser.add_argument("--axis-length", type=float, default=0.25, help="Camera frame axis length in meters.")
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap Y and Z axes before logging to rerun.",
    )
    parser.add_argument("--flip-x", action="store_true", help="Flip X axis before logging to rerun.")
    parser.add_argument("--flip-y", action="store_true", help="Flip Y axis before logging to rerun.")
    parser.add_argument("--flip-z", action="store_true", help="Flip Z axis before logging to rerun.")
    parser.add_argument(
        "--isaac-axis-fix",
        action="store_true",
        help="Convenience axis remap for common IsaacSim->Rerun mismatch: swap Y/Z and flip Y.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    skip_labels = set(DEFAULT_SKIP_LABELS if args.skip_label is None else [s.lower() for s in args.skip_label])

    loader = IsaacSimSceneLoader(
        scene_dir=str(args.scene_dir),
        load_rgb=True,
        load_depth=True,
        skip_labels=skip_labels,
        require_all_ids=not args.allow_incomplete_ids,
    )
    if not loader.frame_indices:
        raise RuntimeError(f"No frames found in {args.scene_dir}")

    frame_indices = _select_frames(
        frame_indices=loader.frame_indices,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
    )
    if not frame_indices:
        raise RuntimeError("No frames selected after applying frame range filters.")

    first_frame = loader.get_frame_data(frame_indices[0])
    if first_frame.rgb is not None:
        img_h, img_w = first_frame.rgb.shape[:2]
    else:
        img_h, img_w = int(args.image_height), int(args.image_width)

    cx = float(args.cx) if args.cx is not None else (img_w / 2.0)
    cy = float(args.cy) if args.cy is not None else (img_h / 2.0)

    swap_yz = bool(args.swap_yz or args.isaac_axis_fix)
    flip_x = bool(args.flip_x)
    flip_y = bool(args.flip_y or args.isaac_axis_fix)
    flip_z = bool(args.flip_z)
    axis_remap = _build_axis_remap_matrix(
        swap_yz=swap_yz,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_z=flip_z,
    )

    if args.headless:
        rr.init(args.recording_id, spawn=False)
        if args.connect is not None:
            rr.connect_grpc(args.connect)
        else:
            rr.connect_grpc()
    else:
        rr.init(args.recording_id, spawn=True)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            resolution=[img_w, img_h],
            principal_point=[cx, cy],
            focal_length=[args.fx, args.fy],
            image_plane_distance=args.image_plane_distance,
        ),
        static=True,
    )

    rr.log(
        "world/camera/frame",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * float(args.axis_length),
            colors=np.array(
                [
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                ],
                dtype=np.uint8,
            ),
        ),
        static=True,
    )

    camera_positions: List[np.ndarray] = []
    history_points = np.zeros((0, 3), dtype=np.float32)
    history_colors = np.zeros((0, 3), dtype=np.uint8)
    warned_no_world_pose = False
    for frame_idx in frame_indices:
        frame_data = loader.get_frame_data(frame_idx)
        rr.set_time(timeline="frame", sequence=int(frame_idx))

        cam_t = frame_data.cam_transform_4x4
        cam_t_rerun = _apply_axis_remap_transform(cam_t, axis_remap) if cam_t is not None else None
        if cam_t_rerun is not None:
            rr.log(
                "world/camera",
                rr.Transform3D(
                    mat3x3=cam_t_rerun[:3, :3],
                    translation=cam_t_rerun[:3, 3],
                ),
            )
            camera_positions.append(cam_t_rerun[:3, 3].astype(np.float32))

            rr.log(
                "world/camera_trajectory",
                rr.LineStrips3D(
                    strips=[np.asarray(camera_positions, dtype=np.float32)],
                    colors=np.array([[255, 80, 80]], dtype=np.uint8),
                ),
            )

        if frame_data.rgb is not None:
            rr.log(
                "world/camera/image/rgb",
                rr.Image(cv2.cvtColor(frame_data.rgb, cv2.COLOR_BGR2RGB), color_model=rr.ColorModel.RGB),
            )

        if frame_data.seg is not None:
            rr.log(
                "world/camera/image/segmentation",
                rr.Image(cv2.cvtColor(frame_data.seg, cv2.COLOR_BGR2RGB), color_model=rr.ColorModel.RGB),
            )

        if frame_data.depth is not None and frame_data.rgb is not None:
            depth_m = _depth_to_meters(
                depth_raw=frame_data.depth,
                png_max_value=args.png_max_value,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )

            rr.log(
                "world/camera/image/depth",
                rr.DepthImage(depth_m, meter=1.0, depth_range=[args.min_depth, args.max_depth]),
            )

            points_cam, colors = _depth_rgb_to_pointcloud(
                depth_m=depth_m,
                rgb_bgr=frame_data.rgb,
                fx=args.fx,
                fy=args.fy,
                cx=cx,
                cy=cy,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                stride=args.pc_stride,
                max_points=args.pc_max_points,
            )

            log_world_points = args.pointcloud_frame == "world" and cam_t is not None
            if args.pointcloud_frame == "world" and cam_t is None and not warned_no_world_pose:
                print(
                    "[WARN] pointcloud-frame=world requested but camera pose is missing for some frames. "
                    "Falling back to camera frame for those frames."
                )
                warned_no_world_pose = True

            if log_world_points:
                points = _transform_points(points_cam, cam_t)
                points = _apply_axis_remap_points(points, axis_remap)
                current_pcd_path = "world/pointcloud/current"
            else:
                points = points_cam
                current_pcd_path = "world/camera/pointcloud/current"

            if (
                not args.disable_pc_history
                and history_points.shape[0] > 0
                and log_world_points
            ):
                rr.log(
                    "world/pointcloud/history",
                    rr.Points3D(
                        history_points,
                        colors=_rgb_to_rgba(history_colors, args.pc_history_alpha),
                        radii=np.full(history_points.shape[0], float(args.pc_point_radius), dtype=np.float32),
                    ),
                )
            elif hasattr(rr, "Clear"):
                rr.log("world/pointcloud/history", rr.Clear(recursive=False))

            if points.shape[0] > 0:
                rr.log(
                    current_pcd_path,
                    rr.Points3D(
                        points,
                        colors=colors,
                        radii=np.full(points.shape[0], float(args.pc_point_radius), dtype=np.float32),
                    ),
                )
                if hasattr(rr, "Clear"):
                    if current_pcd_path == "world/pointcloud/current":
                        rr.log("world/camera/pointcloud/current", rr.Clear(recursive=False))
                    else:
                        rr.log("world/pointcloud/current", rr.Clear(recursive=False))

                if not args.disable_pc_history and log_world_points:
                    history_points, history_colors = _append_history(
                        history_points=history_points,
                        history_colors=history_colors,
                        new_points=points,
                        new_colors=colors,
                        max_points=args.pc_history_max_points,
                    )
            elif hasattr(rr, "Clear"):
                rr.log(current_pcd_path, rr.Clear(recursive=False))

        strips = []
        strip_colors = []
        box_centers = []
        box_center_colors = []
        box_labels = []

        for obj in frame_data.gt_objects:
            corners_world = _box_corners_world(
                aabb_xyzmin_xyzmax=obj.box_3d_aabb_xyzmin_xyzmax,
                box_transform_4x4=obj.box_3d_transform_4x4,
                bbox_frame=args.bbox_frame,
                cam_transform_4x4=cam_t,
            )
            if corners_world is None:
                continue
            corners_world = _apply_axis_remap_points(corners_world, axis_remap)

            size_xyz = np.max(corners_world, axis=0) - np.min(corners_world, axis=0)
            if float(np.max(size_xyz)) > float(args.max_box_edge):
                continue

            color = _stable_color_u8(obj.track_id)
            edge_points = corners_world[_BOX_EDGES]
            strips.extend(edge_points)
            strip_colors.extend([color] * edge_points.shape[0])

            center = np.mean(corners_world, axis=0)
            box_centers.append(center.astype(np.float32))
            box_center_colors.append(color)
            box_labels.append(f"{obj.class_name} | T:{obj.track_id}")

        if strips:
            rr.log(
                "world/gt_boxes",
                rr.LineStrips3D(
                    strips=np.asarray(strips, dtype=np.float32),
                    colors=np.asarray(strip_colors, dtype=np.uint8),
                ),
            )
        elif hasattr(rr, "Clear"):
            rr.log("world/gt_boxes", rr.Clear(recursive=False))

        if box_centers:
            rr.log(
                "world/gt_boxes/labels",
                rr.Points3D(
                    np.asarray(box_centers, dtype=np.float32),
                    colors=np.asarray(box_center_colors, dtype=np.uint8),
                    labels=box_labels,
                    radii=np.full(len(box_centers), 0.03, dtype=np.float32),
                ),
            )
        elif hasattr(rr, "Clear"):
            rr.log("world/gt_boxes/labels", rr.Clear(recursive=False))

    print(f"Logged {len(frame_indices)} frames to rerun recording '{args.recording_id}'.")


if __name__ == "__main__":
    main()
