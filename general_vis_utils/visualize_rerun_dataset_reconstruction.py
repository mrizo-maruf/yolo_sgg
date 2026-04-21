#!/usr/bin/env python3
"""Standalone Rerun visualization for dataset RGB + depth + trajectory.

Supports IsaacSim and ScanNet++ scene folders without requiring the full
tracking pipeline. The script auto-detects the RGB folder (``rgb`` or
``images``), reads a user-specified trajectory file, reads depth PNGs from a
user-specified depth folder, reconstructs colored point clouds incrementally,
and logs the current camera frustum, camera trajectory, and world frame.

To keep the Rerun session bounded, only a sliding window of live points is
kept in memory/logged at each frame.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import rerun as rr
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'rerun' package is required. Install with: pip install rerun-sdk"
    ) from exc


_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from core.types import CameraIntrinsics
from rerun_utils import (
    _apply_axis_remap_points,
    _apply_axis_remap_transform,
    _build_axis_remap_matrix,
    _rigidize_camera_pose,
)


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class FrameItem:
    seq_idx: int
    frame_key: int
    rgb_path: Path
    depth_path: Path


@dataclass(slots=True)
class PoseEntry:
    pose: np.ndarray
    frame_key: Optional[int] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize incremental colored depth reconstruction in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["isaacsim", "scanetpp"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to the scene directory.",
    )
    parser.add_argument(
        "--traj_file",
        type=str,
        required=True,
        help="Trajectory filename inside the scene directory, or an absolute path.",
    )
    parser.add_argument(
        "--depth_folder",
        type=str,
        required=True,
        help="Depth folder name inside the scene directory, or an absolute path.",
    )
    parser.add_argument(
        "--pose_lookup",
        type=str,
        default="auto",
        choices=["auto", "index", "frame_number", "exact_key"],
        help="How to match poses to frames.",
    )
    parser.add_argument(
        "--max_frame_points",
        type=int,
        default=10000,
        help="Maximum number of points sampled from any single frame.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for visualization.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional cap on the number of visualized frames. 0 means all.",
    )
    parser.add_argument(
        "--point_radius",
        type=float,
        default=0.01,
        help="Rerun point radius in world units.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn the Rerun viewer immediately.",
    )

    # IsaacSim intrinsics
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--focal_length", type=float, default=50.0)
    parser.add_argument("--horizontal_aperture", type=float, default=80.0)
    parser.add_argument("--vertical_aperture", type=float, default=45.0)

    # ScanNet++ intrinsics
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)

    # Depth decoding
    parser.add_argument("--png_max_value", type=int, default=65535)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    return parser.parse_args()


def _resolve_scene_child(scene_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else scene_dir / path


def _discover_rgb_dir(scene_dir: Path) -> Path:
    for name in ("rgb", "images"):
        candidate = scene_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find an RGB directory in {scene_dir}. Expected one of: rgb, images"
    )


def _extract_numeric_key(path: Path) -> Optional[int]:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _sorted_image_files(directory: Path, *, png_only: bool = False) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        p for p in directory.iterdir()
        if p.is_file() and (p.suffix.lower() == ".png" if png_only else p.suffix.lower() in _IMAGE_EXTS)
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in {directory}")
    files.sort(key=lambda p: (_extract_numeric_key(p) is None, _extract_numeric_key(p) or 0, p.name))
    return files


def _build_depth_lookup(depth_files: Sequence[Path]) -> Tuple[dict[int, Path], List[Path]]:
    by_key: dict[int, Path] = {}
    for path in depth_files:
        key = _extract_numeric_key(path)
        if key is not None and key not in by_key:
            by_key[key] = path
    return by_key, list(depth_files)


def _build_frames(rgb_dir: Path, depth_dir: Path) -> List[FrameItem]:
    rgb_files = _sorted_image_files(rgb_dir)
    depth_files = _sorted_image_files(depth_dir, png_only=True)
    depth_by_key, depth_sorted = _build_depth_lookup(depth_files)

    frames: List[FrameItem] = []
    for seq_idx, rgb_path in enumerate(rgb_files):
        frame_key = _extract_numeric_key(rgb_path)
        if frame_key is None:
            frame_key = seq_idx

        depth_path = depth_by_key.get(frame_key)
        if depth_path is None and seq_idx < len(depth_sorted):
            depth_path = depth_sorted[seq_idx]
        if depth_path is None:
            raise FileNotFoundError(
                f"No matching depth PNG found for RGB frame {rgb_path.name} in {depth_dir}"
            )

        frames.append(
            FrameItem(
                seq_idx=seq_idx,
                frame_key=int(frame_key),
                rgb_path=rgb_path,
                depth_path=depth_path,
            )
        )
    return frames


def _parse_pose_line(line: str) -> Optional[PoseEntry]:
    nums = [float(x) for x in _FLOAT_RE.findall(line)]
    if not nums:
        return None

    frame_key: Optional[int] = None
    payload = nums
    if len(nums) in (13, 17):
        maybe_key = nums[0]
        if abs(maybe_key - round(maybe_key)) < 1e-6:
            frame_key = int(round(maybe_key))
            payload = nums[1:]

    if len(payload) == 12:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = np.asarray(payload, dtype=np.float32).reshape(3, 4)
        return PoseEntry(pose=pose, frame_key=frame_key)
    if len(payload) == 16:
        pose = np.asarray(payload, dtype=np.float32).reshape(4, 4)
        return PoseEntry(pose=pose, frame_key=frame_key)
    return None


def _load_poses(path: Path) -> List[PoseEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    poses: List[PoseEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = _parse_pose_line(line)
            if entry is not None:
                poses.append(entry)
    if not poses:
        raise ValueError(
            f"No valid poses found in {path}. Expected rows with 12 or 16 numeric values."
        )
    return poses


def _resolve_pose(
    poses: Sequence[PoseEntry],
    dataset: str,
    pose_lookup: str,
    seq_idx: int,
    frame_key: int,
) -> Optional[np.ndarray]:
    keyed = {entry.frame_key: entry.pose for entry in poses if entry.frame_key is not None}
    ordered = [entry.pose for entry in poses]

    if pose_lookup == "exact_key":
        return keyed.get(frame_key)

    if pose_lookup == "index":
        return ordered[seq_idx] if 0 <= seq_idx < len(ordered) else None

    if pose_lookup == "frame_number":
        idx = frame_key - 1 if dataset == "isaacsim" else frame_key
        return ordered[idx] if 0 <= idx < len(ordered) else None

    if frame_key in keyed:
        return keyed[frame_key]
    if dataset == "isaacsim":
        idx = frame_key - 1
        if 0 <= idx < len(ordered):
            return ordered[idx]
    if 0 <= seq_idx < len(ordered):
        return ordered[seq_idx]
    if 0 <= frame_key < len(ordered):
        return ordered[frame_key]
    return None


def _build_intrinsics(args: argparse.Namespace) -> CameraIntrinsics:
    if args.dataset == "isaacsim":
        width = int(args.image_width) if args.image_width is not None else 1280
        height = int(args.image_height) if args.image_height is not None else 720
        return CameraIntrinsics.from_physical(
            focal_length=args.focal_length,
            h_aperture=args.horizontal_aperture,
            v_aperture=args.vertical_aperture,
            width=width,
            height=height,
        )
    width = int(args.image_width) if args.image_width is not None else 920
    height = int(args.image_height) if args.image_height is not None else 690
    return CameraIntrinsics(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=width,
        height=height,
    )


def _load_depth(depth_path: Path, args: argparse.Namespace) -> Optional[np.ndarray]:
    arr = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None

    if args.dataset == "isaacsim":
        depth_m = arr.astype(np.float32) / float(args.png_max_value)
        depth_m *= float(args.max_depth)
    else:
        depth_m = arr.astype(np.float32) / float(args.depth_scale)

    depth_m[(depth_m < float(args.min_depth)) | (depth_m > float(args.max_depth))] = 0.0
    return depth_m


def _unproject_pinhole(
    us: np.ndarray,
    vs: np.ndarray,
    depths: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    xs = (us.astype(np.float32) - intrinsics.cx) * depths / intrinsics.fx
    ys = (vs.astype(np.float32) - intrinsics.cy) * depths / intrinsics.fy
    return np.stack([xs, ys, depths], axis=1).astype(np.float32)


def _reconstruct_frame(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_frame_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth_m.shape[:2]:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid = depth_m > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    vs, us = np.nonzero(valid)
    zs = depth_m[vs, us].astype(np.float32)
    colors = rgb[vs, us].astype(np.uint8)

    if max_frame_points > 0 and len(zs) > max_frame_points:
        idx = rng.choice(len(zs), size=max_frame_points, replace=False)
        us = us[idx]
        vs = vs[idx]
        zs = zs[idx]
        colors = colors[idx]

    pts_cam = _unproject_pinhole(us, vs, zs, intrinsics)
    rot = pose[:3, :3].astype(np.float32)
    trans = pose[:3, 3].astype(np.float32)
    pts_world = (pts_cam @ rot.T) + trans
    return pts_world.astype(np.float32), colors


def _send_blueprint() -> None:
    from rerun.blueprint import Blueprint, Horizontal, Spatial2DView, Spatial3DView

    rr.send_blueprint(
        Blueprint(
            Horizontal(
                Spatial3DView(name="3D Reconstruction", origin="world3d"),
                Spatial2DView(name="RGB", origin="rgb_view"),
                column_shares=[3, 1],
            )
        )
    )


def _log_world_frame() -> None:
    rr.log("world3d", rr.ViewCoordinates.RDF, static=True)
    rr.log("world3d/camera", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "world3d/origin",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32) * 0.35,
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        ),
        static=True,
    )


def _log_camera_intrinsics(intrinsics: CameraIntrinsics) -> None:
    rr.log(
        "world3d/camera/image",
        rr.Pinhole(
            resolution=[int(intrinsics.width), int(intrinsics.height)],
            focal_length=[float(intrinsics.fx), float(intrinsics.fy)],
            principal_point=[float(intrinsics.cx), float(intrinsics.cy)],
            image_plane_distance=0.3,
        ),
        static=True,
    )


def _concat_chunks(chunks: Iterable[np.ndarray], dtype: np.dtype) -> np.ndarray:
    arrays = [chunk for chunk in chunks if chunk.size > 0]
    if not arrays:
        return np.zeros((0, 3), dtype=dtype)
    return np.concatenate(arrays, axis=0).astype(dtype, copy=False)


def main() -> int:
    args = _parse_args()
    scene_dir = Path(args.scene_path).resolve()
    rgb_dir = _discover_rgb_dir(scene_dir)
    depth_dir = _resolve_scene_child(scene_dir, args.depth_folder)
    traj_path = _resolve_scene_child(scene_dir, args.traj_file)

    frames = _build_frames(rgb_dir, depth_dir)
    if args.stride > 1:
        frames = frames[:: args.stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("No frames selected for visualization.")

    poses = _load_poses(traj_path)
    intrinsics = _build_intrinsics(args)
    axis_remap = _build_axis_remap_matrix(swap_yz=True, flip_y=True)
    rng = np.random.default_rng(0)

    rr.init(f"dataset_reconstruction_{args.dataset}", spawn=bool(args.spawn))
    _send_blueprint()
    _log_world_frame()
    _log_camera_intrinsics(intrinsics)

    all_pts: np.ndarray = np.zeros((0, 3), dtype=np.float32)
    all_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
    camera_positions: List[np.ndarray] = []

    print(f"[rerun] scene={scene_dir}")
    print(f"[rerun] rgb_dir={rgb_dir}")
    print(f"[rerun] depth_dir={depth_dir}")
    print(f"[rerun] traj={traj_path}")
    print(f"[rerun] frames={len(frames)} max_frame_points={args.max_frame_points}")

    for vis_idx, frame in enumerate(frames):
        pose = _resolve_pose(
            poses=poses,
            dataset=args.dataset,
            pose_lookup=args.pose_lookup,
            seq_idx=frame.seq_idx,
            frame_key=frame.frame_key,
        )
        if pose is None:
            print(f"[warn] skipping frame {frame.rgb_path.name}: no matching pose")
            continue

        rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            print(f"[warn] skipping unreadable RGB frame: {frame.rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_m = _load_depth(frame.depth_path, args)
        if depth_m is None:
            print(f"[warn] skipping unreadable depth frame: {frame.depth_path}")
            continue

        pts_world, colors = _reconstruct_frame(
            rgb=rgb,
            depth_m=depth_m,
            pose=pose,
            intrinsics=intrinsics,
            max_frame_points=args.max_frame_points,
            rng=rng,
        )

        rr.set_time(timeline="frame", sequence=int(vis_idx))
        rr.log("rgb_view/current", rr.Image(rgb, color_model=rr.ColorModel.RGB))

        if pts_world.size > 0:
            all_pts = np.concatenate([all_pts, pts_world], axis=0)
            all_colors = np.concatenate([all_colors, colors], axis=0)

        pose_vis = _rigidize_camera_pose(pose)
        if pose_vis is not None:
            pose_vis = _apply_axis_remap_transform(pose_vis, axis_remap)
            rr.log(
                "world3d/camera",
                rr.Transform3D(
                    mat3x3=pose_vis[:3, :3].astype(np.float32),
                    translation=pose_vis[:3, 3].astype(np.float32),
                ),
            )
            rr.log(
                "world3d/camera/image/rgb",
                rr.Image(rgb, color_model=rr.ColorModel.RGB),
            )

            camera_positions.append(pose_vis[:3, 3].astype(np.float32))
            rr.log(
                "world3d/camera_positions",
                rr.Points3D(
                    np.asarray(camera_positions, dtype=np.float32),
                    colors=np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (len(camera_positions), 1)),
                    radii=np.full(len(camera_positions), 0.015, dtype=np.float32),
                ),
            )
            if len(camera_positions) >= 2:
                rr.log(
                    "world3d/camera_trajectory",
                    rr.LineStrips3D(
                        strips=[np.asarray(camera_positions, dtype=np.float32)],
                        colors=[[255, 255, 0]],
                    ),
                )

        if all_pts.size > 0:
            rr.log(
                "world3d/reconstruction/points",
                rr.Points3D(
                    _apply_axis_remap_points(all_pts, axis_remap),
                    colors=all_colors,
                    radii=np.full(len(all_pts), args.point_radius, dtype=np.float32),
                ),
            )

        print(
            f"[frame {vis_idx + 1:04d}/{len(frames):04d}] "
            f"rgb={frame.rgb_path.name} depth={frame.depth_path.name} total_points={len(all_pts)}",
            end="\r",
        )

    print()
    print("[rerun] visualization stream complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())