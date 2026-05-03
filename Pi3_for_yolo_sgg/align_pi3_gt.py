#!/usr/bin/env python3
"""
Align Pi3 reconstruction to GT frame using Umeyama Sim(3), then visualize.

What it does:
1) Reconstructs GT and Pi3 world-space points from depth PNGs + poses
2) Builds pixel-wise correspondences (same frame/same pixel, valid in both)
3) Estimates Sim(3):  p_gt ~= s * R * p_pi3 + t
4) Applies alignment to Pi3 point clouds and camera frustums
5) Visualizes GT (red) vs aligned Pi3 (green) in Open3D
"""

import argparse
import os
import re
import json
import numpy as np
import cv2
import open3d as o3d


def load_poses(traj_path):
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    return poses


def parse_cam_params(cam_param_path):
    fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
    png_depth_scale = 0.00015244
    if cam_param_path and os.path.isfile(cam_param_path):
        content = open(cam_param_path).read()
        for name in ["fx", "fy", "cx", "cy", "png_depth_scale"]:
            m = re.search(rf"{name}:\s*([\d.eE\-+]+)", content)
            if m:
                val = float(m.group(1))
                if name == "fx":
                    fx = val
                elif name == "fy":
                    fy = val
                elif name == "cx":
                    cx = val
                elif name == "cy":
                    cy = val
                elif name == "png_depth_scale":
                    png_depth_scale = val
    return fx, fy, cx, cy, png_depth_scale


def parse_depth_scale_from_file(path):
    if path is None or not os.path.isfile(path):
        return None
    content = open(path).read()
    m = re.search(r"png_depth_scale:\s*([\d.eE\-+]+)", content)
    if m:
        return float(m.group(1))
    return None


def depth_to_pointcloud(depth_m, rgb, fx, fy, cx, cy, pose, tint, subsample):
    h, w = depth_m.shape
    us = np.arange(0, w, subsample)
    vs = np.arange(0, h, subsample)
    u, v = np.meshgrid(us, vs)
    u = u.ravel()
    v = v.ravel()

    z = depth_m[v, u]
    valid = np.logical_and(z > 0, np.isfinite(z))
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

    colors = rgb[v, u].astype(np.float64) / 255.0
    colors = 0.6 * colors + 0.4 * np.array(tint, dtype=np.float64).reshape(1, 3)
    colors = np.clip(colors, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_frustum(pose, fx, fy, cx, cy, w, h, scale=0.3, color=(1, 0, 0)):
    corners_uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_uv):
        corners_cam[i] = [(u - cx) / fx * scale, (v - cy) / fy * scale, scale]

    pts_cam = np.vstack([[[0, 0, 0]], corners_cam])
    pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def make_cam_axes(pose, size=0.15):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(pose)
    return axes


def collect_paired_world_points(depth_gt_m, depth_pi3_m, fx, fy, cx, cy, pose_gt, pose_pi3, subsample):
    h, w = depth_gt_m.shape
    us = np.arange(0, w, subsample)
    vs = np.arange(0, h, subsample)
    u, v = np.meshgrid(us, vs)
    u = u.ravel()
    v = v.ravel()

    z_gt = depth_gt_m[v, u]
    z_pi3 = depth_pi3_m[v, u]
    valid_gt = np.logical_and(z_gt > 0, np.isfinite(z_gt))
    valid_pi3 = np.logical_and(z_pi3 > 0, np.isfinite(z_pi3))
    valid = np.logical_and(valid_gt, valid_pi3)

    u = u[valid]
    v = v[valid]
    z_gt = z_gt[valid]
    z_pi3 = z_pi3[valid]

    x_gt = (u - cx) * z_gt / fx
    y_gt = (v - cy) * z_gt / fy
    pts_gt_cam = np.stack([x_gt, y_gt, z_gt], axis=-1)

    x_pi3 = (u - cx) * z_pi3 / fx
    y_pi3 = (v - cy) * z_pi3 / fy
    pts_pi3_cam = np.stack([x_pi3, y_pi3, z_pi3], axis=-1)

    pts_gt_world = (pose_gt[:3, :3] @ pts_gt_cam.T).T + pose_gt[:3, 3]
    pts_pi3_world = (pose_pi3[:3, :3] @ pts_pi3_cam.T).T + pose_pi3[:3, 3]
    return pts_pi3_world, pts_gt_world


def umeyama_sim3(src, dst, eps=1e-12):
    """
    Estimate Sim(3) mapping src -> dst:
        dst ~= s * R * src + t
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be Nx3 with same shape, got {src.shape=} {dst.shape=}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 correspondences, got {n}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src < eps:
        raise ValueError("Source variance is too small for stable Umeyama alignment")

    scale = float(np.sum(D * np.diag(S)) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_sim3_to_points(points, scale, R, t):
    return (scale * (R @ points.T)).T + t


def apply_sim3_to_pose(pose_c2w, scale, R, t):
    sim3 = np.eye(4, dtype=np.float64)
    sim3[:3, :3] = scale * R
    sim3[:3, 3] = t
    return sim3 @ pose_c2w


def main():
    parser = argparse.ArgumentParser(description="Align Pi3 to GT with Umeyama Sim(3) and visualize.")
    parser.add_argument("--scene_dir", required=True, help="Path to scene directory.")
    parser.add_argument("--frame", type=int, default=None, help="Single frame (1-indexed).")
    parser.add_argument("--accumulate", type=int, default=None, help="Accumulate first N frames.")
    parser.add_argument("--subsample", type=int, default=10, help="Pixel stride for PCD rendering.")
    parser.add_argument("--align_subsample", type=int, default=8, help="Pixel stride for correspondence building.")
    parser.add_argument("--max_pairs", type=int, default=250000, help="Max correspondences used in Umeyama.")
    parser.add_argument(
        "--frame_pair_cap",
        type=int,
        default=0,
        help="Optional hard cap of correspondences per frame before global merge. "
             "0 means auto cap based on max_pairs and frame count.",
    )
    parser.add_argument("--frustum_scale", type=float, default=0.3, help="Camera frustum size (m).")
    parser.add_argument("--cam_param", type=str, default=None, help="Path to cam_param.txt (auto-detected).")
    parser.add_argument(
        "--pi3_depth_scale",
        type=float,
        default=None,
        help="Pi3 PNG->meters scale. If omitted, reads pi3 metadata then falls back to GT-based auto-estimation.",
    )
    parser.add_argument(
        "--save_transform",
        action="store_true",
        help="If set, save transform in scene folder as pi3_to_world_transform.json and .npz",
    )
    parser.add_argument(
        "--pi3_only",
        action="store_true",
        help="If set, visualize only aligned Pi3 geometries (hide GT geometries).",
    )
    args = parser.parse_args()

    scene_dir = args.scene_dir

    cam_param_path = args.cam_param
    if cam_param_path is None:
        for cand in [
            os.path.join(scene_dir, "cam_param.txt"),
            os.path.join(scene_dir, "..", "cam_param.txt"),
            "cam_param.txt",
        ]:
            if os.path.isfile(cand):
                cam_param_path = cand
                break

    fx, fy, cx, cy, gt_depth_scale = parse_cam_params(cam_param_path)
    print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"GT depth scale: {gt_depth_scale}")

    rgb_dir = os.path.join(scene_dir, "rgb")
    gt_depth_dir = os.path.join(scene_dir, "depth")
    pi3_depth_dir = os.path.join(scene_dir, "pi3_depth")
    gt_traj_path = os.path.join(scene_dir, "traj.txt")
    pi3_traj_path = os.path.join(scene_dir, "pi3_camera_poses.txt")

    gt_poses = load_poses(gt_traj_path)
    pi3_poses = load_poses(pi3_traj_path)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    n_frames = len(rgb_files)
    print(f"Total frames: {n_frames}")

    if args.frame is not None:
        frame_indices = [args.frame - 1]
    elif args.accumulate is not None:
        frame_indices = list(range(min(args.accumulate, n_frames)))
    else:
        # Default: align on the whole scene
        frame_indices = list(range(n_frames))

    if args.accumulate and args.accumulate > 5 and args.subsample == 10:
        render_subsample = max(args.subsample, 2 + args.accumulate // 3)
    else:
        render_subsample = args.subsample
    if len(frame_indices) <= 20:
        print(f"Frames used: {[i + 1 for i in frame_indices]}")
    else:
        print(
            f"Frames used: {len(frame_indices)} "
            f"(from {frame_indices[0] + 1} to {frame_indices[-1] + 1})"
        )
    print(f"Render subsample={render_subsample}, align_subsample={args.align_subsample}")

    pi3_depth_scale = args.pi3_depth_scale
    if pi3_depth_scale is None:
        for meta_path in [
            os.path.join(pi3_depth_dir, "pi3_depth_meta.txt"),
            os.path.join(pi3_depth_dir, "depth_scale.txt"),
        ]:
            scale = parse_depth_scale_from_file(meta_path)
            if scale is not None:
                pi3_depth_scale = scale
                print(f"Pi3 depth scale from metadata ({meta_path}): {pi3_depth_scale:.10f}")
                break

    if pi3_depth_scale is None:
        idx0 = frame_indices[0]
        gt_path = os.path.join(gt_depth_dir, f"depth{idx0 + 1:06d}.png")
        pi3_path = os.path.join(pi3_depth_dir, f"depth{idx0:06d}.png")
        if os.path.isfile(gt_path) and os.path.isfile(pi3_path):
            gt_d = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64) * gt_depth_scale
            pi3_d = cv2.imread(pi3_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_med = np.median(gt_d[gt_d > 0]) if (gt_d > 0).any() else 1.0
            pi3_med = np.median(pi3_d[pi3_d > 0]) if (pi3_d > 0).any() else 1.0
            pi3_depth_scale = gt_med / pi3_med
            print(f"Auto Pi3 depth scale: {pi3_depth_scale:.6f}")
        else:
            pi3_depth_scale = 10.0 / 255.0
            print(f"Fallback Pi3 depth scale: {pi3_depth_scale:.6f}")

    # -----------------------------
    # Build correspondence set
    # -----------------------------
    all_src_pi3 = []
    all_dst_gt = []
    raw_pairs_total = 0
    kept_pairs_total = 0
    if args.frame_pair_cap > 0:
        frame_pair_cap = int(args.frame_pair_cap)
    elif args.max_pairs > 0:
        # Keep memory bounded before global sampling.
        # Oversample a bit per frame to keep spatial diversity.
        frame_pair_cap = max(2000, int(np.ceil((2.0 * args.max_pairs) / max(1, len(frame_indices)))))
    else:
        frame_pair_cap = 50000
    print(f"Per-frame correspondence cap: {frame_pair_cap}")

    for idx in frame_indices:
        fnum = idx + 1
        if idx >= len(gt_poses) or idx >= len(pi3_poses):
            continue

        gt_depth_path = os.path.join(gt_depth_dir, f"depth{fnum:06d}.png")
        pi3_depth_path = os.path.join(pi3_depth_dir, f"depth{idx:06d}.png")
        if not (os.path.isfile(gt_depth_path) and os.path.isfile(pi3_depth_path)):
            continue

        gt_raw = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        pi3_raw = cv2.imread(pi3_depth_path, cv2.IMREAD_UNCHANGED)
        if gt_raw is None or pi3_raw is None:
            continue

        gt_m = gt_raw.astype(np.float64) * gt_depth_scale
        pi3_m = pi3_raw.astype(np.float64) * pi3_depth_scale

        if gt_m.shape != pi3_m.shape:
            pi3_m = cv2.resize(pi3_m, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_NEAREST)

        src_pi3, dst_gt = collect_paired_world_points(
            depth_gt_m=gt_m,
            depth_pi3_m=pi3_m,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            pose_gt=gt_poses[idx],
            pose_pi3=pi3_poses[idx],
            subsample=args.align_subsample,
        )
        if src_pi3.shape[0] > 0:
            raw_pairs_total += int(src_pi3.shape[0])
            if src_pi3.shape[0] > frame_pair_cap:
                rng = np.random.default_rng(42 + idx)
                sel = rng.choice(src_pi3.shape[0], size=frame_pair_cap, replace=False)
                src_pi3 = src_pi3[sel]
                dst_gt = dst_gt[sel]
            all_src_pi3.append(src_pi3)
            all_dst_gt.append(dst_gt)
            kept_pairs_total += int(src_pi3.shape[0])

    if not all_src_pi3:
        raise RuntimeError("No valid point correspondences found for alignment.")

    src = np.concatenate(all_src_pi3, axis=0)
    dst = np.concatenate(all_dst_gt, axis=0)
    print(f"Raw correspondences across frames: {raw_pairs_total:,}")
    print(f"After per-frame cap merge: {kept_pairs_total:,}")
    print(f"Total correspondences before global sampling: {src.shape[0]:,}")

    if args.max_pairs > 0 and src.shape[0] > args.max_pairs:
        rng = np.random.default_rng(42)
        sel = rng.choice(src.shape[0], size=args.max_pairs, replace=False)
        src = src[sel]
        dst = dst[sel]
        print(f"Subsampled correspondences: {src.shape[0]:,}")

    scale, R, t = umeyama_sim3(src=src, dst=dst)
    src_aligned = apply_sim3_to_points(src, scale, R, t)
    rmse = np.sqrt(np.mean(np.sum((src_aligned - dst) ** 2, axis=1)))

    print("\n=== Umeyama Sim(3): Pi3 -> GT ===")
    print(f"Scale: {scale:.10f}")
    print("Rotation R:")
    print(R)
    print("Translation t:")
    print(t)
    print(f"Alignment RMSE (m): {rmse:.6f}")

    sim3 = np.eye(4, dtype=np.float64)
    sim3[:3, :3] = scale * R
    sim3[:3, 3] = t
    print("Sim(3) matrix:")
    print(sim3)

    if args.save_transform:
        json_path = os.path.join(scene_dir, "pi3_to_world_transform.json")
        npz_path = os.path.join(scene_dir, "pi3_to_world_transform.npz")

        payload = {
            "convention": "p_world = scale * R @ p_pi3 + t",
            "scale": float(scale),
            "rotation": R.tolist(),
            "translation": t.tolist(),
            "sim3_matrix_4x4": sim3.tolist(),
            "rmse_m": float(rmse),
            "num_correspondences_used": int(src.shape[0]),
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        np.savez(
            npz_path,
            scale=np.array([scale], dtype=np.float64),
            rotation=R.astype(np.float64),
            translation=t.astype(np.float64),
            sim3_matrix_4x4=sim3.astype(np.float64),
            rmse_m=np.array([rmse], dtype=np.float64),
            num_correspondences_used=np.array([src.shape[0]], dtype=np.int64),
        )
        print(f"Saved transform JSON: {json_path}")
        print(f"Saved transform NPZ : {npz_path}")

    # -----------------------------
    # Visualization
    # -----------------------------
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    for idx in frame_indices:
        fnum = idx + 1
        if idx >= len(gt_poses) or idx >= len(pi3_poses):
            continue

        rgb_path = os.path.join(rgb_dir, f"frame{fnum:06d}.jpg")
        if not os.path.isfile(rgb_path):
            rgb_path = os.path.join(rgb_dir, f"frame{fnum:06d}.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        gt_depth_path = os.path.join(gt_depth_dir, f"depth{fnum:06d}.png")
        pi3_depth_path = os.path.join(pi3_depth_dir, f"depth{idx:06d}.png")
        if not (os.path.isfile(gt_depth_path) and os.path.isfile(pi3_depth_path)):
            continue

        gt_raw = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        pi3_raw = cv2.imread(pi3_depth_path, cv2.IMREAD_UNCHANGED)
        if gt_raw is None or pi3_raw is None:
            continue

        gt_m = gt_raw.astype(np.float64) * gt_depth_scale
        pi3_m = pi3_raw.astype(np.float64) * pi3_depth_scale
        if gt_m.shape != pi3_m.shape:
            pi3_m = cv2.resize(pi3_m, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_pose = gt_poses[idx]
        pi3_pose_aligned = apply_sim3_to_pose(pi3_poses[idx], scale, R, t)

        pcd_pi3 = depth_to_pointcloud(
            pi3_m, rgb, fx, fy, cx, cy, pi3_pose_aligned, tint=[0.3, 1.0, 0.3], subsample=render_subsample
        )

        geometries.append(pcd_pi3)

        geometries.append(make_frustum(pi3_pose_aligned, fx, fy, cx, cy, w, h, scale=args.frustum_scale, color=(0, 1, 0)))

        geometries.append(make_cam_axes(pi3_pose_aligned))

        if not args.pi3_only:
            pcd_gt = depth_to_pointcloud(
                gt_m, rgb, fx, fy, cx, cy, gt_pose, tint=[0.5, 0.3, 0.3], subsample=render_subsample
            )
            geometries.append(pcd_gt)
            geometries.append(make_frustum(gt_pose, fx, fy, cx, cy, w, h, scale=args.frustum_scale, color=(1, 0, 0)))
            geometries.append(make_cam_axes(gt_pose))

    if args.pi3_only:
        print("\nLegend: GREEN=Pi3 aligned (GT hidden by --pi3_only)")
    else:
        print("\nLegend: RED=GT, GREEN=Pi3 aligned")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT (Red) vs Pi3 aligned to GT (Green)",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    main()
