#!/usr/bin/env python3
"""
Visualize GT depth and Pi3 depth as colored point clouds in Open3D.

  GT  → reddish tinted + RGB point cloud, red camera frustums
  Pi3 → greenish tinted + RGB point cloud, green camera frustums

Usage:
    # Single frame (frame number from filename, 1-indexed)
    python visualize_depth.py --scene_dir scene_3 --frame 5

    # First frame (default)
    python visualize_depth.py --scene_dir scene_3

    # Accumulate N frames
    python visualize_depth.py --scene_dir scene_3 --accumulate 10

    # Accumulate with custom subsample
    python visualize_depth.py --scene_dir scene_3 --accumulate 20 --subsample 8
"""

import argparse
import os
import re
import numpy as np
import cv2
import open3d as o3d


def load_poses(traj_path):
    """Load camera poses from trajectory file (each line: 16 floats, row-major 4x4)."""
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def parse_cam_params(cam_param_path):
    """Parse cam_param.txt and return fx, fy, cx, cy, png_depth_scale."""
    fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
    png_depth_scale = 0.00015244

    if cam_param_path and os.path.isfile(cam_param_path):
        content = open(cam_param_path).read()
        for name, var in [('fx', 'fx'), ('fy', 'fy'), ('cx', 'cx'), ('cy', 'cy'),
                          ('png_depth_scale', 'png_depth_scale')]:
            m = re.search(rf'{name}:\s*([\d.eE\-+]+)', content)
            if m:
                locals()[var]  # just to verify name exists
                if name == 'fx': fx = float(m.group(1))
                elif name == 'fy': fy = float(m.group(1))
                elif name == 'cx': cx = float(m.group(1))
                elif name == 'cy': cy = float(m.group(1))
                elif name == 'png_depth_scale': png_depth_scale = float(m.group(1))
    return fx, fy, cx, cy, png_depth_scale


def parse_depth_scale_from_file(path):
    """Parse a text file for `png_depth_scale: <float>`."""
    if path is None or not os.path.isfile(path):
        return None
    content = open(path).read()
    m = re.search(r'png_depth_scale:\s*([\d.eE\-+]+)', content)
    if m:
        return float(m.group(1))
    return None


def depth_to_pointcloud(depth_m, rgb, fx, fy, cx, cy, pose, tint, subsample):
    """
    Back-project depth map to a world-frame point cloud with tinted RGB colors.

    Args:
        depth_m:  (H, W) float64, depth in meters (0 = invalid)
        rgb:      (H, W, 3) uint8, RGB image
        fx,fy,cx,cy: intrinsics
        pose:     (4, 4) camera-to-world
        tint:     (3,) float in [0,1], color to blend in
        subsample: pixel stride

    Returns:
        open3d PointCloud
    """
    H, W = depth_m.shape
    us = np.arange(0, W, subsample)
    vs = np.arange(0, H, subsample)
    u, v = np.meshgrid(us, vs)
    u = u.ravel()
    v = v.ravel()

    z = depth_m[v, u]
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    # Back-project
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)  # (N, 3)

    # To world
    pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

    # Colors: blend 60% RGB + 40% tint
    colors = rgb[v, u].astype(np.float64) / 255.0
    colors = 0.6 * colors + 0.4 * np.array(tint).reshape(1, 3)
    colors = np.clip(colors, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_frustum(pose, fx, fy, cx, cy, w, h, scale=0.3, color=(1, 0, 0)):
    """Camera frustum wireframe as a LineSet."""
    # Image corners at z=scale in camera space
    corners_uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_cam = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_uv):
        corners_cam[i] = [(u - cx) / fx * scale, (v - cy) / fy * scale, scale]

    pts_cam = np.vstack([[[0, 0, 0]], corners_cam])  # 0=origin, 1-4=corners
    pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],   # rays
             [1, 2], [2, 3], [3, 4], [4, 1]]     # rectangle
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_world),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def make_cam_axes(pose, size=0.15):
    """Small coordinate frame at camera position."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(pose)
    return axes


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GT (red) and Pi3 (green) depth as point clouds')
    parser.add_argument('--scene_dir', required=True,
                        help='Path to scene directory (e.g. scene_3)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame number (1-indexed, matching filenames). '
                             'Default: frame 1')
    parser.add_argument('--accumulate', type=int, default=None,
                        help='Accumulate N frames starting from frame 1')
    parser.add_argument('--subsample', type=int, default=10,
                        help='Point subsampling stride (default: 10)')
    parser.add_argument('--frustum_scale', type=float, default=0.3,
                        help='Frustum size in meters (default: 0.3)')
    parser.add_argument('--pi3_depth_scale', type=float, default=None,
                        help='Scale factor for Pi3 depth PNG values → meters. '
                             'If not given: use pi3_depth metadata if available, otherwise auto-estimate from GT.')
    parser.add_argument('--cam_param', type=str, default=None,
                        help='Path to cam_param.txt (auto-detected)')
    args = parser.parse_args()

    scene_dir = args.scene_dir

    # --- Locate cam_param.txt ---
    cam_param_path = args.cam_param
    if cam_param_path is None:
        for cand in [os.path.join(scene_dir, 'cam_param.txt'),
                     os.path.join(scene_dir, '..', 'cam_param.txt'),
                     'cam_param.txt']:
            if os.path.isfile(cand):
                cam_param_path = cand
                break

    fx, fy, cx, cy, png_depth_scale = parse_cam_params(cam_param_path)
    print(f"Intrinsics  fx={fx}  fy={fy}  cx={cx}  cy={cy}")
    print(f"GT depth scale: {png_depth_scale}")

    # --- Paths ---
    rgb_dir       = os.path.join(scene_dir, 'rgb')
    gt_depth_dir  = os.path.join(scene_dir, 'depth')
    pi3_depth_dir = os.path.join(scene_dir, 'pi3_depth')
    gt_traj_path  = os.path.join(scene_dir, 'traj.txt')
    pi3_traj_path = os.path.join(scene_dir, 'pi3_camera_poses.txt')

    gt_poses  = load_poses(gt_traj_path)
    pi3_poses = load_poses(pi3_traj_path)

    # --- Determine which frames to visualize ---
    rgb_files = sorted([f for f in os.listdir(rgb_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    n_frames = len(rgb_files)
    print(f"Total frames: {n_frames}")

    if args.frame is not None:
        frame_indices = [args.frame - 1]          # user gives 1-indexed
    elif args.accumulate is not None:
        n = min(args.accumulate, n_frames)
        frame_indices = list(range(n))
    else:
        frame_indices = [0]

    # Auto-increase subsample for many accumulated frames
    subsample = args.subsample
    if args.accumulate and args.accumulate > 5 and args.subsample == 10:
        subsample = max(subsample, 2 + args.accumulate // 3)
        print(f"Auto subsample → {subsample}  (for {len(frame_indices)} frames)")

    print(f"Frames to render: {[i + 1 for i in frame_indices]}  subsample={subsample}")

    # --- Resolve Pi3 depth scale ---
    pi3_depth_scale = args.pi3_depth_scale
    if pi3_depth_scale is None:
        # Prefer explicit metadata written by the Pi3 exporter
        meta_candidates = [
            os.path.join(pi3_depth_dir, 'pi3_depth_meta.txt'),
            os.path.join(pi3_depth_dir, 'depth_scale.txt'),
        ]
        for meta_path in meta_candidates:
            scale = parse_depth_scale_from_file(meta_path)
            if scale is not None:
                pi3_depth_scale = scale
                print(f"Pi3 depth scale from metadata ({meta_path}): {pi3_depth_scale:.10f}")
                break

    if pi3_depth_scale is None:
        # Backward-compatible fallback for older exports (auto estimate from GT)
        idx0 = frame_indices[0]
        gt_path = os.path.join(gt_depth_dir, f'depth{idx0 + 1:06d}.png')
        pi3_path = os.path.join(pi3_depth_dir, f'depth{idx0:06d}.png')
        if os.path.isfile(gt_path) and os.path.isfile(pi3_path):
            gt_d = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64) * png_depth_scale
            pi3_d = cv2.imread(pi3_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_med  = np.median(gt_d[gt_d > 0])   if (gt_d > 0).any()  else 1.0
            pi3_med = np.median(pi3_d[pi3_d > 0])  if (pi3_d > 0).any() else 1.0
            pi3_depth_scale = gt_med / pi3_med
            print(f"Auto Pi3 depth scale: {pi3_depth_scale:.6f}  "
                  f"(GT median={gt_med:.3f}m, Pi3 median={pi3_med:.1f})")
        else:
            pi3_depth_scale = 10.0 / 255.0  # fallback ~0.039
            print(f"Fallback Pi3 depth scale: {pi3_depth_scale:.6f}")

    # --- Build geometries ---
    geometries = []

    # World frame
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    for idx in frame_indices:
        fnum = idx + 1                         # 1-indexed filename number

        # RGB
        rgb_path = os.path.join(rgb_dir, f'frame{fnum:06d}.jpg')
        if not os.path.isfile(rgb_path):
            rgb_path = os.path.join(rgb_dir, f'frame{fnum:06d}.png')
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"  [skip] cannot read {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # ---- GT ----
        gt_depth_path = os.path.join(gt_depth_dir, f'depth{fnum:06d}.png')
        if os.path.isfile(gt_depth_path) and idx < len(gt_poses):
            gt_raw = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
            gt_m = gt_raw.astype(np.float64) * png_depth_scale
            gt_pose = gt_poses[idx]

            pcd_gt = depth_to_pointcloud(
                gt_m, rgb, fx, fy, cx, cy, gt_pose,
                tint=[1.0, 0.3, 0.3], subsample=subsample)
            geometries.append(pcd_gt)
            geometries.append(make_frustum(
                gt_pose, fx, fy, cx, cy, w, h,
                scale=args.frustum_scale, color=(1, 0, 0)))
            geometries.append(make_cam_axes(gt_pose))
            print(f"  frame {fnum}: GT  {len(pcd_gt.points):,} pts")

        # ---- Pi3 ----
        pi3_depth_path = os.path.join(pi3_depth_dir, f'depth{idx:06d}.png')
        if os.path.isfile(pi3_depth_path) and idx < len(pi3_poses):
            pi3_raw = cv2.imread(pi3_depth_path, cv2.IMREAD_UNCHANGED)
            pi3_m = pi3_raw.astype(np.float64) * pi3_depth_scale
            pi3_pose = pi3_poses[idx]

            pcd_pi3 = depth_to_pointcloud(
                pi3_m, rgb, fx, fy, cx, cy, pi3_pose,
                tint=[0.3, 1.0, 0.3], subsample=subsample)
            geometries.append(pcd_pi3)
            geometries.append(make_frustum(
                pi3_pose, fx, fy, cx, cy, w, h,
                scale=args.frustum_scale, color=(0, 1, 0)))
            geometries.append(make_cam_axes(pi3_pose))
            print(f"  frame {fnum}: Pi3 {len(pcd_pi3.points):,} pts")

    total_pts = sum(len(np.asarray(g.points))
                    for g in geometries if isinstance(g, o3d.geometry.PointCloud))
    print(f"\nTotal points: {total_pts:,}")
    print("Legend:  RED frustums / reddish cloud = GT")
    print("         GREEN frustums / greenish cloud = Pi3")
    print("         RGB axes at origin = world frame")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT (Red) vs Pi3 (Green) Depth",
        width=1400, height=900)


if __name__ == '__main__':
    main()
