"""
Visualize GT data: RGB + masks + reprojected 3D bboxes (matplotlib),
then GT point cloud + 3D bboxes + world/camera frames (Open3D).
"""

import numpy as np
import cv2
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import open3d as o3d
from pathlib import Path


def load_poses(traj_path):
    poses = []
    with open(traj_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(4, 4))
    return poses


def backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


# ── consistent colors per track_id ──
def id_to_color(track_id, as_float=True):
    """Deterministic color from track_id using golden-ratio hashing."""
    hue = (track_id * 0.618033988749895) % 1.0
    # HSV -> RGB via matplotlib
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    if as_float:
        return (r, g, b)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── 3D bbox corners ──
def aabb_corners(aabb):
    """Return 8 corners from [xmin,ymin,zmin,xmax,ymax,zmax]."""
    xn, yn, zn, xx, yx, zx = aabb
    return np.array([
        [xn, yn, zn], [xn, yn, zx], [xn, yx, zn], [xn, yx, zx],
        [xx, yn, zn], [xx, yn, zx], [xx, yx, zn], [xx, yx, zx],
    ])

BBOX_EDGES = [
    [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],
    [2,6],[3,7],[4,5],[4,6],[5,7],[6,7]
]


def project_3d_bbox_to_2d(aabb, pose, K, img_shape):
    """
    Project 3D AABB corners to 2D image coords.
    pose: camera-to-world (4x4), so world-to-camera = inv(pose).
    Returns projected 2D corners (8x2) and visibility mask.
    """
    corners_w = aabb_corners(aabb)  # (8, 3)
    T_wc = np.linalg.inv(pose)     # world-to-camera

    ones = np.ones((8, 1))
    corners_cam = (T_wc @ np.hstack([corners_w, ones]).T).T[:, :3]

    # filter points behind camera
    valid = corners_cam[:, 2] > 0.01
    if valid.sum() < 2:
        return None, None

    pts_2d = (K @ corners_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

    h, w = img_shape[:2]
    return pts_2d, valid


# ── matplotlib visualization ──
def show_2d(rgb, mask, boxes, pose, K, frame_idx):
    """Show RGB + mask overlay + reprojected 3D bboxes."""
    h, w = rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # base image
    ax.imshow(rgb)

    # overlay masks with transparency
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    legend_handles = []
    id_to_col = {}

    for oid in obj_ids:
        c = id_to_color(int(oid))
        id_to_col[int(oid)] = c
        m = mask == oid
        overlay[m] = [c[0], c[1], c[2], 0.35]

        # 2D mask centroid for label
        ys, xs = np.where(m)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            ax.text(cx, cy, f"#{oid}", color='white', fontsize=7,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc=c, alpha=0.7, ec='none'))

    ax.imshow(overlay)

    # project 3D bboxes to 2D and draw edges
    for box in boxes:
        tid = box["track_id"]
        aabb = box["aabb_xyzmin_xyzmax"]
        c = id_to_col.get(tid, id_to_color(tid))

        pts_2d, valid = project_3d_bbox_to_2d(aabb, pose, K, rgb.shape)
        if pts_2d is None:
            continue

        # draw edges where both endpoints are in front of camera
        segments = []
        for i0, i1 in BBOX_EDGES:
            if valid[i0] and valid[i1]:
                x0, y0 = pts_2d[i0]
                x1, y1 = pts_2d[i1]
                # clip to generous image bounds
                if (min(x0, x1) > -w and max(x0, x1) < 2 * w and
                    min(y0, y1) > -h and max(y0, y1) < 2 * h):
                    segments.append([(x0, y0), (x1, y1)])

        if segments:
            lc = LineCollection(segments, colors=[c], linewidths=1.5, alpha=0.9)
            ax.add_collection(lc)

        # label at mean projected position
        front_pts = pts_2d[valid]
        if len(front_pts) > 0:
            mx, my = front_pts.mean(axis=0)
            if 0 <= mx < w and 0 <= my < h:
                ax.text(mx, my - 8, f"T{tid}", color=c, fontsize=6,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, ec='none'))

        legend_handles.append(mpatches.Patch(color=c, label=f"Track {tid}"))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f"Frame {frame_idx} — RGB + Masks + Reprojected 3D BBoxes   (press Q for 3D view)")
    ax.axis('off')

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=6,
                  ncol=2, framealpha=0.7)

    plt.tight_layout()
    plt.show()


# ── open3d visualization ──
def show_3d(depth_path, rgb_path, pose, K, boxes, stride=4):
    """Show GT point cloud + 3D bboxes + world frame + camera frustum."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pts_cam = backproject(depth, K)
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

    # world frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0]))

    # camera frame at pose position
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
    fr_lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_w)
    frustum.lines = o3d.utility.Vector2iVector(fr_lines)
    frustum.paint_uniform_color([0.2, 0.2, 0.8])
    geometries.append(frustum)

    # 3D bboxes
    for box in boxes:
        tid = box["track_id"]
        aabb = box["aabb_xyzmin_xyzmax"]
        c = id_to_color(tid)
        corners = aabb_corners(aabb)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(BBOX_EDGES)
        ls.paint_uniform_color(list(c))
        geometries.append(ls)

    print("Open3D 3D view:")
    print("  Point cloud = GT depth + GT pose (true RGB)")
    print("  Colored wireframes = 3D bboxes (same colors as 2D)")
    print("  Small axes at camera position, large axes at world origin")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT 3D View — Point Cloud + BBoxes",
        width=1280, height=720,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize GT labels: 2D (matplotlib) then 3D (Open3D)")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame index (0-based)")
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for 3D subsampling")
    parser.add_argument("--fx", type=float, default=692.52)
    parser.add_argument("--fy", type=float, default=693.83)
    parser.add_argument("--cx", type=float, default=459.76)
    parser.add_argument("--cy", type=float, default=344.76)
    args = parser.parse_args()

    scene = Path(args.scene)
    K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]])

    gt_poses = load_poses(scene / "traj.txt")

    image_files = sorted((scene / "images").glob("*.jpg"))
    if not image_files:
        image_files = sorted((scene / "images").glob("*.png"))
    depth_files = sorted((scene / "gt_depth").glob("*.png"))
    mask_files = sorted((scene / "masks").glob("*.npy"))
    bbox_files = sorted((scene / "bbox").glob("bboxes*_info.json"))

    n = min(len(image_files), len(depth_files), len(mask_files), len(bbox_files), len(gt_poses))
    print(f"Total frames: {n}")

    i = args.frame
    assert 0 <= i < n, f"Frame {i} out of range [0, {n})"

    print(f"Frame {i}: {image_files[i].name}")

    rgb = cv2.cvtColor(cv2.imread(str(image_files[i])), cv2.COLOR_BGR2RGB)
    mask = np.load(str(mask_files[i]))
    pose = gt_poses[i]

    with open(bbox_files[i]) as f:
        boxes = json.load(f)["bboxes"]["bbox_3d"]["boxes"]

    print(f"  Objects in mask: {len(np.unique(mask)) - 1}")
    print(f"  Boxes: {len(boxes)}")

    # step 1: 2D matplotlib view
    show_2d(rgb, mask, boxes, pose, K, i)

    # step 2: 3D open3d view (after matplotlib window is closed)
    show_3d(depth_files[i], image_files[i], pose, K, boxes, stride=args.stride)


if __name__ == "__main__":
    main()
