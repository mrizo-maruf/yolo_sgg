"""
Visualize IsaacSim Replicator 3D bboxes + point cloud in Open3D.

Assumptions:
- You saved per-frame:
    rgb/frameXXXXXX.jpg
    depth/depthXXXXXX.png   (uint16, mapped linearly from [MIN_DEPTH, MAX_DEPTH])
    bbox/bboxesXXXXXX_info.json
    traj.txt                (optional; currently ROS camera pose flattened 4x4 per frame)

- Your JSON format (as in your code):
  boxes["bboxes"]["bbox_3d"]["boxes"] is a list of dicts with:
    - aabb_xyzmin_xyzmax: [x_min,y_min,z_min,x_max,y_max,z_max]
    - transform_4x4: 4x4 (optional usage)
    - prim_path, label, track_id, etc.

This script supports TWO common cases:
  (A) bbox extents already in WORLD coords -> set BBOX_FRAME="world"
  (B) bbox extents are in CAMERA coords -> set BBOX_FRAME="camera" and provide camera pose

Also supports optional axis swapping for Open3D visualization if needed.
"""

import os
import json
import glob
import numpy as np
import cv2
import open3d as o3d

# -------------------------
# -------------------------
# User-configurable settings
# -------------------------
BASE_DIR = "/home/maribjonov_mr/IsaacSim_bench/cabinet_complex"  # folder with rgb/depth/bbox
FRAME_ID = 10  # frame index (0-based)

# Image + depth settings (must match how you saved data)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
PNG_MAX_VALUE = 65535

# Intrinsics (copy from capture log)
FX = 800
FY = 800
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

# Bbox interpretation: "world" if already transformed, "camera" if still in camera frame
# "use_transform" will apply the transform_4x4 from each box to convert local->world
BBOX_FRAME = "world"  # Use "world" - boxes are now saved in world coords after fix

# Use traj.txt camera pose to move the point cloud into world frame
USE_TRAJ = True
TRAJ_PATH = "/home/maribjonov_mr/IsaacSim_bench/cabinet_complex/traj.txt"  # None -> BASE_DIR/traj.txt

# Visualization tweaks
SWAP_YZ = False  # try True if axes look rotated
MAX_BOX_EDGE = 20.0  # meters; filter very large boxes
IGNORE_PRIM_PREFIXES = ["/World/env"]


# -------------------------
# Helpers
# -------------------------
def load_depth_meters(depth_png_path: str, png_depth_scale: float, min_depth: float) -> np.ndarray:
    d16 = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    if d16 is None:
        raise FileNotFoundError(depth_png_path)
    if d16.ndim == 3:
        d16 = d16[:, :, 0]
    d = d16.astype(np.float32) * png_depth_scale + min_depth
    return d

def rgb_to_o3d(rgb_bgr_path: str) -> o3d.geometry.Image:
    bgr = cv2.imread(rgb_bgr_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_bgr_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return o3d.geometry.Image(rgb)

def depth_to_o3d(depth_m: np.ndarray) -> o3d.geometry.Image:
    # Open3D expects depth in uint16 (in mm) or float (in meters) depending on API usage.
    # We'll use float meters via create_from_color_and_depth with depth_scale=1.0.
    return o3d.geometry.Image(depth_m.astype(np.float32))

def make_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    if fx is None or fy is None:
        raise ValueError("Set fx and fy to the values printed during capture (fx, fy).")
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, fx, fy, cx, cy)
    return intr

def apply_swap_yz(points: np.ndarray) -> np.ndarray:
    # (x,y,z) -> (x,z,y)
    p = points.copy()
    p[:, [1, 2]] = p[:, [2, 1]]
    return p

def load_boxes_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    boxes = data["bboxes"]["bbox_3d"]["boxes"]
    return boxes

def read_traj_pose(frame_id: int, traj_path: str) -> np.ndarray:
    frame_id = frame_id - 1
    """
    Reads traj.txt line 'frame_id' as a flattened 4x4.
    WARNING: your traj is in ROS camera axes (camera_axes="ros").
    Use only if you know how your bbox frame aligns with it.
    """
    with open(traj_path, "r") as f:
        lines = f.readlines()
    if frame_id >= len(lines):
        raise IndexError("traj.txt has fewer lines than frame_id")
    vals = np.fromstring(lines[frame_id].strip(), sep=" ", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"traj line {frame_id} does not have 16 values")
    T = vals.reshape(4, 4)
    return T

def aabb_lineset_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax, swap_yz: bool):
    pts = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)
    if swap_yz:
        pts = apply_swap_yz(pts)

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    # don't set colors explicitly (Open3D defaults are fine); you can set if you want.
    return ls

def transform_points(T, pts3):
    pts_h = np.concatenate([pts3, np.ones((pts3.shape[0], 1), dtype=pts3.dtype)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]

def camera_aabb_to_world_lineset(aabb, T_world_from_cam, swap_yz: bool):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)
    corners_w = transform_points(T_world_from_cam, corners)

    # Build lineset from transformed corners
    if swap_yz:
        corners_w = apply_swap_yz(corners_w)

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_w)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def local_aabb_to_world_lineset(aabb, T_world_from_local, swap_yz: bool):
    """
    Transform a LOCAL-frame AABB using the object's transform_4x4,
    then create an ORIENTED bounding box lineset (not axis-aligned after transform).
    """
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    # 8 corners in local frame
    corners_local = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=np.float64)
    
    # Transform to world
    corners_world = transform_points(T_world_from_local, corners_local)
    
    if swap_yz:
        corners_world = apply_swap_yz(corners_world)
    
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ], dtype=np.int32)
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls

# -------------------------
# Main
# -------------------------
def parse_args():
    raise RuntimeError("CLI parsing removed; set variables at top of file.")


def main():
    base_dir = BASE_DIR
    frame_id = FRAME_ID
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    fx = FX
    fy = FY
    cx = CX
    cy = CY
    min_depth = MIN_DEPTH
    max_depth = MAX_DEPTH
    png_depth_scale = (max_depth - min_depth) / float(PNG_MAX_VALUE)

    bbox_frame = BBOX_FRAME
    swap_yz = SWAP_YZ
    max_box_edge = MAX_BOX_EDGE
    ignore_prefixes = IGNORE_PRIM_PREFIXES

    traj_path = TRAJ_PATH if TRAJ_PATH is not None else os.path.join(base_dir, "traj.txt")

    rgb_path = os.path.join(base_dir, "rgb", f"frame{frame_id:06d}.jpg")
    depth_path = os.path.join(base_dir, "depth", f"depth{frame_id:06d}.png")
    bbox_path = os.path.join(base_dir, "bbox", f"bboxes{frame_id:06d}_info.json")

    if not os.path.exists(bbox_path):
        raise FileNotFoundError(bbox_path)

    depth_m = load_depth_meters(depth_path, png_depth_scale, min_depth)
    color_o3d = rgb_to_o3d(rgb_path)
    depth_o3d = depth_to_o3d(depth_m)

    intr = make_intrinsics(width, height, fx, fy, cx, cy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    if swap_yz:
        pts = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(apply_swap_yz(pts))

    boxes = load_boxes_json(bbox_path)

    T_world_from_cam = None
    if USE_TRAJ:
        T_world_from_cam = read_traj_pose(frame_id, traj_path)
        pcd.transform(T_world_from_cam)

    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    print("\n" + "="*60)
    print(f"DEBUG: Loaded {len(boxes)} boxes from JSON")
    print(f"DEBUG: BBOX_FRAME = '{bbox_frame}'")
    print(f"DEBUG: USE_TRAJ = {USE_TRAJ}")
    if T_world_from_cam is not None:
        print(f"DEBUG: Camera translation (from traj): {T_world_from_cam[:3, 3]}")
    print("="*60 + "\n")

    for b in boxes:
        prim = b.get("prim_path", "")

        aabb = b["aabb_xyzmin_xyzmax"]
        xmin, ymin, zmin, xmax, ymax, zmax = map(float, aabb)

        sx, sy, sz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
        if max(sx, sy, sz) > max_box_edge:
            print(f"SKIP (too large): '{prim}' size=({sx:.2f}, {sy:.2f}, {sz:.2f})")
            continue

        # Debug: print raw data from JSON
        T_raw = b.get("transform_4x4", None)
        print(f"\n--- Box: '{prim}' ---")
        print(f"  Raw AABB: [{xmin:.3f}, {ymin:.3f}, {zmin:.3f}, {xmax:.3f}, {ymax:.3f}, {zmax:.3f}]")
        print(f"  AABB center: ({(xmin+xmax)/2:.3f}, {(ymin+ymax)/2:.3f}, {(zmin+zmax)/2:.3f})")
        print(f"  AABB size: ({sx:.3f}, {sy:.3f}, {sz:.3f})")
        if T_raw is not None:
            T_arr = np.array(T_raw, dtype=np.float64).reshape(4, 4)
            print(f"  Transform translation: {T_arr[:3, 3]}")
        else:
            print(f"  Transform: NONE/MISSING")

        if bbox_frame == "world":
            # Assume boxes are already in world coords (e.g., from fixed data collection)
            ls = aabb_lineset_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax, swap_yz)
            print(f"  -> Using raw AABB as world coords")
        elif bbox_frame == "use_transform":
            # Use transform_4x4 from the JSON to convert local AABB to world
            T_local_to_world = np.array(b.get("transform_4x4", np.eye(4).tolist()), dtype=np.float64)
            if T_local_to_world.shape != (4, 4):
                T_local_to_world = T_local_to_world.reshape(4, 4)
            ls = local_aabb_to_world_lineset(aabb, T_local_to_world, swap_yz)
            # Compute transformed center
            center_local = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, 1.0])
            center_world = T_local_to_world @ center_local
            print(f"  -> Transformed center: ({center_world[0]:.3f}, {center_world[1]:.3f}, {center_world[2]:.3f})")
        else:  # camera
            if T_world_from_cam is None:
                raise ValueError("BBOX_FRAME='camera' needs T_world_from_cam (set --use_traj).")
            ls = camera_aabb_to_world_lineset(aabb, T_world_from_cam, swap_yz)
            print(f"  -> Transformed via camera pose")

        # Color the box green for visibility
        ls.paint_uniform_color([0, 1, 0])
        geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
