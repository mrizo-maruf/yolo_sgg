"""
Visualize IsaacSim Replicator data with Open3D (3D) and Matplotlib (2D).

Features:
- 3D visualization: Point cloud + 3D bounding boxes in Open3D
- 2D visualization: RGB + semantic masks + 2D bboxes + track IDs + semantic IDs in Matplotlib

Assumptions:
- You saved per-frame:
    rgb/frameXXXXXX.jpg
    depth/depthXXXXXX.png   (uint16, mapped linearly from [MIN_DEPTH, MAX_DEPTH])
    bbox/bboxesXXXXXX_info.json (contains both 2D and 3D bboxes)
    seg/semanticXXXXXX.png + semanticXXXXXX_info.json (semantic segmentation)
    traj.txt                (optional; currently ROS camera pose flattened 4x4 per frame)

- Your JSON format (as in your code):
  boxes["bboxes"]["bbox_3d"]["boxes"] is a list of dicts with:
    - aabb_xyzmin_xyzmax: [x_min,y_min,z_min,x_max,y_max,z_max]
    - transform_4x4: 4x4 (optional usage)
    - prim_path, label, track_id, semantic_id, etc.
  boxes["bboxes"]["bbox_2d_tight"]["boxes"] (or bbox_2d_loose/bbox_2d) with:
    - xyxy: [x1, y1, x2, y2]
    - track_id, semantic_id, label, prim_path, etc.

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb

# -------------------------
# -------------------------
# User-configurable settings
# -------------------------
scene = "scene_7"
BASE_DIR = f"/home/maribjonov_mr/IsaacSim_bench/{scene}"  # folder with rgb/depth/bbox
TRAJ_PATH = f"/home/maribjonov_mr/IsaacSim_bench/{scene}/traj.txt"  # None -> BASE_DIR/traj.txt
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

# Visualization tweaks
SWAP_YZ = False  # try True if axes look rotated
MAX_BOX_EDGE = 20.0  # meters; filter very large boxes
# Prim paths to ignore (be specific - e.g. "/World/env/ground" not "/World/env")
# Set to empty list [] to show all objects
IGNORE_PRIM_PREFIXES = []  # Was ["/World/env"] which filtered everything

# 2D Visualization settings (matplotlib)
SHOW_2D_VIS = True  # Show matplotlib 2D visualization
SHOW_SEMANTIC_MASK = True  # Show semantic segmentation masks
SHOW_2D_BBOX = True  # Show 2D bounding boxes
SHOW_TRACK_ID = True  # Show track IDs on boxes
SHOW_SEMANTIC_ID = True  # Show semantic IDs on boxes
MASK_ALPHA = 0.4  # Transparency of semantic masks
SHOW_3D_VIS = True  # Show Open3D 3D visualization

# Labels to skip when extracting masks (e.g., structural elements)
SKIP_LABELS = {
    'wall', 'floor', 'ground', 'ceiling', 'roof',
    'background', 'unlabeled', 'unknown'
}


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


def load_full_bbox_json(json_path: str):
    """Load the full bbox JSON including 2D and 3D bboxes."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_2d_bboxes(json_path: str):
    """
    Load 2D bounding boxes from bbox JSON.
    
    Returns:
        List of dicts with keys: xyxy, track_id, semantic_id, label, prim_path
    """
    data = load_full_bbox_json(json_path)
    bboxes_2d = []
    
    if "bboxes" not in data:
        return bboxes_2d
    
    # Try different 2D bbox keys
    for key in ["bbox_2d_tight", "bbox_2d_loose", "bbox_2d"]:
        if key in data["bboxes"]:
            for box in data["bboxes"][key].get("boxes", []):
                bbox_info = {
                    "xyxy": box.get("xyxy", [0, 0, 0, 0]),
                    "track_id": box.get("bbox_id", box.get("track_id", -1)),
                    "semantic_id": box.get("semantic_id", -1),
                    "label": extract_label(box.get("label", {})),
                    "prim_path": box.get("prim_path", ""),
                    "occlusion": box.get("visibility_or_occlusion", box.get("occlusion", 0.0))
                }
                bboxes_2d.append(bbox_info)
            break
    
    return bboxes_2d


def extract_label(label_data):
    """Extract label string from various label formats."""
    if isinstance(label_data, str):
        return label_data
    if isinstance(label_data, dict):
        # Try common keys
        for key in ["class", "name", "label"]:
            if key in label_data:
                return str(label_data[key])
        # Fallback: return first value
        for k, v in label_data.items():
            return str(v) if v else str(k)
    return "unknown"


def load_semantic_mask(seg_dir: str, frame_id: int):
    """
    Load semantic segmentation mask and info.
    
    Returns:
        seg_map: (H, W, 3) BGR color-coded segmentation
        seg_info: Dict mapping semantic_id -> {class, color_bgr}
    """
    seg_png_path = os.path.join(seg_dir, f"semantic{frame_id:06d}.png")
    seg_json_path = os.path.join(seg_dir, f"semantic{frame_id:06d}_info.json")
    
    seg_map = None
    seg_info = {}
    
    if os.path.exists(seg_png_path):
        seg_map = cv2.imread(seg_png_path, cv2.IMREAD_COLOR)
        if seg_map is not None:
            seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    if os.path.exists(seg_json_path):
        with open(seg_json_path, "r") as f:
            raw_info = json.load(f)
            for sem_id_str, info in raw_info.items():
                try:
                    sem_id = int(sem_id_str)
                except ValueError:
                    continue
                
                label_data = info.get("label", {})
                if isinstance(label_data, str):
                    cls_name = label_data
                elif isinstance(label_data, dict):
                    cls_name = extract_label(label_data)
                else:
                    cls_name = str(label_data) if label_data else "unknown"
                
                seg_info[sem_id] = {
                    "class": cls_name,
                    "color_bgr": tuple(info.get("color_bgr", [0, 0, 0]))
                }
    
    return seg_map, seg_info


def generate_color_for_id(id_val: int):
    """Generate a consistent color for a given ID using golden ratio."""
    golden_ratio = 0.618033988749895
    hue = (id_val * golden_ratio) % 1.0
    saturation = 0.7 + (id_val % 3) * 0.1
    value = 0.8 + (id_val % 2) * 0.1
    rgb = hsv_to_rgb([hue, saturation, value])
    return rgb


def extract_mask_for_bbox(seg_map, bbox_2d, seg_info, skip_labels=None):
    """
    Extract a binary mask for a specific bbox based on dominant color in the bbox region.
    
    This uses the 2D bbox region to find the object's color, which is more reliable
    than relying on semantic_id (which may differ between segmentation and bbox annotators).
    
    Args:
        seg_map: Semantic segmentation map (H, W, 3) RGB
        bbox_2d: Dict with 'xyxy' key containing [x1, y1, x2, y2]
        seg_info: Dict mapping semantic_id -> {class, color_bgr}
        skip_labels: Optional set of labels to skip (e.g., {'wall', 'floor'})
    
    Returns:
        Binary mask (H, W) with True where the object is, or None if extraction fails
    """
    if seg_map is None:
        return None
    
    if skip_labels is None:
        skip_labels = set()
    
    H, W = seg_map.shape[:2]
    xyxy = bbox_2d.get("xyxy")
    if xyxy is None:
        return None
    
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Build set of colors to skip (colors belonging to wall, floor, etc.)
    skipped_colors = set()
    for sem_id, info in seg_info.items():
        cls_name = info.get('class', '')
        for skip_label in skip_labels:
            if skip_label in cls_name.lower():
                color = info.get('color_bgr')
                if color and tuple(color) != (0, 0, 0):
                    # Convert BGR to RGB for comparison with seg_map
                    skipped_colors.add((color[2], color[1], color[0]))
                break
    
    # Get the region inside the bbox
    region = seg_map[y1:y2, x1:x2]
    
    # Find unique colors and their counts
    region_flat = region.reshape(-1, 3)
    unique_colors, counts = np.unique(region_flat, axis=0, return_counts=True)
    
    # Find dominant color (excluding black and skipped colors)
    best_color = None
    best_count = 0
    
    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(color.tolist())
        # Skip black/very dark (background)
        if all(c < 10 for c in color_tuple):
            continue
        # Skip colors belonging to filtered classes
        if color_tuple in skipped_colors:
            continue
        if count > best_count:
            best_count = count
            best_color = color
    
    if best_color is None:
        return None
    
    # Create full-image mask for pixels matching dominant color
    mask = np.all(seg_map == best_color, axis=2)
    
    return mask


def visualize_2d_matplotlib(
    rgb_path: str,
    bboxes_2d: list,
    seg_map: np.ndarray,
    seg_info: dict,
    frame_id: int,
    show_mask: bool = True,
    show_bbox: bool = True,
    show_track_id: bool = True,
    show_semantic_id: bool = True,
    mask_alpha: float = 0.4,
    ignore_prefixes: list = None,
    skip_labels: set = None
):
    """
    Visualize RGB image with semantic masks, 2D bboxes, track IDs, and semantic IDs using matplotlib.
    
    Args:
        rgb_path: Path to RGB image
        bboxes_2d: List of 2D bbox dicts
        seg_map: Semantic segmentation map (H, W, 3) RGB
        seg_info: Dict mapping semantic_id -> {class, color_bgr}
        frame_id: Frame ID for title
        show_mask: Show semantic masks
        show_bbox: Show 2D bounding boxes
        show_track_id: Show track ID labels
        show_semantic_id: Show semantic ID labels
        mask_alpha: Transparency for mask overlay
        ignore_prefixes: List of prim_path prefixes to ignore
        skip_labels: Set of labels to skip when extracting masks (e.g., {'wall', 'floor'})
    """
    if ignore_prefixes is None:
        ignore_prefixes = []
    if skip_labels is None:
        skip_labels = set()
    
    # Load RGB image
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        print(f"Warning: Could not load RGB image: {rgb_path}")
        return
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    H, W = rgb.shape[:2]
    
    # Create figure with subplots
    if show_mask and seg_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_rgb = axes[0]
        ax_mask = axes[1]
        ax_overlay = axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_rgb = axes[0]
        ax_overlay = axes[1]
        ax_mask = None
    
    # Panel 1: Raw RGB
    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f"RGB - Frame {frame_id}")
    ax_rgb.axis("off")
    
    # Panel 2: Semantic Mask (if available)
    if ax_mask is not None and seg_map is not None:
        ax_mask.imshow(seg_map)
        ax_mask.set_title("Semantic Segmentation")
        ax_mask.axis("off")
    
    # Panel 3: RGB + Masks + 2D Bboxes + Labels
    overlay_img = rgb.copy().astype(np.float32)
    mask_overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    # Filter boxes
    filtered_bboxes = []
    skipped_count = 0
    for bbox in bboxes_2d:
        prim = bbox.get("prim_path", "")
        skip = False
        for prefix in ignore_prefixes:
            if prim.startswith(prefix):
                skip = True
                break
        if skip:
            skipped_count += 1
        else:
            filtered_bboxes.append(bbox)
    
    print(f"[2D VIS] Total boxes: {len(bboxes_2d)}, Kept: {len(filtered_bboxes)}, Skipped: {skipped_count}")
    if len(filtered_bboxes) == 0 and len(bboxes_2d) > 0:
        print(f"[WARNING] All boxes filtered out! Check IGNORE_PRIM_PREFIXES: {ignore_prefixes}")
    
    # Draw masks
    if show_mask and seg_map is not None:
        for bbox in filtered_bboxes:
            mask = extract_mask_for_bbox(seg_map, bbox, seg_info, skip_labels)
            if mask is not None:
                color = generate_color_for_id(bbox["track_id"])
                mask_overlay[mask] = color
        
        # Blend mask with RGB
        overlay_img = overlay_img * (1 - mask_alpha) + mask_overlay * mask_alpha * 255
    
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    ax_overlay.imshow(overlay_img)
    
    # Draw 2D bboxes and labels
    for bbox in filtered_bboxes:
        x1, y1, x2, y2 = bbox["xyxy"]
        track_id = bbox["track_id"]
        semantic_id = bbox["semantic_id"]
        label = bbox["label"]
        
        color = generate_color_for_id(track_id)
        
        if show_bbox:
            # Draw bbox rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax_overlay.add_patch(rect)
        
        # Build label text
        label_parts = []
        if show_track_id:
            label_parts.append(f"T:{track_id}")
        if show_semantic_id:
            label_parts.append(f"S:{semantic_id}")
        label_parts.append(label[:15] if len(label) > 15 else label)  # Truncate long labels
        
        label_text = " | ".join(label_parts)
        
        # Draw label background and text
        ax_overlay.text(
            x1 + 2, y1 - 5,
            label_text,
            fontsize=7,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
        )
    
    title = f"Frame {frame_id} - RGB + "
    parts = []
    if show_mask:
        parts.append("Masks")
    if show_bbox:
        parts.append("2D BBoxes")
    if show_track_id:
        parts.append("Track IDs")
    if show_semantic_id:
        parts.append("Semantic IDs")
    title += " + ".join(parts)
    title += f" ({len(filtered_bboxes)} objects)"
    
    ax_overlay.set_title(title)
    ax_overlay.axis("off")
    
    plt.tight_layout()
    plt.show()

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
    seg_dir = os.path.join(base_dir, "seg")

    if not os.path.exists(bbox_path):
        raise FileNotFoundError(bbox_path)

    # -------------------------
    # 2D Visualization (Matplotlib)
    # -------------------------
    if SHOW_2D_VIS:
        print("\n" + "="*60)
        print("2D VISUALIZATION (Matplotlib)")
        print("="*60)
        
        # Load 2D bboxes
        bboxes_2d = load_2d_bboxes(bbox_path)
        print(f"Loaded {len(bboxes_2d)} 2D bounding boxes")
        
        # Load semantic mask
        seg_map, seg_info = load_semantic_mask(seg_dir, frame_id)
        if seg_map is not None:
            print(f"Loaded semantic mask: {seg_map.shape}")
            print(f"Semantic info: {len(seg_info)} classes")
        else:
            print("No semantic mask found")
        
        # Show 2D visualization
        visualize_2d_matplotlib(
            rgb_path=rgb_path,
            bboxes_2d=bboxes_2d,
            seg_map=seg_map,
            seg_info=seg_info,
            frame_id=frame_id,
            show_mask=SHOW_SEMANTIC_MASK,
            show_bbox=SHOW_2D_BBOX,
            show_track_id=SHOW_TRACK_ID,
            show_semantic_id=SHOW_SEMANTIC_ID,
            mask_alpha=MASK_ALPHA,
            ignore_prefixes=ignore_prefixes,
            skip_labels=SKIP_LABELS
        )

    # -------------------------
    # 3D Visualization (Open3D)
    # -------------------------
    if SHOW_3D_VIS:
        print("\n" + "="*60)
        print("3D VISUALIZATION (Open3D)")
        print("="*60)
        
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
