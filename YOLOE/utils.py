import numpy as np
import cv2
from pathlib import Path
from typing import List
from PIL import Image
import open3d as o3d
import os
import pickle
import json
from typing import Union, Tuple, List, Optional

from shapely import points

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 
from ultralytics import YOLOE
from utils import *
import os
import pickle
import time
import networkx as nx

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0  # meters
PNG_MAX_VALUE = 65535  # 16-bit depth image
FOCAL_LENGTH = 50
HORIZONTAL_APARTURE = 80
VERTICAL_APARTURE = 45

# Compute camera intrinsics
fx = FOCAL_LENGTH / HORIZONTAL_APARTURE * IMAGE_WIDTH   # fx = 50/80 * 1280 = 800.0
fy = FOCAL_LENGTH / VERTICAL_APARTURE * IMAGE_HEIGHT    # fy = 50/45 * 720 = 800.0
cx = IMAGE_WIDTH / 2.0                                  # cx = 640.0
cy = IMAGE_HEIGHT / 2.0   

DEPTH_PATHS = []
RGB_PATHS = []
TRACKER_CFG = "botsort.yaml"
DEVICE = "0"

def list_png_paths(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        print(f"DEBUG[utils.list_png_paths]: Folder {folder} does not exist or is not a directory.")
        return []

    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.png']

    # sort by path (natural/alphanumeric order if filenames are zero-padded)
    files_sorted = sorted(files)
    
    DEPTH_PATHS = [f.as_posix() for f in files_sorted]
    # print(f'DEBUG[utils.list_png_paths]: Found {len(DEPTH_PATHS)} PNG files in {folder}.')
    return DEPTH_PATHS

def list_jpg_paths(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        print(f"DEBUG[utils.list_jpg_paths]: Folder {folder} does not exist or is not a directory.")
        return []

    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.jpg']

    # sort by path (natural/alphanumeric order if filenames are zero-padded)
    files_sorted = sorted(files)
    
    RGB_PATHS = [f.as_posix() for f in files_sorted]
    return RGB_PATHS

def load_camera_poses(traj_path: str):
    poses = []
    if not os.path.isfile(traj_path):
        print(f"[WARN][prep_frames] traj file not found: {traj_path}")
        return poses
    with open(traj_path, 'r') as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            M = np.array(list(map(float, vals)), dtype=np.float64).reshape(4, 4)
            poses.append(M)
    if len(poses) == 0:
        print(f"[WARN][prep_frames] No valid poses found in traj file: {traj_path}")
    return poses

def get_pose(poses,frame_idx):
    T_w_c = None
    if len(poses) > 0:
        T_w_c = poses[min(frame_idx, len(poses)-1)]
        if T_w_c is None:
            print("[WARN][prep_frames] No poses loaded. Keeping points in camera frame.")
    else:
        print(f"[WARN][prep_frames] No pose found for frame {frame_idx}. Keeping points in camera frame.")
        T_w_c = None
    
    return T_w_c

def load_depth_as_meters(depth_path):
    # Loads a 16-bit PNG and converts to meters using the PNG_MAX_VALUE & MAX_DEPTH scale.
    # NOTE: This assumes the depth file is linearly scaled 0..PNG_MAX_VALUE -> 0..MAX_DEPTH.
    # If your depth is actually in millimeters or different scale, change this function accordingly.
    d = np.array(Image.open(depth_path))
    if d.dtype != np.uint16:
        # allow PNGs saved as 8-bit (rare for depth) - convert
        d = d.astype(np.uint16)
    depth_m = (d.astype(np.float32) / float(PNG_MAX_VALUE)) * float(MAX_DEPTH)
    # clamp
    depth_m[depth_m < MIN_DEPTH] = 0.0
    depth_m[depth_m > MAX_DEPTH] = 0.0
    return depth_m  # shape (H, W) float32, zeros mark invalid

def masks_to_binary_by_polygons(masks, orig_shape):
    """
    Prefer polygon-based conversion (xyn or xy). Returns list of HxW uint8 masks (0/255).
    masks: ultralytics.engine.results.Masks object
    orig_shape: (H, W)
    """
    H, W = orig_shape
    bin_masks = []
    
    # 2) Fallback: try masks.xy (pixel coords). Similar rasterization (no normalization)
    xy = getattr(masks, "xy", None)
    if xy is not None and len(xy) > 0:
        for poly in xy:
            if poly is None or len(poly) == 0:
                bin_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            all_polys = poly if isinstance(poly, (list, tuple)) else [poly]
            mask = np.zeros((H, W), dtype=np.uint8)
            for p in all_polys:
                pts = np.asarray(p, dtype=np.int32)
                # If pts appear as (N,2) float but already in pixel coords, clip & fill
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    pts[:, 0] = np.clip(pts[:, 0], 0, W-1)  # x
                    pts[:, 1] = np.clip(pts[:, 1], 0, H-1)  # y
                    cv2.fillPoly(mask, [pts], color=255)
            bin_masks.append(mask)
        return bin_masks

def track_objects_in_video_stream(rgb_dir_path, depth_path_list,
                                  model_path='yoloe-11l-seg-pf.pt',
                                  conf=0.3,
                                  iou=0.5):
    """Track objects in a video stream.
    Args:
        rgb_dir_path (_type_): _description_
        depth_dir_path (_type_): _description_
        model_path (str, optional): _description_. Defaults to 'yoloe-11l-seg-pf.pt'.
        conf (float, optional): _description_. Defaults to 0.3.
        iou (float, optional): _description_. Defaults to 0.5.

    Yields:
        _type_: _description_
    """
    print(f'DEBUG tracking init')
    rgb_paths = list_jpg_paths(rgb_dir_path)
    depth_paths = depth_path_list

    model = YOLOE(model_path)
    for ip, rgb_p in enumerate(rgb_paths):
        bgr = cv2.imread(rgb_p)
        if bgr is None:
            print(f"[WARN][prep_frames] Could not read image: {rgb_p}")
            continue
        
        out = model.track(
            source=[bgr],
            tracker=TRACKER_CFG,
            device=DEVICE,
            conf=conf,
            verbose=False,
            persist=True,
        )
         
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out        
        yield res, rgb_p, depth_paths[ip]

def preprocess_mask(yolo_res, index, KERNEL_SIZE, alpha = 0.5, show=True, fast: bool = False):
    import numpy as np
    import torch
    import cv2
    import matplotlib.pyplot as plt

    img = yolo_res.orig_img

    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()

    # ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    H_orig, W_orig = yolo_res.orig_shape  # (H, W)

    # --- prepare boxes info if present ---
    boxes = getattr(yolo_res, "boxes", None)
    N = 0
    if boxes is not None:
        # determine number of boxes in a few common ways
        if hasattr(boxes, "xyxy"):
            bx = boxes.xyxy
            try:
                N = int(len(bx))
            except Exception:
                N = 0
        elif hasattr(boxes, "data"):
            try:
                N = int(len(boxes.data))
            except Exception:
                N = 0

    # --- prepare masks if present ---
    masks = getattr(yolo_res, "masks", None)

    if masks is not None and getattr(masks, "data", None) is not None:
        bin_masks = masks_to_binary_by_polygons(masks, yolo_res.orig_shape)
        cleaned_masks = [morph_mask(mask, method='erode', kernel_size=KERNEL_SIZE, iterations=1) for mask in bin_masks]
    else:
        print(f"[WARN][prep_frames] No masks found at frame {index}.")
        bin_masks = [None] * N
        cleaned_masks = [None] * N

    if fast:
        # Skip all visualization work, return masks only
        return bin_masks, cleaned_masks

    # visualization of original mask and cleaned mask side by side with image
    if show and len(bin_masks) > 0:
        vis_img = img.copy()
        if vis_img.ndim == 2:
            vis_img = np.stack([vis_img]*3, axis=-1)
        elif vis_img.shape[2] == 4:
            vis_img = vis_img[..., :3]

        def random_color():
            return tuple(np.random.randint(0, 255, size=3).tolist())

        def deterministic_color_from_id(tid):
            # stable color per integer id
            tid_i = abs(hash(str(tid))) % (2**31)
            
            v = (tid_i * 123457) % 0xFFFFFF
            r = (v >> 2) & 255
            g = (v >> 4) & 255
            b = (v) & 255
            
            return (int(r), int(g), int(b))

        def overlay(image, masks_list, colors, alpha=0.5):
            out = image.copy().astype(np.float32)
            for m, color in zip(masks_list, colors):
                if m is None:
                    continue
                mask = (m > 0).astype(np.uint8)
                color_arr = np.zeros_like(out)
                color_arr[..., 0] = color[0]
                color_arr[..., 1] = color[1]
                color_arr[..., 2] = color[2]
                mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
                out[mask_3c == 1] = (1 - alpha) * out[mask_3c == 1] + alpha * color_arr[mask_3c == 1]
            return np.clip(out / 255.0, 0, 1)

        def get_color_label(boxes_obj):
            if boxes_obj is None:
                print("[WARN][preprocess_mask] boxes object is None.")
                return (0, 0, 0)

            if hasattr(boxes_obj, "xyxy"):
                xyxy = boxes_obj.xyxy
            else:
                print("[WARN][preprocess_mask] boxes object has no xyxy attribute.")
            
            # ensure numpy
            xyxy = xyxy.cpu().numpy()
            
            conf = getattr(boxes_obj, "conf", None)
            ids = getattr(boxes_obj, "id", None)
            
            colors = []
            labels = []
            bboxes = []
            for tid, conf_val, xyxy_val in zip(ids, conf, xyxy):
                tid = int(tid)
                color = deterministic_color_from_id(tid)
                label = f"{tid}: {conf_val:.1f}"
                colors.append(color)
                labels.append(label)
                bboxes.append(xyxy_val)

            return colors, labels, bboxes

        # draw boxes helper (works on uint8 images)
        def draw_boxes_on_uint8(img_uint8, boxes_obj, colors, labels, xyxys):
            
            im = img_uint8.copy()

            for color, label, xyxy in zip(colors, labels, xyxys):
                
                try:
                    x1, y1, x2, y2 = [int(round(float(x))) for x in xyxy[:4]]
                except Exception:
                    continue
                # clip
                x1 = max(0, min(W_orig-1, x1))
                x2 = max(0, min(W_orig-1, x2))
                y1 = max(0, min(H_orig-1, y1))
                y2 = max(0, min(H_orig-1, y2))

                # draw rectangle
                cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=2)
                if label:
                    # put label above box if possible
                    ((w_text, h_text), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y_text = y1 - 6 if (y1 - 6) > h_text else y1 + h_text + 6
                    cv2.rectangle(im, (x1, y_text - h_text - 4), (x1 + w_text + 4, y_text + 2), color, -1)
                    cv2.putText(im, label, (x1 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1, lineType=cv2.LINE_AA)

            return im

        # print(f"boxes: {boxes}, yolo: {yolo_res.boxes}")
        # print(f"boxes: {boxes.conf}, yolo: {yolo_res.boxes.conf}")
        # print(f"boxes: {boxes.id}, yolo: {yolo_res.boxes.id}")
        
        colors, labels, xyxys = get_color_label(boxes)

        orig_overlay = overlay(vis_img, bin_masks, colors=colors, alpha=alpha)
        cln_overlay = overlay(vis_img, cleaned_masks, colors=colors, alpha=alpha)

        # Convert overlays to uint8 for drawing boxes
        orig_u8 = (orig_overlay * 255).astype(np.uint8)
        cln_u8 = (cln_overlay * 255).astype(np.uint8)

        # draw boxes on both overlays
        orig_u8_with_boxes = draw_boxes_on_uint8(orig_u8, boxes, colors, labels, xyxys)
        cln_u8_with_boxes = draw_boxes_on_uint8(cln_u8, boxes, colors, labels, xyxys)

        # Convert back to floats in [0,1] for matplotlib
        orig_im_show = orig_u8_with_boxes.astype(np.float32) / 255.0
        cln_im_show = cln_u8_with_boxes.astype(np.float32) / 255.0

        # Compute total mask coverage (safe: ensure non-empty stack)
        try:
            orig_union = np.any(np.stack([(m > 0) for m in bin_masks]), axis=0)
            cln_union = np.any(np.stack([(m > 0) for m in cleaned_masks]), axis=0)
            orig_count = int(np.sum(orig_union))
            cln_count = int(np.sum(cln_union))
        except Exception:
            orig_count = cln_count = 0

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Frame {index} — All Masks (boxes shown when available)", fontsize=15)

        axes[0].imshow(orig_im_show)
        axes[0].set_title(f"Original masks (pixels covered: {orig_count})")
        axes[0].axis('off')

        axes[1].imshow(cln_im_show)
        axes[1].set_title(f"Cleaned masks erosion kernel {KERNEL_SIZE} (pixels covered: {cln_count})")
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return bin_masks, cleaned_masks


def extract_points_from_mask( depth_m: np.ndarray,
    mask,
    frame_idx,
    max_points = None,
    random_state = None):
    """
    Convert depth + mask -> 3D points in camera coordinates.

    Args:
        depth_m: (H, W) float32 depth in meters. Invalid pixels should be 0.
        mask: (H, W) binary mask (0/255 or bool) OR an iterable/list/array of masks shape (N, H, W).
        max_points: optionally cap the number of output points. If mask yields >max_points, random sampling is applied.
        random_state: int seed for deterministic sampling (if max_points used).
        return_pixels: if True, also return (u,v) pixel coordinates for each 3D point.

    Returns:
        If input mask is single (H,W):
            points: (M, 3) ndarray float32 in camera coords (X,Y,Z).
            (optionally) pixels: (M, 2) ints (u, v) — column,row
        If input mask is stack/list of masks:
            list_of_points: [ (M_i,3) numpy arrays ... ]
            (optionally) list_of_pixels: [ (M_i,2) arrays ... ]

    Notes:
        - Pixel coordinates convention: u = x (column), v = y (row).
        - Unprojection formula:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth_m[v, u]
    """
    # TO-DO: later look on removing outliers from point cloud
    
    rng = np.random.default_rng(random_state)

    results_points = []

    # normalize mask to boolean
    if mask.dtype == np.uint8 or mask.dtype == np.int32 or mask.dtype == np.int64:
        mask_bool = (mask > 0)

    mask_bool = np.squeeze(mask_bool)
    
    # get valid pixels where mask==True and depth>0
    valid_mask = mask_bool & (depth_m > 0)
    if not np.any(valid_mask):
        # empty
        pts = np.zeros((0,3), dtype=np.float32)
        results_points = pts

    # get pixel indices (v, u)
    vs, us = np.nonzero(valid_mask)  # arrays of equal length M
    zs = depth_m[vs, us].astype(np.float32)  # shape (M,)

    # optional sampling to cap points
    M = zs.shape[0]
    if max_points is not None and M > int(max_points):
        idx = rng.choice(M, size=int(max_points), replace=False)
        us = us[idx]
        vs = vs[idx]
        zs = zs[idx]
        M = zs.shape[0]
    else:
        # keep 70% of points
        idx = rng.choice(M, size=int(M * 0.7), replace=False)
        us = us[idx]
        vs = vs[idx]
        zs = zs[idx]

    # unproject
    us_f = us.astype(np.float32)
    vs_f = vs.astype(np.float32)
    X = (us_f - cx) * zs / fx
    Y = (vs_f - cy) * zs / fy
    Z = zs

    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (M,3)

    if pts is None or pts.size == 0:
        print(f"[utils.extract_points_from_mask] Warning: no points extracted from mask in frame {frame_idx},\
              {frame_idx} Extracted {pts.shape[0]}.")
    return pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Clean with statistical outlier removal
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.7)
    # pcd = pcd.select_by_index(ind)

    cleaned_pts_np = np.asarray(pcd.points)
    results_points = cleaned_pts_np
    
    return results_points

def cam_to_world(points_cam: np.ndarray, T_w_c: np.ndarray):
    
    # TO-DO: veryfy correctness of transformations
    
    """Apply camera->world transform to Nx3 points."""
    if points_cam is None or points_cam.size == 0:
        print("[utils.cam_to_world] Warning: empty points_cam input.")
        return points_cam
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3]
    
    ret_pts = (points_cam @ R.T) + t
    
    return ret_pts

def estimate_scene_height(depth_m: np.ndarray, T_w_c: np.ndarray, stride: int = 1):
    """
    Estimate world-frame scene height from a single depth image by:
    1) sampling depth pixels on a grid;
    2) unprojecting to camera frame using intrinsics (fx, fy, cx, cy);
    3) transforming to world frame via T_w_c;
    4) returning min(Z), max(Z), and height = max - min in world coordinates.

    Args:
        depth_m: (H,W) depth map in meters, 0 indicates invalid.
        T_w_c: 4x4 camera->world transform.
        stride: subsampling step to reduce compute (default 8).

    Returns:
        (z_min, z_max, height). If no valid points, returns (nan, nan, 0.0).
    """
    if depth_m is None or depth_m.size == 0:
        return float('nan'), float('nan'), 0.0
    if T_w_c is None or not isinstance(T_w_c, np.ndarray) or T_w_c.shape != (4, 4):
        return float('nan'), float('nan'), 0.0

    H, W = depth_m.shape[:2]
    vv = np.arange(0, H, max(1, int(stride)), dtype=np.int32)
    uu = np.arange(0, W, max(1, int(stride)), dtype=np.int32)
    V, U = np.meshgrid(vv, uu, indexing='ij')
    D = depth_m[V, U].astype(np.float32)
    valid = D > 0
    if not np.any(valid):
        return float('nan'), float('nan'), 0.0

    us = U[valid].astype(np.float32)
    vs = V[valid].astype(np.float32)
    zs = D[valid]

    X = (us - cx) * zs / fx
    Y = (vs - cy) * zs / fy
    Z = zs
    pts_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

    pts_w = cam_to_world(pts_cam, T_w_c)
    if pts_w is None or pts_w.size == 0:
        return float('nan'), float('nan'), 0.0

    z_vals = pts_w[:, 2]
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    
    x_min = float(np.min(pts_w[:, 0]))
    x_max = float(np.max(pts_w[:, 0]))

    y_min = float(np.min(pts_w[:, 1]))
    y_max = float(np.max(pts_w[:, 1]))
    
    height = max(0.0, z_max - z_min)
    # return x_min, x_max, y_min, y_max, z_min, z_max, height
    return height

def compute_3d_bboxes(points, fast_mode: bool = True):
    
    if points.shape[0] == 0:
        return {
            'aabb': None,
            'obb': None
        }
    
    if fast_mode:
        # Fast AABB only
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        obb_center = (min_vals + max_vals) / 2.0
        obb_extent = (max_vals - min_vals)

        return {
            'aabb': {'min': min_vals, 'max': max_vals},
            'obb': {'center': obb_center, 'extent': obb_extent, 'R': np.eye(3)}
        }
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
                
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    
    aabb_min = np.asarray(aabb.get_min_bound())
    aabb_max = np.asarray(aabb.get_max_bound())
    obb_center = np.asarray(obb.center).tolist()
    obb_extent = np.asarray(obb.extent).tolist()
    obb_R = np.asarray(obb.R).tolist()
    
    return {
        'aabb': {'min': aabb_min, 'max': aabb_max},
        'obb': {'center': obb_center, 'extent': obb_extent, 'R': obb_R}
    }

def create_3d_objects(track_ids, masks_clean, max_points_per_obj, depth_m, T_w_c, frame_idx):
    frame_objs = []
    graph = nx.DiGraph()
    for t_id, m_clean in zip(track_ids, masks_clean):
        if m_clean is None:
            continue
        
        # 1. extract points from mask in camera frame (cap points)
        points_cam = extract_points_from_mask(depth_m, m_clean,frame_idx=frame_idx, max_points=max_points_per_obj, random_state=int(t_id))
        
        # 2. transform to world frame if pose available
        if points_cam is None:
            print("[WARN][prep_frames] No points extracted from mask. Skipping object.")
            points_world = None
            pts_for_bbox = points_cam
            continue
        elif points_cam.size <= 0:
            print("[WARN][prep_frames] No valid points extracted from mask. Skipping object.")
            points_world = None
            pts_for_bbox = points_cam
            continue
        elif T_w_c is None:
            print("[WARN][prep_frames] No poses loaded. Keeping points in camera frame.")
            points_world = None
            pts_for_bbox = points_cam
        else:
            points_world = cam_to_world(points_cam, T_w_c)
            pts_for_bbox = points_world

        # 3. compute 3d bbox (in world if transformed, else in camera)
        bbox3d = compute_3d_bboxes(pts_for_bbox)
        
        
        obj = {
            'track_id': int(t_id),
            'points': pts_for_bbox,
            'bbox_3d': bbox3d
        }
        
        graph.add_node(int(t_id), data=obj)
        # graph.add_node(obj)
        frame_objs.append(obj)
            
    return frame_objs, graph

def _yoloe_utils__generate_color(track_id: int):
    import random
    if track_id == -1:
        return [0.6, 0.6, 0.6]
    random.seed(int(track_id))
    return [random.random(), random.random(), random.random()]

def _yoloe_utils__create_aabb_lineset(aabb_data):
    mn = aabb_data.get('min') if isinstance(aabb_data, dict) else None
    mx = aabb_data.get('max') if isinstance(aabb_data, dict) else None
    if mn is None or mx is None:
        return None
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls

def _yoloe_utils__create_obb_lineset(obb_data):
    center = np.asarray(obb_data.get('center'), dtype=np.float64)
    extent = np.asarray(obb_data.get('extent'), dtype=np.float64)
    R = np.asarray(obb_data.get('R'), dtype=np.float64)
    corners_local = np.array([
        [-extent[0]/2, -extent[1]/2, -extent[2]/2],
        [+extent[0]/2, -extent[1]/2, -extent[2]/2],
        [+extent[0]/2, +extent[1]/2, -extent[2]/2],
        [-extent[0]/2, +extent[1]/2, -extent[2]/2],
        [-extent[0]/2, -extent[1]/2, +extent[2]/2],
        [+extent[0]/2, -extent[1]/2, +extent[2]/2],
        [+extent[0]/2, +extent[1]/2, +extent[2]/2],
        [-extent[0]/2, +extent[1]/2, +extent[2]/2],
    ], dtype=np.float64)
    corners_world = (R @ corners_local.T).T + center
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls

def visualize_frame_objects_open3d(
    frame_objs: list,
    frame_index: int,
    show_points: bool = True,
    show_aabb: bool = True,
    show_obb: bool = False,
    add_labels: bool = True,
    width: int = 1280,
    height: int = 720,
    point_size: float = 2.0,
    line_width: float = 2.0,
):
    """
    Visualize per-object point clouds and 3D boxes using Open3D.

    Each element in frame_objs is expected to be a dict with keys:
      - 'track_id': int
      - 'points': (N,3) numpy array (in world or camera)
      - 'bbox_3d': { 'aabb': {'min': [..], 'max': [..]}, 'obb': {'center':[..], 'extent':[..], 'R':[[..],[..],[..]]} }

    - AABB is drawn in red.
    - OBB is drawn in green.
    - Points are colored by track id.
    - Labels (track IDs) are added as small spheres at OBB/AABB center; if 3D text mesh is available, it's also added.
    """
    if not isinstance(frame_objs, (list, tuple)) or len(frame_objs) == 0:
        print("[visualize_frame_objects_open3d] Nothing to visualize.")
        return

    # Determine global bounds to scale spheres/frames
    gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for obj in frame_objs:
        bbox = obj.get('bbox_3d') or {}
        aabb = bbox.get('aabb') if isinstance(bbox, dict) else None
        pts = obj.get('points', None)
        if aabb and 'min' in aabb and 'max' in aabb:
            mn = np.asarray(aabb['min'], dtype=np.float64)
            mx = np.asarray(aabb['max'], dtype=np.float64)
            gmin = np.minimum(gmin, mn)
            gmax = np.maximum(gmax, mx)
        elif isinstance(pts, np.ndarray) and pts.size >= 3:
            gmin = np.minimum(gmin, pts.min(axis=0))
            gmax = np.maximum(gmax, pts.max(axis=0))
    if not np.all(np.isfinite(gmin)) or not np.all(np.isfinite(gmax)):
        gmin = np.array([-1,-1,-1], dtype=np.float64)
        gmax = np.array([ 1, 1, 1], dtype=np.float64)
    diag = float(np.linalg.norm(gmax - gmin)) if np.all(np.isfinite(gmax-gmin)) else 1.0
    sphere_r = max(1e-3, 0.01 * diag)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame Objects (Open3D) - Frame {frame_index}", width=width, height=height, visible=True)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = float(point_size)
    try:
        opt.line_width = float(line_width)
    except Exception:
        pass

    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * diag, origin=[0, 0, 0])
    vis.add_geometry(coord)

    # Geometries accumulator for potential cleanup
    geoms = [coord]

    for obj in frame_objs:
        tid = int(obj.get('track_id', -1))
        col = _yoloe_utils__generate_color(tid)
        pts = obj.get('points', None)
        if show_points and isinstance(pts, np.ndarray) and pts.size >= 3:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color(col)
            vis.add_geometry(pcd)
            geoms.append(pcd)
        bbox = obj.get('bbox_3d', {}) or {}
        aabb = bbox.get('aabb')
        obb = bbox.get('obb')
        label_pos = None
        if show_aabb and aabb is not None:
            ls = _yoloe_utils__create_aabb_lineset(aabb)
            if ls is not None:
                ls.paint_uniform_color([1.0, 0.0, 0.0])  # red
                vis.add_geometry(ls)
                geoms.append(ls)
                # center from aabb
                mn = np.asarray(aabb['min'], dtype=float)
                mx = np.asarray(aabb['max'], dtype=float)
                label_pos = (mn + mx) / 2.0
        if show_obb and obb is not None:
            ls2 = _yoloe_utils__create_obb_lineset(obb)
            if ls2 is not None:
                ls2.paint_uniform_color([0.0, 1.0, 0.0])  # green
                vis.add_geometry(ls2)
                geoms.append(ls2)
                # prefer OBB center for label
                label_pos = np.asarray(obb.get('center'), dtype=float)

    # Fit view to global bounds once
    gbox = o3d.geometry.AxisAlignedBoundingBox(gmin, gmax)
    vis.add_geometry(gbox)
    vis.poll_events(); vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.set_lookat(gbox.get_center())
    ctr.set_front([-1.0, -1.0, -1.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(1.0)
    vis.remove_geometry(gbox, reset_bounding_box=False)

    print("[visualize_frame_objects_open3d] Controls: mouse rotate, wheel zoom, right-button pan, R reset, Q quit")
    vis.run()
    vis.destroy_window()

def get_track_ids(yolo_res):
    boxes = getattr(yolo_res, "boxes", None)
    if boxes is None:
        return []
    track_ids = yolo_res.boxes.id.detach().cpu().numpy().astype(np.int64)
    if track_ids is None:
        print("[utils.get_track_ids] Warning: No track IDs found.")
        return []
    return track_ids

