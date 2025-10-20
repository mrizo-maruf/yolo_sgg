import numpy as np
import cv2
from pathlib import Path
from typing import List
from PIL import Image
import open3d as o3d
import os
import pickle

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
cy = IMAGE_HEIGHT / 2.0                                 # cy = 360.0

# Helper: load depth PNG (16-bit) -> depth in meters
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

    # 1) Try masks.xyn (normalized polygons) — best option
    xyn = getattr(masks, "xyn", None)
    if xyn is not None and len(xyn) > 0:
        # xyn is typically a list-like of arrays, each array Nx2 with values in [0..1]
        for poly_norm in xyn:
            # handle empty
            if poly_norm is None or len(poly_norm) == 0:
                bin_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            # poly_norm may be nested (list of polygons) or a single polygon
            # ensure we iterate over each sub-polygon
            all_polys = poly_norm if isinstance(poly_norm, (list, tuple)) else [poly_norm]
            mask = np.zeros((H, W), dtype=np.uint8)
            for p in all_polys:
                arr = np.asarray(p, dtype=np.float32)
                if arr.size == 0:
                    continue
                # arr shape (N,2) with [x_norm, y_norm] or maybe [y,x] in some edge cases -> we assume [x,y]
                pts = np.round(arr * np.array([W, H])).astype(np.int32)  # scale to pixels
                # cv2.fillPoly expects list of arrays shaped (N,1,2) or (N,2)
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], color=255)
            bin_masks.append(mask)
        return bin_masks

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

    # 3) Last resort: use masks.data but un-letterbox it correctly.
    # This may still be off if you don't have the exact letterbox parameters.
    # We'll try a reasonable approach: resize mask to orig_shape with nearest neighbor (same as before)
    mask_tensor = getattr(masks, "data", None)
    if mask_tensor is None:
        return []

    mask_np = mask_tensor.detach().cpu().numpy()  # shape (N, h_mask, w_mask)
    for m in mask_np:
        m_bin = (m >= 0.5).astype(np.uint8) * 255
        # resize — this is imperfect if letterbox is present, but it's the last resort
        m_resized = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        bin_masks.append(m_resized)
    return bin_masks

def morph_mask(mask, method='erode', kernel_size=3, iterations=1):
    """
    Apply morphological operation to a binary mask (0/255 uint8).
    Args:
        mask: HxW uint8 binary mask (0 or 255).
        method: 'erode', 'dilate', 'open', 'close'.
        kernel_size: odd int >= 1 (3 is typical).
        iterations: int; number of times to apply (for erode/dilate).
    Returns:
        new_mask: HxW uint8 binary mask.
    """
    assert method in ('erode', 'dilate', 'open', 'close')
    if kernel_size <= 1:
        return mask.copy()
    # ensure odd kernel
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if method == 'erode':
        out = cv2.erode(mask, kernel, iterations=iterations)
    elif method == 'dilate':
        out = cv2.dilate(mask, kernel, iterations=iterations)
    elif method == 'open':
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif method == 'close':
        out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return out

def list_png_paths(folder: str) -> List[str]:
    """
    Return a sorted list of paths to all PNG files in `folder`.

    Args:
        folder: path to folder (can be relative like "depth" or absolute).
        recursive: if True, search subdirectories recursively.

    Returns:
        List of POSIX-style path strings, e.g. ["depth/00001.png", ...].
    """
    
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        print(f"DEBUG[utils.list_png_paths]: Folder {folder} does not exist or is not a directory.")
        return []

    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == '.png']

    # sort by path (natural/alphanumeric order if filenames are zero-padded)
    files_sorted = sorted(files)
    return [f.as_posix() for f in files_sorted]

# Helper: compute bbox 3d (AABB+OBB) from points using Open3D
def compute_3d_bboxes(points):
    if points.shape[0] == 0:
        return {
            'aabb': None,
            'obb': None
        }
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # clean with statistical outlier removal
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.7)
    # pcd = pcd.select_by_index(ind)
                
    aabb = pcd.get_axis_aligned_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    
    # return serializable versions
    aabb_min = np.asarray(aabb.get_min_bound()).tolist()
    aabb_max = np.asarray(aabb.get_max_bound()).tolist()
    obb_center = np.asarray(obb.center).tolist()
    obb_extent = np.asarray(obb.extent).tolist()
    obb_R = np.asarray(obb.R).tolist()  # 3x3 rotation matrix
    return {
        'aabb': {'min': aabb_min, 'max': aabb_max},
        'obb': {'center': obb_center, 'extent': obb_extent, 'R': obb_R}
    }

def save_pkl(data, name, path="/home/rizo/mipt_ccm/yle_sc_sg/out_pkls2"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(data, f)

def extract_points_from_mask( depth_m: np.ndarray,
    mask,
    max_points = None,
    random_state = None,
    return_pixels = False):
    """
    Convert depth + mask -> 3D points in camera coordinates.

    Args:
        depth_m: (H, W) float32 depth in meters. Invalid pixels should be 0.
        mask: (H, W) binary mask (0/255 or bool) OR an iterable/list/array of masks shape (N, H, W).
        K: camera intrinsics either 3x3 matrix or tuple (fx, fy, cx, cy).
        max_points: optionally cap the number of output points. If mask yields >max_points,
                    random sampling is applied.
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
    if not isinstance(depth_m, np.ndarray):
        raise ValueError("depth_m must be a numpy array")
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    
    H, W = depth_m.shape

    single_mask_input = False
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        masks = [mask]
        single_mask_input = True
    else:
        # try to treat mask as iterable of masks (list, tuple, or 3D ndarray)
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            masks = [mask[i] for i in range(mask.shape[0])]
        else:
            # assume list-like
            masks = list(mask)

    rng = np.random.default_rng(random_state)

    results_points = []
    results_pixels = [] if return_pixels else None

    for m in masks:
        if not isinstance(m, np.ndarray):
            m = np.array(m)
            
        # normalize mask to boolean
        if m.dtype == np.uint8 or m.dtype == np.int32 or m.dtype == np.int64:
            mask_bool = (m > 0)
        else:
            mask_bool = (m != 0)

        if mask_bool.shape != depth_m.shape:
            raise ValueError(f"Mask shape {mask_bool.shape} does not match depth shape {depth_m.shape}")

        # get valid pixels where mask==True and depth>0
        valid_mask = mask_bool & (depth_m > 0)
        if not np.any(valid_mask):
            # empty
            pts = np.zeros((0,3), dtype=np.float32)
            results_points.append(pts)
            if return_pixels:
                results_pixels.append(np.zeros((0,2), dtype=np.int64))
            continue

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

        # unproject
        us_f = us.astype(np.float32)
        vs_f = vs.astype(np.float32)
        X = (us_f - cx) * zs / fx
        Y = (vs_f - cy) * zs / fy
        Z = zs

        pts = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (M,3)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Clean with statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.7)
        pcd = pcd.select_by_index(ind)

        cleaned_pts_np = np.asarray(pcd.points)
        results_points.append(cleaned_pts_np)
        
        if return_pixels:
            pix = np.stack([us.astype(np.int64), vs.astype(np.int64)], axis=1)
            results_pixels.append(pix)

    if single_mask_input:
        if return_pixels:
            return results_points[0], results_pixels[0]
        return results_points[0]
    else:
        if return_pixels:
            return results_points, results_pixels
        return results_points

import numpy as np
import open3d as o3d
from typing import Union, Tuple, List, Optional

def _colormap(n):
    """Return n distinct colors in [0,1] as Nx3 numpy array (deterministic)."""
    rng = np.random.default_rng(0)
    cols = rng.random((n, 3))
    # brighten
    cols = 0.6 + 0.4 * cols
    return cols.clip(0,1)

def extract_points_from_mask1(
    depth_m: np.ndarray,
    mask,
    max_points: Optional[int] = None,
    random_state: Optional[int] = None,
    return_pixels: bool = False,
    vis: bool = False,
    vis_combined: bool = True,
    point_size: int = 2
) -> Union[np.ndarray, List[np.ndarray], Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]]:
    """
    Convert depth + mask -> 3D points in camera coordinates (optionally visualize with Open3D).

    Args:
        depth_m: (H, W) float32 depth in meters. Invalid pixels should be 0.
        mask: (H, W) binary mask (0/255 or bool) OR an iterable/list/array of masks shape (N, H, W).
        K: camera intrinsics either 3x3 matrix or tuple (fx, fy, cx, cy).
        max_points: optionally cap the number of output points per mask. If mask yields >max_points,
                    random sampling is applied.
        random_state: int seed for deterministic sampling (if max_points used).
        return_pixels: if True, also return (u,v) pixel coordinates for each 3D point.
        vis: if True, show point cloud(s) in an Open3D window.
        vis_combined: if True and multiple masks, show single combined cloud (colored per mask).
        point_size: visualization point size.

    Returns:
        Same semantics as your original function:
        - Single mask input -> (M,3) points or (points, pixels) if return_pixels
        - Multiple masks -> list of (M_i,3) arrays or tuple(list_points, list_pixels)
    """
    """
    Extracts 3D points and 3D Oriented Bounding Boxes (OBBs) for objects defined by 2D masks.
    """
    if not isinstance(depth_m, np.ndarray):
        raise ValueError("depth_m must be a numpy array")
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    H, W = depth_m.shape

    # Process mask input to ensure 'masks' is a list of 2D numpy arrays
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        masks = [mask]
    elif isinstance(mask, np.ndarray) and mask.ndim == 3:
        masks = [mask[i] for i in range(mask.shape[0])]
    else:
        masks = list(mask)

    rng = np.random.default_rng(random_state)

    results_points = []
    results_pixels = [] if return_pixels else None
    results_bboxes = [] # New: Store the calculated OBBs

    # For visualization: accumulate point clouds and colors
    vis_pcs = []
    vis_colors = []

    for mi, m in enumerate(masks):
        if not isinstance(m, np.ndarray):
            m = np.array(m)

        # normalize mask to boolean
        if m.dtype == np.uint8 or m.dtype == np.int32 or m.dtype == np.int64:
            mask_bool = (m > 0)
        else:
            mask_bool = (m != 0)

        if mask_bool.shape != depth_m.shape:
            raise ValueError(f"Mask shape {mask_bool.shape} does not match depth shape {depth_m.shape}")

        # valid pixels where mask==True and depth>0
        valid_mask = mask_bool & (depth_m > 0)
        
        # --- Point Cloud Extraction (Original Logic) ---
        if not np.any(valid_mask):
            pts = np.zeros((0,3), dtype=np.float32)
            results_points.append(pts)
            results_bboxes.append(None) # Store None for bbox
            if return_pixels:
                results_pixels.append(np.zeros((0,2), dtype=np.int64))
            vis_pcs.append(None)
            vis_colors.append(None)
            continue

        vs, us = np.nonzero(valid_mask)   # v = row (y), u = col (x)
        zs = depth_m[vs, us].astype(np.float32)

        M = zs.shape[0]
        if max_points is not None and M > int(max_points):
            idx = rng.choice(M, size=int(max_points), replace=False)
            us = us[idx]; vs = vs[idx]; zs = zs[idx]; M = zs.shape[0]

        us_f = us.astype(np.float32)
        vs_f = vs.astype(np.float32)
        
        # Back-projection to camera frame (X, Y, Z are in meters)
        X = (us_f - cx) * zs / fx
        Y = (vs_f - cy) * zs / fy
        Z = zs
        pts = np.stack([X, Y, Z], axis=1).astype(np.float64) # Use float64 for Open3D consistency
        results_points.append(pts)
        
        if return_pixels:
            pix = np.stack([us.astype(np.int64), vs.astype(np.int64)], axis=1)
            results_pixels.append(pix)
            
        vis_pcs.append(pts)
        vis_colors.append(None)

        # --- NEW: Outlier Removal and 3D BBox Calculation ---
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(pts)
        
        # Apply Statistical Outlier Removal (SOR) to clean the point cloud for a better fit
        cl, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        cleaned_pts = pcd_temp.select_by_index(ind)
        
        if len(cleaned_pts.points) > 5: # Ensure enough points remain to fit a box
            # Calculate the Oriented Bounding Box (OBB) from the *cleaned* points
            obb = cleaned_pts.get_oriented_bounding_box()
            obb.color = (1, 0, 0) # Default color for OBB
            results_bboxes.append(obb)
        else:
            results_bboxes.append(None)
            
        print(f"[extract_points_from_mask] mask {mi}: {M} points -> {len(cleaned_pts.points)} after SOR. OBB calculated.")


    # --- Visualization block (Modified to include BBoxes) ---
    if vis:
        n_masks = len(masks)
        cols = _colormap(n_masks)
        geom_list = []
        
        # Add a placeholder for the bounding boxes to draw later
        bboxes_to_draw = []

        # Logic for combined or separate point clouds
        if vis_combined:
            print("combined visualization")
            all_pts = []
            all_cols = []
            for i, pts in enumerate(vis_pcs):
                if pts is None or pts.shape[0] == 0:
                    continue
                # Color is for the original point cloud data
                color = cols[i % len(cols)]
                
                # We need to re-run the clean up to get the *same* cleaned point cloud used for the OBB
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(pts)
                cl, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.7)
                cleaned_pcd = pcd_temp.select_by_index(ind)

                # Assign color to the *cleaned* points
                color_arr = np.tile(color.reshape(1,3), (len(cleaned_pcd.points),1))
                cleaned_pcd.colors = o3d.utility.Vector3dVector(color_arr)
                
                all_pts.append(np.asarray(cleaned_pcd.points))
                all_cols.append(np.asarray(cleaned_pcd.colors))
                
                if results_bboxes[i] is not None:
                    bboxes_to_draw.append((results_bboxes[i], cols[i % len(cols)]))

            if len(all_pts) > 0:
                all_pts = np.vstack(all_pts)
                all_cols = np.vstack(all_cols)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                pcd.colors = o3d.utility.Vector3dVector(all_cols)
                geom_list.append(pcd)
        else:
            # make separate clouds per mask
            for i, pts in enumerate(vis_pcs):
                if pts is None or pts.shape[0] == 0:
                    continue
                color = cols[i % len(cols)]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1,3), (pts.shape[0],1)))
                geom_list.append(pcd)
                
                if results_bboxes[i] is not None:
                    bboxes_to_draw.append((results_bboxes[i], cols[i % len(cols)]))

        # Add all Bounding Boxes to the geometry list
        for obb, color in bboxes_to_draw:
            obb.color = color
            geom_list.append(obb)

        # Add coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(H, W)/1000.0 * 0.5, origin=[0,0,0])
        geom_list.append(frame)

        if not geom_list:
             print("[extract_points_with_3d_bbox] nothing to visualize (no points).")
        else:
            # Open3D Visualization setup
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Extracted Points and 3D BBoxes (Open3D)", width=1024, height=768)
            for g in geom_list:
                vis.add_geometry(g)
            render_opt = vis.get_render_option()
            render_opt.point_size = float(point_size)
            vis.run()
            vis.destroy_window()

    return results_points, results_pixels, results_bboxes

    # Return results in same structure as original
    # if single_mask_input:
    #     if return_pixels:
    #         return results_points[0], results_pixels[0]
    #     return results_points[0]
    # else:
    #     if return_pixels:
    #         return results_points, results_pixels
    #     return results_points
    
    
