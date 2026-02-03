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
import itertools as it
    
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

# ============================================================================
# DEFAULT CLASSES TO SKIP (structural elements, rooms, large spaces)
# ============================================================================
DEFAULT_SKIP_CLASSES = {
    # Structural elements
    'wall', 'floor', 'ceiling', 'roof', 'door', 'window',
    'stairway', 'stairs', 'stair', 'escalator', 'elevator',
    
    # Rooms and spaces
    'room', 'kitchen', 'bathroom', 'bedroom', 'living room',
    'dining room', 'office', 'hallway', 'corridor', 'lobby',
    'garage', 'basement', 'attic',
    
    # Large venues/areas
    'basketball court', 'tennis court', 'football field',
    'soccer field', 'baseball field', 'stadium', 'arena',
    'mall', 'store', 'shop', 'market', 'supermarket',
    'restaurant', 'cafe', 'bar', 'gym', 'pool',
    'parking lot', 'parking', 'road', 'street', 'sidewalk',
    'highway', 'bridge', 'tunnel',
    
    # Outdoor elements
    'sky', 'ground', 'grass', 'field', 'lawn',
    'mountain', 'hill', 'river', 'lake', 'ocean', 'sea',
    'beach', 'forest', 'tree', 'trees',
    
    # Building types
    'building', 'house', 'apartment', 'skyscraper',
    'warehouse', 'factory', 'school', 'hospital',
    'church', 'temple', 'mosque',
    
    # Other large elements
    'platform', 'stage', 'runway', 'track',
}


def filter_detections_by_class(
    masks: list,
    track_ids: np.ndarray,
    class_names: list,
    skip_classes: set = None,
    verbose: bool = False
) -> tuple:
    """
    Filter out detections belonging to unwanted classes.
    
    Args:
        masks: List of binary masks
        track_ids: Array of track IDs
        class_names: List of class names
        skip_classes: Set of class names to filter out (lowercase).
                      If None, uses DEFAULT_SKIP_CLASSES.
        verbose: If True, print filtered classes
    
    Returns:
        (filtered_masks, filtered_track_ids, filtered_class_names)
    """
    if skip_classes is None:
        skip_classes = DEFAULT_SKIP_CLASSES
    else:
        skip_classes = set(c.lower() for c in skip_classes)
    
    if not skip_classes or class_names is None:
        return masks, track_ids, class_names
    
    keep_indices = []
    for i, cls_name in enumerate(class_names):
        if cls_name is None:
            keep_indices.append(i)
        elif cls_name.lower() not in skip_classes:
            keep_indices.append(i)
        elif verbose:
            print(f"    [filter] Skipping: {cls_name}")
    
    if len(keep_indices) == len(class_names):
        return masks, track_ids, class_names
    
    # Filter
    filtered_masks = [masks[i] for i in keep_indices] if masks else []
    filtered_track_ids = track_ids[keep_indices] if track_ids is not None and len(track_ids) > 0 else np.array([], dtype=np.int64)
    filtered_class_names = [class_names[i] for i in keep_indices]
    
    if verbose:
        print(f"    [filter] Kept {len(keep_indices)}/{len(class_names)} detections")
    
    return filtered_masks, filtered_track_ids, filtered_class_names


class GlobalObjectRegistry:
    """
    Central registry for all tracked objects with reconstruction support.
    
    Design principles:
    - Point clouds are stored centrally for reconstruction (not passed around)
    - Only lightweight 3D bboxes are used for edge prediction
    - Each object has visibility status for current frame
    - Objects persist even when not visible (for re-identification)
    
    Matching Priority:
    1. YOLO track_id mapping (verified with spatial check)
    2. Previous frame objects (temporal continuity)
    3. Global registry lookup (re-observation after occlusion)
    4. New object assignment
    
    Anti-Containment Logic:
    - Volume ratio check: Objects with vastly different volumes (>10x) are not matched
    - Size similarity check: Individual dimensions must be within 5x of each other
    - This prevents matching small objects (chair, table) to large containers (room, cabinet)
    
    Reprojection-Based Visibility:
    - YOLO detection alone is unreliable (may miss visible objects)
    - Objects with >=20% of their 3D bbox visible in camera view are considered "present"
    - These objects participate in edge prediction even if YOLO didn't detect them
    - This handles cases where YOLO misses objects that are clearly in view
    
    Object structure:
    {
        'global_id': int,
        'class_name': str or None,
        'points_accumulated': np.ndarray,  # Full PCD for reconstruction
        'bbox_3d': dict,  # Lightweight bbox for edge prediction
        'first_seen_frame': int,
        'last_seen_frame': int,
        'observation_count': int,
        'visible_current_frame': bool,  # Visibility in current frame
        'last_mask': np.ndarray or None,  # Last clean 2D mask (H, W) for metrics
        'last_mask_frame': int,  # Frame when last_mask was captured
    }
    """
    
    def __init__(self, 
                 overlap_threshold: float = 0.3,
                 distance_threshold: float = 0.5,
                 max_points_per_object: int = 10000,
                 inactive_frames_limit: int = 0,
                 volume_ratio_threshold: float = 0.1,
                 reprojection_visibility_threshold: float = 0.2):
        """
        Args:
            overlap_threshold: 3D IoU threshold for matching
            distance_threshold: Max centroid distance (meters) for candidate matching
            max_points_per_object: Cap accumulated points per object
            inactive_frames_limit: Remove objects not seen for this many frames (0 = never remove)
            volume_ratio_threshold: Min volume ratio (smaller/larger) to consider objects
                                    as potential matches. Prevents matching small objects
                                    to large containers (room/cabinet). Default 0.1 means
                                    volumes can differ by at most 10x.
            reprojection_visibility_threshold: Min fraction of object bbox that must be
                                               visible in camera view to consider object
                                               as "present" even if YOLO didn't detect it.
                                               Default 0.2 means 20% of bbox must be visible.
        """
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.max_points_per_object = max_points_per_object
        self.inactive_frames_limit = inactive_frames_limit
        self.volume_ratio_threshold = volume_ratio_threshold
        self.reprojection_visibility_threshold = reprojection_visibility_threshold
        
        # Global object storage: global_id -> object_data
        self.objects = {}
        
        # Counter for assigning new global IDs
        self._next_global_id = 1
        
        # Previous frame's objects for temporal continuity matching
        self._prev_frame_objects = {}  # global_id -> object_data
        
        # Mapping from YOLO track_id to global_id (for current session)
        self.yolo_to_global = {}
        
        # Current frame index
        self.current_frame = -1
        
        # Set of object IDs visible in current frame
        self._visible_this_frame = set()
    
    def _generate_global_id(self) -> int:
        """Generate a new unique global ID."""
        gid = self._next_global_id
        self._next_global_id += 1
        return gid
    
    def _compute_overlap_score(self, pcd1: np.ndarray, pcd2: np.ndarray, 
                                bbox1: dict, bbox2: dict) -> tuple:
        """
        Compute overlap score between two objects using IoU, centroid distance,
        and volume ratio check to prevent matching objects of very different sizes
        (e.g., small object inside a room).
        
        Returns:
            (iou_score, centroid_distance, is_candidate)
        """
        if pcd1 is None or pcd2 is None or len(pcd1) == 0 or len(pcd2) == 0:
            return 0.0, float('inf'), False
        
        # Get centroids and extents from bbox or compute from points
        if bbox1 and 'obb' in bbox1 and bbox1['obb']:
            center1 = np.array(bbox1['obb']['center'])
            extent1 = np.array(bbox1['obb'].get('extent', [1, 1, 1]))
        else:
            center1 = np.mean(pcd1, axis=0)
            extent1 = np.max(pcd1, axis=0) - np.min(pcd1, axis=0)
        
        if bbox2 and 'obb' in bbox2 and bbox2['obb']:
            center2 = np.array(bbox2['obb']['center'])
            extent2 = np.array(bbox2['obb'].get('extent', [1, 1, 1]))
        else:
            center2 = np.mean(pcd2, axis=0)
            extent2 = np.max(pcd2, axis=0) - np.min(pcd2, axis=0)
        
        centroid_dist = np.linalg.norm(center1 - center2)
        
        # Quick rejection based on distance
        if centroid_dist > self.distance_threshold:
            return 0.0, centroid_dist, False
        
        # === VOLUME RATIO CHECK ===
        # Prevent matching objects of vastly different sizes (containment case)
        # e.g., a chair inside a room should NOT match the room
        volume1 = np.prod(np.maximum(extent1, 1e-6))  # Avoid zero
        volume2 = np.prod(np.maximum(extent2, 1e-6))
        
        # Volume ratio: smaller/larger (always <= 1)
        volume_ratio = min(volume1, volume2) / max(volume1, volume2)
        
        # If volumes differ by more than 10x (ratio < 0.1), likely different objects
        # This catches the room-contains-objects case
        if volume_ratio < self.volume_ratio_threshold:
            return 0.0, centroid_dist, False
        
        # === SIZE SIMILARITY CHECK ===
        # Compare dimensions individually - objects should have similar proportions
        size_ratios = np.minimum(extent1, extent2) / np.maximum(extent1, extent2 + 1e-6)
        # If any dimension differs by more than 5x, likely different objects
        if np.any(size_ratios < 0.2):
            return 0.0, centroid_dist, False
        
        # Compute 3D IoU
        iou = calculate_3d_iou(pcd1, pcd2)
        
        is_candidate = iou >= self.overlap_threshold
        
        return iou, centroid_dist, is_candidate
    
    def _compute_reprojection_visibility(self, bbox_3d: dict, T_w_c: np.ndarray,
                                          image_width: int = IMAGE_WIDTH,
                                          image_height: int = IMAGE_HEIGHT) -> float:
        """
        Compute what fraction of an object's 3D bbox is visible when reprojected to camera view.
        
        Args:
            bbox_3d: Object's 3D bounding box with 'aabb' or 'obb'
            T_w_c: 4x4 camera-to-world transformation matrix
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Visibility fraction [0, 1]. 0 = not visible, 1 = fully visible
        """
        if bbox_3d is None or T_w_c is None:
            return 0.0
        
        # Get 8 corners of the 3D bounding box in world coordinates
        corners_world = self._get_bbox_corners(bbox_3d)
        if corners_world is None or len(corners_world) == 0:
            return 0.0
        
        # Transform from world to camera coordinates
        # T_w_c is camera-to-world, so we need its inverse (world-to-camera)
        try:
            T_c_w = np.linalg.inv(T_w_c)
        except np.linalg.LinAlgError:
            return 0.0
        
        R = T_c_w[:3, :3]
        t = T_c_w[:3, 3]
        
        corners_cam = (corners_world @ R.T) + t  # (8, 3)
        
        # Check if any corner is behind camera (Z <= 0)
        z_vals = corners_cam[:, 2]
        
        # If all points are behind camera, not visible
        if np.all(z_vals <= 0):
            return 0.0
        
        # Project to 2D (only points in front of camera)
        valid_mask = z_vals > 0.01  # Small epsilon to avoid division issues
        if not np.any(valid_mask):
            return 0.0
        
        corners_valid = corners_cam[valid_mask]
        
        # Project: u = fx * X/Z + cx, v = fy * Y/Z + cy
        u = fx * corners_valid[:, 0] / corners_valid[:, 2] + cx
        v = fy * corners_valid[:, 1] / corners_valid[:, 2] + cy
        
        # Compute 2D bounding box of projected points
        u_min, u_max = np.min(u), np.max(u)
        v_min, v_max = np.min(v), np.max(v)
        
        # Clip to image bounds
        u_min_clipped = max(0, u_min)
        u_max_clipped = min(image_width, u_max)
        v_min_clipped = max(0, v_min)
        v_max_clipped = min(image_height, v_max)
        
        # Compute areas
        projected_area = max(0, u_max - u_min) * max(0, v_max - v_min)
        visible_area = max(0, u_max_clipped - u_min_clipped) * max(0, v_max_clipped - v_min_clipped)
        
        if projected_area <= 0:
            return 0.0
        
        # Also check if the object is within reasonable depth range
        min_depth = np.min(corners_valid[:, 2])
        if min_depth > MAX_DEPTH:
            return 0.0
        
        visibility_fraction = visible_area / projected_area
        
        # Additional check: if projected bbox is entirely outside image, return 0
        if u_max < 0 or u_min > image_width or v_max < 0 or v_min > image_height:
            return 0.0
        
        return visibility_fraction
    
    def _get_bbox_corners(self, bbox_3d: dict) -> np.ndarray:
        """
        Get 8 corners of the 3D bounding box in world coordinates.
        
        Returns:
            (8, 3) array of corner points, or None if bbox is invalid
        """
        if bbox_3d is None:
            return None
        
        # Try AABB first (simpler)
        aabb = bbox_3d.get('aabb')
        if aabb is not None and aabb.get('min') is not None and aabb.get('max') is not None:
            mn = np.array(aabb['min'])
            mx = np.array(aabb['max'])
            
            # Generate 8 corners
            corners = np.array([
                [mn[0], mn[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mx[1], mx[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mn[2]],
                [mx[0], mx[1], mx[2]],
            ])
            return corners
        
        # Fallback to OBB
        obb = bbox_3d.get('obb')
        if obb is not None and obb.get('center') is not None and obb.get('extent') is not None:
            center = np.array(obb['center'])
            extent = np.array(obb['extent'])
            R = np.array(obb.get('R', np.eye(3)))
            
            # Half extents
            half = extent / 2.0
            
            # Local corners (before rotation)
            local_corners = np.array([
                [-half[0], -half[1], -half[2]],
                [-half[0], -half[1], +half[2]],
                [-half[0], +half[1], -half[2]],
                [-half[0], +half[1], +half[2]],
                [+half[0], -half[1], -half[2]],
                [+half[0], -half[1], +half[2]],
                [+half[0], +half[1], -half[2]],
                [+half[0], +half[1], +half[2]],
            ])
            
            # Rotate and translate to world
            corners = (local_corners @ R.T) + center
            return corners
        
        return None
    
    def get_reprojection_visible_objects(self, T_w_c: np.ndarray, 
                                          excluded_ids: set = None) -> list:
        """
        Get all objects that are visible by reprojection but not detected by YOLO.
        
        Args:
            T_w_c: Camera-to-world transformation matrix
            excluded_ids: Set of global_ids to exclude (already detected by YOLO)
            
        Returns:
            List of lightweight object dicts for objects visible by reprojection
        """
        if excluded_ids is None:
            excluded_ids = set()
        
        visible_by_reprojection = []
        
        for gid, obj in self.objects.items():
            if gid in excluded_ids:
                continue
            
            bbox_3d = obj.get('bbox_3d')
            if bbox_3d is None:
                continue
            
            visibility = self._compute_reprojection_visibility(bbox_3d, T_w_c)
            
            if visibility >= self.reprojection_visibility_threshold:
                # Object is visible by reprojection but wasn't detected by YOLO
                result_obj = {
                    'global_id': gid,
                    'track_id': gid,
                    'yolo_track_id': -1,  # Not detected by YOLO
                    'class_name': obj.get('class_name'),
                    'bbox_3d': bbox_3d,
                    'visible_current_frame': True,  # Visible by reprojection
                    'match_source': 'reprojection',
                    'observation_count': obj['observation_count'],
                    'reprojection_visibility': visibility
                }
                visible_by_reprojection.append(result_obj)
                
                # Mark as visible in registry
                self.objects[gid]['visible_current_frame'] = True
                self._visible_this_frame.add(gid)
        
        return visible_by_reprojection
    
    def _merge_point_clouds(self, pcd_existing: np.ndarray, pcd_new: np.ndarray) -> np.ndarray:
        """Merge two point clouds with downsampling to limit size."""
        if pcd_existing is None or len(pcd_existing) == 0:
            merged = pcd_new
        elif pcd_new is None or len(pcd_new) == 0:
            merged = pcd_existing
        else:
            merged = np.vstack([pcd_existing, pcd_new])
        
        # Downsample if exceeds limit
        if merged is not None and len(merged) > self.max_points_per_object:
            indices = np.random.choice(len(merged), 
                                       size=self.max_points_per_object, 
                                       replace=False)
            merged = merged[indices]
        
        return merged
    
    def process_frame(self, frame_idx: int, detections: list) -> list:
        """
        Process detections from a single frame and update the registry.
        
        Updates visibility status for ALL objects:
        - Objects detected this frame: visible_current_frame = True
        - Objects not detected: visible_current_frame = False
        
        Args:
            frame_idx: Current frame index
            detections: List of dicts with keys:
                - 'yolo_track_id': int (from YOLO tracker)
                - 'points': np.ndarray (N, 3) in world coordinates
                - 'bbox_3d': dict with 'aabb' and 'obb'
                - 'class_name': str (optional)
        
        Returns:
            List of lightweight object dicts for edge prediction (bbox only, no heavy PCDs)
        """
        self.current_frame = frame_idx
        
        # Reset visibility for all objects
        self._visible_this_frame = set()
        for gid in self.objects:
            self.objects[gid]['visible_current_frame'] = False
        
        # Store current frame objects for next iteration
        current_frame_objects = {}
        
        # Track which global objects were matched this frame
        matched_global_ids = set()
        
        # Results to return (lightweight, for edge prediction)
        processed_objects = []
        
        for det in detections:
            yolo_id = det.get('yolo_track_id', -1)
            points = det.get('points')
            bbox_3d = det.get('bbox_3d', {})
            class_name = det.get('class_name', None)
            
            if points is None or len(points) == 0:
                continue
            
            global_id = None
            match_source = None
            
            # === LEVEL 1: Try to match using YOLO track_id mapping ===
            if yolo_id >= 0 and yolo_id in self.yolo_to_global:
                candidate_gid = self.yolo_to_global[yolo_id]
                if candidate_gid in self.objects:
                    # Verify with spatial overlap (YOLO might reuse IDs incorrectly)
                    existing = self.objects[candidate_gid]
                    iou, dist, is_match = self._compute_overlap_score(
                        points, existing['points_accumulated'],
                        bbox_3d, existing['bbox_3d']
                    )
                    if is_match or dist < self.distance_threshold * 0.5:
                        global_id = candidate_gid
                        match_source = 'yolo_mapping'
            
            # === LEVEL 2: Try to match with previous frame objects (temporal continuity) ===
            if global_id is None:
                best_match = None
                best_iou = 0
                
                for gid, prev_obj in self._prev_frame_objects.items():
                    if gid in matched_global_ids:
                        continue
                    
                    iou, dist, is_candidate = self._compute_overlap_score(
                        points, prev_obj['points_accumulated'],
                        bbox_3d, prev_obj['bbox_3d']
                    )
                    
                    if is_candidate and iou > best_iou:
                        best_iou = iou
                        best_match = gid
                
                if best_match is not None:
                    global_id = best_match
                    match_source = 'prev_frame'
            
            # === LEVEL 3: Try to match with global registry (re-observation) ===
            if global_id is None:
                best_match = None
                best_iou = 0
                
                for gid, obj in self.objects.items():
                    if gid in matched_global_ids:
                        continue
                    
                    iou, dist, is_candidate = self._compute_overlap_score(
                        points, obj['points_accumulated'],
                        bbox_3d, obj['bbox_3d']
                    )
                    
                    if is_candidate and iou > best_iou:
                        best_iou = iou
                        best_match = gid
                
                if best_match is not None:
                    global_id = best_match
                    match_source = 'global_registry'
            
            # === LEVEL 4: No match found - create new object ===
            if global_id is None:
                global_id = self._generate_global_id()
                match_source = 'new'
                self.objects[global_id] = {
                    'global_id': global_id,
                    'class_name': class_name,
                    'points_accumulated': points.copy(),
                    'bbox_3d': bbox_3d,
                    'first_seen_frame': frame_idx,
                    'last_seen_frame': frame_idx,
                    'observation_count': 0,
                    'visible_current_frame': True,
                    'last_mask': None,
                    'last_mask_frame': -1
                }
            
            # === Update the object ===
            matched_global_ids.add(global_id)
            self._visible_this_frame.add(global_id)
            
            # Update YOLO mapping
            if yolo_id >= 0:
                self.yolo_to_global[yolo_id] = global_id
            
            # Merge point clouds (reconstruction)
            existing_points = self.objects[global_id].get('points_accumulated')
            merged_points = self._merge_point_clouds(existing_points, points)
            
            # Recompute bbox from merged points
            merged_bbox = compute_3d_bboxes(merged_points)
            
            # Update class name if provided and not already set
            if class_name and not self.objects[global_id].get('class_name'):
                self.objects[global_id]['class_name'] = class_name
            
            # Get mask from detection if provided
            det_mask = det.get('mask', None)
            
            # Update registry
            self.objects[global_id].update({
                'points_accumulated': merged_points,
                'bbox_3d': merged_bbox,
                'last_seen_frame': frame_idx,
                'observation_count': self.objects[global_id].get('observation_count', 0) + 1,
                'visible_current_frame': True
            })
            
            # Update last_mask if provided
            if det_mask is not None:
                self.objects[global_id]['last_mask'] = det_mask.copy() if hasattr(det_mask, 'copy') else det_mask
                self.objects[global_id]['last_mask_frame'] = frame_idx
            
            # Store for current frame (for temporal continuity next frame)
            current_frame_objects[global_id] = self.objects[global_id]
            
            # Build LIGHTWEIGHT result object (for edge prediction - NO heavy PCDs)
            result_obj = {
                'global_id': global_id,
                'track_id': global_id,  # For compatibility with existing code
                'yolo_track_id': yolo_id,
                'class_name': self.objects[global_id].get('class_name'),
                'bbox_3d': merged_bbox,  # Only bbox, not PCD
                'visible_current_frame': True,
                'match_source': match_source,
                'observation_count': self.objects[global_id]['observation_count']
            }
            processed_objects.append(result_obj)
        
        # Update previous frame reference
        self._prev_frame_objects = current_frame_objects
        
        # Optional: Clean up very old objects
        if self.inactive_frames_limit > 0:
            self._cleanup_inactive_objects(frame_idx)
        
        return processed_objects
    
    def _cleanup_inactive_objects(self, current_frame: int):
        """Remove objects not seen for too long (optional)."""
        to_remove = []
        for gid, obj in self.objects.items():
            frames_inactive = current_frame - obj.get('last_seen_frame', current_frame)
            if frames_inactive > self.inactive_frames_limit:
                to_remove.append(gid)
        
        for gid in to_remove:
            del self.objects[gid]
            # Also clean up YOLO mappings
            self.yolo_to_global = {k: v for k, v in self.yolo_to_global.items() if v != gid}
    
    def get_all_objects(self) -> dict:
        """Return all objects in the registry (includes PCDs for reconstruction)."""
        return self.objects
    
    def get_visible_objects(self) -> dict:
        """Return only objects visible in current frame."""
        return {
            gid: obj for gid, obj in self.objects.items()
            if obj.get('visible_current_frame', False)
        }
    
    def get_invisible_objects(self) -> dict:
        """Return objects NOT visible in current frame (but tracked)."""
        return {
            gid: obj for gid, obj in self.objects.items()
            if not obj.get('visible_current_frame', False)
        }
    
    def get_active_objects(self, frame_idx: int, max_inactive_frames: int = 10) -> dict:
        """Return objects seen recently (within max_inactive_frames)."""
        return {
            gid: obj for gid, obj in self.objects.items()
            if frame_idx - obj.get('last_seen_frame', 0) <= max_inactive_frames
        }
    
    def get_object_pcd(self, global_id: int) -> np.ndarray:
        """Get accumulated point cloud for a specific object (for visualization)."""
        if global_id in self.objects:
            return self.objects[global_id].get('points_accumulated')
        return None
    
    def get_object_bbox(self, global_id: int) -> dict:
        """Get 3D bbox for a specific object (for edge prediction)."""
        if global_id in self.objects:
            return self.objects[global_id].get('bbox_3d')
        return None
    
    def get_all_pcds_for_visualization(self) -> list:
        """
        Get all accumulated PCDs with their track_ids for visualization.
        Returns list of dicts with 'global_id', 'points', 'class_name', 'visible'.
        """
        result = []
        for gid, obj in self.objects.items():
            result.append({
                'global_id': gid,
                'points': obj.get('points_accumulated'),
                'class_name': obj.get('class_name'),
                'bbox_3d': obj.get('bbox_3d'),
                'visible_current_frame': obj.get('visible_current_frame', False),
                'observation_count': obj.get('observation_count', 0)
            })
        return result
    
    def get_visible_ids(self) -> set:
        """Return set of global_ids visible in current frame."""
        return self._visible_this_frame.copy()
    
    def get_object_mask(self, global_id: int) -> np.ndarray:
        """Get the last stored mask for an object."""
        if global_id in self.objects:
            return self.objects[global_id].get('last_mask', None)
        return None
    
    def get_all_objects_with_masks(self) -> dict:
        """
        Get all objects that have masks stored.
        
        Returns:
            dict: global_id -> object_data (only objects with masks)
        """
        return {
            gid: obj for gid, obj in self.objects.items() 
            if obj.get('last_mask') is not None
        }
    
    def get_tracking_summary(self) -> dict:
        """Get summary statistics for tracking."""
        visible = len(self._visible_this_frame)
        total = len(self.objects)
        with_masks = sum(1 for obj in self.objects.values() if obj.get('last_mask') is not None)
        return {
            'total_objects': total,
            'visible_objects': visible,
            'invisible_objects': total - visible,
            'objects_with_masks': with_masks,
            'current_frame': self.current_frame
        }


def draw_labeled_multigraph(G, attr_name='label', ax=None):
    """
    Draw a multigraph with labeled edges.
    Red labels for 'proximity' type, blue for others.
    """
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )
    
    # Group edges by (u, v) to track which connectionstyle index to use
    edge_count = {}
    
    for u, v, key, attrs in G.edges(keys=True, data=True):
        edge_pair = (u, v)
        if edge_pair not in edge_count:
            edge_count[edge_pair] = 0
        
        # Get the appropriate connectionstyle for this specific edge
        cs_index = edge_count[edge_pair]
        cs = connectionstyle[cs_index] if cs_index < len(connectionstyle) else connectionstyle[-1]
        
        label = attrs[attr_name]
        label_class = attrs.get('label_class', '')
        color = "red" if (label_class == 'proximity' or label_class == 'middle_furniture') else "blue"
        
        # Draw this specific edge label
        nx.draw_networkx_edge_labels(
            G,
            pos,
            {(u, v, key): label},
            connectionstyle=[cs],
            label_pos=0.3,
            font_color=color,
            bbox={"alpha": 0},
            ax=ax,
        )
        
        edge_count[edge_pair] += 1

    # Create custom legend with text colors
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='r', markerfacecolor='w', 
                   markersize=0, label='egocentric'),
        plt.Line2D([0], [0], marker='o', color='b', markerfacecolor='w', 
                   markersize=0, label='allocentric')
    ], loc='upper right')


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

def remove_mask_overlaps(masks):
    """
    Remove overlapping regions from masks. When masks overlap, 
    the overlapping pixels are removed from both masks.
    
    Args:
        masks: list of binary masks (H, W) with values 0/255 or bool
        
    Returns:
        list of masks with overlaps removed
    """
    if not masks or len(masks) < 2:
        return masks
    
    # Handle None masks
    if all(m is None for m in masks):
        return masks
    
    # Find first non-None mask to get shape
    shape = None
    for m in masks:
        if m is not None:
            shape = m.shape
            break
    
    if shape is None:
        return masks
    
    # Convert all masks to boolean for easier processing
    bool_masks = []
    for m in masks:
        if m is None:
            bool_masks.append(None)
        elif m.dtype == bool:
            bool_masks.append(m.copy())
        else:
            bool_masks.append(m > 0)
    
    # Find all overlapping pixels across all masks
    n = len(bool_masks)
    overlap_map = np.zeros(shape, dtype=bool)
    
    for i in range(n):
        if bool_masks[i] is None:
            continue
        for j in range(i + 1, n):
            if bool_masks[j] is None:
                continue
            # Find overlap between mask i and j
            overlap = bool_masks[i] & bool_masks[j]
            overlap_map |= overlap
    
    # Remove overlaps from all masks
    result_masks = []
    for i, m in enumerate(bool_masks):
        if m is None:
            result_masks.append(None)
        else:
            # Remove overlap regions
            cleaned = m & (~overlap_map)
            # Convert back to uint8 format (0/255)
            result_masks.append(cleaned.astype(np.uint8) * 255)
    
    return result_masks

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
    # print(f'DEBUG tracking init')
    rgb_paths = list_jpg_paths(rgb_dir_path)
    depth_paths = depth_path_list

    model = YOLOE(model_path)
    for ip, rgb_p in enumerate(rgb_paths):
        # read in rgb
        rgb = cv2.imread(rgb_p)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if rgb is None:
            print(f"[WARN][prep_frames] Could not read image: {rgb_p}")
            continue
        
        out = model.track(
            source=[rgb],
            tracker=TRACKER_CFG,
            device=DEVICE,
            conf=conf,
            verbose=False,
            persist=True,
            agnostic_nms=True
        )
         
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out        
        yield res, rgb_p, depth_paths[ip]

def preprocess_mask(yolo_res, index, KERNEL_SIZE, alpha = 0.5, show=True, fast: bool = False):
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
        # Remove overlapping regions from masks
        cleaned_masks = remove_mask_overlaps(cleaned_masks)
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

def _guided_filter_denoise(points: np.ndarray, radius: float, epsilon: float):
    """
    Apply guided filter denoising to a point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        radius: search radius for neighbors
        epsilon: regularization parameter
        
    Returns:
        (N, 3) array of denoised points
    """
    if points.shape[0] < 3:
        return points
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Build KD-tree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    num_points = len(pcd.points)
    
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue
        neighbors = points[idx, :]
        mean = np.mean(neighbors, axis=0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))
        A = cov @ e
        b = mean - A @ mean
        points_copy[i] = A @ points[i] + b
    
    return points_copy.astype(np.float32)

def extract_points_from_mask( depth_m: np.ndarray,
    mask,
    frame_idx,
    o3_nb_neighbors,
    o3std_ratio,
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
        idx = rng.choice(M, size=int(M * 0.5), replace=False)
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
    # return pts

    # Very SLOW: Apply denoising if requested
    # denoise_radius = 0.02
    # denoise_epsilon = 0.5
    # if pts.shape[0] > 0:
    #     pts = _guided_filter_denoise(pts, denoise_radius, denoise_epsilon)
    #     pts = _guided_filter_denoise(pts, denoise_radius, denoise_epsilon)
    #     pts = _guided_filter_denoise(pts, denoise_radius, denoise_epsilon)
    
    # return pts

    # OK SLOW
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Clean with statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=o3_nb_neighbors, std_ratio=o3std_ratio)
    pcd = pcd.select_by_index(ind)

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

def create_3d_objects(track_ids, masks_clean, max_points_per_obj, depth_m, T_w_c, frame_idx, o3_nb_neighbors, o3std_ratio, accumulated_points_dict=None, max_accumulated_points=10000):
    # t_total_start = time.perf_counter()
    # timings = {
    #     'extract_points': 0,
    #     'cam_to_world': 0,
    #     'compute_bbox': 0,
    #     'graph_ops': 0,
    # }
    n_objects = 0
    
    if accumulated_points_dict is None:
        accumulated_points_dict = {}
    
    frame_objs = []
    graph = nx.MultiDiGraph()
    for t_id, m_clean in zip(track_ids, masks_clean):
        if m_clean is None:
            continue
        
        # 1. extract points from mask in camera frame (cap points)
        # t_start = time.perf_counter()
        points_cam = extract_points_from_mask(
            depth_m, 
            m_clean, 
            frame_idx=frame_idx, 
            max_points=max_points_per_obj,
            o3_nb_neighbors=o3_nb_neighbors,
            o3std_ratio=o3std_ratio, 
            random_state=int(t_id)
        )
        # timings['extract_points'] += (time.perf_counter() - t_start) * 1000

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
            # t_start = time.perf_counter()
            points_world = cam_to_world(points_cam, T_w_c)
            pts_for_bbox = points_world
            # timings['cam_to_world'] += (time.perf_counter() - t_start) * 1000

        # 2.5. Accumulate points across frames for this track_id
        if t_id not in accumulated_points_dict:
            # First time seeing this object
            accumulated_points_dict[t_id] = pts_for_bbox.copy()
        else:
            # Accumulate with previous points
            accumulated_points_dict[t_id] = np.vstack([accumulated_points_dict[t_id], pts_for_bbox])
            
            # Limit total accumulated points to avoid memory issues
            if accumulated_points_dict[t_id].shape[0] > max_accumulated_points:
                # Downsample: randomly sample to avoid temporal bias
                indices = np.random.choice(accumulated_points_dict[t_id].shape[0], 
                                          size=max_accumulated_points, 
                                          replace=False)
                accumulated_points_dict[t_id] = accumulated_points_dict[t_id][indices]
        
        # Use accumulated points for bbox computation (more robust)
        pts_accumulated = accumulated_points_dict[t_id]

        # 3. compute 3d bbox (in world if transformed, else in camera) using accumulated points
        # t_start = time.perf_counter()
        bbox3d = compute_3d_bboxes(pts_accumulated)
        # timings['compute_bbox'] += (time.perf_counter() - t_start) * 1000
        
        # t_start = time.perf_counter()
        obj = {
            'track_id': int(t_id),
            'points': pts_for_bbox,  # Current frame points
            'points_accumulated': pts_accumulated,  # Accumulated points from all frames
            'bbox_3d': bbox3d  # Computed from accumulated points
        }
        
        graph.add_node(int(t_id), data=obj)
        # graph.add_node(obj)
        frame_objs.append(obj)
        # timings['graph_ops'] += (time.perf_counter() - t_start) * 1000
        n_objects += 1
    
    # Calculate total time
    # total_time = (time.perf_counter() - t_total_start) * 1000
    
    # Print timing breakdown
    # print(f"    [create_3d] Objects: {n_objects}, Total: {total_time:.2f} ms")
    # if n_objects > 0:
    #     print(f"      extract_points:  {timings['extract_points']:6.2f} ms ({timings['extract_points']/n_objects:.2f} ms/obj)")
    #     print(f"      cam_to_world:    {timings['cam_to_world']:6.2f} ms ({timings['cam_to_world']/n_objects:.2f} ms/obj)")
    #     print(f"      compute_bbox:    {timings['compute_bbox']:6.2f} ms ({timings['compute_bbox']/n_objects:.2f} ms/obj)")
    #     print(f"      graph_ops:       {timings['graph_ops']:6.2f} ms ({timings['graph_ops']/n_objects:.2f} ms/obj)")
            
    return frame_objs, graph


def create_3d_objects_with_tracking(
    track_ids, 
    masks_clean, 
    max_points_per_obj, 
    depth_m, 
    T_w_c, 
    frame_idx,
    o3_nb_neighbors,
    o3std_ratio,
    object_registry: GlobalObjectRegistry,
    class_names: list = None,
    skip_classes: set = None
):
    """
    Enhanced version of create_3d_objects that uses GlobalObjectRegistry for tracking.
    Provides consistent object IDs across frames even when camera moves away and returns.
    
    Key design:
    - PCDs are stored in registry (not returned) for reconstruction
    - Only lightweight bbox objects are returned for edge prediction
    - Visibility status is updated for all objects
    
    Args:
        track_ids: YOLO track IDs
        masks_clean: Cleaned masks
        max_points_per_obj: Max points to extract per object
        depth_m: Depth map in meters
        T_w_c: Camera to world transform
        frame_idx: Current frame index
        o3_nb_neighbors: Open3D outlier removal param
        o3std_ratio: Open3D outlier removal param
        object_registry: GlobalObjectRegistry instance
        class_names: Optional list of class names corresponding to track_ids
        skip_classes: Optional set of class names to filter out (uses DEFAULT_SKIP_CLASSES if None)
    
    Returns:
        frame_objs: List of LIGHTWEIGHT objects (bbox only) for edge prediction
        graph: NetworkX graph with nodes having global IDs and bbox data
    """
    
    # Filter out unwanted classes if class_names provided
    if class_names is not None and skip_classes is not None:
        masks_clean, track_ids, class_names = filter_detections_by_class(
            masks_clean, track_ids, class_names, skip_classes, verbose=False
        )
    
    # Step 1: Extract detections from current frame
    detections = []
    
    for idx, (t_id, m_clean) in enumerate(zip(track_ids, masks_clean)):
        if m_clean is None:
            continue
        
        # Extract points from mask in camera frame
        points_cam = extract_points_from_mask(
            depth_m, 
            m_clean, 
            frame_idx=frame_idx, 
            max_points=max_points_per_obj,
            o3_nb_neighbors=o3_nb_neighbors,
            o3std_ratio=o3std_ratio, 
            random_state=int(t_id)
        )
        
        if points_cam is None or points_cam.size <= 0:
            continue
        
        # Transform to world frame
        if T_w_c is not None:
            points_world = cam_to_world(points_cam, T_w_c)
        else:
            points_world = points_cam
        
        # Compute initial bbox (will be recomputed after merging)
        bbox_3d = compute_3d_bboxes(points_world)
        
        # Get class name if available
        class_name = None
        if class_names is not None and idx < len(class_names):
            class_name = class_names[idx]
        
        detections.append({
            'yolo_track_id': int(t_id),
            'points': points_world,
            'bbox_3d': bbox_3d,
            'class_name': class_name,
            'mask': m_clean  # Store mask for registry-based metrics
        })
    
    # Step 2: Process through registry for consistent tracking
    # Returns LIGHTWEIGHT objects (bbox only, no PCDs)
    frame_objs = object_registry.process_frame(frame_idx, detections)
    
    # Step 2.5: Check for objects visible by reprojection but not detected by YOLO
    # These objects should still participate in edge prediction
    detected_global_ids = {obj['global_id'] for obj in frame_objs}
    
    reprojection_visible = object_registry.get_reprojection_visible_objects(
        T_w_c, 
        excluded_ids=detected_global_ids
    )
    
    # Add reprojection-visible objects to frame_objs
    frame_objs.extend(reprojection_visible)
    
    # Step 3: Build graph with global IDs (lightweight, for edge prediction)
    graph = nx.MultiDiGraph()
    for obj in frame_objs:
        global_id = obj['global_id']
        # Store only lightweight data in graph node
        graph.add_node(global_id, data=obj)
    
    return frame_objs, graph


def visualize_reconstruction(
    object_registry: GlobalObjectRegistry,
    frame_index: int,
    show_visible_only: bool = False,
    show_aabb: bool = True,
    show_obb: bool = False,
    width: int = 1280,
    height: int = 720,
    point_size: float = 2.0,
    visible_alpha: float = 1.0,
    invisible_alpha: float = 0.3
):
    """
    Visualize all reconstructed point clouds from the registry.
    
    Objects visible in current frame are shown with full opacity.
    Objects not visible are shown with reduced opacity (ghosted).
    
    Args:
        object_registry: GlobalObjectRegistry with accumulated PCDs
        frame_index: Current frame index (for title)
        show_visible_only: If True, only show objects visible in current frame
        show_aabb: Show axis-aligned bounding boxes
        show_obb: Show oriented bounding boxes
        width, height: Window dimensions
        point_size: Size of points in visualization
        visible_alpha: Alpha for visible objects (1.0 = opaque)
        invisible_alpha: Alpha for invisible objects (0.3 = ghosted)
    """
    all_vis_data = object_registry.get_all_pcds_for_visualization()
    
    if not all_vis_data:
        print("[visualize_reconstruction] No objects to visualize.")
        return
    
    # Filter if needed
    if show_visible_only:
        all_vis_data = [d for d in all_vis_data if d['visible_current_frame']]
    
    if not all_vis_data:
        print("[visualize_reconstruction] No visible objects to visualize.")
        return
    
    # Determine global bounds
    gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    
    for obj_data in all_vis_data:
        pts = obj_data.get('points')
        if pts is not None and len(pts) > 0:
            gmin = np.minimum(gmin, np.min(pts, axis=0))
            gmax = np.maximum(gmax, np.max(pts, axis=0))
    
    if not np.all(np.isfinite(gmin)) or not np.all(np.isfinite(gmax)):
        print("[visualize_reconstruction] No valid points found.")
        return
    
    diag = float(np.linalg.norm(gmax - gmin)) if np.all(np.isfinite(gmax - gmin)) else 1.0
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    summary = object_registry.get_tracking_summary()
    title = f"Reconstruction - Frame {frame_index} | Visible: {summary['visible_objects']}/{summary['total_objects']}"
    vis.create_window(window_name=title, width=width, height=height, visible=True)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = float(point_size)
    
    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * diag, origin=[0, 0, 0])
    vis.add_geometry(coord)
    
    # Add all objects
    for obj_data in all_vis_data:
        global_id = obj_data['global_id']
        pts = obj_data.get('points')
        bbox_3d = obj_data.get('bbox_3d')
        is_visible = obj_data.get('visible_current_frame', False)
        class_name = obj_data.get('class_name', '')
        
        if pts is None or len(pts) == 0:
            continue
        
        # Generate color based on track_id
        color = _yoloe_utils__generate_color(global_id)
        
        # Adjust alpha based on visibility
        if not is_visible:
            # Make invisible objects more transparent (darker)
            color = [c * invisible_alpha for c in color]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        vis.add_geometry(pcd)
        
        # Add bounding boxes
        if show_aabb and bbox_3d and bbox_3d.get('aabb'):
            aabb_ls = _yoloe_utils__create_aabb_lineset(bbox_3d['aabb'])
            if aabb_ls:
                # Red for visible, gray for invisible
                box_color = [1.0, 0.0, 0.0] if is_visible else [0.5, 0.5, 0.5]
                aabb_ls.colors = o3d.utility.Vector3dVector([box_color] * len(aabb_ls.lines))
                vis.add_geometry(aabb_ls)
        
        if show_obb and bbox_3d and bbox_3d.get('obb'):
            obb_ls = _yoloe_utils__create_obb_lineset(bbox_3d['obb'])
            if obb_ls:
                box_color = [0.0, 1.0, 0.0] if is_visible else [0.3, 0.5, 0.3]
                obb_ls.colors = o3d.utility.Vector3dVector([box_color] * len(obb_ls.lines))
                vis.add_geometry(obb_ls)
    
    # Fit view
    gbox = o3d.geometry.AxisAlignedBoundingBox(gmin, gmax)
    vis.add_geometry(gbox)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.set_lookat(gbox.get_center())
    ctr.set_front([-1.0, -1.0, -1.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(1.0)
    vis.remove_geometry(gbox, reset_bounding_box=False)
    
    print(f"[visualize_reconstruction] Showing {len(all_vis_data)} objects. "
          f"Visible: {summary['visible_objects']}, Invisible: {summary['invisible_objects']}")
    print("[visualize_reconstruction] Controls: mouse rotate, wheel zoom, right-button pan, R reset, Q quit")
    vis.run()
    vis.destroy_window()


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
    use_accumulated_points: bool = True,
):
    """
    Visualize per-object point clouds and 3D boxes using Open3D.

    Each element in frame_objs is expected to be a dict with keys:
      - 'track_id': int
      - 'points': (N,3) numpy array (current frame points in world or camera)
      - 'points_accumulated': (M,3) numpy array (accumulated points across frames)
      - 'bbox_3d': { 'aabb': {'min': [..], 'max': [..]}, 'obb': {'center':[..], 'extent':[..], 'R':[[..],[..],[..]]} }

    - AABB is drawn in red.
    - OBB is drawn in green.
    - Points are colored by track id.
    - Labels (track IDs) are added as small spheres at OBB/AABB center; if 3D text mesh is available, it's also added.
    - use_accumulated_points: if True, visualize accumulated points instead of current frame points
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
        
        # Choose which points to display
        if use_accumulated_points and 'points_accumulated' in obj:
            pts = obj.get('points_accumulated', None)
        else:
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

def update_graph(current_graph, persistan_graph):
    """
    Update persistent graph with current frame graph.
    Args:
        current_graph: nx.MultiDiGraph of current frame
        persistan_graph: nx.MultiDiGraph of accumulated frames
    Returns:
        updated persistent graph
    """
    if persistan_graph is None:
        persistan_graph = current_graph.copy()
        return persistan_graph
    
    for node_id, data in current_graph.nodes(data=True):
        if not persistan_graph.has_node(node_id):
            persistan_graph.add_node(node_id, data=data['data'])
        else:
            # Optionally, update existing node data if needed
            pass
    return persistan_graph

def match_nodes(persistent_graph, current_graph, iou_threshold=0.3, distance_threshold=0.3):
    """
    Match nodes between persistent and current graphs.
    
    Strategy:
    1. First try matching by track_id (most reliable)
    2. Fall back to spatial proximity for objects without reliable tracking
    3. Unmatched current nodes will be added as new nodes
    
    Args:
        persistent_graph: nx.MultiDiGraph - accumulated graph
        current_graph: nx.MultiDiGraph - current frame graph
        iou_threshold: float - 3D IoU threshold for spatial matching
        distance_threshold: float - Euclidean distance threshold (meters)
    
    Returns:
        node_mapping: dict - maps current_graph node IDs to persistent_graph node IDs
    """
    node_mapping = {}
    matched_persistent_nodes = set()
    
    # Step 1: Match by track_id (highest priority)
    for curr_node, curr_data in current_graph.nodes(data=True):
        curr_track_id = curr_data.get('data', {}).get('track_id', None)
        
        # Skip invalid track_ids
        if curr_track_id is None or curr_track_id == -10 or curr_track_id < 0:
            continue
        
        # Search for matching track_id in persistent graph
        for persist_node, persist_data in persistent_graph.nodes(data=True):
            persist_track_id = persist_data.get('data', {}).get('track_id', None)
            
            if persist_track_id == curr_track_id and persist_node not in matched_persistent_nodes:
                node_mapping[curr_node] = persist_node
                matched_persistent_nodes.add(persist_node)
                break
    
    # Step 2: Match remaining nodes by spatial proximity
    for curr_node, curr_data in current_graph.nodes(data=True):
        # Skip if already matched
        if curr_node in node_mapping:
            continue
        
        # Skip if no valid 3D bounding box
        curr_bbox_3d = curr_data.get('data', {}).get('bbox_3d', {})
        curr_obb = curr_bbox_3d.get('obb')
        if curr_obb is None:
            continue
        
        curr_center = np.array(curr_obb['center'])
        curr_points = np.array(curr_data.get('data', {}).get('points', []))
        
        best_match = None
        best_score = 0
        
        for persist_node, persist_data in persistent_graph.nodes(data=True):
            # Skip already matched nodes
            if persist_node in matched_persistent_nodes:
                continue
            
            persist_bbox_3d = persist_data.get('data', {}).get('bbox_3d', {})
            persist_obb = persist_bbox_3d.get('obb')
            if persist_obb is None:
                continue
            
            persist_center = np.array(persist_obb['center'])
            persist_points = np.array(persist_data.get('data', {}).get('points', []))
            
            # Calculate center distance
            distance = np.linalg.norm(curr_center - persist_center)
            
            # If too far, skip
            if distance > distance_threshold:
                continue
            
            # Calculate 3D IoU if points are available
            if len(curr_points) > 0 and len(persist_points) > 0:
                iou = calculate_3d_iou(curr_points, persist_points)
                score = iou
            else:
                # Fall back to distance-based score (inverse distance)
                score = 1.0 / (1.0 + distance)
            
            # Keep track of best match
            if score > best_score and score > iou_threshold:
                best_score = score
                best_match = persist_node
        
        # If found a good match, add to mapping
        if best_match is not None:
            # print(f"[match_nodes] Matched current node {curr_node} to persistent node {best_match} with score {best_score:.3f}")
            node_mapping[curr_node] = best_match
            matched_persistent_nodes.add(best_match)
    
    # Step 3: Assign new IDs to unmatched current nodes
    for curr_node in current_graph.nodes():
        if curr_node not in node_mapping:
            node_mapping[curr_node] = curr_node
    
    return node_mapping


def calculate_3d_iou(points1, points2):
    """
    Calculate 3D Intersection over Union between two point clouds.
    Simple approximation using axis-aligned bounding boxes.
    
    Args:
        points1: np.array of shape (N, 3)
        points2: np.array of shape (M, 3)
    
    Returns:
        iou: float - IoU score between 0 and 1
    """
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # Get axis-aligned bounding boxes
    min1 = np.min(points1, axis=0)
    max1 = np.max(points1, axis=0)
    min2 = np.min(points2, axis=0)
    max2 = np.max(points2, axis=0)
    
    # Calculate intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    
    # Check if there's an intersection
    if np.any(inter_min >= inter_max):
        return 0.0
    
    # Calculate volumes
    inter_volume = np.prod(inter_max - inter_min)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    
    # Calculate IoU
    union_volume = volume1 + volume2 - inter_volume
    iou = inter_volume / union_volume if union_volume > 0 else 0.0
    
    return iou


def calculate_bbox_iou_3d(obb1, obb2):
    """
    Alternative: Calculate IoU using oriented bounding boxes.
    For more accurate matching with rotated objects.
    
    Args:
        obb1: dict with 'center', 'extent', 'rotation'
        obb2: dict with 'center', 'extent', 'rotation'
    
    Returns:
        iou: float - approximate IoU
    """
    # This is a simplified version
    # For production, consider using libraries like Open3D for accurate OBB IoU
    
    center1 = np.array(obb1['center'])
    center2 = np.array(obb2['center'])
    extent1 = np.array(obb1.get('extent', [1, 1, 1]))
    extent2 = np.array(obb2.get('extent', [1, 1, 1]))
    
    # Simple approximation: treat as axis-aligned for quick check
    min1 = center1 - extent1 / 2
    max1 = center1 + extent1 / 2
    min2 = center2 - extent2 / 2
    max2 = center2 + extent2 / 2
    
    # Calculate intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    
    if np.any(inter_min >= inter_max):
        return 0.0
    
    inter_volume = np.prod(inter_max - inter_min)
    volume1 = np.prod(extent1)
    volume2 = np.prod(extent2)
    union_volume = volume1 + volume2 - inter_volume
    
    return inter_volume / union_volume if union_volume > 0 else 0.0

def merge_scene_graphs(persistent_graph, current_graph):
    """
    Merge current frame's scene graph into persistent graph.
    Handles egocentric vs allocentric relationships appropriately.
    """
    
    # Step 1: Match nodes (using track_id + spatial proximity fallback)
    node_mapping = match_nodes(persistent_graph, current_graph, 
                               iou_threshold=0.3)
    
    # Step 2: Update node attributes
    for curr_node, persist_node in node_mapping.items():
        if persist_node in persistent_graph:
            # Update with latest observation
            persistent_graph.nodes[persist_node].update(
                current_graph.nodes[curr_node]
            )
        else:
            # Add new node
            persistent_graph.add_node(persist_node, 
                **current_graph.nodes[curr_node])
    
    # Step 3: Handle EGOCENTRIC edges (camera-dependent)
    egocentric_classes = ['proximity', 'middle_furniture']
    
    # Remove old egocentric edges for all nodes
    # current_nodes = set(node_mapping.values())
    # for u, v, key, data in list(persistent_graph.edges(keys=True, data=True)):
    #     if data['label_class'] in egocentric_classes:
    #         persistent_graph.remove_edge(u, v, key)
    
    # Remove old egocentric edges for nodes in current view only
    current_nodes = set(node_mapping.values())
    for u, v, key, data in list(persistent_graph.edges(keys=True, data=True)):
        if data['label_class'] in egocentric_classes:
            if u in current_nodes and v in current_nodes:
                persistent_graph.remove_edge(u, v, key)
    
    # Add new egocentric edges from current frame
    for u, v, key, data in current_graph.edges(keys=True, data=True):
        if data['label_class'] in egocentric_classes:
            u_mapped = node_mapping[u]
            v_mapped = node_mapping[v]
            persistent_graph.add_edge(u_mapped, v_mapped, **data)

    # remove all aligned furniture edges: because in dynamic scenes these are unreliable, should be updated
    # for every frame
    for u, v, key, data in list(persistent_graph.edges(keys=True, data=True)):
        if data['label_class'] == 'aligned_furniture':
            persistent_graph.remove_edge(u, v, key)
    
    # Step 4: Handle ALLOCENTRIC edges (camera-independent)
    allocentric_classes = ['support', 'embedded', 'hanging', 
                          'oppo_support', 'aligned_furniture']
    
    for u, v, key, data in current_graph.edges(keys=True, data=True):
        if data['label_class'] not in allocentric_classes:
            continue
            
        u_mapped = node_mapping[u]
        v_mapped = node_mapping[v]
        curr_label = data['label']
        curr_class = data['label_class']
        
        # Check for existing edges between these nodes
        edge_found = False
        if persistent_graph.has_edge(u_mapped, v_mapped):
            for edge_key in list(persistent_graph[u_mapped][v_mapped].keys()):
                edge_data = persistent_graph[u_mapped][v_mapped][edge_key]
                persist_class = edge_data.get('label_class')
                persist_label = edge_data.get('label')
                
                # Same label_class
                if persist_class == curr_class:
                    if persist_label == curr_label:
                        # Same edge - update attributes (e.g., confidence)
                        persistent_graph[u_mapped][v_mapped][edge_key].update(data)
                    else:
                        # Different label but same class - replace
                        persistent_graph.remove_edge(u_mapped, v_mapped, edge_key)
                        persistent_graph.add_edge(u_mapped, v_mapped, **data)
                    edge_found = True
                    break
                
                # Conflicting allocentric classes (e.g., support vs embedded)
                elif persist_class in allocentric_classes:
                    # Remove conflicting relationship
                    # (object can't be both ON and INSIDE something)
                    if are_conflicting_relations(persist_class, curr_class):
                        persistent_graph.remove_edge(u_mapped, v_mapped, edge_key)
                        # print(f"[merge_scene_graphs] Removed conflicting allocentric edge (persisted: {persist_class} vs current: {curr_class}).")
                        # print(f"[merge_scene_graphs] {u_mapped}=>{v_mapped},K: {edge_key}, cl: {curr_label}, l:{persist_label}")
                        if persist_class == 'support':
                            # special case: when support is removed, also remove the opposite support edge
                            if persistent_graph.has_edge(v_mapped, u_mapped):
                                # print edges before removal
                                # print(f"all keys between {v_mapped}=>{u_mapped} before removal: {list(persistent_graph[v_mapped][u_mapped].keys())}")
                                for edge_key2 in list(persistent_graph[v_mapped][u_mapped].keys()):
                                    edge_data2 = persistent_graph[v_mapped][u_mapped][edge_key2]
                                    persist_class2 = edge_data2.get('label_class')
                                    persist_label2 = edge_data2.get('label')
                                    # print(f"{v_mapped}->{u_mapped}: k:{edge_key2}, cl:{persist_class2}, l:{persist_label2}")  
                                    if persist_class2 == 'oppo_support':
                                        # print(f"[merge_scene_graphs] Removing opposite support edge with key v_mapped: {v_mapped} -> u_mapped: {u_mapped}: k:{edge_key2}.")
                                        persistent_graph.remove_edge(v_mapped, u_mapped, edge_key2)
                                        break
                                # pass
        
        # Add edge if not found
        if not edge_found:
            persistent_graph.add_edge(u_mapped, v_mapped, **data)
    
    return persistent_graph


def are_conflicting_relations(class1, class2):
    """Check if two allocentric relation types are mutually exclusive"""
    conflicts = {
        'support': ['embedded', 'hanging'],
        'embedded': ['support', 'hanging'],
        'hanging': ['support', 'embedded']
    }
    return class2 in conflicts.get(class1, [])

