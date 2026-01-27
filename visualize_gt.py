"""
Ground truth visualization script for Isaac Sim benchmark dataset.

Use this to inspect GT data before running benchmarks.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# Camera intrinsics
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FOCAL_LENGTH = 50.0
HORIZONTAL_APERTURE = 80.0
VERTICAL_APERTURE = 45.0
PNG_DEPTH_SCALE = 0.00015244


def load_frame_data(scene_path: str, frame_idx: int) -> Dict:
    """Load all data for a single frame."""
    scene_path = Path(scene_path)
    frame_num = frame_idx + 1  # 1-indexed
    
    data = {
        'rgb': None,
        'depth': None,
        'segmentation': None,
        'seg_info': {},
        'bbox_2d': [],
        'bbox_3d': []
    }
    
    # Load RGB
    rgb_path = scene_path / "rgb" / f"frame{frame_num:06d}.jpg"
    if rgb_path.exists():
        data['rgb'] = cv2.imread(str(rgb_path))
        data['rgb'] = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
    
    # Load depth
    depth_path = scene_path / "depth" / f"depth{frame_num:06d}.png"
    if depth_path.exists():
        depth_raw = np.array(Image.open(depth_path))
        data['depth'] = depth_raw.astype(np.float32) * PNG_DEPTH_SCALE
    
    # Load segmentation
    seg_path = scene_path / "seg" / f"semantic{frame_num:06d}.png"
    if seg_path.exists():
        data['segmentation'] = cv2.imread(str(seg_path))
        data['segmentation'] = cv2.cvtColor(data['segmentation'], cv2.COLOR_BGR2RGB)
    
    # Load segmentation info
    seg_info_path = scene_path / "seg" / f"semantic{frame_num:06d}_info.json"
    if seg_info_path.exists():
        with open(seg_info_path, 'r') as f:
            raw_info = json.load(f)
            for sem_id, info in raw_info.items():
                data['seg_info'][int(sem_id)] = {
                    'class': info['label']['class'],
                    'color_rgb': info['color_bgr'][::-1]  # BGR to RGB
                }
    
    # Load bboxes
    bbox_path = scene_path / "bbox" / f"bbox{frame_num:06d}.json"
    if bbox_path.exists():
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)
        
        # 2D bboxes
        if 'bboxes' in bbox_data and 'bbox_2d_tight' in bbox_data['bboxes']:
            for box in bbox_data['bboxes']['bbox_2d_tight']['boxes']:
                data['bbox_2d'].append({
                    'track_id': box.get('bbox_id', -1),
                    'prim_path': box.get('prim_path', ''),
                    'label': _extract_label(box.get('label', {})),
                    'xyxy': box.get('xyxy', [0, 0, 0, 0]),
                    'visibility': 1.0 - box.get('visibility_or_occlusion', 0.0)
                })
        
        # 3D bboxes
        if 'bboxes' in bbox_data and 'bbox_3d' in bbox_data['bboxes']:
            for box in bbox_data['bboxes']['bbox_3d']['boxes']:
                data['bbox_3d'].append({
                    'track_id': box.get('track_id', box.get('bbox_id', -1)),
                    'prim_path': box.get('prim_path', ''),
                    'label': box.get('label', 'unknown'),
                    'aabb': box.get('aabb_xyzmin_xyzmax', []),
                    'transform': box.get('transform_4x4', []),
                    'occlusion': box.get('occlusion_ratio', 0.0)
                })
    
    return data


def _extract_label(label_dict) -> str:
    """Extract class label from label dict."""
    if isinstance(label_dict, str):
        return label_dict
    if isinstance(label_dict, dict):
        for k, v in label_dict.items():
            return str(v) if v else str(k)
    return 'unknown'


def visualize_frame(scene_path: str, frame_idx: int, 
                    show_rgb: bool = True,
                    show_depth: bool = True,
                    show_seg: bool = True,
                    show_bbox: bool = True,
                    figsize: Tuple[int, int] = (20, 10)):
    """Visualize all data for a single frame."""
    
    data = load_frame_data(scene_path, frame_idx)
    
    # Count how many images to show
    n_images = sum([show_rgb and data['rgb'] is not None,
                   show_depth and data['depth'] is not None,
                   show_seg and data['segmentation'] is not None])
    
    if n_images == 0:
        print(f"No data found for frame {frame_idx}")
        return
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # RGB with bboxes
    if show_rgb and data['rgb'] is not None:
        ax = axes[ax_idx]
        img = data['rgb'].copy()
        
        if show_bbox:
            for box in data['bbox_2d']:
                xyxy = box['xyxy']
                label = box['label']
                track_id = box['track_id']
                
                # Skip background
                if label.lower() in ['background', 'unlabelled']:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                text = f"{label} (ID:{track_id})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 0), 1)
        
        ax.imshow(img)
        ax.set_title(f'RGB Frame {frame_idx} ({len(data["bbox_2d"])} objects)')
        ax.axis('off')
        ax_idx += 1
    
    # Depth
    if show_depth and data['depth'] is not None:
        ax = axes[ax_idx]
        depth_vis = data['depth'].copy()
        depth_vis[depth_vis <= 0] = np.nan
        
        im = ax.imshow(depth_vis, cmap='viridis')
        ax.set_title(f'Depth Frame {frame_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Depth (m)')
        ax_idx += 1
    
    # Segmentation
    if show_seg and data['segmentation'] is not None:
        ax = axes[ax_idx]
        ax.imshow(data['segmentation'])
        ax.set_title(f'Segmentation Frame {frame_idx}')
        ax.axis('off')
        
        # Add legend
        if data['seg_info']:
            legend_items = []
            for sem_id, info in data['seg_info'].items():
                color = np.array(info['color_rgb']) / 255
                legend_items.append((info['class'], color))
            
            # Show first 10 classes
            for i, (cls, color) in enumerate(legend_items[:10]):
                ax.plot([], [], 's', color=color, markersize=10, label=cls)
            if len(legend_items) > 10:
                ax.plot([], [], 's', color='gray', markersize=10, label=f'...+{len(legend_items)-10} more')
            ax.legend(loc='upper right', fontsize=8)
        
        ax_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # Print 3D bbox info
    if data['bbox_3d']:
        print(f"\n3D Bounding Boxes for Frame {frame_idx}:")
        print("-" * 60)
        for box in data['bbox_3d']:
            if box['label'].lower() in ['background', 'unlabelled']:
                continue
            aabb = box['aabb']
            if len(aabb) == 6:
                size = [aabb[3] - aabb[0], aabb[4] - aabb[1], aabb[5] - aabb[2]]
                print(f"  {box['label']:20s} ID:{box['track_id']:3d}  "
                      f"Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]  "
                      f"Occl: {box['occlusion']:.1%}")


def visualize_tracking_over_time(scene_path: str, 
                                  track_id: int,
                                  start_frame: int = 0,
                                  end_frame: int = None,
                                  step: int = 5,
                                  figsize: Tuple[int, int] = (20, 4)):
    """Visualize how a specific object (track_id) looks across frames."""
    
    scene_path = Path(scene_path)
    
    # Count frames
    rgb_dir = scene_path / "rgb"
    n_frames = len(list(rgb_dir.glob("frame*.jpg")))
    
    if end_frame is None:
        end_frame = n_frames
    
    frames_to_show = list(range(start_frame, min(end_frame, n_frames), step))
    
    if not frames_to_show:
        print("No frames to show")
        return
    
    fig, axes = plt.subplots(1, len(frames_to_show), figsize=figsize)
    if len(frames_to_show) == 1:
        axes = [axes]
    
    for ax, frame_idx in zip(axes, frames_to_show):
        data = load_frame_data(str(scene_path), frame_idx)
        
        if data['rgb'] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'Frame {frame_idx}')
            continue
        
        img = data['rgb'].copy()
        
        # Find the bbox with matching track_id
        found = False
        for box in data['bbox_2d']:
            if box['track_id'] == track_id:
                found = True
                xyxy = box['xyxy']
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Crop and show
                pad = 50
                crop_x1 = max(0, x1 - pad)
                crop_y1 = max(0, y1 - pad)
                crop_x2 = min(img.shape[1], x2 + pad)
                crop_y2 = min(img.shape[0], y2 + pad)
                
                crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                ax.imshow(crop)
                ax.set_title(f'Frame {frame_idx}\n{box["label"]}')
                break
        
        if not found:
            ax.imshow(img)
            ax.set_title(f'Frame {frame_idx}\nNot visible')
        
        ax.axis('off')
    
    plt.suptitle(f'Tracking Object ID={track_id} Over Time', fontsize=14)
    plt.tight_layout()
    plt.show()


def print_scene_info(scene_path: str):
    """Print summary information about a scene."""
    scene_path = Path(scene_path)
    
    print(f"\nScene: {scene_path.name}")
    print("=" * 60)
    
    # Count frames
    for folder in ['rgb', 'depth', 'seg', 'bbox']:
        folder_path = scene_path / folder
        if folder_path.exists():
            n_files = len(list(folder_path.iterdir()))
            print(f"  {folder:10s}: {n_files} files")
        else:
            print(f"  {folder:10s}: NOT FOUND")
    
    # Load first frame to get object info
    data = load_frame_data(str(scene_path), 0)
    
    if data['bbox_3d']:
        print(f"\nObjects in first frame:")
        print("-" * 60)
        
        # Count by class
        class_counts = {}
        for box in data['bbox_3d']:
            label = box['label']
            if label not in class_counts:
                class_counts[label] = []
            class_counts[label].append(box['track_id'])
        
        for label, ids in sorted(class_counts.items()):
            print(f"  {label:25s}: {len(ids)} instance(s) - IDs: {ids}")
    
    # Check trajectory
    traj_path = scene_path / "traj.txt"
    if traj_path.exists():
        with open(traj_path, 'r') as f:
            n_poses = len(f.readlines())
        print(f"\nCamera poses: {n_poses}")
    else:
        print(f"\nCamera poses: NOT FOUND")


def list_all_objects(scene_path: str) -> Dict[int, Dict]:
    """Get a list of all unique objects across all frames."""
    scene_path = Path(scene_path)
    
    # Count frames
    rgb_dir = scene_path / "rgb"
    n_frames = len(list(rgb_dir.glob("frame*.jpg")))
    
    objects = {}  # track_id -> info
    
    for frame_idx in range(n_frames):
        data = load_frame_data(str(scene_path), frame_idx)
        
        for box in data['bbox_3d']:
            track_id = box['track_id']
            label = box['label']
            
            if track_id not in objects:
                objects[track_id] = {
                    'track_id': track_id,
                    'label': label,
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'total_frames': 1
                }
            else:
                objects[track_id]['last_frame'] = frame_idx
                objects[track_id]['total_frames'] += 1
    
    return objects


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_gt.py <scene_path> [frame_idx]")
        print("       python visualize_gt.py <scene_path> --info")
        print("       python visualize_gt.py <scene_path> --track <track_id>")
        sys.exit(1)
    
    scene_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--info':
            print_scene_info(scene_path)
            print("\nAll objects in scene:")
            objects = list_all_objects(scene_path)
            for track_id, info in sorted(objects.items()):
                print(f"  ID {track_id:3d}: {info['label']:25s} "
                      f"frames {info['first_frame']}-{info['last_frame']} "
                      f"({info['total_frames']} total)")
        elif sys.argv[2] == '--track':
            track_id = int(sys.argv[3])
            visualize_tracking_over_time(scene_path, track_id)
        else:
            frame_idx = int(sys.argv[2])
            visualize_frame(scene_path, frame_idx)
    else:
        # Default: show first frame
        print_scene_info(scene_path)
        visualize_frame(scene_path, 0)
