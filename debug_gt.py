"""
Debug script to visualize ground truth semantic segmentation.
Shows each semantic ID's mask individually to verify data correctness.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------
# Configuration - EDIT THESE
# -------------------------
SEG_INFO_JSON = "/home/maribjonov_mr/IsaacSim_bench/scene_7/seg/semantic000001_info.json"
SEG_PNG = "/home/maribjonov_mr/IsaacSim_bench/scene_7/seg/semantic000001.png"
BBOX_JSON = "/home/maribjonov_mr/IsaacSim_bench/scene_7/bbox/bboxes000001_info.json"


def load_seg_data():
    """Load segmentation info and image."""
    # Load JSON
    with open(SEG_INFO_JSON, 'r') as f:
        seg_info = json.load(f)
    
    # Load PNG as BGR (OpenCV default)
    seg_img = cv2.imread(SEG_PNG, cv2.IMREAD_COLOR)
    if seg_img is None:
        raise RuntimeError(f"Failed to load {SEG_PNG}")
    
    print(f"Loaded segmentation image: {seg_img.shape}")
    print(f"Loaded segmentation info with {len(seg_info)} entries")
    
    return seg_info, seg_img


def load_bbox_data():
    """Load bbox data to see what objects exist."""
    with open(BBOX_JSON, 'r') as f:
        bbox_data = json.load(f)
    
    # Extract 3D bboxes
    bbox_3d = bbox_data.get("bboxes", {}).get("bbox_3d", {}).get("boxes", [])
    
    print(f"\nLoaded {len(bbox_3d)} objects from bbox JSON:")
    for i, b in enumerate(bbox_3d, 1):
        print(f"  {i}. Track:{b.get('track_id'):3d} Sem:{b.get('semantic_id'):3d} Label:'{b.get('label', 'N/A')}'")
    
    return bbox_data, bbox_3d


def visualize_semantic_masks(seg_info, seg_img):
    """Visualize each semantic ID's mask."""
    # Convert to RGB for display
    seg_img_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    
    # Get all semantic IDs
    semantic_ids = sorted([int(k) for k in seg_info.keys() if str(k).isdigit()])
    
    print(f"\n{'='*70}")
    print(f"Visualizing {len(semantic_ids)} semantic IDs")
    print(f"{'='*70}")
    
    for semantic_id in semantic_ids:
        # Get label and color from JSON
        entry = seg_info[str(semantic_id)]
        label_dict = entry.get("label", {})
        
        # Extract label text
        if isinstance(label_dict, dict) and len(label_dict) > 0:
            label = next(iter(label_dict.values()))
        else:
            label = str(label_dict)
        
        color_bgr = entry.get("color_bgr", [0, 0, 0])
        color_bgr_tuple = tuple(color_bgr)
        color_rgb_tuple = (color_bgr[2], color_bgr[1], color_bgr[0])
        
        # Create mask: find all pixels matching this color
        mask = np.all(seg_img == np.array(color_bgr, dtype=np.uint8), axis=2)
        pixel_count = np.sum(mask)
        
        print(f"\nSemantic ID: {semantic_id}")
        print(f"  Label: '{label}'")
        print(f"  Color BGR: {color_bgr}")
        print(f"  Color RGB: {list(color_rgb_tuple)}")
        print(f"  Pixels: {pixel_count}")
        
        if pixel_count == 0:
            print(f"  -> SKIPPING (no pixels found)")
            continue
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Full RGB image
        axes[0].imshow(seg_img_rgb)
        axes[0].set_title("Full Segmentation (RGB)")
        axes[0].axis('off')
        
        # Panel 2: Binary mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"Mask for Semantic ID {semantic_id}\n{pixel_count} pixels")
        axes[1].axis('off')
        
        # Panel 3: Mask overlay on image
        overlay = seg_img_rgb.copy().astype(np.float32)
        mask_color = np.zeros_like(seg_img_rgb, dtype=np.float32)
        mask_color[mask] = [255, 0, 0]  # Red for mask
        overlay = overlay * 0.6 + mask_color * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Mask Overlay (Red)")
        axes[2].axis('off')
        
        # Overall title
        fig.suptitle(
            f"Semantic ID: {semantic_id} | Label: '{label}'\n"
            f"Color BGR: {color_bgr} | Color RGB: {list(color_rgb_tuple)} | Pixels: {pixel_count}",
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"  -> Shown. Close window to continue...")


def check_color_uniqueness(seg_info):
    """Check if colors are unique."""
    print(f"\n{'='*70}")
    print("Checking color uniqueness...")
    print(f"{'='*70}")
    
    color_to_ids = {}
    for semantic_id_str, entry in seg_info.items():
        if not semantic_id_str.isdigit():
            continue
        
        semantic_id = int(semantic_id_str)
        color_bgr = tuple(entry.get("color_bgr", [0, 0, 0]))
        
        if color_bgr not in color_to_ids:
            color_to_ids[color_bgr] = []
        color_to_ids[color_bgr].append(semantic_id)
    
    # Check for duplicates
    duplicates = {color: ids for color, ids in color_to_ids.items() if len(ids) > 1}
    
    if duplicates:
        print(f"⚠ WARNING: Found {len(duplicates)} colors shared by multiple semantic IDs:")
        for color, ids in duplicates.items():
            print(f"  Color BGR {color} -> Semantic IDs: {ids}")
            for sid in ids:
                label_dict = seg_info[str(sid)].get("label", {})
                if isinstance(label_dict, dict) and len(label_dict) > 0:
                    label = next(iter(label_dict.values()))
                else:
                    label = str(label_dict)
                print(f"    - Semantic ID {sid}: '{label}'")
    else:
        print("✓ All colors are unique - each semantic ID has a distinct color")


def main():
    print("="*70)
    print("Ground Truth Segmentation Debug Tool")
    print("="*70)
    
    # Load data
    seg_info, seg_img = load_seg_data()
    bbox_data, bbox_3d = load_bbox_data()
    
    # Check color uniqueness
    check_color_uniqueness(seg_info)
    
    # Visualize each mask
    visualize_semantic_masks(seg_info, seg_img)
    
    print(f"\n{'='*70}")
    print("Debug complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
