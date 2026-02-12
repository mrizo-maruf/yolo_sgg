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
SEG_INFO_JSON = "/home/maribjonov_mr/IsaacSim_bench/scene_1/seg/semantic000002_info.json"
SEG_PNG = "/home/maribjonov_mr/IsaacSim_bench/scene_1/seg/semantic000002.png"
BBOX_JSON = "/home/maribjonov_mr/IsaacSim_bench/scene_1/bbox/bboxes000002_info.json"


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
    # Extract 2D bboxes
    bbox_2d = bbox_data.get("bboxes", {}).get("bbox_2d_tight", {}).get("boxes", [])
    
    print(f"\nLoaded {len(bbox_3d)} 3D boxes and {len(bbox_2d)} 2D boxes from bbox JSON:")
    print("\n3D Boxes:")
    for i, b in enumerate(bbox_3d, 1):
        print(f"  {i}. bbox_3d_id:{b.get('bbox_3d_id', 'N/A'):3} "
              f"bbox_2d_id:{b.get('bbox_2d_id', 'N/A'):3} "
              f"instance_seg_id:{b.get('instance_seg_id', 'N/A'):3} "
              f"Label:'{b.get('label', 'N/A')}'")
    
    print("\n2D Boxes:")
    for i, b in enumerate(bbox_2d, 1):
        print(f"  {i}. bbox_2d_id:{b.get('bbox_2d_id', 'N/A'):3} "
              f"bbox_3d_id:{b.get('bbox_3d_id', 'N/A'):3} "
              f"instance_seg_id:{b.get('instance_seg_id', 'N/A'):3} "
              f"Label:{b.get('label', 'N/A')}")
    
    return bbox_data, bbox_3d, bbox_2d


def visualize_semantic_masks(seg_info, seg_img, only_valid_ids: bool = True):
    """Visualize each semantic ID's mask.
    
    Args:
        seg_info: Segmentation info dict
        seg_img: Segmentation image (BGR)
        only_valid_ids: If True, only show objects with all IDs (bbox_2d_id, bbox_3d_id) >= 0
    """
    # Convert to RGB for display
    seg_img_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    
    # Get all instance IDs
    instance_ids = sorted([int(k) for k in seg_info.keys() if str(k).isdigit()])
    
    print(f"\n{'='*70}")
    print(f"Found {len(instance_ids)} instance segmentation IDs")
    print(f"{'='*70}")
    
    valid_count = 0
    skipped_count = 0
    
    for instance_id in instance_ids:
        # Get entry from JSON
        entry = seg_info[str(instance_id)]
        label_dict = entry.get("label", {})
        
        # Extract label text
        if isinstance(label_dict, dict) and len(label_dict) > 0:
            label = next(iter(label_dict.values()))
        else:
            label = str(label_dict)
        
        # Get cross-reference IDs (new format)
        bbox_2d_id = entry.get("bbox_2d_id", -1)
        bbox_3d_id = entry.get("bbox_3d_id", -1)
        
        # Filter: only show objects with all IDs valid
        if only_valid_ids:
            if bbox_2d_id < 0 or bbox_3d_id < 0:
                print(f"\nInstance ID {instance_id}: SKIPPED (incomplete IDs: 2d={bbox_2d_id}, 3d={bbox_3d_id})")
                skipped_count += 1
                continue
        
        color_bgr = entry.get("color_bgr", [0, 0, 0])
        color_bgr_tuple = tuple(color_bgr)
        color_rgb_tuple = (color_bgr[2], color_bgr[1], color_bgr[0])
        
        # Create mask: find all pixels matching this color
        mask = np.all(seg_img == np.array(color_bgr, dtype=np.uint8), axis=2)
        pixel_count = np.sum(mask)
        
        print(f"\nInstance ID: {instance_id}")
        print(f"  Label: '{label}'")
        print(f"  Cross-refs: bbox_2d_id={bbox_2d_id}, bbox_3d_id={bbox_3d_id}")
        print(f"  Color BGR: {color_bgr}")
        print(f"  Pixels: {pixel_count}")
        
        if pixel_count == 0:
            print(f"  -> SKIPPING (no pixels found)")
            continue
        
        valid_count += 1
        
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
            f"Instance ID: {instance_id} | Label: '{label}'\n"
            f"bbox_2d_id: {bbox_2d_id} | bbox_3d_id: {bbox_3d_id} | Pixels: {pixel_count}",
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"  -> Shown. Close window to continue...")
    
    print(f"\n{'='*70}")
    print(f"Summary: {valid_count} valid objects shown, {skipped_count} skipped (incomplete IDs)")
    print(f"{'='*70}")


def check_color_uniqueness(seg_info):
    """Check if colors are unique."""
    print(f"\n{'='*70}")
    print("Checking color uniqueness...")
    print(f"{'='*70}")
    
    color_to_ids = {}
    for instance_id_str, entry in seg_info.items():
        if not instance_id_str.isdigit():
            continue
        
        instance_id = int(instance_id_str)
        color_bgr = tuple(entry.get("color_bgr", [0, 0, 0]))
        
        if color_bgr not in color_to_ids:
            color_to_ids[color_bgr] = []
        color_to_ids[color_bgr].append(instance_id)
    
    # Check for duplicates
    duplicates = {color: ids for color, ids in color_to_ids.items() if len(ids) > 1}
    
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} colors shared by multiple instance IDs:")
        for color, ids in duplicates.items():
            print(f"  Color BGR {color} -> Instance IDs: {ids}")
            for iid in ids:
                entry = seg_info[str(iid)]
                label_dict = entry.get("label", {})
                if isinstance(label_dict, dict) and len(label_dict) > 0:
                    label = next(iter(label_dict.values()))
                else:
                    label = str(label_dict)
                bbox_2d_id = entry.get("bbox_2d_id", -1)
                bbox_3d_id = entry.get("bbox_3d_id", -1)
                print(f"    - Instance {iid}: '{label}' (2d={bbox_2d_id}, 3d={bbox_3d_id})")
    else:
        print("All colors are unique - each instance ID has a distinct color")


def main():
    print("="*70)
    print("Ground Truth Segmentation Debug Tool")
    print("="*70)
    
    # Load data
    seg_info, seg_img = load_seg_data()
    bbox_data, bbox_3d, bbox_2d = load_bbox_data()
    
    # Check color uniqueness
    check_color_uniqueness(seg_info)
    
    # Visualize each mask (only objects with all IDs valid)
    visualize_semantic_masks(seg_info, seg_img, only_valid_ids=True)
    
    print(f"\n{'='*70}")
    print("Debug complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
