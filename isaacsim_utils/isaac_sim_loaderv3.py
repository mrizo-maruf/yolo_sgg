import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class GTObject:
    track_id: int
    # Cross-reference IDs (all must be >= 0 for valid objects)
    instance_seg_id: int  # instance ID from segmentation (unique per object)
    bbox_2d_id: int       # ID from 2D bbox annotator
    bbox_3d_id: int       # ID from 3D bbox annotator
    
    class_name: str
    prim_path: Optional[str]  # USD prim path, e.g., "/World/env/bowl"

    bbox2d_xyxy: Optional[Tuple[float, float, float, float]]  # (x1,y1,x2,y2)
    box_3d_aabb_xyzmin_xyzmax: Tuple[float, float, float, float, float, float]
    box_3d_transform_4x4: np.ndarray  # (4,4)

    visibility: Optional[float]   # from 2D (visibility_or_occlusion)
    occlusion: Optional[float]    # from 3D (occlusion_ratio)

    mask: np.ndarray              # HxW bool


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    gt_objects: List[GTObject]

    # Optional extras if you want them later
    rgb: Optional[np.ndarray] = None   # HxWx3 BGR
    depth: Optional[np.ndarray] = None # HxW uint16 or float (depends on your decode)
    cam_transform_4x4: Optional[np.ndarray] = None  # from traj.txt, 4x4 matrix
    seg: Optional[np.ndarray] = None   # HxWx3 BGR semantic segmentation (for visualization)

# -----------------------------
# Loader
# -----------------------------

class IsaacSimSceneLoader:
    """
    Loads one Isaac Sim scene folder:
      scene/
        bbox/
        seg/
        rgb/
        depth/
        traj.txt
    """

    def __init__(
        self,
        scene_dir: str,
        load_rgb: bool = False,
        load_depth: bool = False,
        skip_labels: Optional[Set[str]] = None,
        require_all_ids: bool = True,  # Only load objects with all IDs >= 0
    ):
        self.scene_dir = Path(scene_dir)
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"Scene dir not found: {self.scene_dir}")

        self.bbox_dir = self.scene_dir / "bbox"
        self.seg_dir = self.scene_dir / "seg"
        self.rgb_dir = self.scene_dir / "rgb"
        self.depth_dir = self.scene_dir / "depth"

        for d in [self.bbox_dir, self.seg_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Required folder missing: {d}")

        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.skip_labels = skip_labels if skip_labels is not None else set()
        self.require_all_ids = require_all_ids

        # Optional: infer available frame indices from bbox files (robust)
        self.frame_indices = self._discover_frames()

        self.traj_data = self.load_traj()  # Optional: load trajectory if needed

    # ---------- public API ----------

    def load_traj(self) -> Optional[List[Dict[str, Any]]]:
        """
        Loads trajectory data from traj.txt.
        Each line is a 4x4 camera transform matrix flattened in row-major order.
        Returns a list of dicts with "frame_idx" and "cam_transform_4x4" keys.
        """
        traj_path = self.scene_dir / "traj.txt"
        if not traj_path.exists():
            return None
        
        traj_data = []
        with open(traj_path, "r") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                values = line.split()
                if len(values) != 16:
                    raise ValueError(f"Expected 16 values in traj.txt line {idx}, got {len(values)}")
                
                # Parse as floats and reshape to 4x4 matrix (row-major)
                transform = np.array([float(v) for v in values], dtype=np.float32).reshape(4, 4)
                
                traj_data.append({
                    "frame_idx": idx,
                    "cam_transform_4x4": transform
                })
        
        return traj_data
    
    def get_frame_data(self, frame_idx: int) -> FrameData:
        """
        Returns FrameData for the given frame index (1-based),
        including gt_objects built from bbox_3d entries.
        """
        bbox_path = self._bbox_json_path(frame_idx)
        seg_png_path = self._seg_png_path(frame_idx)
        seg_info_path = self._seg_info_json_path(frame_idx)

        bbox_data = self._read_json(bbox_path)
        seg_info = self._read_json(seg_info_path)

        seg_img = self._read_seg_bgr(seg_png_path)  # HxWx3 BGR

        # Build maps
        bbox2d_by_id = self._parse_bbox2d_tight(bbox_data)  # keyed by bbox_2d_id
        bbox3d_list = self._parse_bbox3d(bbox_data)

        # instance_id -> color_bgr (tuple) for mask extraction
        color_by_instance_id = self._parse_instance_color_map(seg_info)

        gt_objects: List[GTObject] = []
        for b3d in bbox3d_list:
            # Extract cross-reference IDs (new format)
            bbox_3d_id = int(b3d.get("bbox_3d_id", -1))
            bbox_2d_id = int(b3d.get("bbox_2d_id", -1))
            instance_seg_id = int(b3d.get("instance_seg_id", -1))
            track_id = int(b3d["track_id"])
            prim_path = b3d.get("prim_path", None)

            # Filter: require all IDs to be valid (>= 0) if enabled
            if self.require_all_ids:
                if bbox_3d_id < 0 or bbox_2d_id < 0 or instance_seg_id < 0:
                    print(f"  [LOADER] SKIPPING object '{prim_path}': incomplete IDs "
                          f"(3d={bbox_3d_id}, 2d={bbox_2d_id}, seg={instance_seg_id})")
                    continue

            # class name: prefer b3d["label"] (string), else derive from 2D label dict, else fallback
            b2d = bbox2d_by_id.get(bbox_2d_id)  # Look up 2D box by bbox_2d_id
            class_name = self._infer_class_name(b3d=b3d, b2d=b2d)
            class_name_lower = class_name.lower()
            if any(skip_label in class_name_lower for skip_label in self.skip_labels):
                print(f"  [LOADER] SKIPPING GT CLASS '{class_name_lower}'")
                continue

            # 2D info (from the matched 2D box)
            bbox2d_xyxy = None
            visibility = None
            if b2d is not None:
                xyxy = b2d.get("xyxy", None)
                if xyxy is not None and len(xyxy) == 4:
                    bbox2d_xyxy = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                visibility = b2d.get("visibility_or_occlusion", None)
                visibility = float(visibility) if visibility is not None else None

            # 3D info
            aabb = b3d["aabb_xyzmin_xyzmax"]
            aabb_xyzmin_xyzmax = (
                float(aabb[0]), float(aabb[1]), float(aabb[2]),
                float(aabb[3]), float(aabb[4]), float(aabb[5]),
            )

            T = np.array(b3d["transform_4x4"], dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError(f"Bad transform shape in {bbox_path}: {T.shape}")

            occlusion = b3d.get("occlusion_ratio", None)
            occlusion = float(occlusion) if occlusion is not None else None

            # Mask extraction: use instance_seg_id to look up color from segmentation
            color = color_by_instance_id.get(instance_seg_id, None)
            if color is None:
                # If missing mapping, make empty mask
                print(f"  [LOADER] WARNING: No color mapping for instance_seg_id={instance_seg_id}, class='{class_name}'")
                mask = np.zeros(seg_img.shape[:2], dtype=bool)
            else:
                # seg_img is BGR; color is (b,g,r)
                mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=2)

            gt_objects.append(
                GTObject(
                    track_id=track_id,
                    instance_seg_id=instance_seg_id,
                    bbox_2d_id=bbox_2d_id,
                    bbox_3d_id=bbox_3d_id,
                    class_name=class_name,
                    prim_path=prim_path,
                    bbox2d_xyxy=bbox2d_xyxy,
                    box_3d_aabb_xyzmin_xyzmax=aabb_xyzmin_xyzmax,
                    box_3d_transform_4x4=T,
                    visibility=visibility,
                    occlusion=occlusion,
                    mask=mask,
                )
            )

        # Debug: print all class names before filtering
        print(f"  [LOADER] Frame {frame_idx}: Found {len(gt_objects)} objects before filtering")
        if gt_objects:
            class_names = [obj.class_name for obj in gt_objects]
            print(f"  [LOADER] Class names: {class_names}")
        
        rgb = self._read_rgb_bgr(self._rgb_path(frame_idx)) if self.load_rgb else None
        depth = self._read_depth(self._depth_path(frame_idx)) if self.load_depth else None

        # Load camera transform from trajectory data (frame_idx is 1-based, list is 0-based)
        cam_transform_4x4 = None
        if self.traj_data is not None and len(self.traj_data) >= frame_idx:
            cam_transform_4x4 = self.traj_data[frame_idx - 1]["cam_transform_4x4"]
        
        return FrameData(
            frame_idx=frame_idx,
            gt_objects=gt_objects,
            rgb=rgb,
            depth=depth,
            cam_transform_4x4=cam_transform_4x4,
            seg=seg_img,
        )

    # ---------- internals ----------

    def _discover_frames(self) -> List[int]:
        """
        Discovers frame indices by scanning bbox/*.json.
        """
        if not self.bbox_dir.exists():
            return []
        frames = []
        for p in sorted(self.bbox_dir.glob("bboxes*_info.json")):
            # expects bboxes000001_info.json
            stem = p.name.replace("bboxes", "").replace("_info.json", "")
            if stem.isdigit():
                frames.append(int(stem))
        return frames

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_seg_bgr(path: Path) -> np.ndarray:
        """
        Reads semanticXXXXXX.png as BGR uint8 (keeps BGR since your json is color_bgr).
        """
        if not path.exists():
            raise FileNotFoundError(f"Missing seg png: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"Failed to read seg png: {path}")
        return img

    @staticmethod
    def _read_rgb_bgr(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing rgb jpg: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"Failed to read rgb jpg: {path}")
        return img

    @staticmethod
    def _read_depth(path: Path) -> np.ndarray:
        """
        Depth is stored as PNG with max value 65535 (uint16).
        You can later convert to meters if you have min/max mapping.
        """
        if not path.exists():
            raise FileNotFoundError(f"Missing depth png: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # likely uint16
        if img is None:
            raise RuntimeError(f"Failed to read depth png: {path}")
        return img

    @staticmethod
    def _parse_bbox2d_tight(bbox_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Returns bbox_2d_id -> 2D box dict.
        Uses the new 'bbox_2d_id' field for keying.
        """
        b2d = (
            bbox_data.get("bboxes", {})
                    .get("bbox_2d_tight", {})
                    .get("boxes", [])
        )
        out: Dict[int, Dict[str, Any]] = {}
        for b in b2d:
            # Use new format key 'bbox_2d_id'
            bbox_2d_id = b.get("bbox_2d_id", b.get("bbox_id", None))
            if bbox_2d_id is not None:
                out[int(bbox_2d_id)] = b
        return out

    @staticmethod
    def _parse_bbox3d(bbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns list of 3D box dicts.
        Uses new format with bbox_3d_id, bbox_2d_id, instance_seg_id.
        """
        b3d = (
            bbox_data.get("bboxes", {})
                    .get("bbox_3d", {})
                    .get("boxes", [])
        )
        # Enforce required keys for new format
        required = {"bbox_3d_id", "track_id", "aabb_xyzmin_xyzmax", "transform_4x4"}
        for i, b in enumerate(b3d):
            missing = required - set(b.keys())
            if missing:
                raise ValueError(f"bbox_3d[{i}] missing keys {missing}")
        return b3d

    @staticmethod
    def _parse_instance_color_map(seg_info: Dict[str, Any]) -> Dict[int, Tuple[int, int, int]]:
        """
        seg_info keys are instance seg IDs as strings ("0","1","12",...).
        Each entry has: instance_seg_id, bbox_2d_id, bbox_3d_id, prim_path, label, color_bgr.
        Returns instance_seg_id(int) -> (b,g,r) color tuple.
        """
        out: Dict[int, Tuple[int, int, int]] = {}
        for k, v in seg_info.items():
            if not str(k).isdigit():
                continue
            instance_id = int(k)
            color = v.get("color_bgr", None)
            if color is None or len(color) != 3:
                continue
            out[instance_id] = (int(color[0]), int(color[1]), int(color[2]))
        return out

    @staticmethod
    def _infer_class_name(b3d: Dict[str, Any], b2d: Optional[Dict[str, Any]]) -> str:
        """
        Prefer 3D label string. If not present, 2D label dict like {"wall":"wall"} -> take first value.
        """
        # 3D example: "label": "wall"
        if "label" in b3d and isinstance(b3d["label"], str) and b3d["label"]:
            return b3d["label"]

        # 2D example: "label": {"wall":"wall"} or {"goal":"bowl"}
        if b2d is not None:
            lab = b2d.get("label", None)
            if isinstance(lab, dict) and len(lab) > 0:
                # take first value if exists, else key
                k0 = next(iter(lab.keys()))
                v0 = lab.get(k0, k0)
                return str(v0)

        return "xxxxxxx"

    def _frame_str(self, frame_idx: int) -> str:
        return f"{frame_idx:06d}"

    def _bbox_json_path(self, frame_idx: int) -> Path:
        return self.bbox_dir / f"bboxes{self._frame_str(frame_idx)}_info.json"

    def _seg_png_path(self, frame_idx: int) -> Path:
        return self.seg_dir / f"semantic{self._frame_str(frame_idx)}.png"

    def _seg_info_json_path(self, frame_idx: int) -> Path:
        return self.seg_dir / f"semantic{self._frame_str(frame_idx)}_info.json"

    def _rgb_path(self, frame_idx: int) -> Path:
        return self.rgb_dir / f"frame{self._frame_str(frame_idx)}.jpg"

    def _depth_path(self, frame_idx: int) -> Path:
        return self.depth_dir / f"depth{self._frame_str(frame_idx)}.png"
