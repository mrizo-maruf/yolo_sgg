import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class GTObject:
    track_id: int
    semantic_id: int
    class_name: str

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
    cam_transform_4x4: List[np.ndarray] = None  # from traj.txt

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
        bbox2d_by_bbox_id = self._parse_bbox2d_tight(bbox_data)
        bbox3d_list = self._parse_bbox3d(bbox_data)

        # semantic_id -> color_bgr (tuple)
        color_by_semantic_id = self._parse_semantic_color_map(seg_info)

        gt_objects: List[GTObject] = []
        for b3d in bbox3d_list:
            bbox_id = b3d["bbox_id"]
            track_id = int(b3d["track_id"])
            semantic_id = int(b3d["semantic_id"])

            # class name: prefer b3d["label"] (string), else derive from 2D label dict, else fallback
            class_name = self._infer_class_name(b3d=b3d, b2d=bbox2d_by_bbox_id.get(bbox_id))

            # 2D info (optional, may not exist)
            b2d = bbox2d_by_bbox_id.get(bbox_id)
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

            # Mask extraction: ONLY for objects with 3D bbox (we're iterating 3D list already)
            color = color_by_semantic_id.get(semantic_id, None)
            if color is None:
                # If missing mapping, make empty mask (or raise; your choice)
                mask = np.zeros(seg_img.shape[:2], dtype=bool)
            else:
                # seg_img is BGR; color is (b,g,r)
                mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=2)

            gt_objects.append(
                GTObject(
                    track_id=track_id,
                    semantic_id=semantic_id,
                    class_name=class_name,
                    bbox2d_xyxy=bbox2d_xyxy,
                    box_3d_aabb_xyzmin_xyzmax=aabb_xyzmin_xyzmax,
                    box_3d_transform_4x4=T,
                    visibility=visibility,
                    occlusion=occlusion,
                    mask=mask,
                )
            )

        rgb = self._read_rgb_bgr(self._rgb_path(frame_idx)) if self.load_rgb else None
        depth = self._read_depth(self._depth_path(frame_idx)) if self.load_depth else None
        seg = seg_img if self.load_seg else None

        # Load camera transform from trajectory data (frame_idx is 1-based)
        cam_transform_4x4 = None
        if self.traj_data is not None and len(self.traj_data) >= frame_idx:
            cam_transform_4x4 = self.traj_data[frame_idx]["cam_transform_4x4"]
        
        return FrameData(
            frame_idx=frame_idx,
            gt_objects=gt_objects,
            rgb=rgb,
            depth=depth,
            seg=seg,
            cam_transform_4x4=cam_transform_4x4,
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
        Returns bbox_id -> 2D box dict.
        """
        b2d = (
            bbox_data.get("bboxes", {})
                    .get("bbox_2d_tight", {})
                    .get("boxes", [])
        )
        out: Dict[int, Dict[str, Any]] = {}
        for b in b2d:
            if "bbox_id" in b:
                out[int(b["bbox_id"])] = b
        return out

    @staticmethod
    def _parse_bbox3d(bbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns list of 3D box dicts (each must have bbox_id, track_id, semantic_id).
        """
        b3d = (
            bbox_data.get("bboxes", {})
                    .get("bbox_3d", {})
                    .get("boxes", [])
        )
        # Optionally enforce required keys
        required = {"bbox_id", "track_id", "semantic_id", "aabb_xyzmin_xyzmax", "transform_4x4"}
        for i, b in enumerate(b3d):
            missing = required - set(b.keys())
            if missing:
                raise ValueError(f"bbox_3d[{i}] missing keys {missing}")
        return b3d

    @staticmethod
    def _parse_semantic_color_map(seg_info: Dict[str, Any]) -> Dict[int, Tuple[int, int, int]]:
        """
        seg_info keys are strings ("0","1",...), each has color_bgr.
        Returns semantic_id(int) -> (b,g,r)
        """
        out: Dict[int, Tuple[int, int, int]] = {}
        for k, v in seg_info.items():
            if not str(k).isdigit():
                continue
            sid = int(k)
            color = v.get("color_bgr", None)
            if color is None or len(color) != 3:
                continue
            out[sid] = (int(color[0]), int(color[1]), int(color[2]))
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
