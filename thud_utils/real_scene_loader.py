"""
THUD Real_Scenes data loader + visualization.

Supports dataset structure:
Real_Scenes/<SceneName>/static/Capture_1/
  - RGB/frame-000000.color.png
  - Depth/frame-000000.depth.png
  - Camera_intrinsics/camera-intrinsics.txt
  - Label/2D_Object_Detection/frame-000000.json
  - Label/3D_Object_Detection/frame-000000.txt
  - Label/Pose/frame-000000.pose.txt
  - Label/Semantic/frame-000000.png
  - Pointcloud/frame-000000.point.ply (optional)

When `visualize_frame` is called, this loader can show:
1) RGB scene
2) RGB + semantic segmentation + semantic class labels
3) RGB + 2D bounding boxes + class names
4) Open3D reprojected depth point cloud with RGB + 3D boxes (prints class names)
5) Optional built-in PLY/PCD visualization in Open3D
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import ast
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class CameraIntrinsics:
    matrix: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraPose:
    transform_matrix: np.ndarray
    translation: np.ndarray
    rotation_matrix: np.ndarray


@dataclass
class Object2D:
    class_name: str
    bbox_xyxy: List[float]
    track_id: Optional[int] = None


@dataclass
class Object3D:
    track_id: int
    class_name: str
    truncation: float = 0.0
    occlusion: int = 0
    alpha: float = 0.0
    bbox_2d: Optional[List[float]] = None
    bbox_3d_size: Optional[List[float]] = None  # [h, w, l]
    bbox_3d_center: Optional[List[float]] = None  # [x, y, z]
    rotation_y: float = 0.0
    score: float = 1.0


@dataclass
class FrameData:
    frame_idx: int
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    semantic_seg: Optional[np.ndarray] = None
    objects_2d: List[Object2D] = field(default_factory=list)
    objects_3d: List[Object3D] = field(default_factory=list)
    camera_pose: Optional[CameraPose] = None


class RealSceneDataLoader:
    def __init__(self, sequence_path: str, verbose: bool = True):
        self.sequence_path = Path(sequence_path)
        self.verbose = verbose

        self.rgb_dir = self.sequence_path / "RGB"
        self.depth_dir = self.sequence_path / "Depth"
        self.label_dir = self.sequence_path / "Label"
        self.det2d_dir = self.label_dir / "2D_Object_Detection"
        self.det3d_dir = self.label_dir / "3D_Object_Detection"
        self.pose_dir = self.label_dir / "Pose"
        self.semantic_dir = self.label_dir / "Semantic"
        self.intrinsics_dir = self.sequence_path / "Camera_intrinsics"
        self.pointcloud_dir = self.sequence_path / "Pointcloud"

        self.scene_root = self.sequence_path.parent
        self.semantic_color_map = self._load_semantic_color_map(self.scene_root)

        self._validate_paths()
        self.camera_intrinsics = self._load_camera_intrinsics()
        self.frame_indices = self._discover_frames()
        self.image_height, self.image_width = self._get_image_size()

        self._class_names = set()
        self._scan_known_class_names()

        if self.verbose:
            print(f"[RealSceneDataLoader] Sequence: {self.sequence_path}")
            print(f"  Frames: {len(self.frame_indices)}")
            print(f"  Image size: {self.image_width}x{self.image_height}")
            if self.semantic_color_map:
                print(f"  Semantic labels from sheet: {len(self.semantic_color_map)}")

    @staticmethod
    def _infer_scene_root(scene_path: Path) -> Path:
        if scene_path.parent.name == "static":
            return scene_path.parent.parent
        return scene_path

    def _warn(self, msg: str) -> None:
        if self.verbose:
            print(f"[WARN] {msg}")

    def _validate_paths(self) -> None:
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.sequence_path}")
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")

    def _load_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        path = self.intrinsics_dir / "camera-intrinsics.txt"
        if not path.exists():
            self._warn(f"Missing intrinsics: {path}")
            return None

        try:
            matrix = np.loadtxt(path, dtype=np.float32)
            if matrix.shape != (3, 3):
                self._warn(f"Invalid intrinsics shape: {matrix.shape}")
                return None
            return CameraIntrinsics(
                matrix=matrix,
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
            )
        except Exception as exc:
            self._warn(f"Could not parse intrinsics: {exc}")
            return None

    def _discover_frames(self) -> List[int]:
        frame_ids = set()
        for rgb_path in self.rgb_dir.glob("frame-*.color.png"):
            m = re.search(r"frame-(\d+)\.color\.png", rgb_path.name)
            if m:
                frame_ids.add(int(m.group(1)))
        return sorted(frame_ids)

    def _get_image_size(self) -> Tuple[int, int]:
        if not self.frame_indices:
            return 540, 960
        p = self.rgb_dir / f"frame-{self.frame_indices[0]:06d}.color.png"
        if not p.exists():
            return 540, 960
        with Image.open(p) as img:
            return int(img.height), int(img.width)

    @staticmethod
    def _parse_rgb_triplet(value: object) -> Optional[Tuple[int, int, int]]:
        nums = re.findall(r"\d+", str(value))
        if len(nums) < 3:
            return None
        r, g, b = [max(0, min(255, int(x))) for x in nums[:3]]
        return r, g, b

    def _load_semantic_color_map(self, scene_root: Path) -> Dict[Tuple[int, int, int], str]:

        mapping: Dict[Tuple[int, int, int], str] = {}
        if not scene_root.exists():
            print(f'RealSceneDataLoader._load_semantic_color_map scene rood doesnt exist')
            return mapping

        xlsx_files = list(scene_root.glob("*.xlsx"))

        if not xlsx_files:
            print("No xlsx files in {scene_root}")

        for xlsx in sorted(scene_root.glob("*.xlsx")):
            try:
                df = pd.read_excel(xlsx, skiprows=1)
            except Exception as e:
                print(f"RealSceneDataLoader._load_semantic_color_map failed to read {xlsx.name}, exception {e}")
                continue
            for sem_label, sem_color in df.itertuples(index=False):
                if not isinstance(sem_color, str):
                    print(f"RealSceneDataLoader._load_semantic_color_map: {sem_label} has non str RGB {sem_color}")
                    continue
                parts = sem_color.split(".")

                if len(parts) != 3:
                    print(f"RealSceneDataLoader._load_semantic_color_map: {sem_label} has non str RGB {sem_color}")
                    continue

                rgb = tuple(int(p) for p in parts)

                mapping[rgb] = sem_label

            if mapping:
                if self.verbose:
                    print(f"Loaded semantic label map from {xlsx.name}")
                break

        return mapping

    def _scan_known_class_names(self) -> None:
        sample_ids = self.frame_indices[: min(15, len(self.frame_indices))]
        for idx in sample_ids:
            for obj in self.get_2d_objects(idx):
                self._class_names.add(obj.class_name)
            for obj in self.get_3d_objects(idx):
                self._class_names.add(obj.class_name)

    def _frame_stem(self, frame_idx: int) -> str:
        return f"frame-{frame_idx:06d}"

    def get_frame_count(self) -> int:
        return len(self.frame_indices)

    def get_frame_indices(self) -> List[int]:
        return list(self.frame_indices)

    def get_class_names(self) -> List[str]:
        return sorted(self._class_names)

    def get_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self.camera_intrinsics

    def get_intrinsics_matrix(self) -> Optional[np.ndarray]:
        """Return the 3x3 camera intrinsics matrix (or None)."""
        if self.camera_intrinsics is not None:
            return self.camera_intrinsics.matrix.copy()
        return None

    def get_rgb_path(self, frame_idx: int) -> str:
        """Return the filesystem path to the RGB image for *frame_idx*."""
        return str(self.rgb_dir / f"{self._frame_stem(frame_idx)}.color.png")

    def get_depth_path(self, frame_idx: int) -> str:
        """Return the filesystem path to the depth image for *frame_idx*."""
        return str(self.depth_dir / f"{self._frame_stem(frame_idx)}.depth.png")

    def get_instance_masks(
        self,
        frame_idx: int,
        objects_2d: Optional[List[Object2D]] = None,
    ) -> List[np.ndarray]:
        """Generate per-object binary masks from semantic segmentation + 2D boxes.

        For each 2D object the method:
        1. Loads the semantic segmentation image.
        2. Looks up the semantic RGB colour for the object's class name.
        3. Builds a binary mask = (pixels matching that colour) AND (inside bbox).

        If the semantic colour is unknown for a class, falls back to a filled
        bbox mask so every object always gets *some* mask.

        Returns a list parallel to *objects_2d* (or ``self.get_2d_objects()``).
        """
        if objects_2d is None:
            objects_2d = self.get_2d_objects(frame_idx)

        semantic = self.load_semantic_segmentation(frame_idx)
        if semantic is None:
            # No semantic image – fall back to bbox-filled masks
            h, w = self.image_height, self.image_width
            masks = []
            for obj in objects_2d:
                m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]
                m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
                masks.append(m)
            return masks

        h, w = semantic.shape[:2]
        # Build inverse map: class_name -> RGB colour
        name_to_color: Dict[str, Tuple[int, int, int]] = {}
        for color_rgb, label in self.semantic_color_map.items():
            name_to_color[label] = color_rgb

        masks: List[np.ndarray] = []
        for obj in objects_2d:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            color = name_to_color.get(obj.class_name)
            if color is not None:
                # Semantic colour match inside the bbox
                color_arr = np.array(color, dtype=np.uint8)
                roi = semantic[y1:y2, x1:x2]
                roi_mask = np.all(roi == color_arr, axis=2).astype(np.uint8)
                m = np.zeros((h, w), dtype=np.uint8)
                m[y1:y2, x1:x2] = roi_mask
            else:
                # Fallback: fill the bbox
                m = np.zeros((h, w), dtype=np.uint8)
                m[y1:y2, x1:x2] = 1
            masks.append(m)
        return masks

    def load_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self.rgb_dir / f"{self._frame_stem(frame_idx)}.color.png"
        if not path.exists():
            self._warn(f"Missing RGB: {path}")
            return None
        arr = np.array(Image.open(path))
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def load_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self.depth_dir / f"{self._frame_stem(frame_idx)}.depth.png"
        if not path.exists():
            self._warn(f"Missing depth: {path}")
            return None

        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            self._warn(f"Could not read depth: {path}")
            return None

        depth = depth.astype(np.float32)
        if depth.max() > 100.0:
            depth /= 1000.0
        return depth

    def load_semantic_segmentation(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self.semantic_dir / f"{self._frame_stem(frame_idx)}.png"
        if not path.exists():
            self._warn(f"Missing semantic: {path}")
            return None
        arr = np.array(Image.open(path))
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def load_camera_pose(self, frame_idx: int) -> Optional[CameraPose]:
        path = self.pose_dir / f"{self._frame_stem(frame_idx)}.pose.txt"
        if not path.exists():
            self._warn(f"Missing pose: {path}")
            return None
        try:
            t = np.loadtxt(path, dtype=np.float32)
            if t.shape != (4, 4):
                self._warn(f"Invalid pose shape {t.shape} in {path.name}")
                return None
            return CameraPose(
                transform_matrix=t,
                translation=t[:3, 3].copy(),
                rotation_matrix=t[:3, :3].copy(),
            )
        except Exception as exc:
            self._warn(f"Failed to parse pose: {exc}")
            return None

    def get_2d_objects(self, frame_idx: int) -> List[Object2D]:
        path = self.det2d_dir / f"{self._frame_stem(frame_idx)}.color.json"
        if not path.exists():
            print(f"2d object labels path doesn't exist, {path}")
            return []

        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            self._warn(f"2D JSON parse failed ({path.name}): {exc}")
            return []

        objects: List[Object2D] = []
        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type", "rectangle")
            if shape_type != "rectangle":
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x_min, x_max = sorted([float(x1), float(x2)])
            y_min, y_max = sorted([float(y1), float(y2)])
            label = str(shape.get("label", "Unknown"))
            objects.append(Object2D(label, [x_min, y_min, x_max, y_max]))
            self._class_names.add(label)
        return objects

    def get_3d_objects(self, frame_idx: int) -> List[Object3D]:
        path = self.det3d_dir / f"{self._frame_stem(frame_idx)}.txt"
        if not path.exists():
            return []

        objects: List[Object3D] = []
        try:
            with path.open("r") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    try:
                        class_name = parts[0]
                        trunc = float(parts[1])
                        occ = int(float(parts[2]))
                        alpha = float(parts[3])
                        bbox_2d = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                        size_hwl = [float(parts[8]), float(parts[9]), float(parts[10])]
                        center_xyz = [float(parts[11]), float(parts[12]), float(parts[13])]
                        ry = float(parts[14])
                        score = float(parts[15]) if len(parts) > 15 else 1.0
                        objects.append(
                            Object3D(
                                track_id=idx,
                                class_name=class_name,
                                truncation=trunc,
                                occlusion=occ,
                                alpha=alpha,
                                bbox_2d=bbox_2d,
                                bbox_3d_size=size_hwl,
                                bbox_3d_center=center_xyz,
                                rotation_y=ry,
                                score=score,
                            )
                        )
                        self._class_names.add(class_name)
                    except Exception:
                        continue
        except Exception as exc:
            self._warn(f"3D TXT parse failed ({path.name}): {exc}")

        return objects

    def get_gt_objects(self, frame_idx: int) -> List[Object3D]:
        return self.get_3d_objects(frame_idx)

    def load_pointcloud(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self.pointcloud_dir.exists():
            return None

        stem = self._frame_stem(frame_idx)
        candidates = [
            self.pointcloud_dir / f"{stem}.point.ply",
            self.pointcloud_dir / f"{stem}.point.pcd",
            self.pointcloud_dir / f"{stem}.ply",
            self.pointcloud_dir / f"{stem}.pcd",
            self.pointcloud_dir / f"{stem}.npy",
            self.pointcloud_dir / f"{stem}.bin",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                suffix = path.suffix.lower()
                if suffix == ".npy":
                    arr = np.load(path)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        return arr[:, :3]
                elif suffix == ".bin":
                    arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                    return arr[:, :3]
                elif HAS_OPEN3D and suffix in [".ply", ".pcd"]:
                    pcd = o3d.io.read_point_cloud(str(path))
                    return np.asarray(pcd.points)
            except Exception as exc:
                self._warn(f"Point cloud load failed ({path.name}): {exc}")

        return None

    def load_frame(
        self,
        frame_idx: int,
        load_rgb: bool = True,
        load_depth: bool = True,
        load_semantic: bool = True,
    ) -> FrameData:
        frame = FrameData(frame_idx=frame_idx)
        if load_rgb:
            frame.rgb = self.load_rgb(frame_idx)
        if load_depth:
            frame.depth = self.load_depth(frame_idx)
        if load_semantic:
            frame.semantic_seg = self.load_semantic_segmentation(frame_idx)
        frame.objects_2d = self.get_2d_objects(frame_idx)
        frame.objects_3d = self.get_3d_objects(frame_idx)
        frame.camera_pose = self.load_camera_pose(frame_idx)
        return frame

    @staticmethod
    def _stable_color(label: str) -> Tuple[int, int, int]:
        rng = np.random.default_rng(abs(hash(label)) % (2**32))
        c = rng.integers(40, 235, size=3)
        return int(c[0]), int(c[1]), int(c[2])

    def _draw_2d_boxes(self, rgb: np.ndarray, objects_2d: List[Object2D]) -> np.ndarray:
        out = rgb.copy()
        for obj in objects_2d:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]
            # Black bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), 2)
            # Measure text for background
            text = f"[{obj.track_id}] {obj.class_name}" if obj.track_id is not None else obj.class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_y = max(th + 4, y1 - 5)
            # Black background rectangle behind text
            cv2.rectangle(out, (x1, text_y - th - 4), (x1 + tw + 4, text_y + baseline), (0, 0, 0), cv2.FILLED)
            # White text
            cv2.putText(out, text, (x1 + 2, text_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return out

    def _semantic_overlay_with_labels(self, rgb: np.ndarray, semantic: np.ndarray, alpha: float,
                                       ax=None) -> np.ndarray:
        vis = cv2.addWeighted(rgb, 1.0 - alpha, semantic, alpha, 0.0)
        if not self.semantic_color_map:
            return vis

        legend_patches = []
        for color_rgb, label in self.semantic_color_map.items():
            mask = np.all(semantic == np.array(color_rgb, dtype=np.uint8), axis=2)
            if int(mask.sum()) < 60:
                continue
            ys, xs = np.where(mask)
            x = int(xs.mean())
            y = int(ys.mean())
            text_color = self._stable_color(label)
            cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
            # Build legend entry
            if HAS_MATPLOTLIB and ax is not None:
                from matplotlib.patches import Patch
                legend_patches.append(
                    Patch(facecolor=np.array(color_rgb) / 255.0, edgecolor="black", label=label)
                )

        if HAS_MATPLOTLIB and ax is not None and legend_patches:
            ax.legend(handles=legend_patches, loc="lower left", fontsize=7,
                      framealpha=0.75, fancybox=True, title="Semantic Classes", title_fontsize=8)
        return vis

    @staticmethod
    def _iou(box_a: List[float], box_b: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _center_to_world(self, obj: Object3D, cam_pose: CameraPose) -> np.ndarray:
        """Transform a 3D object center from upright-camera frame to world frame.

        The label stores centres in upright-camera convention
        (X-right, Y-forward, Z-up).  We first convert to standard camera
        (X-right, Y-down, Z-forward) then apply the camera extrinsic.
        """
        x_u, y_u, z_u = obj.bbox_3d_center
        # Upright-cam -> standard-cam
        center_cam = np.array([x_u, -z_u, y_u], dtype=np.float32)
        # Standard-cam -> world
        center_world = cam_pose.rotation_matrix @ center_cam + cam_pose.translation
        return center_world

    def assign_tracking_ids(self, distance_threshold: float = 0.3) -> Dict[int, List[Object3D]]:
        """Assign consistent tracking IDs to objects across all frames.

        For **3D objects**: transforms centres to world frame and matches
        across frames by (class_name + Euclidean proximity) using greedy
        nearest-neighbour assignment.

        For **2D objects**: propagates track IDs from matched 3D objects
        via class-name + 2D bbox IoU.  2D objects with no 3D counterpart
        are matched across consecutive frames by class-name + IoU.

        Args:
            distance_threshold: Max world-frame distance (metres) to
                consider two detections the same object.

        Returns:
            Dict mapping frame_idx -> list of Object3D with assigned
            ``track_id`` values.
        """
        tracks: Dict[int, dict] = {}   # tid -> {class_name, world_center}
        next_id = 0
        result_3d: Dict[int, List[Object3D]] = {}
        result_2d: Dict[int, List[Object2D]] = {}
        prev_2d: List[Object2D] = []

        for frame_idx in self.frame_indices:
            objects_3d = self.get_3d_objects(frame_idx)
            cam_pose = self.load_camera_pose(frame_idx)

            # --- 3D tracking ---
            for obj in objects_3d:
                if obj.bbox_3d_center is None or cam_pose is None:
                    continue

                world_center = self._center_to_world(obj, cam_pose)

                best_tid: Optional[int] = None
                best_dist = distance_threshold
                for tid, track in tracks.items():
                    if track["class_name"] != obj.class_name:
                        continue
                    dist = float(np.linalg.norm(world_center - track["world_center"]))
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    obj.track_id = best_tid
                    # Exponential moving average for robustness
                    alpha = 0.3
                    tracks[best_tid]["world_center"] = (
                        alpha * world_center
                        + (1 - alpha) * tracks[best_tid]["world_center"]
                    )
                else:
                    obj.track_id = next_id
                    tracks[next_id] = {
                        "class_name": obj.class_name,
                        "world_center": world_center.copy(),
                    }
                    next_id += 1

            result_3d[frame_idx] = objects_3d

            # --- 2D tracking (propagate from 3D, fall back to IoU) ---
            objects_2d = self.get_2d_objects(frame_idx)
            for obj2d in objects_2d:
                # Try matching to a 3D object with same class via 2D bbox IoU
                best_iou = 0.1
                best_tid_2d: Optional[int] = None
                for obj3d in objects_3d:
                    if obj3d.bbox_2d is None or obj3d.track_id is None:
                        continue
                    if obj2d.class_name != obj3d.class_name:
                        continue
                    iou = self._iou(obj2d.bbox_xyxy, obj3d.bbox_2d)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid_2d = obj3d.track_id

                if best_tid_2d is not None:
                    obj2d.track_id = best_tid_2d
                else:
                    # Fall back: match against previous frame's 2D objects
                    best_iou_prev = 0.15
                    for prev_obj in prev_2d:
                        if prev_obj.track_id is None:
                            continue
                        if obj2d.class_name != prev_obj.class_name:
                            continue
                        iou = self._iou(obj2d.bbox_xyxy, prev_obj.bbox_xyxy)
                        if iou > best_iou_prev:
                            best_iou_prev = iou
                            best_tid_2d = prev_obj.track_id
                    if best_tid_2d is not None:
                        obj2d.track_id = best_tid_2d
                    else:
                        # Truly new 2D-only object
                        obj2d.track_id = next_id
                        next_id += 1

            result_2d[frame_idx] = objects_2d
            prev_2d = objects_2d

        self._tracked_3d = result_3d
        self._tracked_2d = result_2d
        self._num_tracks = next_id

        if self.verbose:
            print(
                f"[Tracking] Assigned {next_id} unique track IDs "
                f"across {len(self.frame_indices)} frames"
            )
        return result_3d

    def get_tracked_3d_objects(self, frame_idx: int) -> List[Object3D]:
        """Return 3D objects with consistent track IDs (call ``assign_tracking_ids`` first)."""
        if not hasattr(self, "_tracked_3d") or self._tracked_3d is None:
            raise RuntimeError("Call assign_tracking_ids() before get_tracked_3d_objects()")
        return self._tracked_3d.get(frame_idx, [])

    def get_tracked_2d_objects(self, frame_idx: int) -> List[Object2D]:
        """Return 2D objects with consistent track IDs (call ``assign_tracking_ids`` first)."""
        if not hasattr(self, "_tracked_2d") or self._tracked_2d is None:
            raise RuntimeError("Call assign_tracking_ids() before get_tracked_2d_objects()")
        return self._tracked_2d.get(frame_idx, [])

    def get_num_tracks(self) -> int:
        """Return total number of unique tracks assigned."""
        return getattr(self, "_num_tracks", 0)

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation around Z axis (vertical in upright convention)."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Rotation around X axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[1, 0, 0],
            [0, c, -s],
            [0, s,  c]],
            dtype=np.float32
        )

    def _bbox3d_corners(self, obj: Object3D) -> Optional[np.ndarray]:
        """Return 8 corners in *standard camera* frame (X=right, Y=down, Z=forward).

        The label file stores boxes in upright-camera convention
        (X=right, Y=forward, Z=up) with KITTI-style ordering:
            parts[8]  = h   (height, extent along Z-up)  → dz
            parts[9]  = w   (width ,  extent along X)     → dx
            parts[10] = l   (length, extent along Y-fwd)  → dy
        Center is the true box center.  Rotation is around the upright Z axis.

        After building corners in upright-cam we convert to standard-cam:
            x_cam =  x_u
            y_cam = -z_u
            z_cam =  y_u
        """
        if obj.bbox_3d_size is None or obj.bbox_3d_center is None:
            return None

        dz, dy, dx = obj.bbox_3d_size  # h, w, l → height, width, length
        x, y, z = obj.bbox_3d_center

        # Corners in upright-camera frame (before rotation), centered on the box center
        x_c = np.array([dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2], dtype=np.float32)
        y_c = np.array([dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2], dtype=np.float32)
        z_c = np.array([-dz/2, -dz/2, -dz/2, -dz/2, dz/2, dz/2, dz/2, dz/2], dtype=np.float32)

        corners_u = np.vstack([x_c, y_c, z_c])          # 3x8 upright
        rot = self._rotation_z(obj.rotation_y)
        corners_u = rot @ corners_u
        corners_u[0, :] += x
        corners_u[1, :] += y
        corners_u[2, :] += z

        # Convert upright-cam → standard-cam
        corners_cam = np.vstack([
            corners_u[0, :],    #  x_cam =  x_u
            -corners_u[2, :],   #  y_cam = -z_u
            corners_u[1, :],    #  z_cam =  y_u
        ])  # 3x8

        return corners_cam.T  # 8x3

    def _depth_to_xyzrgb(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        intr: CameraIntrinsics,
        stride: int = 1,
        d_min: float = 0.05,
        d_max: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = depth.shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]

        z = depth[::stride, ::stride]
        valid = (z > d_min) & (z < d_max) & np.isfinite(z)

        z = z[valid]
        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)

        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy

        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = (rgb[::stride, ::stride][valid].astype(np.float32) / 255.0).reshape(-1, 3)
        return xyz, colors

    def _make_3d_box_lineset(self, obj: Object3D, cam_pose: Optional[CameraPose] = None):
        corners = self._bbox3d_corners(obj)  # 8x3 in standard-camera frame
        if corners is None:
            return None

        # Transform bbox corners from camera frame to world frame
        if cam_pose is not None:
            R = cam_pose.rotation_matrix
            t = cam_pose.translation
            corners = (R @ corners.T).T + t

        lines = np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int32,
        )
        c = np.array(self._stable_color(obj.class_name), dtype=np.float32) / 255.0
        line_colors = np.tile(c[None, :], (lines.shape[0], 1))
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(line_colors)
        return ls

    def _show_open3d_reprojected(
        self,
        frame_idx: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        objects_3d: List[Object3D],
        point_stride: int,
    ) -> None:
        if not HAS_OPEN3D:
            self._warn("Open3D not installed, skipping reprojection view")
            return
        if self.camera_intrinsics is None:
            self._warn("Camera intrinsics required for depth reprojection")
            return

        xyz, colors = self._depth_to_xyzrgb(depth, rgb, self.camera_intrinsics, stride=max(1, point_stride))
        if xyz.size == 0:
            self._warn("No valid depth points found")
            return

        # Transform PCD from standard-camera frame to world frame using camera pose
        cam_pose = self.load_camera_pose(frame_idx)
        if cam_pose is not None:
            R = cam_pose.rotation_matrix  # 3x3
            t = cam_pose.translation      # 3
            xyz = (R @ xyz.T).T + t
            if self.verbose:
                print(f"[Frame {frame_idx}] Transformed PCD to world frame")
        else:
            self._warn("Camera pose missing — PCD stays in camera frame; 3D boxes may not align")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # World-frame coordinate axes at the origin
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geoms = [pcd, world_frame]

        # Camera-frame axes at the camera position
        if cam_pose is not None:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            cam_frame.transform(cam_pose.transform_matrix)
            geoms.append(cam_frame)

        print(f"\n[Frame {frame_idx}] 3D box classes:")
        if not objects_3d:
            print("  (none)")

        for obj in objects_3d:
            tid = obj.track_id if obj.track_id is not None else "?"
            center = obj.bbox_3d_center or []
            size = obj.bbox_3d_size or []
            print(f"  track_id={tid}  class={obj.class_name}  center={center}  size={size}")
            ls = self._make_3d_box_lineset(obj, cam_pose)
            if ls is not None:
                geoms.append(ls)

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Frame {frame_idx} - Reprojected Depth + 3D Boxes (world frame)",
            width=1280,
            height=820,
        )

    def _show_open3d_builtin_pcd(
        self,
        frame_idx: int,
        rgb: Optional[np.ndarray] = None,
        objects_3d: Optional[List[Object3D]] = None,
    ) -> None:

        stem = self._frame_stem(frame_idx)
        candidates = [
            self.pointcloud_dir / f"{stem}.point.ply",
            self.pointcloud_dir / f"{stem}.point.pcd",
            self.pointcloud_dir / f"{stem}.ply",
            self.pointcloud_dir / f"{stem}.pcd",
        ]

        pc_path = next((p for p in candidates if p.exists()), None)
        if pc_path is None:
            self._warn(f"No built-in PLY/PCD for frame {frame_idx}")
            return

        pcd = o3d.io.read_point_cloud(str(pc_path))
        if pcd.is_empty():
            self._warn(f"Built-in point cloud is empty: {pc_path.name}")
            return

        # Colour PLY points from the RGB image via back-projection
        cam_pose = self.load_camera_pose(frame_idx)
        intr = self.camera_intrinsics
        if rgb is not None and cam_pose is not None and intr is not None:
            pts_world = np.asarray(pcd.points)  # Nx3 in world frame
            R = cam_pose.rotation_matrix
            t = cam_pose.translation
            # World → camera: p_cam = R^T @ (p_world - t)
            pts_cam = (R.T @ (pts_world - t).T).T  # Nx3
            z_cam = pts_cam[:, 2]
            valid = z_cam > 0.01
            u = (intr.fx * pts_cam[:, 0] / z_cam + intr.cx).astype(np.int32)
            v = (intr.fy * pts_cam[:, 1] / z_cam + intr.cy).astype(np.int32)
            h, w = rgb.shape[:2]
            in_frame = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            colors = np.zeros((len(pts_world), 3), dtype=np.float64)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors).copy()
            colors[in_frame] = rgb[v[in_frame], u[in_frame]].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # World-frame coordinate axes at the origin
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geoms = [pcd, world_frame]

        # Camera-frame axes at the camera position
        if cam_pose is not None:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            cam_frame.transform(cam_pose.transform_matrix)
            geoms.append(cam_frame)

        # 3D bounding boxes
        if objects_3d:
            print(f"\n[Frame {frame_idx}] Built-in PCD — 3D box classes:")
            for obj in objects_3d:
                tid = obj.track_id if obj.track_id is not None else "?"
                print(f"  track_id={tid}  class={obj.class_name}")
                ls = self._make_3d_box_lineset(obj, cam_pose)
                if ls is not None:
                    geoms.append(ls)

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Frame {frame_idx} - Built-in Pointcloud ({pc_path.name})",
            width=1280,
            height=820,
        )

    def visualize_frame(
        self,
        frame_idx: int,
        semantic_alpha: float = 0.45,
        show_open3d_reprojected: bool = True,
        show_open3d_builtin_pcd: bool = False,
        point_stride: int = 1,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 6),
    ) -> None:
        """Run all requested visualization modes for one frame."""
        rgb = self.load_rgb(frame_idx)
        if rgb is None:
            self._warn(f"Cannot visualize frame {frame_idx}: missing RGB")
            return

        semantic = self.load_semantic_segmentation(frame_idx)
        # Use tracked objects (with IDs) if available, otherwise raw
        if hasattr(self, "_tracked_2d") and self._tracked_2d is not None:
            objects_2d = self.get_tracked_2d_objects(frame_idx)
        else:
            objects_2d = self.get_2d_objects(frame_idx)
        if hasattr(self, "_tracked_3d") and self._tracked_3d is not None:
            objects_3d = self.get_tracked_3d_objects(frame_idx)
        else:
            objects_3d = self.get_3d_objects(frame_idx)

        # 1 + 2 + 3 in matplotlib panels
        if HAS_MATPLOTLIB:
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            axes[0].imshow(rgb)
            axes[0].set_title(f"Frame {frame_idx} - RGB")
            axes[0].axis("off")

            if semantic is not None and semantic.shape[:2] == rgb.shape[:2]:
                sem_vis = self._semantic_overlay_with_labels(rgb, semantic, semantic_alpha, ax=axes[1])
                axes[1].imshow(sem_vis)
                axes[1].set_title(f"Frame {frame_idx} - RGB + Semantic")
            else:
                axes[1].imshow(rgb)
                axes[1].set_title(f"Frame {frame_idx} - Semantic missing")
            axes[1].axis("off")

            box_vis = self._draw_2d_boxes(rgb, objects_2d)
            axes[2].imshow(box_vis)
            axes[2].set_title(f"Frame {frame_idx} - RGB + 2D Boxes ({len(objects_2d)})")
            axes[2].axis("off")

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=160, bbox_inches="tight")
                if self.verbose:
                    print(f"Saved visualization to {save_path}")
            plt.show()
        else:
            self._warn("Matplotlib not installed: skipping RGB/semantic/2D box panels")

        # 4 reprojected depth in Open3D + 3D boxes
        if show_open3d_reprojected:
            depth = self.load_depth(frame_idx)
            if depth is None:
                self._warn("Depth missing, skipping reprojected point cloud view")
            else:
                self._show_open3d_reprojected(frame_idx, rgb, depth, objects_3d, point_stride)

        # 5 optional built-in pcd
        if show_open3d_builtin_pcd:
            self._show_open3d_builtin_pcd(frame_idx, rgb=rgb, objects_3d=objects_3d)

    def visualize_3d_objects(self, frame_idx: int) -> None:
        objs = self.get_3d_objects(frame_idx)
        print(f"\n=== Frame {frame_idx} - 3D Objects ({len(objs)}) ===")
        for i, obj in enumerate(objs):
            tid = obj.track_id if obj.track_id is not None else "?"
            center = ""
            if obj.bbox_3d_center is not None:
                x, y, z = obj.bbox_3d_center
                center = f"  center=({x:.3f}, {y:.3f}, {z:.3f})"
            size = ""
            if obj.bbox_3d_size is not None:
                h, w, l = obj.bbox_3d_size
                size = f"  size=({h:.3f}, {w:.3f}, {l:.3f})"
            print(f"  track_id={tid}  class={obj.class_name}{center}{size}")


def discover_real_scenes(thud_root: str) -> List[str]:
    root = Path(thud_root) / "Real_Scenes"
    if not root.exists():
        return []

    captures = []
    for cap in root.rglob("Capture_*"):
        if cap.is_dir() and (cap / "RGB").exists():
            captures.append(str(cap))
    return sorted(captures)


def get_real_scene_info(scene_path: str) -> Dict:
    p = Path(scene_path)
    info = {
        "path": str(p),
        "name": p.name,
        "scene": p.parent.parent.name if p.parent.name == "static" else "unknown",
        "rgb_count": 0,
        "depth_count": 0,
        "semantic_count": 0,
        "det2d_count": 0,
        "det3d_count": 0,
        "pose_count": 0,
        "pointcloud_count": 0,
    }

    if (p / "RGB").exists():
        info["rgb_count"] = len(list((p / "RGB").glob("frame-*.color.png")))
    if (p / "Depth").exists():
        info["depth_count"] = len(list((p / "Depth").glob("frame-*.depth.png")))
    if (p / "Label" / "Semantic").exists():
        info["semantic_count"] = len(list((p / "Label" / "Semantic").glob("frame-*.png")))
    if (p / "Label" / "2D_Object_Detection").exists():
        info["det2d_count"] = len(list((p / "Label" / "2D_Object_Detection").glob("frame-*.json")))
    if (p / "Label" / "3D_Object_Detection").exists():
        info["det3d_count"] = len(list((p / "Label" / "3D_Object_Detection").glob("frame-*.txt")))
    if (p / "Label" / "Pose").exists():
        info["pose_count"] = len(list((p / "Label" / "Pose").glob("frame-*.pose.txt")))
    if (p / "Pointcloud").exists():
        info["pointcloud_count"] = len(list((p / "Pointcloud").glob("frame-*.*")))

    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THUD Real_Scenes loader demo")
    scene_path = "/home/yehia/rizo/THUD_Robot/Real_Scenes/1004/static/Capture_1"
    frame_idx = 0
    no_open3d = False
    show_builtin_pcd = True
    stride = 1

    loader = RealSceneDataLoader(scene_path, verbose=True)
    print(f"Frames available: {loader.get_frame_count()}")
    print(f"Class names: {loader.get_class_names()}")

    # Assign consistent tracking IDs across all frames
    loader.assign_tracking_ids(distance_threshold=0.3)

    if frame_idx not in loader.get_frame_indices() and loader.get_frame_indices():
        frame_idx = loader.get_frame_indices()[0]
        print(f"Requested frame not found; using frame {frame_idx}")

    for idx in loader.get_frame_indices():
        loader.visualize_frame(
            frame_idx=idx,
            show_open3d_reprojected=not no_open3d,
            show_open3d_builtin_pcd=show_builtin_pcd,
            point_stride=max(1, stride),
        )
