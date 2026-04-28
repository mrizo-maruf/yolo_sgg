"""Offline Pi3 depth providers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import DepthProvider
from .pose_utils import load_poses_txt, lookup_pose
from .sequence_sync import OrderedIndexMap, sorted_files_with_ids


_DEPTH_SCALE_RE = re.compile(r"png_depth_scale:\s*([0-9eE+.\-]+)")
_FORMAT_RE = re.compile(r"format:\s*(\S+)")
_GLOBAL_MIN_RE = re.compile(r"global_depth_min_m:\s*([0-9eE+.\-]+)")
_GLOBAL_MAX_RE = re.compile(r"global_depth_max_m:\s*([0-9eE+.\-]+)")


class IsaacSimOfflinePi3DepthProvider(DepthProvider):
    """Load offline Pi3 depth and align poses to GT world via Sim(3).

    Depth format (from Pi3 export):
        depth_m = depth_png_uint16 * png_depth_scale

    Pose alignment:
        T_world_cam_aligned = T_sim3_pi3_to_world @ T_pi3_world_cam
    """

    def __init__(
        self,
        depth_dir: str,
        pose_path: Optional[str] = None,
        transform_path: Optional[str] = None,
        png_depth_scale: Optional[float] = None,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
        pose_lookup: str = "frame_number",
        require_transform: bool = True,
        depth_glob: str = "depth*.png",
        use_rank: bool = False,
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._pose_lookup = pose_lookup
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._use_rank = use_rank
        self._poses = load_poses_txt(pose_path)
        if self._poses is None:
            print(
                f"[IsaacSimOfflinePi3DepthProvider] WARNING: pose_path={pose_path!r} not found "
                f"or empty — get_pose() will return None for all frames."
            )
        self._depth_files, self._depth_ids = sorted_files_with_ids(self._depth_dir, depth_glob)
        self._sync = OrderedIndexMap(self._depth_ids)

        meta = self._read_meta()
        if png_depth_scale is None:
            self._png_depth_scale = meta.get("png_depth_scale", 0.001)
        else:
            self._png_depth_scale = float(png_depth_scale)
        if self._png_depth_scale <= 0.0:
            raise ValueError(f"png_depth_scale must be > 0, got {self._png_depth_scale}")

        meta_max = meta.get("global_depth_max_m")
        meta_min = meta.get("global_depth_min_m")
        meta_fmt = meta.get("format")
        meta_src = meta.get("_source")
        n_files = len(self._depth_files)
        scale_origin = "cfg" if png_depth_scale is not None else (
            "meta" if meta_src else "default(0.001)"
        )
        print(
            f"[Pi3Offline] dir={self._depth_dir} files={n_files} "
            f"png_depth_scale={self._png_depth_scale:.6g} ({scale_origin}) "
            f"format={meta_fmt or 'unknown'} "
            f"meta_min={meta_min} meta_max={meta_max} "
            f"clamp=[{self._min_depth},{self._max_depth}]"
        )
        if meta_max is not None and meta_max > self._max_depth:
            print(
                f"[Pi3Offline] WARNING: meta global_depth_max_m={meta_max:.3f} > "
                f"clamp max_depth={self._max_depth:.3f} — pixels above will be zeroed."
            )

        if transform_path is None:
            if require_transform:
                raise FileNotFoundError(
                    "transform_path is required for IsaacSimOfflinePi3DepthProvider."
                )
            self._sim3 = np.eye(4, dtype=np.float32)
        else:
            self._sim3 = self._load_sim3_matrix(transform_path, require_transform)

    def _read_meta(self) -> dict:
        """Parse Pi3 depth metadata file (if present).

        Returns a dict with any of: png_depth_scale, format,
        global_depth_min_m, global_depth_max_m, _source (filename).
        Empty dict if no meta file is found.

        Raises ``ValueError`` if a meta file exists but ``png_depth_scale``
        is present with an unparseable / non-positive value — silent
        fall-through to the 0.001 default has caused encoding
        mismatches in the past.
        """
        for name in ("pi3_depth_meta.txt", "depth_scale.txt", "meta.txt"):
            path = self._depth_dir / name
            if not path.exists():
                continue
            try:
                txt = path.read_text(encoding="utf-8")
            except Exception:
                continue

            out: dict = {"_source": name}

            m = _DEPTH_SCALE_RE.search(txt)
            if m:
                try:
                    value = float(m.group(1))
                except ValueError as e:
                    raise ValueError(
                        f"Malformed png_depth_scale in {path}: {m.group(1)!r}"
                    ) from e
                if value <= 0:
                    raise ValueError(
                        f"png_depth_scale must be > 0 in {path}, got {value}"
                    )
                out["png_depth_scale"] = value

            m = _FORMAT_RE.search(txt)
            if m:
                out["format"] = m.group(1).strip()

            m = _GLOBAL_MIN_RE.search(txt)
            if m:
                try:
                    out["global_depth_min_m"] = float(m.group(1))
                except ValueError:
                    pass

            m = _GLOBAL_MAX_RE.search(txt)
            if m:
                try:
                    out["global_depth_max_m"] = float(m.group(1))
                except ValueError:
                    pass

            return out

        return {}

    @staticmethod
    def _load_sim3_matrix(path_str: str, require: bool) -> np.ndarray:
        path = Path(path_str)
        if not path.exists():
            if require:
                raise FileNotFoundError(f"Pi3 alignment transform not found: {path}")
            return np.eye(4, dtype=np.float32)

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "sim3_matrix_4x4" in payload:
            sim3 = np.asarray(payload["sim3_matrix_4x4"], dtype=np.float64)
            if sim3.shape != (4, 4):
                raise ValueError(
                    f"sim3_matrix_4x4 must be 4x4, got shape {sim3.shape}"
                )
        else:
            scale = float(payload["scale"])
            rotation = np.asarray(payload["rotation"], dtype=np.float64)
            translation = np.asarray(payload["translation"], dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError(f"rotation must be 3x3, got shape {rotation.shape}")
            if translation.shape != (3,):
                raise ValueError(
                    f"translation must be shape (3,), got shape {translation.shape}"
                )
            sim3 = np.eye(4, dtype=np.float64)
            sim3[:3, :3] = scale * rotation
            sim3[:3, 3] = translation

        if not np.isfinite(sim3).all():
            raise ValueError(f"Invalid values in Sim(3) transform: {path}")

        return sim3.astype(np.float32)

    def _depth_path(self, frame_idx: int) -> Path:
        if self._use_rank:
            # Sequential index — loader passes 0-based frame_idx directly.
            idx = int(frame_idx)
            if 0 <= idx < len(self._depth_files):
                return self._depth_files[idx]
            return self._depth_dir / f"frame_{frame_idx:06d}.png"
        # In frame-number mode, loader passes 1-based frame keys while
        # depth/pose lists are naturally ordered 0..N-1. Align by rank first.
        if self._pose_lookup == "frame_number":
            ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
        else:
            ord_idx = self._sync.resolve_index(int(frame_idx))
        if ord_idx is not None and 0 <= ord_idx < len(self._depth_files):
            return self._depth_files[ord_idx]

        return self._depth_dir / f"depth{frame_idx:06d}.png"

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self._depth_path(frame_idx)
        if not path.exists():
            return None

        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]

        dm = arr.astype(np.float32) * self._png_depth_scale
        dm[~np.isfinite(dm)] = 0.0
        dm[dm < self._min_depth] = 0.0
        if self._max_depth > 0.0:
            dm[dm > self._max_depth] = 0.0
        return dm.astype(np.float32)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        pose = None
        if self._use_rank:
            idx = int(frame_idx)
            if self._poses is not None and 0 <= idx < len(self._poses):
                pose = self._poses[idx]
        else:
            if self._pose_lookup == "frame_number":
                ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
            else:
                ord_idx = self._sync.resolve_index(int(frame_idx))
            if self._poses is not None and ord_idx is not None and 0 <= ord_idx < len(self._poses):
                pose = self._poses[ord_idx]
            if pose is None:
                pose = lookup_pose(self._poses, frame_idx, self._pose_lookup)
        if pose is None:
            return None
        pose = pose.astype(np.float32)
        return (self._sim3 @ pose).astype(np.float32)

    def get_sync_debug(self, frame_idx: int) -> dict:
        if self._use_rank:
            idx = int(frame_idx)
            depth_path = str(self._depth_files[idx]) if 0 <= idx < len(self._depth_files) else None
            pose_index = idx if (self._poses is not None and 0 <= idx < len(self._poses)) else None
        else:
            if self._pose_lookup == "frame_number":
                ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
            else:
                ord_idx = self._sync.resolve_index(int(frame_idx))
            depth_path = str(self._depth_path(frame_idx))
            pose_index = int(ord_idx) if (ord_idx is not None and self._poses is not None and 0 <= ord_idx < len(self._poses)) else None
        return {
            "frame_key": int(frame_idx),
            "depth_path": depth_path,
            "pose_index": pose_index,
        }

    def get_sim3_matrix(self) -> np.ndarray:
        return self._sim3.copy()

    @property
    def png_depth_scale(self) -> float:
        return self._png_depth_scale
