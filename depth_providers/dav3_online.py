"""Online DepthAnythingV3 streaming depth provider.

This mirrors the public shape of ``Pi3OnlineDepthProvider`` while keeping
DAv3-specific model calls self-contained.
"""
from __future__ import annotations

import logging
import json
from collections import OrderedDict
from pathlib import Path
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from .base import DepthProvider, OnlineDepthProvider

log = logging.getLogger("dav3_online")

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

_STOP = object()
_DRAIN = object()


def _resolve_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    s = str(device).strip().lower()
    if s in ("cpu", "mps") or s.startswith("cuda"):
        return s
    if s.isdigit():
        return f"cuda:{s}"
    return s


class DAv3OnlineDepthProvider(OnlineDepthProvider):
    """Chunked online DAv3 provider with Pi3-like feeder semantics.

    Frames are fed with :meth:`feed_frame`; a background worker batches them
    into overlapping chunks and caches predicted depth and poses by frame id.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-LARGE",
        chunk_size: int = 5,
        overlap: int = 3,
        device: Optional[str] = None,
        max_cache: int = 512,
        min_depth: float = 0.01,
        max_depth: float = 0.0,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_ray_pose: bool = True,
        intrinsics: Optional[np.ndarray] = None,
        intrinsics_image_size: Optional[tuple[int, int]] = None,
        gt_depth_provider: Optional[DepthProvider] = None,
        scale_mode: str = "none",
        fixed_depth_scale: float = 1.0,
        scale_clip_min: float = 0.05,
        scale_clip_max: float = 20.0,
        conf_percentile: Optional[float] = None,
        mask_sky: bool = False,
        sim3_transform_path: Optional[str] = None,
        require_transform: bool = False,
    ) -> None:
        self._model_name = model_name
        self._chunk_size = max(1, int(chunk_size))
        self._overlap = min(max(0, int(overlap)), self._chunk_size - 1)
        self._device = _resolve_device(device)
        self._max_cache = max(1, int(max_cache))
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._process_res = int(process_res)
        self._process_res_method = str(process_res_method)
        self._use_ray_pose = bool(use_ray_pose)
        self._gt_depth_provider = gt_depth_provider
        self._scale_mode = str(scale_mode or "none").lower()
        self._fixed_depth_scale = float(fixed_depth_scale)
        self._scale_clip_min = float(scale_clip_min)
        self._scale_clip_max = float(scale_clip_max)
        self._conf_percentile = (
            None if conf_percentile is None else float(conf_percentile)
        )
        self._mask_sky = bool(mask_sky)
        self._warned_no_gt_scale = False

        if self._scale_clip_min <= 0.0 or self._scale_clip_max < self._scale_clip_min:
            raise ValueError(
                "scale clip bounds must satisfy 0 < min <= max, got "
                f"{self._scale_clip_min}, {self._scale_clip_max}"
            )
        if self._scale_mode not in ("none", "fixed", "gt_median_chunk"):
            raise ValueError(
                "scale_mode must be one of: none, fixed, gt_median_chunk; "
                f"got {self._scale_mode!r}"
            )

        if intrinsics is not None:
            intrinsics = np.asarray(intrinsics, dtype=np.float32)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"intrinsics must be finite 3x3, got {intrinsics.shape}")
        self._base_K = intrinsics
        self._base_K_size: Optional[tuple[int, int]] = None
        if intrinsics_image_size is not None:
            h, w = int(intrinsics_image_size[0]), int(intrinsics_image_size[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"intrinsics_image_size must be positive, got ({h}, {w})")
            self._base_K_size = (h, w)
        self._sim3 = _load_sim3_matrix(sim3_transform_path, require_transform)

        self._model = None

        self._frame_buffer: list[tuple[int, np.ndarray]] = []
        self._is_first_chunk = True

        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._fed: set[int] = set()

        self._queue: queue.Queue = queue.Queue()
        self._events: dict[int, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._worker_error: Optional[BaseException] = None
        self._worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        self._ensure_model()
        self._ensure_worker()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if rgb is None:
            return
        with self._events_lock:
            if frame_idx in self._fed:
                return
            self._fed.add(frame_idx)
            self._events.setdefault(frame_idx, threading.Event())

        self._ensure_worker()
        self._queue.put((int(frame_idx), rgb))

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        self._wait_for_frame(frame_idx)
        self._raise_worker_error()
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        self._wait_for_frame(frame_idx)
        self._raise_worker_error()
        return self._pose_cache.get(frame_idx)

    def get_sync_debug(self, frame_idx: int) -> dict:
        return {
            "frame_key": int(frame_idx),
            "depth_path": None,
            "pose_index": int(frame_idx) if frame_idx in self._pose_cache else None,
        }

    def drain(self, timeout: float = 300.0) -> None:
        self._raise_worker_error()
        done = threading.Event()
        self._queue.put((_DRAIN, done))
        done.wait(timeout=timeout)
        self._raise_worker_error()
        log.info("[DAv3] drain complete")

    def close(self, join_timeout: float = 120.0) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker_stop.set()
            self._queue.put(_STOP)
            self._worker.join(timeout=join_timeout)
            if self._worker.is_alive():
                log.warning(
                    "[DAv3] worker did not exit within %.1fs - model may leak",
                    join_timeout,
                )
            self._worker = None

        self._model = None
        self._frame_buffer.clear()
        self._is_first_chunk = True
        self._depth_cache.clear()
        self._pose_cache.clear()
        self._fed.clear()
        self._worker_error = None
        with self._events_lock:
            self._events.clear()

        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Model / inference
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise RuntimeError("DAv3OnlineDepthProvider requires PyTorch")
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable for device {self._device!r}")

        DepthAnything3 = _import_depth_anything_3()
        log.info("[DAv3] loading %s on %s ...", self._model_name, self._device)
        if hasattr(DepthAnything3, "from_pretrained"):
            model = DepthAnything3.from_pretrained(self._model_name)
        else:
            model = DepthAnything3(self._model_name)
        self._model = model.to(self._device).eval()
        log.info("[DAv3] model ready")

    def _push_frame_impl(self, frame_idx: int, rgb: np.ndarray) -> None:
        self._ensure_model()
        self._frame_buffer.append((int(frame_idx), np.ascontiguousarray(rgb)))
        if len(self._frame_buffer) >= self._chunk_size:
            self._process_buffer(final=False)

    def _flush_buffer(self) -> None:
        if not self._frame_buffer:
            return
        if not self._is_first_chunk and len(self._frame_buffer) <= self._overlap:
            return
        self._process_buffer(final=True)

    def _process_buffer(self, final: bool) -> None:
        if not self._frame_buffer:
            return

        indices = [idx for idx, _ in self._frame_buffer]
        frames = [rgb for _, rgb in self._frame_buffer]
        raw_shapes = [rgb.shape[:2] for rgb in frames]

        prediction = self._model.inference(
            frames,
            intrinsics=self._build_intrinsics_batch(raw_shapes),
            export_format="mini_npz",
            use_ray_pose=self._use_ray_pose,
            process_res=self._process_res,
            process_res_method=self._process_res_method,
        )

        depth_raw = getattr(prediction, "depth", None)
        if depth_raw is None:
            raise RuntimeError("DepthAnything3 prediction has no depth output")
        depths = np.asarray(depth_raw)
        if depths.size == 0:
            raise RuntimeError("DepthAnything3 prediction returned empty depth output")
        if depths.ndim == 2:
            depths = depths[None, ...]

        poses = getattr(prediction, "extrinsics", None)
        poses_np = np.asarray(poses) if poses is not None else None
        if poses_np is not None and poses_np.ndim == 2:
            poses_np = poses_np[None, ...]
        conf_np = _as_optional_array(getattr(prediction, "conf", None))
        sky_np = _as_optional_array(getattr(prediction, "sky", None))

        start = 0 if self._is_first_chunk else self._overlap
        emit_count = max(0, len(indices) - start)
        emit_indices = indices[start : start + emit_count]

        pending: list[tuple[int, int, np.ndarray, Optional[np.ndarray]]] = []
        for local_i, frame_idx in enumerate(emit_indices, start=start):
            if local_i >= depths.shape[0]:
                break
            h, w = raw_shapes[local_i]
            dm = self._resize_depth(depths[local_i], h, w)
            invalid = self._invalid_mask(conf_np, sky_np, local_i, h, w)
            pending.append((frame_idx, local_i, dm, invalid))

        scale = self._resolve_depth_scale(pending)

        for frame_idx, local_i, dm, invalid in pending:
            self._depth_cache[frame_idx] = self._finalize_depth(dm, scale, invalid)
            pose = self._postprocess_pose(poses_np, local_i, scale)
            if pose is not None:
                self._pose_cache[frame_idx] = pose

        with self._events_lock:
            for frame_idx in emit_indices:
                ev = self._events.get(frame_idx)
                if ev is not None:
                    ev.set()

        log.debug(
            "[DAv3] emitted frames %s..%s (%d)",
            emit_indices[0] if emit_indices else "?",
            emit_indices[-1] if emit_indices else "?",
            len(emit_indices),
        )

        keep = 0 if final else self._overlap
        self._frame_buffer = self._frame_buffer[-keep:] if keep > 0 else []
        self._is_first_chunk = False

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _build_intrinsics_batch(self, raw_shapes: list[tuple[int, int]]) -> Optional[np.ndarray]:
        if self._base_K is None:
            return None

        base_h, base_w = self._base_K_size or raw_shapes[0]
        Ks = []
        for raw_h, raw_w in raw_shapes:
            K = self._base_K.copy()
            if (raw_h, raw_w) != (base_h, base_w):
                sx = raw_w / float(base_w)
                sy = raw_h / float(base_h)
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
            Ks.append(K)
        return np.stack(Ks, axis=0).astype(np.float32)

    def _resize_depth(self, depth: np.ndarray, raw_h: int, raw_w: int) -> np.ndarray:
        dm = np.asarray(depth, dtype=np.float32).squeeze()
        if dm.ndim != 2:
            raise ValueError(f"Expected 2D DAv3 depth map, got shape {dm.shape}")
        if dm.shape != (raw_h, raw_w):
            dm = cv2.resize(dm, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        dm[~np.isfinite(dm)] = 0.0
        return dm.astype(np.float32)

    def _finalize_depth(
        self,
        depth: np.ndarray,
        scale: float,
        invalid_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        dm = (depth.astype(np.float32) * float(scale)).astype(np.float32)
        if invalid_mask is not None:
            dm[invalid_mask] = 0.0
        dm[~np.isfinite(dm)] = 0.0
        if self._min_depth > 0:
            dm[dm < self._min_depth] = 0.0
        if self._max_depth > 0:
            dm[dm > self._max_depth] = 0.0
        return dm.astype(np.float32)

    def _invalid_mask(
        self,
        conf: Optional[np.ndarray],
        sky: Optional[np.ndarray],
        local_i: int,
        raw_h: int,
        raw_w: int,
    ) -> Optional[np.ndarray]:
        invalid = None

        if self._conf_percentile is not None and conf is not None and local_i < conf.shape[0]:
            conf_i = self._resize_aux(conf[local_i], raw_h, raw_w)
            finite = np.isfinite(conf_i)
            if finite.any():
                threshold = np.percentile(conf_i[finite], self._conf_percentile)
                invalid = conf_i < threshold

        if self._mask_sky and sky is not None and local_i < sky.shape[0]:
            sky_i = self._resize_aux(sky[local_i], raw_h, raw_w)
            sky_mask = sky_i >= 0.5
            invalid = sky_mask if invalid is None else np.logical_or(invalid, sky_mask)

        return invalid

    def _resize_aux(self, arr: np.ndarray, raw_h: int, raw_w: int) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32).squeeze()
        if out.ndim != 2:
            return np.zeros((raw_h, raw_w), dtype=np.float32)
        if out.shape != (raw_h, raw_w):
            out = cv2.resize(out, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        out[~np.isfinite(out)] = 0.0
        return out

    def _resolve_depth_scale(
        self,
        pending: list[tuple[int, int, np.ndarray, Optional[np.ndarray]]],
    ) -> float:
        if self._scale_mode == "none":
            return 1.0
        if self._scale_mode == "fixed":
            return self._clip_scale(self._fixed_depth_scale)
        if self._gt_depth_provider is None:
            if not self._warned_no_gt_scale:
                log.warning("[DAv3] GT depth unavailable; falling back to scale=1.0")
                self._warned_no_gt_scale = True
            return 1.0

        ratios = []
        for frame_idx, _, dm, invalid in pending:
            gt = self._gt_depth_provider.get_depth(frame_idx)
            if gt is None:
                continue
            gt = np.asarray(gt, dtype=np.float32)
            if gt.shape != dm.shape:
                gt = cv2.resize(gt, (dm.shape[1], dm.shape[0]), interpolation=cv2.INTER_NEAREST)

            valid = np.isfinite(dm) & np.isfinite(gt) & (dm > 1e-6) & (gt > self._min_depth)
            if self._max_depth > 0:
                valid &= gt <= self._max_depth
            if invalid is not None:
                valid &= ~invalid
            if valid.any():
                ratios.append((gt[valid] / dm[valid]).astype(np.float32))

        if not ratios:
            if not self._warned_no_gt_scale:
                log.warning("[DAv3] Could not estimate GT median scale; falling back to scale=1.0")
                self._warned_no_gt_scale = True
            return 1.0

        scale = float(np.median(np.concatenate(ratios)))
        return self._clip_scale(scale)

    def _clip_scale(self, scale: float) -> float:
        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0
        return float(np.clip(scale, self._scale_clip_min, self._scale_clip_max))

    def _postprocess_pose(
        self,
        poses: Optional[np.ndarray],
        local_i: int,
        depth_scale: float,
    ) -> Optional[np.ndarray]:
        if poses is None or local_i >= poses.shape[0]:
            return None
        pose = np.asarray(poses[local_i], dtype=np.float32)
        if pose.shape == (3, 4):
            full = np.eye(4, dtype=np.float32)
            full[:3, :4] = pose
            pose = full
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None
        if not self._use_ray_pose:
            pose = np.linalg.inv(pose).astype(np.float32)
        pose = pose.copy()
        pose[:3, 3] *= float(depth_scale)
        return (self._sim3 @ pose).astype(np.float32)

    # ------------------------------------------------------------------
    # Worker / cache helpers
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="dav3-worker")
        self._worker.start()
        log.info("[DAv3] worker started")

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if item is _STOP:
                    break

                if isinstance(item, tuple) and len(item) == 2 and item[0] is _DRAIN:
                    self._flush_buffer()
                    item[1].set()
                    continue

                frame_idx, rgb = item
                self._push_frame_impl(frame_idx, rgb)
            except BaseException as exc:
                self._record_worker_error(exc)
                if isinstance(item, tuple) and len(item) == 2 and item[0] is _DRAIN:
                    item[1].set()
                break

        if self._worker_error is None:
            try:
                self._flush_buffer()
            except BaseException as exc:
                self._record_worker_error(exc)
        log.info("[DAv3] worker exiting")

    def _wait_for_frame(self, frame_idx: int, timeout: float = 120.0) -> None:
        if frame_idx in self._depth_cache:
            return
        self._raise_worker_error()
        with self._events_lock:
            ev = self._events.get(frame_idx)
        if ev is not None:
            ev.wait(timeout=timeout)
        self._raise_worker_error()

    def _evict_old(self, cache: OrderedDict) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)

    def get_sim3_matrix(self) -> np.ndarray:
        return self._sim3.copy()

    def set_sim3_transform(self, sim3: np.ndarray) -> None:
        sim3 = np.asarray(sim3, dtype=np.float32)
        if sim3.shape != (4, 4):
            raise ValueError(f"sim3 must be 4x4, got {sim3.shape}")
        self._sim3 = sim3
        log.info("[DAv3] Sim(3) transform set externally")

    def _record_worker_error(self, exc: BaseException) -> None:
        log.exception("[DAv3] worker failed")
        self._worker_error = exc
        with self._events_lock:
            for ev in self._events.values():
                ev.set()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("DAv3 online worker failed") from self._worker_error


def _import_depth_anything_3():
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:
        raise RuntimeError("DepthAnything3 is not installed.") from exc
    return DepthAnything3


def _as_optional_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def _load_sim3_matrix(path_str: Optional[str], require: bool = False) -> np.ndarray:
    if path_str is None:
        return np.eye(4, dtype=np.float32)

    path = Path(path_str)
    if not path.exists():
        if require:
            raise FileNotFoundError(f"DAv3 Sim(3) transform not found: {path}")
        return np.eye(4, dtype=np.float32)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "sim3_matrix_4x4" in payload:
        sim3 = np.asarray(payload["sim3_matrix_4x4"], dtype=np.float64)
    else:
        s = float(payload["scale"])
        R = np.asarray(payload["rotation"], dtype=np.float64)
        t = np.asarray(payload["translation"], dtype=np.float64)
        sim3 = np.eye(4, dtype=np.float64)
        sim3[:3, :3] = s * R
        sim3[:3, 3] = t

    if sim3.shape != (4, 4) or not np.isfinite(sim3).all():
        raise ValueError(f"Invalid DAv3 Sim(3) transform: {path}")
    return sim3.astype(np.float32)
