"""Pi3 online / streaming depth provider backed by Pi3XVOStream.

Delegates all inference, chunking, inter-chunk Sim3 stitching, confidence
filtering, and depth-edge masking to ``Pi3XVOStream``.  This class handles:

- RGB preprocessing (pad / resize + normalisation)
- Async background inference via a worker thread
- Per-frame depth / pose caching with ``Event``-based blocking
- External Sim(3) transform (Pi3 world -> GT world)
- Depth post-processing (un-pad / un-resize, min / max clamping)
"""
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
from pathlib import Path
import queue
import threading
import time as _time
from typing import Optional, Sequence

log = logging.getLogger("pi3_online")

import cv2
import numpy as np

from .base import OnlineDepthProvider

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _load_sim3_matrix(path_str: str, require: bool) -> np.ndarray:
    """Load a Sim(3) 4x4 matrix from a JSON file."""
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
            raise ValueError(f"sim3_matrix_4x4 must be 4x4, got shape {sim3.shape}")
    else:
        scale = float(payload["scale"])
        rotation = np.asarray(payload["rotation"], dtype=np.float64)
        translation = np.asarray(payload["translation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation must be (3,), got {translation.shape}")
        sim3 = np.eye(4, dtype=np.float64)
        sim3[:3, :3] = scale * rotation
        sim3[:3, 3] = translation

    if not np.isfinite(sim3).all():
        raise ValueError(f"Invalid values in Sim(3) transform: {path}")
    return sim3.astype(np.float32)


def _normalize_torch_device(device: Optional[str]) -> str:
    if device is None:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    raw = str(device).strip()
    low = raw.lower()
    if low in {"cpu", "mps"}:
        return low
    if low.startswith("cuda"):
        return low
    if raw.isdigit():
        return f"cuda:{raw}"
    return raw


def _validate_intrinsics(K: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {K.shape}")
    if not np.isfinite(K).all():
        raise ValueError("intrinsics contains non-finite values")
    return K


def _validate_intrinsics_size(
    size: Optional[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    if size is None:
        return None
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"intrinsics_image_size must be positive, got {(h, w)}")
    return (h, w)


# ------------------------------------------------------------------
#  Provider
# ------------------------------------------------------------------

class Pi3OnlineDepthProvider(OnlineDepthProvider):
    """Streaming depth+pose provider backed by ``Pi3XVOStream``."""

    def __init__(
        self,
        model_name: str = "yyfz233/Pi3X",
        chunk_size: int = 30,
        overlap: int = 10,
        target_size: Optional[tuple[int, int]] = None,
        device: Optional[str] = None,
        max_cache: int = 128,
        min_depth: float = 0.01,
        max_depth: float = 0.0,
        conf_threshold: float = 0.05,
        inject_condition: Optional[Sequence[str]] = None,
        intrinsics: Optional[np.ndarray] = None,
        intrinsics_image_size: Optional[tuple[int, int]] = None,
        use_original_size: bool = False,
        pixel_limit: int = 255_000,
        patch_size: int = 14,
        async_inference: bool = True,
        transform_path: Optional[str] = None,
        require_transform: bool = False,
    ) -> None:
        self._model_name = str(model_name)
        self._chunk_size = max(1, int(chunk_size))
        self._overlap = max(0, int(overlap))
        if self._overlap >= self._chunk_size:
            self._overlap = self._chunk_size - 1

        self._target_size = target_size
        self._device = _normalize_torch_device(device)
        self._max_cache = int(max_cache)

        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._conf_threshold = float(conf_threshold)
        self._inject_condition: list[str] = [
            str(x) for x in (inject_condition or [])
        ]
        self._use_original_size = bool(use_original_size)
        self._pixel_limit = int(max(1, pixel_limit))
        self._patch_size = int(max(1, patch_size))

        self._base_K = _validate_intrinsics(intrinsics)
        self._base_intrinsics_size = _validate_intrinsics_size(intrinsics_image_size)

        # External Sim(3): Pi3 world -> GT world
        self._sim3: np.ndarray = (
            _load_sim3_matrix(transform_path, require_transform)
            if transform_path
            else np.eye(4, dtype=np.float32)
        )
        if not np.allclose(self._sim3, np.eye(4)):
            log.info("[Pi3] Sim(3) loaded from %s", transform_path)

        # Model / stream pipeline (lazy-loaded)
        self._model = None
        self._stream = None  # Pi3XVOStream instance

        # Preprocessing state (computed from first frame)
        self._raw_h: Optional[int] = None
        self._raw_w: Optional[int] = None
        self._model_h: int = 0
        self._model_w: int = 0
        self._preprocess_mode: str = "none"  # "none" | "pad" | "resize"

        # Frame-index buffer (kept in sync with Pi3XVOStream's internal buffer)
        self._index_buffer: list[int] = []
        self._is_first_chunk: bool = True

        # Result caches
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Dedup: track which frame indices have been fed
        self._fed_frames: set[int] = set()

        # Async worker
        self._async_inference = bool(async_inference)
        self._input_queue: queue.Queue = queue.Queue()
        self._frame_events: dict[int, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

    # ==============================================================
    #  Public API
    # ==============================================================

    def warmup(self) -> None:
        log.info("[Pi3] warmup()")
        self._load_model()
        if self._async_inference and self._worker_thread is None:
            self._start_worker()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if rgb is None:
            return
        if frame_idx in self._fed_frames:
            return
        self._fed_frames.add(frame_idx)
        with self._events_lock:
            self._frame_events.setdefault(frame_idx, threading.Event())
        if self._async_inference:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._start_worker()
            self._input_queue.put((frame_idx, rgb))
            return
        # Synchronous path
        self._push_frame(frame_idx, rgb)

    def drain(self, timeout: float = 300.0) -> None:
        """Flush remaining buffered frames and wait for all results.

        Call this after feeding all frames to ensure Pi3XVOStream processes
        any partial chunk that hasn't reached ``chunk_size`` yet.
        """
        if not self._async_inference:
            # Sync path: flush directly
            if self._stream is not None and len(self._stream._buffer) > 0:
                results = self._stream.flush()
                if results is not None:
                    self._handle_results(results)
            return
        # Async path: send drain sentinel and wait
        done = threading.Event()
        self._input_queue.put(("__drain__", done))
        done.wait(timeout=timeout)
        log.info("[Pi3] drain complete")

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._depth_cache:
            if self._async_inference:
                ev = self._get_event(frame_idx)
                if ev is not None:
                    log.debug("[Pi3] get_depth(%d) waiting ...", frame_idx)
                    ev.wait(timeout=120.0)
            else:
                self._force_process()
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._pose_cache:
            if self._async_inference:
                ev = self._get_event(frame_idx)
                if ev is not None:
                    ev.wait(timeout=120.0)
            else:
                self._force_process()
        return self._pose_cache.get(frame_idx)

    def get_sim3_matrix(self) -> np.ndarray:
        return self._sim3.copy()

    def set_sim3_transform(self, sim3: np.ndarray) -> None:
        sim3 = np.asarray(sim3, dtype=np.float32)
        if sim3.shape != (4, 4):
            raise ValueError(f"sim3 must be 4x4, got {sim3.shape}")
        self._sim3 = sim3
        log.info("[Pi3] Sim(3) transform set (externally)")

    def close(self) -> None:
        log.info("[Pi3] close()")
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_stop.set()
            self._input_queue.put(None)
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        self._model = None
        self._stream = None
        self._depth_cache.clear()
        self._pose_cache.clear()
        self._index_buffer.clear()
        self._is_first_chunk = True
        self._raw_h = self._raw_w = None
        self._fed_frames.clear()
        with self._events_lock:
            self._frame_events.clear()
        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ==============================================================
    #  Model loading
    # ==============================================================

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise RuntimeError("Pi3OnlineDepthProvider requires torch")
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {self._device!r} requested but CUDA unavailable"
            )

        Pi3X, _ = self._import_pi3_modules()
        log.info("[Pi3] loading %s on %s ...", self._model_name, self._device)

        self._model = Pi3X.from_pretrained(self._model_name).to(self._device).eval()
        log.info("[Pi3] model loaded")

    @staticmethod
    def _import_pi3_modules():
        try:
            from pi3.models.pi3x import Pi3X
            from pi3.pipe.pi3x_vo_stream import Pi3XVOStream
            return Pi3X, Pi3XVOStream
        except Exception:
            import sys
            root = Path(__file__).resolve().parent.parent / "Pi3_for_yolo_sgg"
            if root.exists():
                rstr = str(root)
                if rstr not in sys.path:
                    sys.path.insert(0, rstr)
            from pi3.models.pi3x import Pi3X
            from pi3.pipe.pi3x_vo_stream import Pi3XVOStream
            return Pi3X, Pi3XVOStream

    def _select_dtype(self):
        if not self._device.startswith("cuda"):
            return torch.float32
        dev_idx = None
        if self._device.startswith("cuda:"):
            s = self._device.split(":", 1)[1]
            if s.isdigit():
                dev_idx = int(s)
        cap = torch.cuda.get_device_capability(dev_idx)
        return torch.bfloat16 if cap[0] >= 8 else torch.float16

    # ==============================================================
    #  Preprocessing (computed once from first frame)
    # ==============================================================

    def _setup_on_first_frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        self._raw_h, self._raw_w = h, w
        self._model_h, self._model_w = self._compute_model_size(h, w)

        mh, mw = self._model_h, self._model_w
        if self._use_original_size and (mh != h or mw != w):
            self._preprocess_mode = "pad"
        elif (not self._use_original_size) and (mh != h or mw != w):
            self._preprocess_mode = "resize"
        else:
            self._preprocess_mode = "none"

        log.info(
            "[Pi3] first frame %dx%d -> model %dx%d (%s)",
            w, h, mw, mh, self._preprocess_mode,
        )

        # Create the Pi3XVOStream with intrinsics known upfront
        K_tensor = self._build_intrinsics_tensor()
        _, Pi3XVOStream = self._import_pi3_modules()
        self._stream = Pi3XVOStream(
            model=self._model,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            conf_thre=self._conf_threshold,
            inject_condition=self._inject_condition or [],
            intrinsics=K_tensor,
            dtype=self._select_dtype(),
        )
        log.info("[Pi3] Pi3XVOStream created (intrinsics=%s)",
                 "set" if K_tensor is not None else "none")

    def _compute_model_size(self, raw_h: int, raw_w: int) -> tuple[int, int]:
        p = self._patch_size
        if self._use_original_size:
            return int(math.ceil(raw_h / p) * p), int(math.ceil(raw_w / p) * p)
        if raw_h <= 0 or raw_w <= 0:
            return p, p
        scale = math.sqrt(self._pixel_limit / float(raw_h * raw_w))
        k = max(1, round(raw_h * scale / p))
        m = max(1, round(raw_w * scale / p))
        while (k * p) * (m * p) > self._pixel_limit and (k > 1 or m > 1):
            if k / max(1, m) > (raw_h / max(1e-9, raw_w)):
                k -= 1
            else:
                m -= 1
            k, m = max(1, k), max(1, m)
        return int(k * p), int(m * p)

    def _build_intrinsics_tensor(self):
        """Build (1, 1, 3, 3) intrinsics scaled to model resolution."""
        if self._base_K is None or torch is None:
            return None
        K = self._base_K.copy()
        if self._base_intrinsics_size is not None:
            base_h, base_w = self._base_intrinsics_size
        else:
            base_h, base_w = self._raw_h, self._raw_w
        if self._preprocess_mode == "resize":
            sx = float(self._model_w) / float(base_w)
            sy = float(self._model_h) / float(base_h)
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
        return torch.as_tensor(
            K, dtype=torch.float32, device=self._device
        ).view(1, 1, 3, 3)

    def _preprocess_frame(self, rgb: np.ndarray):
        """RGB uint8 (H,W,3) -> float32 tensor (3, model_h, model_w) on device."""
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        if self._preprocess_mode == "pad":
            canvas = np.zeros(
                (self._model_h, self._model_w, 3), dtype=np.uint8
            )
            h = min(self._raw_h, rgb.shape[0])
            w = min(self._raw_w, rgb.shape[1])
            canvas[:h, :w] = rgb[:h, :w]
            rgb = canvas
        elif self._preprocess_mode == "resize":
            rgb = cv2.resize(
                rgb, (self._model_w, self._model_h),
                interpolation=cv2.INTER_LINEAR,
            )
        t = torch.from_numpy(np.ascontiguousarray(rgb))
        return (
            t.permute(2, 0, 1)
             .to(device=self._device, dtype=torch.float32)
             .div_(255.0)
        )

    # ==============================================================
    #  Core: push frames / handle results
    # ==============================================================

    def _push_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if self._model is None:
            self._load_model()
        if self._raw_h is None:
            self._setup_on_first_frame(rgb)

        tensor = self._preprocess_frame(rgb)
        self._index_buffer.append(frame_idx)

        results = self._stream.push_frame(tensor)
        if results is not None:
            self._handle_results(results)

    def _handle_results(self, results: dict) -> None:
        n_new = results["depth"].shape[0]
        if n_new == 0:
            self._index_buffer = self._index_buffer[-self._overlap:]
            self._is_first_chunk = False
            return

        n_start = 0 if self._is_first_chunk else self._overlap
        result_indices = self._index_buffer[n_start : n_start + n_new]

        depth_np = results["depth"].cpu().numpy()
        poses_np = results["poses"].cpu().numpy()

        for i, idx in enumerate(result_indices):
            self._depth_cache[idx] = self._postprocess_depth(depth_np[i])
            self._pose_cache[idx] = (self._sim3 @ poses_np[i]).astype(np.float32)

        with self._events_lock:
            for idx in result_indices:
                ev = self._frame_events.get(idx)
                if ev is not None:
                    ev.set()

        log.debug("[Pi3] emitted %d frames [%s..%s]",
                  len(result_indices),
                  result_indices[0] if result_indices else "?",
                  result_indices[-1] if result_indices else "?")

        self._index_buffer = self._index_buffer[-self._overlap:]
        self._is_first_chunk = False

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

    def _postprocess_depth(self, dm: np.ndarray) -> np.ndarray:
        dm = dm.astype(np.float32)
        if self._preprocess_mode == "pad":
            dm = dm[: self._raw_h, : self._raw_w]
        elif self._preprocess_mode == "resize":
            dm = cv2.resize(
                dm, (self._raw_w, self._raw_h),
                interpolation=cv2.INTER_LINEAR,
            )
        if self._target_size is not None:
            dm = cv2.resize(
                dm, (self._target_size[1], self._target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        dm[~np.isfinite(dm)] = 0.0
        dm[dm < self._min_depth] = 0.0
        if self._max_depth > 0.0:
            dm[dm > self._max_depth] = 0.0
        return dm

    def _force_process(self) -> None:
        """Sync path: force stream to process its current buffer."""
        if self._stream is None:
            return
        n_buf = len(self._stream._buffer)
        min_needed = 1 if self._is_first_chunk else self._overlap + 1
        if n_buf < min_needed:
            return
        results = self._stream._process_chunk()
        if results is not None:
            self._handle_results(results)

    def _get_event(self, frame_idx: int) -> Optional[threading.Event]:
        with self._events_lock:
            return self._frame_events.get(frame_idx)

    def _evict_old(self, cache: OrderedDict) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)

    # ==============================================================
    #  Async worker
    # ==============================================================

    def _start_worker(self) -> None:
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._async_worker, daemon=True, name="pi3-worker",
        )
        self._worker_thread.start()
        log.info("[Pi3] worker thread started")

    def _async_worker(self) -> None:
        log.info("[Pi3] async worker running")
        while not self._worker_stop.is_set():
            try:
                item = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            # Drain sentinel: flush remaining frames
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__drain__":
                done_event = item[1]
                if self._stream is not None and len(self._stream._buffer) > 0:
                    log.info("[Pi3] drain: flushing %d buffered frames",
                             len(self._stream._buffer))
                    results = self._stream.flush()
                    if results is not None:
                        self._handle_results(results)
                done_event.set()
                continue

            frame_idx, rgb = item
            self._push_frame(frame_idx, rgb)

        # Final drain on shutdown
        if self._stream is not None and len(self._stream._buffer) > 0:
            log.info("[Pi3] shutdown drain: %d frames", len(self._stream._buffer))
            results = self._stream.flush()
            if results is not None:
                self._handle_results(results)
        log.info("[Pi3] async worker exiting")
