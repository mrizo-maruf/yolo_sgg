"""Pi3 online streaming depth provider backed by Pi3XVOStream.

Handles:
- RGB preprocessing (resize / pad to model resolution)
- Async background inference via a dedicated worker thread
- Per-frame depth / pose caching with Event-based blocking
- Optional Sim(3) transform (Pi3 world -> GT world)
- Depth post-processing (un-pad / un-resize, min / max clamping)
"""
from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
import queue
import threading
from typing import Optional, Sequence

import cv2
import numpy as np

from .base import OnlineDepthProvider

log = logging.getLogger("pi3_online")

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Queue sentinel objects
_STOP = object()
_DRAIN = object()


# ------------------------------------------------------------------
#  Public helpers (used by new_run.py for external Sim(3) loading)
# ------------------------------------------------------------------

def load_sim3_matrix(path_str: str, require: bool = True) -> np.ndarray:
    """Load a Sim(3) 4x4 matrix from a JSON file.

    Accepts either ``{"sim3_matrix_4x4": [[...]]}`` or decomposed
    ``{"scale": ..., "rotation": [[...]], "translation": [...]}``.
    """
    path = Path(path_str)
    if not path.exists():
        if require:
            raise FileNotFoundError(f"Sim(3) transform not found: {path}")
        return np.eye(4, dtype=np.float32)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "sim3_matrix_4x4" in payload:
        sim3 = np.asarray(payload["sim3_matrix_4x4"], dtype=np.float64)
        if sim3.shape != (4, 4):
            raise ValueError(f"sim3_matrix_4x4 must be 4x4, got {sim3.shape}")
    else:
        s = float(payload["scale"])
        R = np.asarray(payload["rotation"], dtype=np.float64)
        t = np.asarray(payload["translation"], dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"translation must be (3,), got {t.shape}")
        sim3 = np.eye(4, dtype=np.float64)
        sim3[:3, :3] = s * R
        sim3[:3, 3] = t

    if not np.isfinite(sim3).all():
        raise ValueError(f"Non-finite values in Sim(3) transform: {path}")
    return sim3.astype(np.float32)


# Backward-compat alias used by new_run.py
_load_sim3_matrix = load_sim3_matrix


# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    s = str(device).strip().lower()
    if s in ("cpu", "mps") or s.startswith("cuda"):
        return s
    if s.isdigit():
        return f"cuda:{s}"
    return s


def _select_dtype(device: str):
    if not device.startswith("cuda"):
        return torch.float32
    idx = int(device.split(":", 1)[1]) if ":" in device else None
    return torch.bfloat16 if torch.cuda.get_device_capability(idx)[0] >= 8 else torch.float16


# ------------------------------------------------------------------
#  Provider
# ------------------------------------------------------------------

class Pi3OnlineDepthProvider(OnlineDepthProvider):
    """Streaming depth + pose provider backed by ``Pi3XVOStream``.

    Frames are fed via :meth:`feed_frame` (called from the data loader's
    ``get_rgb``).  A background worker pushes them into ``Pi3XVOStream``,
    which emits depth + pose results every ``chunk_size`` frames.
    :meth:`get_depth` / :meth:`get_pose` block until the requested frame
    is ready.
    """

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
    ) -> None:
        # -- Config --
        self._model_name = model_name
        self._chunk_size = max(1, chunk_size)
        self._overlap = min(max(0, overlap), self._chunk_size - 1)
        self._target_size = target_size
        self._device = _resolve_device(device)
        self._max_cache = max_cache
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._conf_threshold = conf_threshold
        self._inject_condition = list(inject_condition or [])
        self._use_original_size = use_original_size
        self._pixel_limit = max(1, pixel_limit)
        self._patch_size = max(1, patch_size)

        # Intrinsics (at original image resolution)
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

        # Sim(3): Pi3 world -> GT world (set externally via set_sim3_transform)
        self._sim3 = np.eye(4, dtype=np.float32)

        # -- Lazy state (initialised from first frame) --
        self._model = None
        self._stream = None  # Pi3XVOStream
        self._raw_h: Optional[int] = None
        self._raw_w: Optional[int] = None
        self._model_h = 0
        self._model_w = 0
        self._preprocess_mode = "none"  # "none" | "pad" | "resize"

        # Frame-index buffer (mirrors Pi3XVOStream's internal buffer)
        self._index_buffer: list[int] = []
        self._is_first_chunk = True

        # -- Caches --
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # -- Dedup --
        self._fed: set[int] = set()

        # -- Threading --
        self._queue: queue.Queue = queue.Queue()
        self._events: dict[int, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

    # ================================================================
    #  Public API
    # ================================================================

    def warmup(self) -> None:
        """Load the model and start the inference worker."""
        self._ensure_model()
        self._ensure_worker()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        """Push an RGB frame for background inference.

        Duplicate frame indices are silently ignored.  Thread-safe.
        """
        if rgb is None:
            return
        # Atomic check-and-add under lock (feeder thread + YOLO can race)
        with self._events_lock:
            if frame_idx in self._fed:
                return
            self._fed.add(frame_idx)
            self._events.setdefault(frame_idx, threading.Event())

        self._ensure_worker()
        self._queue.put((frame_idx, rgb))

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        self._wait_for_frame(frame_idx)
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        self._wait_for_frame(frame_idx)
        return self._pose_cache.get(frame_idx)

    def drain(self, timeout: float = 300.0) -> None:
        """Flush remaining buffered frames and block until all results are cached."""
        done = threading.Event()
        self._queue.put((_DRAIN, done))
        done.wait(timeout=timeout)
        log.info("[Pi3] drain complete")

    def close(self) -> None:
        """Shut down the worker and free resources."""
        if self._worker is not None and self._worker.is_alive():
            self._worker_stop.set()
            self._queue.put(_STOP)
            self._worker.join(timeout=5.0)
            self._worker = None

        self._model = None
        self._stream = None
        self._depth_cache.clear()
        self._pose_cache.clear()
        self._index_buffer.clear()
        self._is_first_chunk = True
        self._raw_h = self._raw_w = None
        self._fed.clear()
        with self._events_lock:
            self._events.clear()

        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    # -- Sim(3) management --

    def get_sim3_matrix(self) -> np.ndarray:
        return self._sim3.copy()

    def set_sim3_transform(self, sim3: np.ndarray) -> None:
        sim3 = np.asarray(sim3, dtype=np.float32)
        if sim3.shape != (4, 4):
            raise ValueError(f"sim3 must be 4x4, got {sim3.shape}")
        self._sim3 = sim3
        log.info("[Pi3] Sim(3) transform set externally")

    # ================================================================
    #  Model loading (lazy)
    # ================================================================

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise RuntimeError("Pi3OnlineDepthProvider requires PyTorch")
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable for device {self._device!r}")

        Pi3X = _import_pi3x()
        log.info("[Pi3] loading %s on %s ...", self._model_name, self._device)
        self._model = Pi3X.from_pretrained(self._model_name).to(self._device).eval()
        log.info("[Pi3] model ready")

    # ================================================================
    #  First-frame initialisation
    # ================================================================

    def _init_on_first_frame(self, rgb: np.ndarray) -> None:
        """Compute preprocessing params and create the Pi3XVOStream."""
        h, w = rgb.shape[:2]
        self._raw_h, self._raw_w = h, w
        self._model_h, self._model_w = self._compute_model_size(h, w)

        mh, mw = self._model_h, self._model_w
        if self._use_original_size and (mh != h or mw != w):
            self._preprocess_mode = "pad"
        elif mh != h or mw != w:
            self._preprocess_mode = "resize"
        else:
            self._preprocess_mode = "none"

        log.info("[Pi3] %dx%d -> %dx%d (%s)", w, h, mw, mh, self._preprocess_mode)

        Pi3XVOStream = _import_vo_stream()
        self._stream = Pi3XVOStream(
            model=self._model,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            conf_thre=self._conf_threshold,
            inject_condition=self._inject_condition,
            intrinsics=self._build_intrinsics_tensor(),
            dtype=_select_dtype(self._device),
        )

    def _compute_model_size(self, h: int, w: int) -> tuple[int, int]:
        p = self._patch_size
        if self._use_original_size:
            return math.ceil(h / p) * p, math.ceil(w / p) * p

        scale = math.sqrt(self._pixel_limit / max(1, h * w))
        rows = max(1, round(h * scale / p))
        cols = max(1, round(w * scale / p))
        while (rows * p) * (cols * p) > self._pixel_limit and (rows > 1 or cols > 1):
            if rows / max(1, cols) > h / max(1e-9, w):
                rows -= 1
            else:
                cols -= 1
            rows, cols = max(1, rows), max(1, cols)
        return rows * p, cols * p

    def _build_intrinsics_tensor(self):
        """Scale base intrinsics to model resolution -> (1, 1, 3, 3) tensor."""
        if self._base_K is None or torch is None:
            return None
        K = self._base_K.copy()

        base_h, base_w = self._base_K_size or (self._raw_h, self._raw_w)
        if self._preprocess_mode == "resize":
            sx = self._model_w / base_w
            sy = self._model_h / base_h
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        return torch.as_tensor(K, dtype=torch.float32, device=self._device).view(1, 1, 3, 3)

    # ================================================================
    #  Preprocessing
    # ================================================================

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H,W,3) -> float32 tensor (3, model_h, model_w) on device."""
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

        if self._preprocess_mode == "pad":
            canvas = np.zeros((self._model_h, self._model_w, 3), dtype=np.uint8)
            h = min(self._raw_h, rgb.shape[0])
            w = min(self._raw_w, rgb.shape[1])
            canvas[:h, :w] = rgb[:h, :w]
            rgb = canvas
        elif self._preprocess_mode == "resize":
            rgb = cv2.resize(rgb, (self._model_w, self._model_h), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(np.ascontiguousarray(rgb))
        return t.permute(2, 0, 1).to(device=self._device, dtype=torch.float32).div_(255.0)

    # ================================================================
    #  Core inference
    # ================================================================

    def _push_frame_impl(self, frame_idx: int, rgb: np.ndarray) -> None:
        """Preprocess, push to Pi3XVOStream, and store any emitted results."""
        self._ensure_model()
        if self._raw_h is None:
            self._init_on_first_frame(rgb)

        tensor = self._preprocess(rgb)
        self._index_buffer.append(frame_idx)

        results = self._stream.push_frame(tensor)
        if results is not None:
            self._store_results(results)
        del tensor, results

    def _flush_stream(self) -> None:
        """Flush any remaining frames in the Pi3XVOStream buffer."""
        if self._stream is None or len(self._stream._buffer) == 0:
            return
        results = self._stream.flush()
        if results is not None:
            self._store_results(results)
        del results

    def _store_results(self, results: dict) -> None:
        """Unpack a Pi3XVOStream result dict into the per-frame caches."""
        n_new = results["depth"].shape[0]
        if n_new == 0:
            self._index_buffer = self._index_buffer[-self._overlap:]
            self._is_first_chunk = False
            return

        start = 0 if self._is_first_chunk else self._overlap
        indices = self._index_buffer[start : start + n_new]

        # Move only what we need to CPU; discard GPU tensors immediately
        depth_np = results["depth"].cpu().numpy()
        poses_np = results["poses"].cpu().numpy()
        conf_np = results["conf"].cpu().numpy() if "conf" in results else None
        # Release all GPU tensors (points, conf, depth, poses) now
        results.clear()
        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Zero out low-confidence depth pixels (edge artifacts, unreliable
        # regions).  run_stream.py does this via conf_thre in build_frame_pcd;
        # we mirror it here so the tracker only sees reliable depth.
        if conf_np is not None and self._conf_threshold > 0:
            for i in range(depth_np.shape[0]):
                depth_np[i][conf_np[i] < self._conf_threshold] = 0.0

        sim3 = self._sim3

        for i, idx in enumerate(indices):
            self._depth_cache[idx] = self._postprocess_depth(depth_np[i])
            self._pose_cache[idx] = (sim3 @ poses_np[i]).astype(np.float32)

        # Signal waiting consumers
        with self._events_lock:
            for idx in indices:
                ev = self._events.get(idx)
                if ev is not None:
                    ev.set()

        log.debug("[Pi3] emitted frames %s..%s (%d)",
                  indices[0] if indices else "?",
                  indices[-1] if indices else "?",
                  len(indices))

        self._index_buffer = self._index_buffer[-self._overlap:]
        self._is_first_chunk = False

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

    def _postprocess_depth(self, dm: np.ndarray) -> np.ndarray:
        """Resize depth back to original / target resolution and clamp."""
        dm = dm.astype(np.float32)

        # Undo preprocessing
        if self._preprocess_mode == "pad":
            dm = dm[: self._raw_h, : self._raw_w]
        elif self._preprocess_mode == "resize":
            dm = cv2.resize(dm, (self._raw_w, self._raw_h), interpolation=cv2.INTER_NEAREST)

        # Optional explicit target size
        if self._target_size is not None:
            th, tw = self._target_size
            if dm.shape[0] != th or dm.shape[1] != tw:
                dm = cv2.resize(dm, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Clamp
        dm[~np.isfinite(dm)] = 0.0
        if self._min_depth > 0:
            dm[dm < self._min_depth] = 0.0
        if self._max_depth > 0:
            dm[dm > self._max_depth] = 0.0
        return dm

    # ================================================================
    #  Async worker
    # ================================================================

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="pi3-worker")
        self._worker.start()
        log.info("[Pi3] worker started")

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            # Drain sentinel: flush buffered frames, then signal done
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _DRAIN:
                self._flush_stream()
                item[1].set()  # done_event
                continue

            frame_idx, rgb = item
            self._push_frame_impl(frame_idx, rgb)

        # Final flush on shutdown
        self._flush_stream()
        log.info("[Pi3] worker exiting")

    # ================================================================
    #  Blocking / cache helpers
    # ================================================================

    def _wait_for_frame(self, frame_idx: int, timeout: float = 120.0) -> None:
        if frame_idx in self._depth_cache:
            return
        with self._events_lock:
            ev = self._events.get(frame_idx)
        if ev is not None:
            ev.wait(timeout=timeout)

    def _evict_old(self, cache: OrderedDict) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)


# ------------------------------------------------------------------
#  Lazy imports (Pi3 may not be installed at module-load time)
# ------------------------------------------------------------------

def _ensure_pi3_on_path() -> None:
    import sys
    root = Path(__file__).resolve().parent.parent / "Pi3_for_yolo_sgg"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _import_pi3x():
    try:
        from pi3.models.pi3x import Pi3X
        return Pi3X
    except ImportError:
        _ensure_pi3_on_path()
        from pi3.models.pi3x import Pi3X
        return Pi3X


def _import_vo_stream():
    try:
        from pi3.pipe.pi3x_vo_stream import Pi3XVOStream
        return Pi3XVOStream
    except ImportError:
        _ensure_pi3_on_path()
        from pi3.pipe.pi3x_vo_stream import Pi3XVOStream
        return Pi3XVOStream
