"""Pi3 online / streaming depth provider."""
from __future__ import annotations

from collections import OrderedDict
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

try:  # Optional dependency for environments that do not use Pi3 online.
    import torch
except Exception:  # pragma: no cover
    torch = None


def _normalize_torch_device(device: Optional[str]) -> str:
    """Normalize user/device-config strings into a torch-friendly device."""
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


class Pi3OnlineDepthProvider(OnlineDepthProvider):
    """Streaming depth+pose provider using Pi3XVO.

    Notes
    -----
    - Maintains a temporal RGB buffer.
    - Runs Pi3 inference on buffered frames with configurable
      ``window_size`` and ``overlap``.
    - Caches per-frame depth + pose for random access by frame index.
    - When ``async_inference=True`` (default), Pi3 inference runs in a
      background thread so YOLO + 3-D tracking can overlap with it.
      ``get_depth`` blocks only if the result is not cached yet (ideally
      it arrives before tracking needs it).
    """

    def __init__(
        self,
        model_name: str = "yyfz233/Pi3X",
        window_size: int = 13,
        overlap: int = 5,
        target_size: Optional[tuple[int, int]] = None,  # (H, W)
        device: Optional[str] = None,
        max_cache: int = 128,
        min_depth: float = 0.01,
        max_depth: float = 0.0,
        conf_threshold: float = 0.05,
        inject_condition: Optional[Sequence[str]] = None,
        intrinsics: Optional[np.ndarray] = None,          # 3x3 (base resolution)
        intrinsics_image_size: Optional[tuple[int, int]] = None,  # (H, W)
        use_original_size: bool = True,
        pixel_limit: int = 255000,
        patch_size: int = 14,
        async_inference: bool = True,
    ) -> None:
        self._model_name = str(model_name)

        self._window_size = max(1, int(window_size))
        self._overlap = max(0, int(overlap))
        if self._overlap >= self._window_size:
            self._overlap = self._window_size - 1

        self._target_size = target_size
        self._device = _normalize_torch_device(device)
        self._max_cache = int(max_cache)

        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._conf_threshold = float(conf_threshold)
        self._inject_condition = [str(x) for x in (inject_condition or [])]
        self._use_original_size = bool(use_original_size)
        self._pixel_limit = int(max(1, pixel_limit))
        self._patch_size = int(max(1, patch_size))

        self._base_K = self._validate_intrinsics(intrinsics)
        self._base_intrinsics_size = self._validate_intrinsics_image_size(
            intrinsics_image_size
        )

        self._model = None
        self._pipe = None

        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Async inference — background worker thread
        self._async_inference = bool(async_inference)
        self._input_queue: queue.Queue = queue.Queue()
        self._frame_events: dict[int, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

    @staticmethod
    def _validate_intrinsics(intrinsics: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if intrinsics is None:
            return None
        K = np.asarray(intrinsics, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be 3x3, got {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("intrinsics contains non-finite values")
        return K

    @staticmethod
    def _validate_intrinsics_image_size(
        size: Optional[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        if size is None:
            return None
        h, w = int(size[0]), int(size[1])
        if h <= 0 or w <= 0:
            raise ValueError(
                f"intrinsics_image_size must be positive, got {(h, w)}"
            )
        return (h, w)

    def warmup(self) -> None:
        log.info("[Pi3] warmup() called, async=%s", self._async_inference)
        self._load_model()
        if self._async_inference and self._worker_thread is None:
            self._worker_stop.clear()
            self._worker_thread = threading.Thread(
                target=self._async_worker, daemon=True, name="pi3-worker"
            )
            self._worker_thread.start()
            log.info("[Pi3] worker thread started (warmup)")

    def _load_model(self) -> None:
        """Load Pi3X model + VO pipe (idempotent)."""
        if self._pipe is not None:
            return
        log.info("[Pi3] loading model %s on %s ...", self._model_name, self._device)
        if torch is None:
            raise RuntimeError(
                "Pi3OnlineDepthProvider requires torch. Install torch to use pi3_online."
            )
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Pi3OnlineDepthProvider device is {self._device!r}, "
                "but CUDA is not available."
            )

        Pi3X, Pi3XVO = self._import_pi3_modules()

        model = Pi3X.from_pretrained(self._model_name)
        if hasattr(model, "to"):
            model = model.to(self._device)
        if hasattr(model, "eval"):
            model = model.eval()

        self._model = model
        self._pipe = Pi3XVO(model)
        log.info("[Pi3] model loaded successfully")

    @staticmethod
    def _import_pi3_modules():
        try:
            from pi3.models.pi3x import Pi3X
            from pi3.pipe.pi3x_vo import Pi3XVO
            return Pi3X, Pi3XVO
        except Exception:
            # Fallback: local bundled source tree.
            import sys

            root = Path(__file__).resolve().parent.parent / "Pi3_for_yolo_sgg"
            if root.exists():
                root_s = str(root)
                if root_s not in sys.path:
                    sys.path.insert(0, root_s)
            from pi3.models.pi3x import Pi3X
            from pi3.pipe.pi3x_vo import Pi3XVO
            return Pi3X, Pi3XVO

    def close(self) -> None:
        log.info("[Pi3] close() called")
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_stop.set()
            self._input_queue.put(None)  # unblock queue.get()
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
            log.info("[Pi3] worker thread joined")

        self._model = None
        self._pipe = None
        self._buffer.clear()
        self._depth_cache.clear()
        self._pose_cache.clear()
        with self._events_lock:
            self._frame_events.clear()

        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if rgb is None:
            return

        # Pre-create the Event so get_depth() can wait on it even if called
        # before the worker has had a chance to process this frame.
        with self._events_lock:
            self._frame_events.setdefault(frame_idx, threading.Event())

        if self._async_inference:
            # Lazy-start the worker thread on first feed_frame() call.
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_stop.clear()
                self._worker_thread = threading.Thread(
                    target=self._async_worker, daemon=True, name="pi3-worker"
                )
                self._worker_thread.start()
                log.info("[Pi3] worker thread started (lazy from feed_frame)")
            # Non-blocking: hand off to the background worker.
            self._input_queue.put((frame_idx, rgb))
            log.debug("[Pi3] feed_frame(%d) queued, qsize=%d", frame_idx, self._input_queue.qsize())
            return

        # Synchronous path (original behaviour).
        if frame_idx in self._buffer:
            self._buffer.pop(frame_idx)
        self._buffer[frame_idx] = rgb
        if len(self._buffer) >= self._window_size:
            self._process_window(force=False)

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._depth_cache:
            if self._async_inference:
                with self._events_lock:
                    ev = self._frame_events.get(frame_idx)
                if ev is not None:
                    log.debug("[Pi3] get_depth(%d) waiting for worker ...", frame_idx)
                    t0 = _time.perf_counter()
                    got = ev.wait(timeout=120.0)
                    wait_ms = (_time.perf_counter() - t0) * 1000
                    if got:
                        log.debug("[Pi3] get_depth(%d) ready after %.1f ms", frame_idx, wait_ms)
                    else:
                        log.warning("[Pi3] get_depth(%d) TIMED OUT after %.1f ms", frame_idx, wait_ms)
            else:
                self._ensure_frame_processed(frame_idx)
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._pose_cache:
            if self._async_inference:
                with self._events_lock:
                    ev = self._frame_events.get(frame_idx)
                if ev is not None:
                    log.debug("[Pi3] get_pose(%d) waiting for worker ...", frame_idx)
                    ev.wait(timeout=120.0)
            else:
                self._ensure_frame_processed(frame_idx)
        return self._pose_cache.get(frame_idx)

    def _ensure_frame_processed(self, _frame_idx: int) -> None:
        if not self._buffer:
            return
        # Force a pass over current buffered frames (supports true streaming).
        self._process_window(force=True)

    def _select_dtype(self):
        if torch is None:
            raise RuntimeError("Pi3 online inference requires torch")
        if not self._device.startswith("cuda"):
            return torch.float32

        cap_dev = None
        if self._device.startswith("cuda:"):
            suffix = self._device.split(":", 1)[1]
            if suffix.isdigit():
                cap_dev = int(suffix)
        cap = torch.cuda.get_device_capability(cap_dev)
        return torch.bfloat16 if cap[0] >= 8 else torch.float16

    def _build_intrinsics_seq(
        self,
        n_frames: int,
        raw_h: int,
        raw_w: int,
        model_h: int,
        model_w: int,
        apply_resize_scaling: bool,
        device: str,
    ):
        if torch is None:
            return None
        if self._base_K is None:
            return None

        if self._base_intrinsics_size is None:
            base_h, base_w = raw_h, raw_w
        else:
            base_h, base_w = self._base_intrinsics_size

        if apply_resize_scaling:
            sx = float(model_w) / float(base_w)
            sy = float(model_h) / float(base_h)
        else:
            sx = 1.0
            sy = 1.0

        K = self._base_K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
        return K_t.view(1, 1, 3, 3).repeat(1, n_frames, 1, 1)

    def _compute_model_size(self, raw_h: int, raw_w: int) -> tuple[int, int]:
        p = self._patch_size
        if self._use_original_size:
            h = int(math.ceil(raw_h / p) * p)
            w = int(math.ceil(raw_w / p) * p)
            return h, w

        if raw_h <= 0 or raw_w <= 0:
            return p, p

        scale = math.sqrt(self._pixel_limit / float(raw_h * raw_w))
        h_t = raw_h * scale
        w_t = raw_w * scale
        k = max(1, round(h_t / p))
        m = max(1, round(w_t / p))
        while (k * p) * (m * p) > self._pixel_limit and (k > 1 or m > 1):
            if (k / max(1, m)) > (h_t / max(1e-9, w_t)):
                k -= 1
            else:
                m -= 1
            k = max(1, k)
            m = max(1, m)
        return int(k * p), int(m * p)

    def _process_window(self, force: bool) -> None:
        if not self._buffer:
            return
        if not force and len(self._buffer) < self._window_size:
            return
        if torch is None:
            raise RuntimeError("Pi3 online inference requires torch")
        if self._pipe is None:
            self.warmup()
        if self._pipe is None:
            return

        indices = list(self._buffer.keys())
        rgbs = list(self._buffer.values())
        if not rgbs:
            return
        log.info(
            "[Pi3] _process_window: %d frames [%d..%d], force=%s",
            len(indices), indices[0], indices[-1], force,
        )

        # Ensure consistent frame shape/channels for model input.
        ref_h, ref_w = rgbs[0].shape[:2]
        rgbs_norm = []
        for rgb in rgbs:
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            if rgb.shape[:2] != (ref_h, ref_w):
                rgb = cv2.resize(
                    rgb, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR
                )
            rgbs_norm.append(np.ascontiguousarray(rgb))

        model_h, model_w = self._compute_model_size(ref_h, ref_w)

        preprocess_mode = "none"
        if self._use_original_size and (model_h != ref_h or model_w != ref_w):
            preprocess_mode = "pad"
        elif (not self._use_original_size) and (model_h != ref_h or model_w != ref_w):
            preprocess_mode = "resize"

        if preprocess_mode == "pad":
            prepped = []
            for rgb in rgbs_norm:
                canvas = np.zeros((model_h, model_w, 3), dtype=rgb.dtype)
                canvas[:ref_h, :ref_w] = rgb
                prepped.append(canvas)
        elif preprocess_mode == "resize":
            prepped = [
                cv2.resize(rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
                for rgb in rgbs_norm
            ]
        else:
            prepped = rgbs_norm

        imgs_np = np.stack(prepped, axis=0)  # (T, H, W, 3)
        imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
        imgs = imgs.to(self._device)

        n_frames = len(indices)
        overlap_eff = min(self._overlap, max(0, n_frames - 1))
        chunk_eff = min(self._window_size, n_frames)
        intrinsics_seq = self._build_intrinsics_seq(
            n_frames=n_frames,
            raw_h=ref_h,
            raw_w=ref_w,
            model_h=model_h,
            model_w=model_w,
            apply_resize_scaling=(preprocess_mode == "resize"),
            device=self._device,
        )

        log.info(
            "[Pi3] inference start: %d frames, chunk=%d, overlap=%d, "
            "img=%dx%d, model=%dx%d, preprocess=%s",
            n_frames, chunk_eff, overlap_eff, ref_h, ref_w, model_h, model_w, preprocess_mode,
        )
        _t_inf = _time.perf_counter()
        try:
            with torch.no_grad():
                results = self._pipe(
                    imgs=imgs[None],
                    chunk_size=chunk_eff,
                    overlap=overlap_eff,
                    conf_thre=self._conf_threshold,
                    inject_condition=self._inject_condition,
                    intrinsics=intrinsics_seq,
                    dtype=self._select_dtype(),
                )
        except Exception as exc:
            if not self._device.startswith("cuda"):
                raise RuntimeError(
                    "Pi3 online inference failed on a non-CUDA device. "
                    "Use a CUDA device (e.g. device='0' or 'cuda:0')."
                ) from exc
            raise

        _inf_ms = (_time.perf_counter() - _t_inf) * 1000
        log.info("[Pi3] inference done: %.1f ms for %d frames (%.1f ms/frame)",
                 _inf_ms, n_frames, _inf_ms / max(1, n_frames))

        points = results["points"][0]
        cam_poses = results["camera_poses"][0]

        # Pi3 outputs global points and camera-to-world poses.
        # Convert points back to each camera frame and read local Z as depth.
        cam_inv = torch.inverse(cam_poses)
        pts_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        local_pts = torch.einsum("tij,thwj->thwi", cam_inv, pts_h)[..., :3]
        depth_maps = local_pts[..., 2].detach().cpu().numpy()
        poses_np = cam_poses.detach().cpu().numpy().astype(np.float32)

        for i, idx in enumerate(indices):
            dm = depth_maps[i].astype(np.float32)
            if preprocess_mode == "pad":
                dm = dm[:ref_h, :ref_w]
            elif preprocess_mode == "resize":
                dm = cv2.resize(dm, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

            if self._target_size is not None:
                dm = cv2.resize(
                    dm,
                    (self._target_size[1], self._target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            dm[~np.isfinite(dm)] = 0.0
            dm[dm < self._min_depth] = 0.0
            if self._max_depth > 0.0:
                dm[dm > self._max_depth] = 0.0

            self._depth_cache[idx] = dm
            self._pose_cache[idx] = poses_np[i]

        # Signal per-frame waiters now that results are in cache.
        log.debug("[Pi3] signalling events for frames [%d..%d]", indices[0], indices[-1])
        with self._events_lock:
            for idx in indices:
                ev = self._frame_events.get(idx)
                if ev is not None:
                    ev.set()

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

        # While warming up (< window_size), keep full causal history so overlap
        # logic can kick in once the first full window is available.
        if force and len(indices) < self._window_size:
            return

        # Slide temporal window: keep only overlap tail for next chunk.
        keep_n = min(self._overlap, max(0, len(indices) - 1))
        new_buf: OrderedDict[int, np.ndarray] = OrderedDict()
        if keep_n > 0:
            for ki in indices[-keep_n:]:
                new_buf[ki] = self._buffer[ki]
        self._buffer = new_buf

    def _evict_old(self, cache: OrderedDict[int, np.ndarray]) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Async worker
    # ------------------------------------------------------------------

    def _async_worker(self) -> None:
        """Background thread: accumulates frames, runs Pi3 windows, sets Events.

        The main thread feeds (frame_idx, rgb) tuples via ``_input_queue``.
        The worker owns ``self._buffer`` exclusively in async mode — no
        locking is needed for it because the main thread never touches it.
        """
        log.info("[Pi3] async worker started (thread=%s)", threading.current_thread().name)
        while not self._worker_stop.is_set():
            try:
                item = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                # Flush any partial buffer that has been sitting idle
                # (e.g. at the end of a sequence with < window_size frames).
                if self._buffer:
                    log.info("[Pi3] worker idle timeout, flushing %d buffered frames", len(self._buffer))
                    self._process_window(force=True)
                continue

            if item is None:  # shutdown sentinel
                log.info("[Pi3] worker received shutdown sentinel")
                break

            frame_idx, rgb = item
            log.debug("[Pi3] worker got frame %d, buffer=%d, qsize=%d",
                      frame_idx, len(self._buffer) + 1, self._input_queue.qsize())

            # Lazy model load: allows feed_frame() to be called before warmup().
            if self._pipe is None:
                self._load_model()

            if frame_idx in self._buffer:
                self._buffer.pop(frame_idx)
            self._buffer[frame_idx] = rgb

            if len(self._buffer) >= self._window_size:
                self._process_window(force=False)

        # Drain: process remaining buffered frames before the thread exits.
        if self._buffer:
            log.info("[Pi3] worker draining %d remaining frames", len(self._buffer))
            self._process_window(force=True)
        log.info("[Pi3] async worker exiting")
