"""Pi3 online / streaming depth provider."""
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

try:  # Optional dependency for environments that do not use Pi3 online.
    import torch
except Exception:  # pragma: no cover
    torch = None


def _load_sim3_matrix(path_str: str, require: bool) -> np.ndarray:
    """Load a Sim(3) 4x4 matrix from a JSON file.

    The JSON may contain either ``sim3_matrix_4x4`` (flat or nested 4x4) or
    separate ``scale``, ``rotation`` (3x3), ``translation`` (3,) keys.
    """
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
        overlap: int = 10,
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
        transform_path: Optional[str] = None,
        require_transform: bool = False,
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

        # Sim(3) alignment: Pi3 world → GT world
        self._sim3 = _load_sim3_matrix(transform_path, require_transform) if transform_path else np.eye(4, dtype=np.float32)
        if not np.allclose(self._sim3, np.eye(4)):
            log.info("[Pi3] Sim(3) alignment transform loaded from %s", transform_path)

        self._model = None
        self._pipe = None

        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Inter-window Sim3 stitching — overlap data from previous window
        # Stored in "global Pi3 frame" (= first window's frame).
        self._prev_overlap_points: Optional[np.ndarray] = None   # (overlap, H, W, 3)
        self._prev_overlap_conf: Optional[np.ndarray] = None     # (overlap, H, W)
        self._prev_overlap_poses: Optional[np.ndarray] = None    # (overlap, 4, 4)
        # Cumulative Sim3 from first window's frame ("global Pi3 frame").
        self._cumulative_sim3 = np.eye(4, dtype=np.float32)

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

    def get_sim3_matrix(self) -> np.ndarray:
        """Return a copy of the Sim(3) alignment matrix (identity if none)."""
        return self._sim3.copy()

    def set_sim3_transform(self, sim3: np.ndarray) -> None:
        """Set the Sim(3) Pi3-world → GT-world alignment matrix."""
        sim3 = np.asarray(sim3, dtype=np.float32)
        if sim3.shape != (4, 4):
            raise ValueError(f"sim3 must be 4x4, got {sim3.shape}")
        self._sim3 = sim3
        log.info("[Pi3] Sim(3) transform set (externally)")

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
        self._prev_overlap_points = None
        self._prev_overlap_conf = None
        self._prev_overlap_poses = None
        self._cumulative_sim3 = np.eye(4, dtype=np.float32)
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

    @staticmethod
    def _depth_edge(
        depth: "torch.Tensor",
        rtol: float = 0.03,
        kernel_size: int = 3,
    ) -> "torch.Tensor":
        """Detect depth discontinuity edges (mirrors Pi3's ``depth_edge``).

        Parameters
        ----------
        depth : Tensor (..., H, W) — linear depth values.
        rtol  : relative tolerance for neighbor depth differences.

        Returns
        -------
        BoolTensor of the same shape — True at edge pixels.
        """
        import torch.nn.functional as F
        shape = depth.shape
        d = depth.reshape(-1, 1, *shape[-2:])
        pad = kernel_size // 2
        diff = (
            F.max_pool2d(d, kernel_size, stride=1, padding=pad)
            + F.max_pool2d(-d, kernel_size, stride=1, padding=pad)
        )
        edge = ((diff / d).nan_to_num_() > rtol)
        return edge.reshape(shape)

    @staticmethod
    def _compute_window_sim3(
        src_pts: "torch.Tensor",
        tgt_pts: "torch.Tensor",
        src_conf: "torch.Tensor | None" = None,
        tgt_conf: "torch.Tensor | None" = None,
        conf_thre: float = 0.05,
    ) -> np.ndarray:
        """Umeyama Sim(3) aligning *src_pts* → *tgt_pts* (overlap frames).

        Parameters
        ----------
        src_pts, tgt_pts : Tensor (T, H, W, 3)
            3-D point maps for the overlap frames in two different frames.
        src_conf, tgt_conf : Tensor (T, H, W) or None
            Per-pixel confidence maps.  Pixels below *conf_thre* are excluded
            from the Sim(3) estimation.
        conf_thre : float
            Confidence threshold (pixels with conf <= conf_thre are masked out).

        Returns
        -------
        np.ndarray (4, 4) float32  — similarity transform.
        """
        device = src_pts.device
        src = src_pts.reshape(-1, 3)
        tgt = tgt_pts.reshape(-1, 3)

        # Valid mask: finite + non-degenerate.
        src_ok = torch.isfinite(src).all(-1) & (src.abs().sum(-1) > 1e-6)
        tgt_ok = torch.isfinite(tgt).all(-1) & (tgt.abs().sum(-1) > 1e-6)
        mask = src_ok & tgt_ok

        # Confidence filtering (mirrors Pi3XVO's _compute_sim3_umeyama_masked).
        if src_conf is not None:
            mask = mask & (src_conf.reshape(-1) > conf_thre)
        if tgt_conf is not None:
            mask = mask & (tgt_conf.reshape(-1) > conf_thre)

        src_m = src[mask]
        tgt_m = tgt[mask]
        n = src_m.shape[0]

        if n < 10:
            # Fallback: if confident points are too few, relax to top-10 %
            # of whichever confidence map is available.
            if src_conf is not None or tgt_conf is not None:
                geom_mask = src_ok & tgt_ok
                if geom_mask.sum() >= 10:
                    combined_conf = torch.ones(geom_mask.shape[0], device=device)
                    if src_conf is not None:
                        combined_conf = combined_conf * src_conf.reshape(-1)
                    if tgt_conf is not None:
                        combined_conf = combined_conf * tgt_conf.reshape(-1)
                    combined_conf[~geom_mask] = -1
                    k = max(10, int(geom_mask.sum().item() * 0.1))
                    topk_vals, topk_idx = torch.topk(combined_conf, k)
                    src_m = src[topk_idx]
                    tgt_m = tgt[topk_idx]
                    n = src_m.shape[0]
            if n < 10:
                return np.eye(4, dtype=np.float32)

        # Centroids.
        src_mean = src_m.mean(0)
        tgt_mean = tgt_m.mean(0)
        src_c = src_m - src_mean
        tgt_c = tgt_m - tgt_mean

        # SVD of cross-covariance.
        H = src_c.T @ tgt_c
        U, S, Vh = torch.linalg.svd(H)

        # Reflection correction.
        d = torch.det(Vh.T @ U.T)
        diag = torch.ones(3, device=device, dtype=src.dtype)
        diag[2] = torch.sign(d)
        R = Vh.T @ torch.diag(diag) @ U.T

        # Umeyama scale.
        corrected_S = S.clone()
        corrected_S[2] *= diag[2]
        src_total_var = (src_c ** 2).sum()
        scale = corrected_S.sum() / (src_total_var + 1e-8)

        t = tgt_mean - scale * (R @ src_mean)

        sim3 = torch.eye(4, device=device, dtype=src.dtype)
        sim3[:3, :3] = scale * R
        sim3[:3, 3] = t
        return sim3.detach().cpu().numpy().astype(np.float32)

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

        points = results["points"][0]       # (T, H, W, 3)
        cam_poses = results["camera_poses"][0]  # (T, 4, 4)
        # Confidence map — Pi3XVO returns (B, T, H, W) or similar.
        raw_conf = results.get("conf")
        if raw_conf is not None:
            conf = raw_conf[0]  # (T, H, W)
            if conf.dim() == 4:
                conf = conf[..., 0]
        else:
            conf = None

        # ---- Depth edge filtering (mirrors Pi3XVO internal logic) ----
        cam_inv = torch.inverse(cam_poses)
        pts_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        local_pts = torch.einsum("tij,thwj->thwi", cam_inv, pts_h)[..., :3]
        local_depth = local_pts[..., 2]  # (T, H, W)

        if conf is not None:
            edge = self._depth_edge(local_depth, rtol=0.03)
            conf = conf.clone()
            conf[edge] = 0

        # ---- Inter-window Sim3 alignment (Umeyama on overlap) ----
        window_sim3_np = np.eye(4, dtype=np.float32)
        n_overlap_used = min(self._overlap, n_frames)
        if self._prev_overlap_points is not None and n_overlap_used > 0:
            prev_pts = self._prev_overlap_points  # (overlap, H, W, 3)
            n_match = min(prev_pts.shape[0], n_overlap_used)
            src_pts = points[:n_match]
            tgt_pts_np = prev_pts[-n_match:]
            tgt_pts = torch.from_numpy(tgt_pts_np).to(
                device=points.device, dtype=points.dtype
            )
            # Confidence masks for overlap region
            src_conf = conf[:n_match] if conf is not None else None
            tgt_conf = None
            if self._prev_overlap_conf is not None:
                tgt_conf_np = self._prev_overlap_conf[-n_match:]
                tgt_conf = torch.from_numpy(tgt_conf_np).to(
                    device=points.device, dtype=points.dtype
                )
            window_sim3_np = self._compute_window_sim3(
                src_pts, tgt_pts,
                src_conf=src_conf, tgt_conf=tgt_conf,
                conf_thre=self._conf_threshold,
            )
            if not np.allclose(window_sim3_np, np.eye(4), atol=1e-5):
                log.info(
                    "[Pi3] inter-window Sim3 applied (scale=%.4f)",
                    np.linalg.det(window_sim3_np[:3, :3]) ** (1.0 / 3.0),
                )

        # Transform points to global Pi3 frame for overlap storage.
        if not np.allclose(window_sim3_np, np.eye(4), atol=1e-6):
            ws_t = torch.from_numpy(window_sim3_np).to(
                device=points.device, dtype=points.dtype
            )
            pts_flat = points.reshape(-1, 3)
            pts_global = (pts_flat @ ws_t[:3, :3].T + ws_t[:3, 3]).reshape(points.shape)
        else:
            pts_global = points

        # Save last-overlap frames' global points + conf for next window.
        if self._overlap > 0:
            n_save = min(self._overlap, n_frames)
            self._prev_overlap_points = (
                pts_global[-n_save:].detach().cpu().numpy().astype(np.float32)
            )
            if conf is not None:
                self._prev_overlap_conf = (
                    conf[-n_save:].detach().cpu().numpy().astype(np.float32)
                )
            self._prev_overlap_poses = (
                cam_poses[-n_save:].detach().cpu().numpy().astype(np.float32)
            )

        # Convert points back to each camera frame, read local Z as depth.
        depth_maps = local_depth.detach().cpu().numpy()
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
            # Apply inter-window Sim3 alignment, then external Pi3→GT transform.
            aligned_pose = (window_sim3_np @ poses_np[i]).astype(np.float32)
            self._pose_cache[idx] = (self._sim3 @ aligned_pose).astype(np.float32)

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
