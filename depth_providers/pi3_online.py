"""Pi3 online / streaming depth provider."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

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

        self._base_K = self._validate_intrinsics(intrinsics)
        self._base_intrinsics_size = self._validate_intrinsics_image_size(
            intrinsics_image_size
        )

        self._model = None
        self._pipe = None

        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

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
        if self._pipe is not None:
            return
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
        self._model = None
        self._pipe = None
        self._buffer.clear()
        self._depth_cache.clear()
        self._pose_cache.clear()

        if torch is not None and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if rgb is None:
            return

        # Keep latest copy for this index (OrderedDict insertion order matters).
        if frame_idx in self._buffer:
            self._buffer.pop(frame_idx)
        self._buffer[frame_idx] = rgb

        # Opportunistic batch inference once enough frames accumulate.
        if len(self._buffer) >= self._window_size:
            self._process_window(force=False)

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._depth_cache:
            self._ensure_frame_processed(frame_idx)
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._pose_cache:
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
        frame_h: int,
        frame_w: int,
        device: str,
    ):
        if torch is None:
            return None
        if self._base_K is None:
            return None

        if self._base_intrinsics_size is None:
            base_h, base_w = frame_h, frame_w
        else:
            base_h, base_w = self._base_intrinsics_size

        sx = float(frame_w) / float(base_w)
        sy = float(frame_h) / float(base_h)

        K = self._base_K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
        return K_t.view(1, 1, 3, 3).repeat(1, n_frames, 1, 1)

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

        imgs_np = np.stack(rgbs_norm, axis=0)  # (T, H, W, 3)
        imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
        imgs = imgs.to(self._device)

        n_frames = len(indices)
        overlap_eff = min(self._overlap, max(0, n_frames - 1))
        chunk_eff = min(self._window_size, n_frames)
        intrinsics_seq = self._build_intrinsics_seq(
            n_frames=n_frames,
            frame_h=ref_h,
            frame_w=ref_w,
            device=self._device,
        )

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

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

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
