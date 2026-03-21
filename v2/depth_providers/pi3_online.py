"""Pi3 online / streaming depth provider."""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

from .base import OnlineDepthProvider

try:  # Optional dependency for environments that don't use Pi3 online.
    import torch
except Exception:  # pragma: no cover
    torch = None


class Pi3OnlineDepthProvider(OnlineDepthProvider):
    """Streaming depth and poses via Pi3XVO with a sliding window."""

    def __init__(
        self,
        model_name: str = "yyfz233/Pi3X",
        window_size: int = 13,
        overlap: int = 5,
        target_size: Optional[tuple[int, int]] = None,
        device: Optional[str] = None,
        max_cache: int = 128,
        min_depth: float = 0.01,
    ) -> None:
        self._model_name = model_name
        self._window_size = int(window_size)
        self._overlap = int(overlap)
        self._target_size = target_size  # (H, W)

        if device is not None:
            self._device = device
        else:
            has_cuda = bool(torch is not None and torch.cuda.is_available())
            self._device = "cuda" if has_cuda else "cpu"

        self._max_cache = int(max_cache)
        self._min_depth = float(min_depth)

        self._model = None
        self._pipe = None
        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def warmup(self) -> None:
        if self._model is not None:
            return
        if torch is None:
            raise RuntimeError(
                "Pi3OnlineDepthProvider requires torch. Install torch to use pi3_online."
            )

        from pi3.models.pi3x import Pi3X
        from pi3.pipe.pi3x_vo import Pi3XVO

        self._model = Pi3X.from_pretrained(self._model_name).to(self._device).eval()
        self._pipe = Pi3XVO(self._model)

    def close(self) -> None:
        self._model = None
        self._pipe = None
        self._buffer.clear()
        self._depth_cache.clear()
        self._pose_cache.clear()
        if torch is not None and self._device == "cuda":
            torch.cuda.empty_cache()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        self._buffer[frame_idx] = rgb
        if len(self._buffer) >= self._window_size:
            self._process_window()

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._depth_cache and len(self._buffer) > 0:
            self._process_window()
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx not in self._pose_cache and len(self._buffer) > 0:
            self._process_window()
        return self._pose_cache.get(frame_idx)

    def _select_dtype(self):
        if torch is None:
            raise RuntimeError("Pi3 online inference requires torch")
        if self._device != "cuda":
            return torch.float32
        cap = torch.cuda.get_device_capability()
        return torch.bfloat16 if cap[0] >= 8 else torch.float16

    def _process_window(self) -> None:
        if not self._buffer:
            return
        if torch is None:
            raise RuntimeError("Pi3 online inference requires torch")
        if self._pipe is None:
            self.warmup()
        if self._pipe is None:
            return

        indices = list(self._buffer.keys())
        rgbs = list(self._buffer.values())

        tensors = []
        for rgb in rgbs:
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        imgs = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            results = self._pipe(
                imgs=imgs[None],
                chunk_size=min(self._window_size, len(rgbs)),
                overlap=min(self._overlap, max(0, len(rgbs) - 1)),
                conf_thre=0.05,
                inject_condition=[],
                dtype=self._select_dtype(),
            )

        points = results["points"][0]
        cam_poses = results["camera_poses"][0]

        cam_inv = torch.inverse(cam_poses)
        pts_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        local_pts = torch.einsum("nij,nhwj->nhwi", cam_inv, pts_h)[..., :3]
        depths = local_pts[..., 2].cpu().numpy()

        poses_np = cam_poses.detach().cpu().numpy().astype(np.float32)

        for i, idx in enumerate(indices):
            dm = depths[i]
            if self._target_size is not None:
                dm = cv2.resize(
                    dm,
                    (self._target_size[1], self._target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            dm = dm.astype(np.float32)
            dm[dm < self._min_depth] = 0.0

            self._depth_cache[idx] = dm
            self._pose_cache[idx] = poses_np[i]

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

        keep_count = min(self._overlap, len(indices))
        keep_indices = indices[-keep_count:] if keep_count > 0 else []
        new_buf: OrderedDict[int, np.ndarray] = OrderedDict()
        for ki in keep_indices:
            new_buf[ki] = self._buffer[ki]
        self._buffer = new_buf

    def _evict_old(self, cache: OrderedDict[int, np.ndarray]) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)
