"""
Pi3X online / streaming depth provider.

Feeds frames one at a time (or in small windows) and returns depth
as soon as available.  Uses the chunk/overlap strategy from Pi3XVO
for temporal consistency.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import torch

from .base import OnlineDepthProvider


class Pi3OnlineDepthProvider(OnlineDepthProvider):
    """Streaming depth via Pi3XVO with a sliding window.

    Usage::

        provider = Pi3OnlineDepthProvider(...)
        provider.warmup()       # loads model
        for idx, rgb in frames:
            provider.feed_frame(idx, rgb)
            depth = provider.get_depth(idx)
    """

    def __init__(
        self,
        model_name: str = "yyfz233/Pi3X",
        window_size: int = 13,
        overlap: int = 5,
        target_size: Optional[tuple] = None,
        device: Optional[str] = None,
        max_cache: int = 64,
    ) -> None:
        self._model_name = model_name
        self._window_size = window_size
        self._overlap = overlap
        self._target_size = target_size  # (H, W) for upscaling
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_cache = max_cache

        self._model = None
        self._pipe = None
        self._buffer: OrderedDict[int, np.ndarray] = OrderedDict()  # idx → rgb
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._last_processed_idx = -1

    # -- lifecycle ----------------------------------------------------------

    def warmup(self) -> None:
        if self._model is not None:
            return
        from pi3.models.pi3x import Pi3X
        from pi3.pipe.pi3x_vo import Pi3XVO

        self._model = Pi3X.from_pretrained(self._model_name).to(self._device).eval()
        self._pipe = Pi3XVO(self._model)

    def close(self) -> None:
        del self._model, self._pipe
        self._model = self._pipe = None
        self._buffer.clear()
        self._depth_cache.clear()
        if self._device == "cuda":
            torch.cuda.empty_cache()

    # -- OnlineDepthProvider interface --------------------------------------

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        self._buffer[frame_idx] = rgb
        # When buffer reaches window size, process a chunk
        if len(self._buffer) >= self._window_size:
            self._process_window()

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        # Flush remaining frames if not yet processed
        if frame_idx not in self._depth_cache and len(self._buffer) > 0:
            self._process_window()
        return self._depth_cache.get(frame_idx)

    # -- internal -----------------------------------------------------------

    def _process_window(self) -> None:
        if self._pipe is None:
            self.warmup()

        indices = list(self._buffer.keys())
        rgbs = list(self._buffer.values())
        if not rgbs:
            return

        # Build tensor (N, 3, H, W)
        tensors = []
        for rgb in rgbs:
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        imgs = torch.stack(tensors).to(self._device)

        dtype = (
            torch.bfloat16
            if self._device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            results = self._pipe(
                imgs=imgs[None],
                chunk_size=min(self._window_size, len(rgbs)),
                overlap=min(self._overlap, max(0, len(rgbs) - 1)),
                conf_thre=0.05,
                inject_condition=[],
                dtype=dtype,
            )

        points = results["points"][0]
        cam_poses = results["camera_poses"][0]
        cam_inv = torch.inverse(cam_poses)
        pts_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        local = torch.einsum("nij,nhwj->nhwi", cam_inv, pts_h)[..., :3]
        depths = local[..., 2].cpu().numpy()

        for i, idx in enumerate(indices):
            dm = depths[i]
            if self._target_size is not None:
                dm = cv2.resize(dm, (self._target_size[1], self._target_size[0]),
                                interpolation=cv2.INTER_LINEAR)
            dm[dm < 0.01] = 0.0
            self._depth_cache[idx] = dm.astype(np.float32)

            # Evict old cache entries
            while len(self._depth_cache) > self._max_cache:
                self._depth_cache.popitem(last=False)

        # Keep overlap frames in buffer for next window
        keep_count = min(self._overlap, len(indices))
        keep_indices = indices[-keep_count:] if keep_count > 0 else []
        new_buf: OrderedDict[int, np.ndarray] = OrderedDict()
        for ki in keep_indices:
            new_buf[ki] = self._buffer[ki]
        self._buffer = new_buf
