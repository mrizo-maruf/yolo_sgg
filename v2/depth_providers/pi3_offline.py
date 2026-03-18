"""
Pi3X offline depth provider.

Pre-computes depth for the entire sequence using the Pi3XVO pipeline
(chunked with overlap for temporal consistency), then serves depth
per-frame from the cache.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .base import DepthProvider


class Pi3OfflineDepthProvider(DepthProvider):
    """Run Pi3X once on the full sequence, cache results.

    Call ``warmup()`` to trigger inference.  After that,
    ``get_depth(idx)`` returns from the cache.
    """

    def __init__(
        self,
        rgb_dir: str,
        output_dir: Optional[str] = None,
        model_name: str = "yyfz233/Pi3X",
        chunk_size: int = 13,
        overlap: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self._rgb_dir = rgb_dir
        self._output_dir = output_dir or str(Path(rgb_dir).parent / "pi3_depth")
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._depth_cache: Dict[int, np.ndarray] = {}
        self._poses: Optional[List[np.ndarray]] = None
        self._warmed_up = False

    # -- DepthProvider interface ---------------------------------------------

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self._warmed_up:
            self.warmup()
        return self._depth_cache.get(frame_idx)

    def warmup(self) -> None:
        if self._warmed_up:
            return

        from pi3.models.pi3x import Pi3X
        from pi3.utils.basic import load_images_as_tensor
        from pi3.pipe.pi3x_vo import Pi3XVO

        model = Pi3X.from_pretrained(self._model_name).to(self._device).eval()
        pipe = Pi3XVO(model)

        imgs = load_images_as_tensor(self._rgb_dir, interval=1).to(self._device)
        dtype = (
            torch.bfloat16
            if self._device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            results = pipe(
                imgs=imgs[None],
                chunk_size=self._chunk_size,
                overlap=self._overlap,
                conf_thre=0.05,
                inject_condition=[],
                dtype=dtype,
            )

        points = results["points"][0]          # (N, H, W, 3)
        cam_poses = results["camera_poses"][0]  # (N, 4, 4)

        # global → local depth
        cam_inv = torch.inverse(cam_poses)
        pts_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        local = torch.einsum("nij,nhwj->nhwi", cam_inv, pts_h)[..., :3]
        depth_maps = local[..., 2].cpu().numpy()  # (N, H, W)

        # Upscale to match RGB resolution
        rgb_files = sorted(glob.glob(os.path.join(self._rgb_dir, "*.[jp][pn][g]")))
        if rgb_files:
            ref = cv2.imread(rgb_files[0])
            tgt_h, tgt_w = ref.shape[:2]
        else:
            tgt_h, tgt_w = depth_maps.shape[1], depth_maps.shape[2]

        for i in range(depth_maps.shape[0]):
            dm = cv2.resize(depth_maps[i], (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            dm[dm < 0.01] = 0.0
            self._depth_cache[i] = dm.astype(np.float32)

        # Store poses
        self._poses = [cam_poses[i].cpu().numpy() for i in range(cam_poses.shape[0])]

        # Free model
        del model, pipe, imgs, results
        if self._device == "cuda":
            torch.cuda.empty_cache()

        self._warmed_up = True

    def get_poses(self) -> Optional[List[np.ndarray]]:
        if not self._warmed_up:
            self.warmup()
        return self._poses

    def close(self) -> None:
        self._depth_cache.clear()
        self._poses = None
        self._warmed_up = False
