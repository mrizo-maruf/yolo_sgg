"""DepthAnything V3 streaming provider with optional pose support."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import cv2
import numpy as np

from .base import OnlineDepthProvider


class DAv3StreamingDepthProvider(OnlineDepthProvider):
    """Single-frame streaming depth provider for DepthAnything V3.

    Notes
    -----
    - DepthAnything models typically predict depth only.
    - If your runtime can also estimate poses, return ``(depth, pose)`` or
      ``{"depth": depth, "pose": pose}`` from ``predict_fn``.
    """

    def __init__(
        self,
        predict_fn: Optional[Callable[..., object]] = None,
        model_name: str = "depth-anything-v3",
        device: str = "cuda",
        target_size: Optional[tuple[int, int]] = None,
        min_depth: float = 0.01,
        max_cache: int = 256,
    ) -> None:
        self._predict_fn = predict_fn
        self._model_name = model_name
        self._device = device
        self._target_size = target_size
        self._min_depth = float(min_depth)
        self._max_cache = int(max_cache)

        self._model = None
        self._depth_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._pose_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def warmup(self) -> None:
        if self._predict_fn is not None:
            return

        try:
            from depth_anything_v3 import DepthAnythingV3  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "DepthAnythingV3 is not installed. Pass a custom predict_fn "
                "or install a DepthAnythingV3 package."
            ) from exc

        if hasattr(DepthAnythingV3, "from_pretrained"):
            model = DepthAnythingV3.from_pretrained(self._model_name)
        else:
            model = DepthAnythingV3(self._model_name)

        if hasattr(model, "to"):
            model = model.to(self._device)
        if hasattr(model, "eval"):
            model = model.eval()

        self._model = model
        self._predict_fn = self._default_predict

    def close(self) -> None:
        self._model = None
        self._predict_fn = None
        self._depth_cache.clear()
        self._pose_cache.clear()

    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        if self._predict_fn is None:
            self.warmup()
        if self._predict_fn is None:
            return

        pred = self._call_predict(self._predict_fn, rgb, frame_idx)
        depth, pose = self._normalize_prediction(pred)

        if depth is None:
            return
        if self._target_size is not None:
            depth = cv2.resize(
                depth,
                (self._target_size[1], self._target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        depth = depth.astype(np.float32)
        depth[depth < self._min_depth] = 0.0

        self._depth_cache[frame_idx] = depth
        if pose is not None:
            self._pose_cache[frame_idx] = pose.astype(np.float32)

        self._evict_old(self._depth_cache)
        self._evict_old(self._pose_cache)

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._depth_cache.get(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._pose_cache.get(frame_idx)

    def _default_predict(self, rgb: np.ndarray, _frame_idx: int) -> object:
        assert self._model is not None

        if hasattr(self._model, "infer_image"):
            return self._model.infer_image(rgb)
        if callable(self._model):
            return self._model(rgb)

        raise RuntimeError(
            "Unsupported DepthAnythingV3 model API. Provide custom predict_fn."
        )

    @staticmethod
    def _call_predict(fn: Callable[..., object], rgb: np.ndarray, frame_idx: int) -> object:
        try:
            return fn(rgb, frame_idx)
        except TypeError:
            return fn(rgb)

    @staticmethod
    def _to_numpy(x) -> Optional[np.ndarray]:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _normalize_prediction(
        self,
        pred: object,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        depth = None
        pose = None

        if isinstance(pred, np.ndarray) or hasattr(pred, "shape"):
            depth = self._to_numpy(pred)

        elif isinstance(pred, tuple) and len(pred) == 2:
            depth = self._to_numpy(pred[0])
            pose = self._to_numpy(pred[1])

        elif isinstance(pred, dict):
            for key in ("depth", "depth_m", "depth_map", "pred_depth"):
                if key in pred:
                    depth = self._to_numpy(pred[key])
                    break
            for key in ("pose", "camera_pose", "camera_poses", "pred_pose"):
                if key in pred:
                    pose = self._to_numpy(pred[key])
                    break

        if depth is not None:
            depth = np.squeeze(depth)
            if depth.ndim != 2:
                raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")

        if pose is not None:
            pose = np.squeeze(pose)
            if pose.shape == (16,):
                pose = pose.reshape(4, 4)
            if pose.shape != (4, 4):
                raise ValueError(f"Expected 4x4 pose, got shape {pose.shape}")

        return depth, pose

    def _evict_old(self, cache: OrderedDict[int, np.ndarray]) -> None:
        while len(cache) > self._max_cache:
            cache.popitem(last=False)
