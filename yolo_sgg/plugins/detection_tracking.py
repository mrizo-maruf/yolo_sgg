from dataclasses import dataclass
from typing import Iterable, Tuple

import YOLOE.utils as yutils


@dataclass
class YoloTrackingAdapter:
    model_path: str
    conf: float
    iou: float
    tracker_cfg: str = "botsort.yaml"
    device: str = "0"

    def stream(self, rgb_dir_path: str, depth_paths: Iterable[str]) -> Iterable[Tuple[object, str, str]]:
        return yutils.track_objects_in_video_stream(
            rgb_dir_path,
            depth_paths,
            model_path=self.model_path,
            conf=self.conf,
            iou=self.iou,
            tracker_cfg=self.tracker_cfg,
            device=self.device,
        )
