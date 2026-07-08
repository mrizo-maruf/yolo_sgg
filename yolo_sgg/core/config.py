from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    rgb_dir: str
    depth_dir: str
    traj_path: str
    yolo_model: str
    conf: float = 0.25
    iou: float = 0.5
    kernel_size: int = 17
    alpha: float = 0.7
    max_points_per_obj: int = 2000
    max_accumulated_points: int = 10000
    fast_mask: bool = True
    o3_nb_neighbors: int = 50
    o3std_ratio: float = 0.1
    tracker_cfg: str = "botsort.yaml"
    device: str = "0"
    edge_predictor: str = "sceneverse"
    vis_graph: bool = False
    show_pcds: bool = False
    print_resource_usage: bool = False
    print_tracking_info: bool = False

    tracking_overlap_threshold: float = 0.3
    tracking_distance_threshold: float = 0.5
    tracking_inactive_limit: int = 0
    tracking_volume_ratio_threshold: float = 0.1
    reprojection_visibility_threshold: float = 0.2

    scene_path: Optional[str] = None


DEFAULT_CONFIG = ExperimentConfig(
    rgb_dir="/path/to/rgb",
    depth_dir="/path/to/depth",
    traj_path="/path/to/traj.txt",
    yolo_model="/path/to/yoloe-11l-seg-pf.pt",
)


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    if config_path is None:
        return DEFAULT_CONFIG

    cfg_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return ExperimentConfig(**cfg_dict)


def save_default_config(output_path: str = "configs/default_run.yaml") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.structured(DEFAULT_CONFIG), f=str(path))
    return path
