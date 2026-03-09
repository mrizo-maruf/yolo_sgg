from utils import process_depth_model
import sys
from pathlib import Path
from omegaconf import OmegaConf


DEFAULT_SCENES = [
    # Isaac Sim scenes:
    '../files/3D_SSGG_IsaacSim/cabinet_complex',
    '../files/3D_SSGG_IsaacSim/cabinet_simple',
    '../files/3D_SSGG_IsaacSim/nk_scene_complex',
    '../files/3D_SSGG_IsaacSim/scene_1',
    '../files/3D_SSGG_IsaacSim/scene_2',
    '../files/3D_SSGG_IsaacSim/scene_3',
    '../files/3D_SSGG_IsaacSim/scene_4',
    '../files/3D_SSGG_IsaacSim/scene_5',
    '../files/3D_SSGG_IsaacSim/scene_6',
    '../files/3D_SSGG_IsaacSim/scene_7',
    '../files/3D_SSGG_IsaacSim/simple_scene'
]

for scene_path in DEFAULT_SCENES:
    # Prepare paths
    rgb_dir = str(Path(scene_path) / 'rgb')
    depth_dir = str(Path(scene_path) / 'depth')
    
    # Process depth with Pi3X model if configured
    temp_cfg = OmegaConf.create({
        'rgb_dir': rgb_dir,
        'depth_dir': depth_dir,
        'traj_path': str(Path(scene_path) / "camera_poses.txt"),
        'depth_model': 'yyfz233/Pi3X'
    })
    temp_cfg = process_depth_model(temp_cfg)


