from utils import process_depth_model
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_PI3X_CKPT = REPO_ROOT / "chkp" / "Pi3X" / "model.safetensors"
USE_LOCAL_CKPT_ONLY = True

if USE_LOCAL_CKPT_ONLY and not LOCAL_PI3X_CKPT.is_file():
    raise FileNotFoundError(
        f"Local Pi3X checkpoint not found at: {LOCAL_PI3X_CKPT}\n"
        "Either place model.safetensors there or set USE_LOCAL_CKPT_ONLY=False."
    )

DEFAULT_SCENES = [
    # Isaac Sim scenes:
    # '/home/maribjonov_mr/IsaacSim_bench/cabinet_complex',
    '/home/maribjonov_mr/IsaacSim_bench/scene_2',
    # '/home/maribjonov_mr/IsaacSim_bench/cabinet_simple',
    # '/home/maribjonov_mr/IsaacSim_bench/nk_scene_complex',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_2',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_3',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_4',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_5',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_6',
    # '/home/maribjonov_mr/IsaacSim_bench/scene_7',
    # '/home/maribjonov_mr/IsaacSim_bench/simple_scene'
]


def main():
    parser = argparse.ArgumentParser(description="Run Pi3X depth + pose processing on default scenes.")
    parser.add_argument(
        "--original_img",
        action="store_true",
        help="Use original image resolution (with minimal padding to meet model constraints).",
    )
    args = parser.parse_args()

    for scene_path in DEFAULT_SCENES:
        # Prepare paths
        rgb_dir = str(Path(scene_path) / 'rgb')
        depth_dir = str(Path(scene_path) / 'depth')

        # Process depth with Pi3X model if configured
        temp_cfg = OmegaConf.create({
            'rgb_dir': rgb_dir,
            'depth_dir': depth_dir,
            'traj_path': str(Path(scene_path) / "camera_poses.txt"),
            'depth_model': 'yyfz233/Pi3X',
            'ckpt': str(LOCAL_PI3X_CKPT) if LOCAL_PI3X_CKPT.is_file() else None,
            'original_img': args.original_img,
            'pi3_png_depth_scale': 0.001,
            'fx': 800.0,
            'fy': 800.0,
            'cx': 640.0,
            'cy': 360.0
        })
        process_depth_model(temp_cfg)


if __name__ == "__main__":
    main()
