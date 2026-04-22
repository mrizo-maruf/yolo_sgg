### Thud Synthetic Visualization
```
python thud_utils visualize_thud_synthetic_new.py --scene /path/to/Capture_1

python thud_utils/visualize_thud_synthetic_new.py \
    --scene /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1

python thud_utils/visualize_thud_synthetic_new.py \
    --scene /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 \
    --save-2d-dir /tmp/thud_vis \
    --show-3d --n-frames 10
```

### Rerun Dataset Reconstruction
```
python general_vis_utils/visualize_rerun_dataset_reconstruction.py \
    --dataset isaacsim \
    --scene_path /path/to/scene_1 \
    --traj_file traj.txt \
    --depth_folder depth \
    --spawn

python general_vis_utils/visualize_rerun_dataset_reconstruction.py \
    --dataset scanetpp \
    --scene_path /path/to/scannetpp_scene \
    --traj_file traj.txt \
    --depth_folder gt_depth \

    --spawn
```

The script auto-detects `rgb` or `images`, reconstructs RGB-colored point clouds
incrementally in world coordinates, logs the camera frustum and trajectory, and
keeps only a bounded live point budget in the Rerun 3D view.

```
python general_vis_utils/visualize_rerun_dataset_reconstruction.py --dataset scanetpp --scene_path /home/maribjonov_mr/Downloads/scanet_pi3/scene0a76e06478 --traj_file traj.txt --depth_folder gt_depth --spawn
```
