running with rerun visualizaiton

```
python run.py --dataset isaacsim --rerun --edge-predictor bs
```

### running in new_run.py
```
python new_run.py --dataset isaacsim --scene_path /home/yehia/rizo/IsaacSim_Dataset/scene_7 --rerun --vis_edge
```

### running isaacsim with pi3_offline depth
```
python new_run.py --dataset isaacsim --scene_path /home/yehia/rizo/IsaacSim_bench_pi3/cabinet_simple --depth_provider pi3_offline
```