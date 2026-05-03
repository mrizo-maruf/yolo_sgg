### Running benchmarking on isaac sim dataset

On all isaac sim dataset
```
python benchmark/benchmark_tracking_isaac.py ~/rizo/3D_SSGG_IsaacSim --multi
```

On spefici scene
```
python benchmark/benchmark_tracking_isaac.py ~/rizo/3D_SSGG_IsaacSim_small/scene_1
```

Other params
```
p.add_argument("--vis", action="store_true", help="Enable debug visualisation.")
p.add_argument("--vis-interval", type=int, default=10, help="Visualise every N frames.")
p.add_argument("--vis-save", type=str, default=None, help="Dir to save vis PNGs.")
p.add_argument("--no-show", action="store_true", help="Don't pop up windows (only save).")
```


### Running benchmark on thud dataset
On all thud synthetic static:
```
python benchmark/benchmark_tracking_thud.py /home/yehia/Downloads/RGB-D --multi
```

### New benchmarking
```python -m benchmark.benchmark --dataset isaacsim --scene_path ~/rizo/3D_SSGG_IsaacSim_small/scene_1 --depth_provider gt --vis --vis_interval 1
```
### New benchmarking on isaacsim with pi3_offile depth
```
python -m benchmark.benchmark --dataset isaacsim --scene_path /home/yehia/rizo/IsaacSim_bench_pi3/cabinet_simple --depth_provider pi3_offline --vis --vis_interval 3

python -m benchmark.benchmark --dataset isaacsim --scene_path /home/yehia/rizo/IsaacSim_bench_pi3 --multi --depth_provider gt

python -m benchmark.benchmark --dataset isaacsim --scene_path /home/yehia/rizo/IsaacSim_bench_pi3 --multi --depth_provider pi3_offline
```