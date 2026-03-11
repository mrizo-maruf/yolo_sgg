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