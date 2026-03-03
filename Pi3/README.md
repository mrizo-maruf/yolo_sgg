# Install requirements

```bash
# in any folder run
conda activate yolo_ssg
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -e .
```

# How its works

File `Pi3/utils.py` provides `process_depth_model` function:

```python
def process_depth_model(cfg):
    """
    Process depth and trajectory using specified depth model.
    
    Args:
        cfg: Config object with fields:
            - rgb_dir: Path to RGB images directory
            - depth_dir: Path to original depth directory
            - traj_path: Path to original trajectory file
            - depth_model: Depth model name (e.g., 'yyfz233/Pi3X') or None
    
    Returns:
        cfg: Updated config with new depth_dir and traj_path if model was used
    """
```

# How to run

1) from `yolo_ssg.py`

At the end of the file you can set up `depth_model` config parameter:
```python
if __name__ == "__main__":
    cfg = OmegaConf.create({
        # ...
        'depth_model': 'yyfz233/Pi3X', # set None if you want to use original depth
        # ...
    })
    main(cfg)

```

2) from `benchmark_thud.py`

The same, at the end of the file you can set up `depth_model` config parameter.


