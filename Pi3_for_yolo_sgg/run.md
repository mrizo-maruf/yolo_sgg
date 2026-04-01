### running with intrinsics
```
python run.py

# original-size mode
python run.py --original_img

```

```
python visualize_gt_vs_pi3x.py --scene_dir /home/maribjonov_mr/IsaacSim_bench/scene_2 --accumulate 29
```

### aligning pi3 to gt pcds
```
python align_pi3_gt.py \
  --scene_dir /home/maribjonov_mr/IsaacSim_bench/scene_2 \
  --save_transform

# saved automatically in scene folder:
#   pi3_to_world_transform.json
#   pi3_to_world_transform.npz

  # if memory is issue
  python align_pi3_gt.py --scene_dir ... --align_subsample 12 --max_pairs 150000 --frame_pair_cap 3000

  # saving align transform
   python align_pi3_gt.py --scene_dir /home/yehia/rizo/IsaacSim_bench_pi3/scene_3 --max_pairs 250000 --save_transform

   # visualizing only pi3 pcds
   python align_pi3_gt.py --scene_dir /home/yehia/rizo/IsaacSim_bench_pi3/scene_3 --pi3_only
```

#### aligning matrices
**scene 2 16-12**:
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1704333623
Rotation R:
[[-0.83297632  0.55292118 -0.02070306]
 [ 0.36008542  0.51330042 -0.77901294]
 [-0.42010587 -0.65635421 -0.62666595]]
Translation t:
[4.84204982 4.33227449 1.34055737]
Alignment RMSE (m): 0.285196
Sim(3) matrix:
[[-0.97494328  0.6471574  -0.02423155  4.84204982]
 [ 0.42145599  0.60078394 -0.91178274  4.33227449]
 [-0.49170592 -0.76821886 -0.73347074  1.34055737]
 [ 0.          0.          0.          1.        ]]
```

**scene 2 full**:
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1598583531
Rotation R:
[[-0.87257264  0.48159678  0.0817406 ]
 [ 0.248681    0.58197978 -0.77424627]
 [-0.42044589 -0.65525878 -0.62758361]]
Translation t:
[6.26369957 4.58098195 1.34531721]
Alignment RMSE (m): 0.283764
Sim(3) matrix:
[[-1.01206066  0.55858405  0.09480752  6.26369957]
 [ 0.28843473  0.67501411 -0.89801601  4.58098195]
 [-0.48765768 -0.76000737 -0.72790809  1.34531721]
 [ 0.          0.          0.          1.        ]]
```
**scene cabinet_simple 60 frames, 16-12**
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.1365642071
Rotation R:
[[-0.9094795   0.28849628  0.29936087]
 [-0.01619221  0.69492627 -0.71889867]
 [-0.41543333 -0.65867092 -0.6273498 ]]
Translation t:
[-0.34668418  0.23423761  1.30144032]
Alignment RMSE (m): 0.094035
Sim(3) matrix:
[[-1.03368185  0.32789455  0.34024285 -0.34668418]
 [-0.01840349  0.78982832 -0.81707449  0.23423761]
 [-0.47216665 -0.74862179 -0.71302333  1.30144032]
 [ 0.          0.          0.          1.        ]]
```
**scene cabinet_simple 60 frames, 30-12**
```
=== Umeyama Sim(3): Pi3 -> GT ===
Scale: 1.0906889340
Rotation R:
[[-0.83284226  0.00201374  0.55350675]
 [-0.3620696   0.75439226 -0.54753806]
 [-0.41866381 -0.65642079 -0.62756064]]
Translation t:
[0.05804596 0.17697884 1.29603352]
Alignment RMSE (m): 0.092250
Sim(3) matrix:
[[-0.90837183  0.00219636  0.60370368  0.05804596]
 [-0.3949053   0.82280729 -0.5971937   0.17697884]
 [-0.45663198 -0.7159509  -0.68447345  1.29603352]
 [ 0.          0.          0.          1.        ]]
```

### how depth saving works (Pi3 output)
- Pi3 predicts per-pixel depth in meters: `D_m(u,v)`.
- We save metric PNGs as `uint16` (not normalized uint8).
- Encoding uses a fixed scale `s = pi3_png_depth_scale` (default `0.001` m/unit):

```text
q(u,v) = round(D_m(u,v) / s)          # quantized depth value
q(u,v) is clipped to [1, 65535] for valid pixels, 0 for invalid
```

- Saved file value is `q(u,v)` in `depth%06d.png`.
- A metadata file `pi3_depth_meta.txt` stores `png_depth_scale`, so decoding is deterministic:

```text
D_m(u,v) = q(u,v) * s
```

### how depth -> point cloud reconstruction works
For each valid depth pixel `(u, v, z)` with `z = D_m(u,v)` and intrinsics `(fx, fy, cx, cy)`:

```text
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
z_cam = z
```

This gives camera-frame 3D point:

```text
p_cam = [x_cam, y_cam, z_cam]^T
```

Then transform to world frame using camera-to-world pose `T_c2w = [R|t]`:

```text
p_world = R * p_cam + t
```

All `p_world` points are accumulated into the Open3D point cloud.
