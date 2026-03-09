#!/bin/bash

# Initialize conda
# source ~/anaconda3/etc/profile.d/conda.sh   # change if your conda path is different

# # Activate environment
# conda activate gen

# Run benchmarks
python ./benchmark/benchmark_tracking_thud.py \
/home/yehia/rizo/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 \
--model /home/yehia/rizo/code/yolo_sgg/yoloe-11l-seg.pt

python ./benchmark/benchmark_tracking_thud.py \
/home/yehia/rizo/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_2 \
--model /home/yehia/rizo/code/yolo_sgg/yoloe-11l-seg.pt

python ./benchmark/benchmark_tracking_thud.py \
/home/yehia/rizo/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_3 \
--model /home/yehia/rizo/code/yolo_sgg/yoloe-11l-seg.pt