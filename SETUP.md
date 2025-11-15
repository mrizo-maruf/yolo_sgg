# YOLO SSG Setup Guide

Complete setup instructions for the YOLO Scene Graph Generation (SSG) project.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option 1: Using Conda (Recommended)](#option-1-using-conda-recommended)
  - [Option 2: Using venv](#option-2-using-venv)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
- [Troubleshooting](#troubleshooting)
- [GPU Setup](#gpu-setup)

---

## Prerequisites

- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, or macOS
- **Python**: 3.10 (recommended) or 3.9-3.11
- **CUDA**: 11.7+ (optional, for GPU acceleration)
- **Git**: For cloning repositories
- **Conda or Python venv**: For environment management

---

## Installation

### Option 1: Using Conda (Recommended)

Conda provides better dependency management and is recommended for this project.

#### Step 1: Install Conda

If you don't have Conda installed, download and install Miniconda or Anaconda:

```bash
# For Linux (Miniconda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# For macOS (Miniconda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

For Windows, download the installer from: https://docs.conda.io/en/latest/miniconda.html

#### Step 2: Create Conda Environment

```bash
# Navigate to project directory
cd /path/to/yolo_ssg

# Create conda environment with Python 3.10
conda create -n yolo_ssg python=3.10 -y

# Activate the environment
conda activate yolo_ssg
```

#### Step 3: Install PyTorch (with CUDA support)

**For GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**For GPU (CUDA 12.1):**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**For CPU only:**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

Verify PyTorch installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Step 4: Install Core Dependencies

```bash
# Install numpy and essential packages
conda install numpy opencv matplotlib pillow -c conda-forge -y

# Install Open3D
pip install open3d>=0.17.0

# Install remaining Python packages
pip install -r requirements.txt
```

#### Step 5: Install Custom YOLOE Package

The project uses a custom YOLOE (YOLO Extension) package. Install it based on your setup:

**Option A: From source (if you have the code)**
```bash
# If YOLOE is in a separate repository
pip install git+https://github.com/your-username/ultralytics-yoloe.git

# OR if you have it locally
cd /path/to/ultralytics-yoloe
pip install -e .
```

**Option B: Use ultralytics as fallback**
```bash
# If YOLOE is not available, you can use standard ultralytics
pip install ultralytics>=8.0.0
# Note: You may need to modify imports in the code
```

#### Step 6: Install Optional Dependencies (VGGT)

If you plan to use VGGT-based processing:

```bash
# Install VGGT from repository
pip install git+https://github.com/ahkhan-repo/VGGT.git

# Or if you have it locally
cd /path/to/VGGT
pip install -e .
```

#### Step 7: Verify Installation

```bash
# Test imports
python -c "import torch; import numpy; import cv2; import open3d; import networkx; print('All core packages imported successfully!')"

# Test YOLOE import
python -c "from ultralytics import YOLOE; print('YOLOE imported successfully!')"
```

---

### Option 2: Using venv

If you prefer using Python's built-in venv:

#### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/yolo_ssg

# Create virtual environment
python3.10 -m venv venv_yolo_ssg

# Activate environment
# On Linux/macOS:
source venv_yolo_ssg/bin/activate

# On Windows:
venv_yolo_ssg\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose appropriate version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install all requirements
pip install -r requirements.txt

# Install YOLOE (as described in Conda Step 5)
```

---

## Project Structure

```
yolo_ssg/
├── yolo_ssg.py              # Main script for traditional RGBD processing
├── yolo_ssg_e.py            # Main script for VGGT-based processing
├── requirements.txt          # Python dependencies
├── SETUP.md                 # This file
├── YOLOE/
│   └── utils.py             # Utility functions for YOLO processing
├── ssg/
│   ├── ssg_main.py          # Scene graph generation logic
│   ├── ssg_utils.py         # SSG utility functions
│   ├── relationships/       # Relationship detection modules
│   └── ssg_data/            # Data structures and visualization
├── robotics_kitchen_dataset_v3/
│   ├── extract_frame_data.py      # VGGT data extraction
│   ├── predict_and_save.py        # VGGT prediction pipeline
│   └── frame_data_vggt/           # Extracted VGGT frame data
└── rendered_frames/                # Output directory for rendered frames
```

---

## Running the Code

### Basic Usage (Traditional RGBD)

```bash
# Activate environment
conda activate yolo_ssg

# Run with default configuration
python yolo_ssg.py

# Run with custom paths
python yolo_ssg.py --rgb_dir /path/to/rgb --depth_dir /path/to/depth --traj_path /path/to/trajectory.txt
```

### VGGT-Based Processing

```bash
# Activate environment
conda activate yolo_ssg

# Run VGGT-based scene graph generation
python yolo_ssg_e.py

# Customize configuration in the script's __main__ section:
# Edit yolo_ssg_e.py and modify the cfg dictionary
```

### Configuration Options

Key configuration parameters in `yolo_ssg_e.py`:

```python
cfg = OmegaConf.create({
    'rgb_dir': '/path/to/rgb',                      # RGB images directory
    'vggt_frame_data_dir': '/path/to/frame_data',  # VGGT frame data
    'yolo_model': 'yoloe-11l-seg-pf-old.pt',       # YOLO model path
    'conf': 0.3,                                    # Confidence threshold
    'iou': 0.5,                                     # IOU threshold
    'max_points_per_obj': 2000,                     # Points per object
    'o3_nb_neighbors': 20,                          # Outlier removal neighbors
    'o3_std_ratio': 2.0,                            # Outlier removal std ratio
    'show_pcds': True,                              # Show 3D visualizations
    'vis_graph': True,                              # Show scene graphs
    'save_rendered_frames': False,                  # Save rendered outputs
})
```

### Running Tests

```bash
# Test VGGT integration
python test_vggt_integration.py

# Test simple VGGT loading
python test_vggt_simple.py
```

---

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'YOLOE'

**Solution:**
```bash
# Ensure YOLOE is in your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/yolo_ssg"

# Or install ultralytics as fallback
pip install ultralytics
```

#### 2. CUDA Out of Memory

**Solution:**
```python
# Reduce batch size or max_points_per_obj
cfg.max_points_per_obj = 1000  # Instead of 2000
```

#### 3. Open3D Visualization Not Working

**Solution:**
```bash
# Install mesa libraries (Linux)
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Or disable visualization
cfg.show_pcds = False
```

#### 4. OpenCV Import Error

**Solution:**
```bash
# Reinstall opencv
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python>=4.8.0
```

#### 5. Shapely Geometry Errors

**Solution:**
```bash
# Update shapely
pip install --upgrade shapely>=2.0.0
```

---

## GPU Setup

### Check GPU Availability

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Install CUDA Toolkit (if needed)

**Ubuntu/Linux:**
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Environment Management

### Exporting Environment

```bash
# Save conda environment
conda env export > environment.yml

# Save pip requirements
pip freeze > requirements_frozen.txt
```

### Recreating Environment

```bash
# From conda yaml
conda env create -f environment.yml

# From pip requirements
pip install -r requirements_frozen.txt
```

### Deactivating Environment

```bash
# Conda
conda deactivate

# venv
deactivate
```

---

## Additional Resources

- **YOLO Documentation**: https://docs.ultralytics.com/
- **Open3D Tutorials**: http://www.open3d.org/docs/release/
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **NetworkX Documentation**: https://networkx.org/documentation/stable/

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{yolo_ssg_2024,
  title={YOLO-based Scene Graph Generation for Robotic Applications},
  author={Your Name},
  year={2024}
}
```

---

## License

[Specify your license here]

---

## Contact

For issues and questions:
- **GitHub Issues**: [Your repository]/issues
- **Email**: your.email@example.com

---

**Last Updated**: November 2025
