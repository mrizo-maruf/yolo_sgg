# from isaacsim import SimulationApp
import carb
from isaacsim import SimulationApp
import sys

# ---------------------- Configuration ----------------------
BACKGROUND_STAGE_PATH = "/World/env"
scene_name = "scene_1"
BACKGROUND_USD_PATH = f"/workspace/isaaclab/SG/is_benchmark_scenes/{scene_name}.usd"
keyframes_move = [
    {'time': 0, 'translation': [0, 4, 1.5], 'euler_angles': [0, 20, 0]},
    {'time': 10, 'translation': [4.5, 4, 1.5], 'euler_angles': [0, 20, 20]},
    {'time': 20, 'translation': [6.5, 6, 1.5], 'euler_angles': [0, 20, -40]},
    {'time': 30, 'translation': [7, 5.5, 1.5], 'euler_angles': [0, 20, -40]},
]
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

# Image / depth settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0  # meters
PNG_MAX_VALUE = 65535  # 16-bit depth image
FOCAL_LENGTH = 50
HORIZONTAL_APARTURE = 80
VERTICAL_APARTURE = 45

# Warm-up steps before recording (improves stability)
WARMUP_STEPS = 50
RENDER_SUBSTEPS = 100  # inner steps per saved frame for better visuals

# ---------------------- Imports ----------------------
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils import stage
from isaacsim.storage.native import get_assets_root_path
import os
import omni.replicator.core as rep
import omni
from isaacsim.core.utils.rotations import euler_angles_to_quat
import cv2
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
import json
from pxr import Sdf, UsdGeom, Usd

# ---------------------- Extensions ----------------------
res = enable_extension("isaacsim.ros2.bridge")

# ---------------------- World ----------------------
physics_dt = 1.0 / 20.0
rendering_dt = 1.0 / 20.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# ---------------------- Environment ----------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
    
# add_reference_to_stage(usd_path=BACKGROUND_USD_PATH, prim_path=BACKGROUND_STAGE_PATH)
add_reference_to_stage(usd_path=BACKGROUND_USD_PATH, prim_path=BACKGROUND_STAGE_PATH)

# ---------------------- Utility Functions ----------------------

def hide_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "invisible")

def show_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "inherited")

def set_prim_visibility_attribute(prim_path: str, value: str):
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
    )

def transformation_matrix(position, orientation):
    # orientation expected as (w, x, y, z)
    w, x, y, z = orientation
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T

def interpolate_keyframes_with_euler(keyframes, i):
    for j in range(len(keyframes) - 1):
        t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
        if t0 <= i <= t1:
            kf0, kf1 = keyframes[j], keyframes[j + 1]
            break
    else:
        return None, None
    alpha = (i - t0) / (t1 - t0)
    next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])
    euler0 = kf0['euler_angles']
    euler1 = kf1['euler_angles']
    interpolated_euler = (1 - alpha) * np.array(euler0) + alpha * np.array(euler1)
    next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)
    return next_translation, next_orientation

def create_color_map(num_classes):
    colors = [[0, 0, 0]]  # background
    import colorsys
    golden_ratio = 0.618033988749895
    for i in range(1, num_classes):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.6 + (i % 4) * 0.1
        value = 0.7 + (i % 3) * 0.15
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(b * 255), int(g * 255), int(r * 255)])
    return np.array(colors, dtype=np.uint8)

def apply_color_map(seg_image, color_map):
    h, w = seg_image.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id in np.unique(seg_image):
        if seg_id < len(color_map):
            colored_seg[seg_image == seg_id] = color_map[seg_id]
    return colored_seg

def compute_intrinsics(camera_prim, width, height):
    focal_length_attr = camera_prim.GetAttribute("focalLength").Get()
    h_aperture_attr = camera_prim.GetAttribute("horizontalAperture").Get()
    camera_prim.GetAttribute("verticalAperture").Set(VERTICAL_APARTURE)
    v_aperture_attr = camera_prim.GetAttribute("verticalAperture").Get()
    

    # focal_length_attr = camera.get_focal_length()
    # h_aperture_attr = camera.get_horizontal_aperture()
    # v_aperture_attr = camera.get_vertical_aperture()
    # resolution = camera.get_resolution()
    
    # print params
    print(f"focal: {focal_length_attr}, hapat: {h_aperture_attr}, \
        vapat: {v_aperture_attr}, res: {resolution}")
    
    print(f"vapat again: ", camera_prim.GetAttribute("verticalAperture").Get())
    print(f"resol: ", )
    fx = focal_length_attr / h_aperture_attr * width
    fy = focal_length_attr / v_aperture_attr * height
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

import json
import numpy as np
from collections import defaultdict

def _to_py(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    return obj


def build_unified_id_mapping(seg_data, twod_bbox_data, threed_bbox_data, max_abs_extent: float = 1e6):
    """
    Build a unified mapping from prim_path to all annotation IDs.
    Only includes 3D bbox entries that pass validation (finite values, reasonable extent).
    
    Returns a dict: prim_path -> {
        'instance_seg_id': int or -1,
        'bbox_2d_id': int or -1,
        'bbox_2d_semantic_id': int or -1,
        'bbox_3d_id': int or -1,
        'bbox_3d_semantic_id': int or -1,
        'semantic_label': dict,
    }
    """
    unified = {}
    
    # 1. Parse instance segmentation: idToLabels maps instance_id -> prim_path
    seg_info = seg_data.get('info', {}) or {}
    seg_id_to_labels = seg_info.get('idToLabels', {}) or {}
    seg_id_to_semantics = seg_info.get('idToSemantics', {}) or {}
    
    for inst_id_str, prim_path in seg_id_to_labels.items():
        if not str(inst_id_str).isdigit():
            continue
        inst_id = int(inst_id_str)
        if prim_path not in unified:
            unified[prim_path] = {
                'instance_seg_id': -1,
                'bbox_2d_id': -1,
                'bbox_2d_semantic_id': -1,
                'bbox_3d_id': -1,
                'bbox_3d_semantic_id': -1,
                'semantic_label': {},
            }
        unified[prim_path]['instance_seg_id'] = inst_id
        unified[prim_path]['semantic_label'] = seg_id_to_semantics.get(inst_id_str, {})
    
    # 2. Parse 2D bbox: primPaths[i] corresponds to bboxIds[i] and data[i]
    if twod_bbox_data is not None:
        twod_info = twod_bbox_data.get('info', {}) or {}
        twod_data = twod_bbox_data.get('data', None)
        twod_prim_paths = twod_info.get('primPaths', [])
        twod_bbox_ids = np.asarray(twod_info.get('bboxIds', []), dtype=np.uint32)
        
        if twod_data is not None and len(twod_data) > 0:
            twod_data = np.asarray(twod_data)
            for idx, prim_path in enumerate(twod_prim_paths):
                if prim_path not in unified:
                    unified[prim_path] = {
                        'instance_seg_id': -1,
                        'bbox_2d_id': -1,
                        'bbox_2d_semantic_id': -1,
                        'bbox_3d_id': -1,
                        'bbox_3d_semantic_id': -1,
                        'semantic_label': {},
                    }
                bbox_id = int(twod_bbox_ids[idx]) if idx < len(twod_bbox_ids) else idx
                semantic_id = int(twod_data[idx][0]) if idx < len(twod_data) else -1
                unified[prim_path]['bbox_2d_id'] = bbox_id
                unified[prim_path]['bbox_2d_semantic_id'] = semantic_id
    
    # 3. Parse 3D bbox: primPaths[i] corresponds to bboxIds[i] and data[i]
    #    ONLY include entries that pass validation (will actually appear in output)
    if threed_bbox_data is not None:
        threed_info = threed_bbox_data.get('info', {}) or {}
        threed_data = threed_bbox_data.get('data', None)
        threed_prim_paths = threed_info.get('primPaths', [])
        threed_bbox_ids = np.asarray(threed_info.get('bboxIds', []), dtype=np.uint32)
        
        if threed_data is not None and len(threed_data) > 0:
            for idx, prim_path in enumerate(threed_prim_paths):
                row = threed_data[idx]
                
                # Apply same validation as build_3d_boxes - skip invalid boxes
                x_min_local = float(row["x_min"]); y_min_local = float(row["y_min"]); z_min_local = float(row["z_min"])
                x_max_local = float(row["x_max"]); y_max_local = float(row["y_max"]); z_max_local = float(row["z_max"])
                vals = np.array([x_min_local, y_min_local, z_min_local, x_max_local, y_max_local, z_max_local], dtype=np.float64)
                
                if (not np.isfinite(vals).all()) or (np.max(np.abs(vals)) > max_abs_extent):
                    print(f"[DEBUG] Skipping invalid 3D bbox for '{prim_path}': vals={vals}")
                    continue
                
                if prim_path not in unified:
                    unified[prim_path] = {
                        'instance_seg_id': -1,
                        'bbox_2d_id': -1,
                        'bbox_2d_semantic_id': -1,
                        'bbox_3d_id': -1,
                        'bbox_3d_semantic_id': -1,
                        'semantic_label': {},
                    }
                bbox_id = int(threed_bbox_ids[idx]) if idx < len(threed_bbox_ids) else idx
                semantic_id = int(row['semanticId'])
                unified[prim_path]['bbox_3d_id'] = bbox_id
                unified[prim_path]['bbox_3d_semantic_id'] = semantic_id
    
    print(f"[DEBUG] Unified ID mapping:")
    for prim_path, ids in unified.items():
        print(f"  {prim_path}: {ids}")
    
    return unified


def build_2d_records(twod_bbox_data, unified_mapping: dict = None):
    """
    Parse 2D bbox data and include cross-reference IDs from unified mapping.
    
    Args:
        twod_bbox_data: bounding_box_2d_tight annotator output
        unified_mapping: mapping from prim_path -> all annotation IDs
    """
    if twod_bbox_data is None:
        return []

    data = twod_bbox_data.get("data", None)
    info = twod_bbox_data.get("info", {}) or {}

    if data is None or len(data) == 0:
        return []

    data = np.asarray(data)
    bbox_ids = np.asarray(info.get("bboxIds", np.arange(len(data))), dtype=np.uint32)
    prim_paths = info.get("primPaths", [None] * len(data))
    id_to_labels = info.get("idToLabels", {}) or {}

    unified_mapping = unified_mapping or {}

    records = []
    for idx, row in enumerate(data):
        bbox_semantic_id = int(row[0])
        
        x_min = float(row[1]); y_min = float(row[2])
        x_max = float(row[3]); y_max = float(row[4])
        vis_or_occ = float(row[5])

        bbox_id = int(bbox_ids[idx]) if idx < len(bbox_ids) else idx
        prim_path = prim_paths[idx] if idx < len(prim_paths) else None

        # Get cross-reference IDs from unified mapping
        ids = unified_mapping.get(prim_path, {})
        instance_seg_id = ids.get('instance_seg_id', -1)
        bbox_3d_id = ids.get('bbox_3d_id', -1)
        semantic_label = ids.get('semantic_label', {})

        # Get label from bbox annotator
        label_dict = id_to_labels.get(str(bbox_semantic_id), {})

        records.append({
            "bbox_2d_id": bbox_id,
            "bbox_2d_semantic_id": bbox_semantic_id,
            "instance_seg_id": instance_seg_id,
            "bbox_3d_id": bbox_3d_id,
            "prim_path": prim_path,
            "label": label_dict,
            "semantic_label": semantic_label,
            "xyxy": [x_min, y_min, x_max, y_max],
            "visibility_or_occlusion": vis_or_occ,
        })
        
        print(f"[DEBUG] 2D Box '{prim_path}': bbox_2d_id={bbox_id}, instance_seg_id={instance_seg_id}, bbox_3d_id={bbox_3d_id}")

    return records
def _label_from_idToLabels(idToLabels: dict, sid: int):
    """
    idToLabels format example:
      {'7': {'box': 'box_purple', 'class': 'klt_bin'}, '6': {'table': 'thos'}, ...}
    We return a compact string label like:
      "box_purple" (plus class if present -> "box_purple|klt_bin")
    """
    d = idToLabels.get(str(int(sid)))
    if not d:
        return None

    # prefer 'class' if it is the only key; otherwise pick first non-'class' key as the main label
    if "class" in d and len(d) == 1:
        return str(d["class"])

    main_val = None
    for k, v in d.items():
        if k == "class":
            continue
        main_val = v
        break

    if main_val is None and "class" in d:
        main_val = d["class"]

    if main_val is None:
        return None

    if "class" in d and str(d["class"]) != str(main_val):
        return f"{main_val}|{d['class']}"
    return str(main_val)


def build_3d_boxes(Nd_bbox_data, unified_mapping: dict = None, max_abs_extent: float = 1e6):
    """Parse Replicator bounding_box_3d output and transform to WORLD coordinates.

    The annotator returns:
    - AABBs in LOCAL/OBJECT coordinates (x_min, y_min, z_min, x_max, y_max, z_max)
    - `transform` is a 4x4 matrix in ROW-MAJOR format with translation in LAST ROW
    
    Args:
        Nd_bbox_data: bounding_box_3d annotator output
        unified_mapping: mapping from prim_path -> all annotation IDs
    """

    data = Nd_bbox_data.get("data", None)
    info = Nd_bbox_data.get("info", {}) or {}

    idToLabels = info.get("idToLabels", {}) or {}
    primPaths = info.get("primPaths", [None] * (len(data) if data is not None else 0))
    bboxIds = np.asarray(info.get("bboxIds", np.arange(len(data) if data is not None else 0)), dtype=np.uint32)
    semanticOcclusion = info.get("semanticOcclusion", None)

    unified_mapping = unified_mapping or {}

    boxes = []
    if data is None or len(data) == 0:
        return boxes

    for idx, row in enumerate(data):
        bbox_semantic_id = int(row["semanticId"])
        prim_path = primPaths[idx] if idx < len(primPaths) else None
        
        # Get cross-reference IDs from unified mapping
        ids = unified_mapping.get(prim_path, {})
        instance_seg_id = ids.get('instance_seg_id', -1)
        bbox_2d_id = ids.get('bbox_2d_id', -1)
        semantic_label = ids.get('semantic_label', {})

        # Local AABB extents (in object frame)
        x_min_local = float(row["x_min"]); y_min_local = float(row["y_min"]); z_min_local = float(row["z_min"])
        x_max_local = float(row["x_max"]); y_max_local = float(row["y_max"]); z_max_local = float(row["z_max"])
        occ_field = float(row.get("occlusionRatio", 0.0)) if hasattr(row, "get") else float(row["occlusionRatio"])

        occ = occ_field
        if semanticOcclusion is not None and idx < len(semanticOcclusion):
            try:
                occ = float(semanticOcclusion[idx])
            except Exception:
                pass

        vals = np.array([x_min_local, y_min_local, z_min_local, x_max_local, y_max_local, z_max_local], dtype=np.float64)
        if (not np.isfinite(vals).all()) or (np.max(np.abs(vals)) > max_abs_extent):
            continue

        # Get transform from annotator - it's ROW-MAJOR with translation in last row
        T_row_major = np.asarray(row["transform"], dtype=np.float64)
        
        # Convert to column-major (standard numpy convention) by transposing
        T_world_local = T_row_major.T
        
        print(f"Object {idx}: translation from transform = {T_world_local[:3, 3]}")

        # Define all 8 corners in local coordinates (homogeneous)
        corners_local = np.array([
            [x_min_local, y_min_local, z_min_local, 1.0],
            [x_max_local, y_min_local, z_min_local, 1.0],
            [x_max_local, y_max_local, z_min_local, 1.0],
            [x_min_local, y_max_local, z_min_local, 1.0],
            [x_min_local, y_min_local, z_max_local, 1.0],
            [x_max_local, y_min_local, z_max_local, 1.0],
            [x_max_local, y_max_local, z_max_local, 1.0],
            [x_min_local, y_max_local, z_max_local, 1.0],
        ]).T  # (4, 8)

        # Transform to world coordinates
        corners_world = (T_world_local @ corners_local)[:3, :].T  # (8, 3)

        # Recompute axis-aligned bounding box in world frame
        x_min_world = float(np.min(corners_world[:, 0]))
        y_min_world = float(np.min(corners_world[:, 1]))
        z_min_world = float(np.min(corners_world[:, 2]))
        x_max_world = float(np.max(corners_world[:, 0]))
        y_max_world = float(np.max(corners_world[:, 1]))
        z_max_world = float(np.max(corners_world[:, 2]))

        bbox_id = int(bboxIds[idx]) if idx < len(bboxIds) else idx
        label = _label_from_idToLabels(idToLabels, bbox_semantic_id)

        boxes.append({
            "track_id": bbox_id,
            "bbox_3d_id": bbox_id,
            "bbox_3d_semantic_id": bbox_semantic_id,
            "instance_seg_id": instance_seg_id,
            "bbox_2d_id": bbox_2d_id,
            "prim_path": prim_path,
            "label": label,
            "semantic_label": semantic_label,
            "aabb_xyzmin_xyzmax": [x_min_world, y_min_world, z_min_world,
                                   x_max_world, y_max_world, z_max_world],
            "transform_4x4": T_world_local.tolist(),
            "occlusion_ratio": occ,
        })
        
        print(f"[DEBUG] 3D Box '{label}' ({prim_path}): bbox_3d_id={bbox_id}, instance_seg_id={instance_seg_id}, bbox_2d_id={bbox_2d_id}")
        
    return boxes
# ---------------------- Camera ----------------------
camera = Camera(
    prim_path="/World/Camera",
    position = np.array([3.5, -2.5, 1]),
    resolution=(IMAGE_HEIGHT, IMAGE_WIDTH),
    orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True)
)

# camera.set_focal_length(FOCAL_LENGTH)
# camera.set_resolution(value=(IMAGE_WIDTH, IMAGE_HEIGHT))

my_world.reset()
camera.initialize()
camera.add_distance_to_camera_to_frame()

stage_ref = get_current_stage()



camera_prim= stage_ref.GetPrimAtPath("/World/Camera")
focal_length = camera_prim.GetAttribute("focalLength")
horizontal_aperture = camera_prim.GetAttribute("horizontalAperture")
vertical_aperture = camera_prim.GetAttribute("verticalAperture")
resolution = camera_prim.GetAttribute("resolution")

# print("before focal length:", focal_length.Get())
# print("before horizontal aperture:", horizontal_aperture.Get())
# print("before vertical aperture:", vertical_aperture.Get())
# print("before resolution: ",resolution.Get())

horizontal_aperture.Set(80)
# Compute and print intrinsics & depth scale BEFORE capturing images
fx, fy, cx, cy = compute_intrinsics(camera_prim, IMAGE_WIDTH, IMAGE_HEIGHT)
png_depth_scale = (MAX_DEPTH - MIN_DEPTH) / PNG_MAX_VALUE
print(f"Camera intrinsics -> fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}, png_depth_scale: {png_depth_scale:.8f}")

print("afeter horizontal aperture:", horizontal_aperture.Get())
print("afeter vertical aperture:", vertical_aperture.Get())
print("afeter focal length:", focal_length.Get())

# ---------------------- Keyframes ----------------------

record_keyframe = keyframes_move

# ---------------------- ROS2 Camera Graph ----------------------
import usdrt.Sdf
CAMERA_STAGE_PATH = "/World/Camera"
ROS_CAMERA_GRAPH_PATH = "/ROS2_Camera"
keys = og.Controller.Keys
(ros_camera_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_STAGE_PATH)]),
        ],
    },
)

from omni.kit.viewport.utility import get_active_viewport
viewport_api = get_active_viewport()
render_product_path = viewport_api.get_render_product_path()

# ---------------------- Output Paths ----------------------
base_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/"
traj_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}"
os.makedirs(base_dir, exist_ok=True)
image_prefix = "frame"
depth_prefix = "depth"
seg_prefix = "semantic"
twod_box = "2d_box"
threed_box = "3d_box"

traj_file_path = os.path.join(traj_dir, "traj.txt")
if not os.path.exists(traj_file_path):
    open(traj_file_path, 'w').close()
    print(f"File '{traj_file_path}' has been created.")
else:
    print(f"File '{traj_file_path}' already exists.")

# ---------------------- Warm-up ----------------------
for _ in range(WARMUP_STEPS):
    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, 0)
    if _ == 0:
        camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    my_world.step(render=True)
    simulation_app.update()

# ---------------------- Main Recording Loop ----------------------
i = 0
frame_index = 0
while simulation_app.is_running():
    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, i)
    if next_translation is None:
        break
    camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    position_, orientation_ = camera.get_local_pose(camera_axes="world")
    _ = transformation_matrix(position_, orientation_)
    print(f"Iteration: {i}")
    i += 1

    # Extra internal render steps (smoother outputs)
    for _ in range(RENDER_SUBSTEPS):
        my_world.step(render=True)
    simulation_app.update()

    viewport_api = get_active_viewport()
    render_product_path = viewport_api.get_render_product_path()

    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_ann.attach([render_product_path])
    depth_image = depth_ann.get_data()

    rgb_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_ann.attach([render_product_path])
    rgba_image = rgb_ann.get_data()

    seg_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
    seg_ann.attach([render_product_path])
    seg_data = seg_ann.get_data()
    seg_image = seg_data['data'].astype(np.uint32)  # Instance IDs can be larger, use uint32
    
    # Instance segmentation provides:
    # - idToLabels: instance_id -> prim_path (e.g., '3': '/World/env/bowl')
    # - idToSemantics: instance_id -> semantic_labels (e.g., '3': {'goal': 'bowl'})
    seg_id_to_labels = seg_data['info'].get('idToLabels', {})  # Prim paths
    seg_id_to_semantics = seg_data['info'].get('idToSemantics', {})  # Semantic labels
    
    print(f"[DEBUG] Instance seg idToLabels (prim paths): {seg_id_to_labels}")
    print(f"[DEBUG] Instance seg idToSemantics: {seg_id_to_semantics}")
    print(f"[DEBUG] Unique instance IDs in image: {np.unique(seg_image)}")

    threed_bbox = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    threed_bbox.attach([render_product_path])
    threed_bbox_data = threed_bbox.get_data()
    
    twod_bbox = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    twod_bbox.attach([render_product_path])
    twod_bbox_data = twod_bbox.get_data()
    
    print(f"[DEBUG] Bbox 3D idToLabels: {threed_bbox_data['info'].get('idToLabels', {})}")
    print(f"[DEBUG] Bbox 2D idToLabels: {twod_bbox_data['info'].get('idToLabels', {})}")
    
    # Build unified mapping from prim_path to all annotation IDs
    unified_mapping = build_unified_id_mapping(seg_data, twod_bbox_data, threed_bbox_data)
    
    base_dir_rgb = base_dir+"rgb"
    base_dir_seg = base_dir+"seg"
    base_dir_depth = base_dir+"depth"
    base_dir_bbox = base_dir+"bbox"

    os.makedirs(base_dir_rgb, exist_ok=True)
    os.makedirs(base_dir_seg, exist_ok=True)
    os.makedirs(base_dir_depth, exist_ok=True)
    os.makedirs(base_dir_bbox, exist_ok=True)

    img_path = os.path.join(base_dir_rgb, f"{image_prefix}{frame_index:06d}.jpg")
    depth_path = os.path.join(base_dir_depth, f"{depth_prefix}{frame_index:06d}.png")
    seg_colored_path = os.path.join(base_dir_seg, f"{seg_prefix}{frame_index:06d}.png")
    seg_info_path = os.path.join(base_dir_seg, f"{seg_prefix}{frame_index:06d}_info.json")
    threed_box_path = os.path.join(base_dir_bbox, f"bboxes{frame_index:06d}_info.json")
    twod_box_path = os.path.join(base_dir_bbox, f"{twod_box}{frame_index:06d}_info.json")

    if depth_image.size != 0 and rgba_image.size != 0:
        clipped_depth = np.clip(depth_image, MIN_DEPTH, MAX_DEPTH)
        normalized_depth = ((clipped_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * PNG_MAX_VALUE
        depth_image_uint16 = normalized_depth.astype("uint16")
        cv2.imwrite(depth_path, depth_image_uint16)

        max_seg_id = np.max(seg_image) if seg_image.size > 0 else 0
        num_classes = max(max_seg_id + 1, len(seg_id_to_labels) if seg_id_to_labels else 0, 1)
        color_map = create_color_map(num_classes)
        colored_seg_image = apply_color_map(seg_image, color_map)
        cv2.imwrite(seg_colored_path, colored_seg_image)

        rgb = rgba_image[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)

        # Gets prim’s pose with respect to the world’s frame (always at [0, 0, 0] and unity quaternion
        # not to be confused with /World Prim)
        # ros is (+Y up, +Z forward) 
        position_ros, orientation_ros = camera.get_world_pose(camera_axes="ros")
        T_ros = transformation_matrix(position_ros, orientation_ros)
        with open(traj_file_path, "a") as traj_file:
            traj_file.write(' '.join(map(str, T_ros.flatten())) + "\n")

        # Build enhanced seg_info using instance segmentation data and unified mapping
        enhanced_seg_info = {}
        for inst_id_str, prim_path in seg_id_to_labels.items():
            if not str(inst_id_str).isdigit():
                continue
            inst_id_int = int(inst_id_str)
            
            # Get semantic label from idToSemantics
            semantic_dict = seg_id_to_semantics.get(inst_id_str, {})
            
            # Get cross-reference IDs from unified mapping
            ids = unified_mapping.get(prim_path, {})
            
            enhanced_seg_info[inst_id_str] = {
                "instance_seg_id": inst_id_int,
                "bbox_2d_id": ids.get('bbox_2d_id', -1),
                "bbox_3d_id": ids.get('bbox_3d_id', -1),
                "prim_path": prim_path,
                "label": semantic_dict,  # e.g., {'goal': 'bowl'}
                "color_bgr": color_map[inst_id_int].tolist() if inst_id_int < len(color_map) else [0, 0, 0]
            }

        # for box_id, 
        with open(seg_info_path, "w") as json_file:
            json.dump(enhanced_seg_info, json_file, indent=4)

        # Parse bbox annotators using unified mapping for cross-references
        boxes2d = build_2d_records(twod_bbox_data, unified_mapping=unified_mapping)
        boxes3d = build_3d_boxes(threed_bbox_data, unified_mapping=unified_mapping)
        
        # print(boxes2d)
        # print(boxes3d)
        boxes = {
            "bboxes": {
                "bbox_2d_tight": {
                    "boxes": boxes2d,
                },
                "bbox_3d": {
                    "boxes": boxes3d,
                }
            }
        }
        
        with open(threed_box_path, "w") as f:
            json.dump(boxes, f, indent=4)
        # save json

    print('='*50)
    print(f"seg info: {seg_data['info']}")
    print(f"2d bbox: {twod_bbox_data['info']}")
    print(f"3d info: {threed_bbox_data['info']}")
    print('='*50)

    frame_index += 1

# ---------------------- Shutdown ----------------------
simulation_app.close()