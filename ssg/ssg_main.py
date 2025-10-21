import json
import pickle
from sympy import re
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import torch
import networkx as nx
import numpy as np

import time

from . import ssg_utils as utils
from .ssg_data import dictionary
from .ssg_data.ssg_visualize import vis_dataset
from .ssg_data.script.ObjNode import ObjNode
from .relationships.support import cal_support_relations
from .relationships.proximity import cal_proximity_relationships
from .relationships.hanging import cal_hanging_relationships
from .relationships.multi_objs import find_aligned_furniture, find_middle_furniture


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size


def predict_new_edges(current_graph, frame_objs, distance_scale: float = 0.5, support_z_tol: float = 0.02):
    """Predict simple spatial relationships (support / near / above / below) between
    objects detected in a single frame and add them as edges to `current_graph`.

    This is a lightweight, heuristic edge predictor intended to provide reasonable
    edges for downstream processing when a learned SceneVerse-style predictor is
    not available. It mutates `current_graph` in-place and also returns a list of
    added edges for convenience.

    Args:
        current_graph (networkx.DiGraph): Graph whose nodes are object ids and
            node attribute 'obj' holds a dict with key 'bbox_3d' containing
            an 'aabb' with 'min' and 'max' (numpy arrays or lists).
        frame_objs (list): List of per-object dicts returned by YOLOE utils
            (each has 'track_id' and 'bbox_3d').
        distance_scale (float): Scale used to determine "near" threshold as a
            fraction of object sizes (default 0.5).
        support_z_tol (float): Vertical tolerance (meters) to consider an object
            sitting on/top-of another (default 0.02 m).

    Returns:
        List[tuple]: list of (src_id, tgt_id, relation_label) edges added.
    """
    added = []
    try:
        import numpy as _np
    except Exception:
        _np = np

    # Build quick lookup for node -> bbox aabb (min,max) and center
    node_data = {}
    for n in current_graph.nodes:
        nd = current_graph.nodes[n].get('obj') if isinstance(current_graph.nodes[n], dict) else None
        if nd is None and isinstance(frame_objs, (list, tuple)):
            # try find in frame_objs by track_id
            for fo in frame_objs:
                if int(fo.get('track_id', -999)) == int(n):
                    nd = fo
                    break
        if nd is None:
            continue
        bbox = nd.get('bbox_3d', {}) or {}
        aabb = bbox.get('aabb')
        if not aabb or aabb.get('min') is None or aabb.get('max') is None:
            continue
        mn = _np.array(aabb['min'], dtype=float)
        mx = _np.array(aabb['max'], dtype=float)
        center = (mn + mx) / 2.0
        size = (mx - mn)
        node_data[n] = {'min': mn, 'max': mx, 'center': center, 'size': size}

    ids = list(node_data.keys())
    for i_idx in range(len(ids)):
        for j_idx in range(i_idx + 1, len(ids)):
            i = ids[i_idx]
            j = ids[j_idx]
            di = node_data[i]
            dj = node_data[j]

            # XY distance between centers
            xy_dist = _np.linalg.norm(di['center'][:2] - dj['center'][:2])

            # size-based threshold: use average diagonal of AABBs
            diag_i = _np.linalg.norm(di['size'])
            diag_j = _np.linalg.norm(dj['size'])
            near_thresh = max(1e-3, distance_scale * 0.5 * (diag_i + diag_j))

            # Decide "near" relation
            if xy_dist <= near_thresh:
                # add bidirectional 'near' edges
                current_graph.add_edge(i, j, label='near')
                current_graph.add_edge(j, i, label='near')
                added.append((i, j, 'near'))
                added.append((j, i, 'near'))

            # Vertical relationship (support / above)
            # If bottom of one is very close to top of the other and their XY projections overlap,
            # declare support: edge from supporting (lower) -> supported (upper).
            i_bottom_z = di['min'][2]
            i_top_z = di['max'][2]
            j_bottom_z = dj['min'][2]
            j_top_z = dj['max'][2]

            # helper: xy overlap of AABB projections
            def _xy_overlap(a_min, a_max, b_min, b_max):
                return not (a_max[0] < b_min[0] or b_max[0] < a_min[0] or a_max[1] < b_min[1] or b_max[1] < a_min[1])

            overlap_xy = _xy_overlap(di['min'], di['max'], dj['min'], dj['max'])

            # j supports i ? (i sits on top of j)
            if overlap_xy and (abs(i_bottom_z - j_top_z) <= support_z_tol) and (i_bottom_z >= j_top_z - support_z_tol):
                current_graph.add_edge(j, i, label='support')
                added.append((j, i, 'support'))
            # i supports j
            if overlap_xy and (abs(j_bottom_z - i_top_z) <= support_z_tol) and (j_bottom_z >= i_top_z - support_z_tol):
                current_graph.add_edge(i, j, label='support')
                added.append((i, j, 'support'))

            # Above / below (non-contact) if one center significantly higher
            z_center_diff = di['center'][2] - dj['center'][2]
            if abs(z_center_diff) > max(di['size'][2], dj['size'][2]) * 0.5:
                if z_center_diff > 0:
                    current_graph.add_edge(i, j, label='above')
                    current_graph.add_edge(j, i, label='below')
                    added.append((i, j, 'above'))
                    added.append((j, i, 'below'))
                else:
                    current_graph.add_edge(j, i, label='above')
                    current_graph.add_edge(i, j, label='below')
                    added.append((j, i, 'above'))
                    added.append((i, j, 'below'))

    # small debug
    if len(added) == 0:
        # don't spam; only print if graph non-empty
        if len(ids) > 0:
            print('[predict_new_edges] No edges added for current frame (heuristic thresholds may be strict).')

    return added

def view_from_pose(T_w_c):
    # T_w_c: 4x4 camera-to-world transform
    R_w_c = T_w_c[:3, :3]
    t_w_c = T_w_c[:3, 3]

    # Camera forward is +Z in camera frame
    forward_cam = np.array([0.0, 0.0, -1.0], dtype=float)
    camera_view = R_w_c @ forward_cam
    nrm = np.linalg.norm(camera_view)
    if nrm > 0:
        camera_view = camera_view / nrm

    camera_pos = t_w_c

    # Planar angle like init_camera_view: relative to +Y, sign by X
    v_xy = camera_view.copy()
    v_xy[2] = 0.0
    vxy_n = np.linalg.norm(v_xy)
    if vxy_n > 0:
        v_xy = v_xy / vxy_n
    else:
        v_xy = np.array([0.0, 1.0, 0.0], dtype=float)

    angle = utils.get_theta(v_xy, [0, 1, 0])
    camera_angle = -angle if v_xy[0] < 0 else angle

    return camera_view, camera_pos, camera_angle

def init_camera_view():
    camera_view = [0, -1, 0]
    camera_pos = [0, 0, 0]
    camera_view = camera_view / np.linalg.norm(camera_view)

    if camera_view[0] < 0:
        camera_angle = -utils.get_theta(camera_view, [0, 1, 0])
    else:
        camera_angle = utils.get_theta(camera_view, [0, 1, 0])

    return camera_view, camera_pos, camera_angle

def filter_bad_label(input_label):
    bad_label_list = ['ceiling', 'wall', 'door', 'doorframe', 'object']
    for o in bad_label_list:
        if o in input_label:
            return False

    return True

def get_obj_room_id (org_id):
    infos = org_id.split('|')
    if infos[1] == 'surface':
        return int(infos[2])
    else:
        return int(infos[1])

def generate_object_info(save_root, scene_name) :
    object_json_list = []

    inst2label_path = save_root / 'instance_id_to_label'
    pcd_path = save_root / 'pcd_with_global_alignment'

    inst_to_label = torch.load(inst2label_path / f"{scene_name}.pth", weights_only=False)
    pcd_data = torch.load(pcd_path / f'{scene_name}.pth', weights_only=False)

    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    pcds = np.concatenate([points, colors], 1)

    x_max, y_max, z_max = points.max(axis=0)
    x_min, y_min, z_min = points.min(axis=0)

    obj_pcds = []
    for i in np.unique(instance_labels):
        if i <= 0:
            continue
        mask = instance_labels == i     # time consuming
        print(inst_to_label)
        obj_pcds.append((pcds[mask], inst_to_label[int(i)], i))

    for _, (obj, obj_label, i) in enumerate(obj_pcds):
        gt_center, gt_size = convert_pc_to_box(obj)
        object_json = {
            'id': int(i),
            'label': obj_label,
            'position': gt_center,
            'size': gt_size,
            'mesh': None
        }
        object_json_list.append(object_json)

    # add scan_id
    object_json_string = {
        'scan': scene_name,
        'point_max': [x_max, y_max, z_max],
        'point_min': [x_min, y_min, z_min],
        'object_json_string': object_json_list,
        'inst_to_label': inst_to_label,
    }

    return object_json_string

def generate_ssg_data(dataset, scene_path, pre_load_path):
    ssg_data = {}
    pre_load_file_save_path = pre_load_path / (dataset + '.pkl')
    if pre_load_file_save_path.exists():
        print('Using preprocessed scene data')
        with open(pre_load_file_save_path, 'rb') as f:
            ssg_data = pickle.load(f)
    else:
        print('Preprocessing scene data')
        scans = [s.stem for s in (scene_path / 'pcd_with_global_alignment').glob('*.pth')]
        print(f'Found {len(scans)} scans in the dataset {dataset} in path {scene_path}')
        scans.sort()
        for scan_id in tqdm(scans):
            object_json_string = generate_object_info(scene_path, scan_id)
            if object_json_string is not None:
                ssg_data[scan_id] = object_json_string
        with open(pre_load_file_save_path, 'wb') as f:
            pickle.dump(ssg_data, f)

    # print ssg_data keys
    print(f"ssg_data.keys(): {ssg_data.keys()}")
    print(f"ssg_data[scene_00000_00].keys(): {ssg_data['scene_00000_00'].keys()}")
    return ssg_data

def main(cfg):
    cfg.rels_save_path.mkdir(parents=True, exist_ok=True)
    ssg_data = generate_ssg_data(cfg.dataset, cfg.scene_path, cfg.rels_save_path)
    scans_all = list(ssg_data.keys())
    
    print(f'Found {len(scans_all)} scans in the dataset {cfg.dataset}')

    ### init camera ###
    camera_view, camera_pos, camera_angle = init_camera_view()
    for scan_id in scans_all:
        objects_save = {}
        relationship_save = {}
        inst_dict = {}

        print('Processing ', scan_id)

        objects_info = ssg_data[scan_id]['object_json_string']
        inst_labels = ssg_data[scan_id]['inst_to_label']
        # bad case
        if len(objects_info) == 0:
            continue

        # Start SG latency timer (graph construction + relation inference)
        t_sg_start = time.perf_counter()

        # construct object graph
        G = nx.DiGraph()
        # create nodes
        ObjNode_dict = {}

        # log objects of the same category
        for inst in inst_labels:
            if inst_labels[inst] not in inst_dict:
                inst_dict[inst_labels[inst]] = 1
            else:
                inst_dict[inst_labels[inst]] += 1

        x_max, y_max, z_max = ssg_data[scan_id]['point_max']
        x_min, y_min, z_min = ssg_data[scan_id]['point_min']
        scene_center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
        # floor bad
        if z_max == z_min:
            z_max = z_min + 5
        scene_high = z_max - z_min

        # generate object node for graph
        obj_z_min = 1000
        floor_idx = -100
        for obj in objects_info:
            if np.array(obj['size']).sum() == 0:
                continue
            if not filter_bad_label(obj['label']):
                continue
            if obj['label'] == 'floor':
                floor_idx = int(obj['id'])
            node = ObjNode(id=int(obj['id']),
                           position=obj['position']-scene_center,
                           label=obj['label'],
                           mesh=obj['mesh'] if 'mesh' in obj else None,
                           size=np.array(obj['size']),
                           children=obj['children'] if 'children' in obj else None,
                           room_id=get_obj_room_id (obj['id_org']) if 'id_org' in obj else None,
                           dataset=cfg.dataset)

            if obj['position'][2] - obj['size'][2]/2 < obj_z_min:
                obj_z_min = obj['position'][2]-obj['size'][2]/2

            obj['count'] = inst_dict[node.label]
            obj['caption'] = ''

            ObjNode_dict[int(obj['id'])] = node
            G.add_node(node.id, label=node.label)

        # added special nodes (wall camera)
        G.add_node(-1, label='CAMERA')
        G.add_node(-2, label='wall')

        # special node for floor
        if floor_idx == -100:
            G.add_node(-3, label='floor')
            fx, fy, fz = scene_center[0], scene_center[1], obj_z_min
            node = ObjNode(id=-3,
                           position=np.array([fx, fy, fz]) - scene_center,
                           label='floor',
                           size=[(x_max-x_min)*1.2, (y_max-y_min)*1.2, (z_max-z_min)*0.1],
                           dataset=cfg.dataset)
            ObjNode_dict[-3] = node
            floor_idx = -3
        else:
            fx, fy, fz = scene_center[0], scene_center[1], obj_z_min
            node_ = ObjNode_dict[floor_idx]
            if node_.size[2] > 0:
                node = ObjNode(id=floor_idx,
                               position= np.array([fx, fy, fz]) - scene_center,
                               label='floor',
                               size=[max((x_max-x_min)*1.2, node_.size[0]),
                                     max((y_max-y_min)*1.2, node_.size[0]),
                                     node_.size[2]],
                               dataset=cfg.dataset)
            else:
                node = ObjNode(id=floor_idx,
                               position= np.array([fx, fy, fz]) - scene_center,
                               label='floor',
                               size=[max((x_max-x_min)*1.2, node_.size[0]),
                                     max((y_max-y_min)*1.2, node_.size[0]),
                                     (z_max-z_min)*0.1],
                               dataset=cfg.dataset)

            ObjNode_dict[floor_idx] = node

        # support embedded relationship
        if cfg.dataset.lower() in ['procthor']:
            support_relations = []
            embedded_relations = []
            hanging_objs_dict = {}
            for src_id, _ in ObjNode_dict.items():
                src_obj = ObjNode_dict[src_id]
                if src_obj.z_min <= ObjNode_dict[floor_idx].z_max and src_obj.id != floor_idx:
                    support_relations.append(utils.generate_relation(floor_idx, src_id,'support'))
                    hanging_objs_dict[src_id] = 1

                if src_obj.children != []:
                    for child in src_obj.children:
                        hanging_objs_dict[child] = 1
                        if child not in ObjNode_dict:
                            continue
                        if ObjNode_dict[child].z_max < src_obj.z_max:
                            embedded_relations.append(utils.generate_relation(src_id, child ,'inside_express'))
                        else:
                            support_relations.append(utils.generate_relation(src_id, child , 'support'))

        else:
            support_relations, embedded_relations, hanging_objs_dict = cal_support_relations(ObjNode_dict, camera_angle)
        for rela in support_relations:
            target_obj_id, obj_id, _ = rela
            G.add_edge(target_obj_id, obj_id, label='support') # optimizer

        # hanging relationships
        hanging_relationships = cal_hanging_relationships(ObjNode_dict, hanging_objs_dict, camera_angle,
                                                          scene_high, dataset=cfg.dataset)

        # iterate graph to cal saptial relationships
        proximity_relations = []
        for node in G:
            neighbor = dict(nx.bfs_successors(G, source=node, depth_limit=1))
            if len(neighbor[node]) > 1:
                neighbor_objs = neighbor[node]
                proximity = cal_proximity_relationships(neighbor_objs, camera_angle, ObjNode_dict, scene_high)
                proximity_relations += proximity

        # added some special relations and oppo support relationships
        oppo_support_relations = []
        objects_rels = support_relations + embedded_relations + hanging_relationships
        for idx, rels in enumerate(objects_rels):
            src, tgt, rela = rels
            if rela == 'support':
                oppo_support_relations.append(utils.generate_relation(src, tgt, 'oppo_support'))

            if src == -2 or tgt == -2:
                continue
            src_label = ObjNode_dict[src].label
            tgt_label = ObjNode_dict[tgt].label

            if src_label in dictionary.added_hanging and dictionary.added_hanging[src_label] == tgt_label:
                objects_rels[idx][2] = 'hanging'
            if tgt_label in dictionary.added_hanging and dictionary.added_hanging[tgt_label] == src_label:
                objects_rels[idx][2] = 'hanging'

        # multi objects
        multi_objs_relationships = []

        # added aligned relationship
        furniture_list = list(ObjNode_dict.keys())
        aligned_furniture = find_aligned_furniture(furniture_list, ObjNode_dict, 0.065)

        for _, aligned_furni in enumerate(aligned_furniture):
            multi_objs_relationships.append([aligned_furni, 'Aligned'])

        # added in the middle of relationship
        middle_relationships = find_middle_furniture(proximity_relations, ObjNode_dict)

        # output json
        relationships_json_string = {
            'scan': scan_id,
            'camera_view': camera_view,
            'camera_position': camera_pos,
            'relationships': objects_rels + proximity_relations + oppo_support_relations,
            'multi_objs_relationships': multi_objs_relationships + middle_relationships,
        }

        # End SG latency timer (exclude visualization and IO)
        sg_latency_ms = (time.perf_counter() - t_sg_start) * 1000.0
        print(f'Latency (scene graph) [{scan_id}]: {sg_latency_ms:.2f} ms')

        np.random.shuffle(objects_rels)
        # visualize scene
        if cfg.visualize:
            vis_dataset(ObjNode_dict=ObjNode_dict,
                        relation=proximity_relations,
                        scene_path=cfg.scene_path,
                        scan_id=scan_id,
                        scene_center=scene_center)


        relationship_save[scan_id] = relationships_json_string
        objects_save[scan_id] = {"objects_info": objects_info,
                                    "inst_to_label" : inst_labels}

        print ('==> DONE')
        print('SCENE ', scan_id)
        print('OBJECTS ', len(ObjNode_dict))

        scan_path = cfg.rels_save_path / scan_id

        scan_path.mkdir(parents=True, exist_ok=True)
        print('SAVE', scan_path)
        with (scan_path / 'relationships.json').open('w') as file:
            json.dump(relationship_save, file, default=default_dump)
        with (scan_path / 'objects.json').open('w') as file:
            json.dump(objects_save, file, default=default_dump)
        print ('=====================\n')

import YOLOE.utils as yutils
 
def obj_min_max(pcd):
    # return x_min, x_max, y_min, y_max, z_min, z_max
    x_min = np.min(pcd[:,0])
    y_min = np.min(pcd[:,1])
    z_min = np.min(pcd[:,2])
    x_max = np.max(pcd[:,0])
    y_max = np.max(pcd[:,1])
    z_max = np.max(pcd[:,2])    
    
    return x_min, x_max, y_min, y_max, z_min, z_max
def edges(current_graph, frame_objs, T_w_c, depth_m):

    ### init camera ###
    camera_view, camera_pos, camera_angle = view_from_pose(T_w_c)
    scene_center = np.array([0, 0, 0])
    # x_min, x_max, y_min, y_max, z_min, z_max, scene_high = yutils.estimate_scene_height(depth_m, T_w_c)
    scene_high = yutils.estimate_scene_height(depth_m, T_w_c)
    ObjNode_dict = {}
    
    # iterate objects of graph and make object node dict

    for node, data in current_graph.nodes(data=True):
        print(node)
        print(data['data'].keys())
        if data['data']['bbox_3d']['obb'] is None:
            ObjNode_dict[node] = ObjNode(
                id=-10,
                label="NONE",
                # bbox=data['data']['bbox_3d'],
                position=np.array([0,0,0]),
                x_min=0,
                x_max=0,
                y_min=0,
                y_max=0,
                z_min=0,
                z_max=0
            )
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = obj_min_max(np.array(data['data']['points']))
            ObjNode_dict[node] = ObjNode(
                id=data['data']['track_id'],
                label="NONE",
                # bbox=data['data']['bbox_3d'],
                position=data['data']['bbox_3d']['obb']['center'],
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max
            )

    # G = nx.DiGraph()
    
    # generate object node for graph

    # added special nodes (wall camera)
    # G.add_node(-1, label='CAMERA')
    # G.add_node(-2, label='wall')

    support_relations, embedded_relations, hanging_objs_dict = cal_support_relations(ObjNode_dict, camera_angle)
    for rela in support_relations:
        target_obj_id, obj_id, _ = rela
        current_graph.add_edge(target_obj_id, obj_id, label='support') # optimizer

    # hanging relationships
    hanging_relationships = cal_hanging_relationships(ObjNode_dict, hanging_objs_dict, camera_angle,
                                                        scene_high)

    # iterate graph to cal saptial relationships
    proximity_relations = []
    for node in current_graph:
        neighbor = dict(nx.bfs_successors(current_graph, source=node, depth_limit=1))
        if len(neighbor[node]) > 1:
            neighbor_objs = neighbor[node]
            proximity = cal_proximity_relationships(neighbor_objs, camera_angle, ObjNode_dict, scene_high)
            proximity_relations += proximity

    # added some special relations and oppo support relationships
    oppo_support_relations = []
    objects_rels = support_relations + embedded_relations + hanging_relationships
    for idx, rels in enumerate(objects_rels):
        src, tgt, rela = rels
        if rela == 'support':
            oppo_support_relations.append(utils.generate_relation(src, tgt, 'oppo_support'))

        if src == -2 or tgt == -2:
            continue
        src_label = ObjNode_dict[src].label
        tgt_label = ObjNode_dict[tgt].label

        if src_label in dictionary.added_hanging and dictionary.added_hanging[src_label] == tgt_label:
            objects_rels[idx][2] = 'hanging'
        if tgt_label in dictionary.added_hanging and dictionary.added_hanging[tgt_label] == src_label:
            objects_rels[idx][2] = 'hanging'

    # multi objects
    multi_objs_relationships = []

    # added aligned relationship
    furniture_list = list(ObjNode_dict.keys())
    aligned_furniture = find_aligned_furniture(furniture_list, ObjNode_dict, 0.065)

    for _, aligned_furni in enumerate(aligned_furniture):
        multi_objs_relationships.append([aligned_furni, 'Aligned'])

    # added in the middle of relationship
    middle_relationships = find_middle_furniture(proximity_relations, ObjNode_dict)

    # output json
    relationships_json_string = {
        'camera_view': camera_view,
        'camera_position': camera_pos,
        'relationships': objects_rels + proximity_relations + oppo_support_relations,
        'multi_objs_relationships': multi_objs_relationships + middle_relationships,
    }

    # End SG latency timer (exclude visualization and IO)

    np.random.shuffle(objects_rels)
    
    print(relationships_json_string['relationships'])
    print(f"")
    
    return relationships_json_string['relationships']


if __name__ == '__main__':
    cfg = OmegaConf.create({
        'dataset': 'MultiScan',
        'scene_path': '/home/rizo/mipt_ccm/SceneVerse/multiscan_preprocessed',
        'rels_save_path': './temp_ssg',
        'visualize': True,
        'num_workers': 1,
    })

    # cfg.scene_path = Path(cfg.scene_path) / cfg.dataset / 'scan_data'
    cfg.scene_path = Path(cfg.scene_path) / cfg.dataset
    # print(cfg.scene_path)
    # cfg.scene_path = "/home/rizo/mipt_ccm/SceneVerse/multiscan_preprocessed/scan_data/pcd_with_global_alignment"

    cfg.rels_save_path = Path(cfg.rels_save_path) / cfg.dataset

    main(cfg)
