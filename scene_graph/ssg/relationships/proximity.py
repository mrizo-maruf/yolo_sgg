import numpy as np
import itertools
import time
from .. import ssg_utils as utils


def get_direction(src_obj, tgt_obj):

    sx, sy = src_obj
    tx, ty = tgt_obj

    y = np.array((tx - sx, ty - sy))
    y = y / np.linalg.norm(y)

    angle_d = utils.get_theta(y, [1, 0])

    direction = round(angle_d / 30)


    if ty > sy : # tgt is up
        if direction == 0: return "3"
        elif direction == 1: return "2"
        elif direction == 2: return "1"
        elif direction == 3: return "12"
        elif direction == 4: return "11"
        elif direction == 5: return "10"
        elif direction == 6: return "9"
    else:
        if direction == 0: return "3"
        elif direction == 1: return "4"
        elif direction == 2: return "5"
        elif direction == 3: return "6"
        elif direction == 4: return "7"
        elif direction == 5: return "8"
        elif direction == 6: return "9"

def get_oppo_direction(direction):

    if direction in ['2', '3', '4']:
        return 'to the left of'
    elif direction in ['8', '9', '10']:
        return 'to the right of'
    elif direction in ['11','12','1']:
        return 'behind'
    else:
        return 'in front of'

def get_space_relations(src, tgt):
    overlap_point = 0
    tgt_rect = tgt.bottom_rect
    for point in tgt_rect:
        if utils.if_inPoly_fast(src.bottom_rect, point): # have overlap
            overlap_point += 1
        
        # if utils.if_inPoly(src.bottom_rect, point) != utils.if_inPoly_fast(src.bottom_rect, point):
        #     print("[ssg_utils.proximity.get_space_relations]BUG in if_inPoly_fast is not same as if_inPoly")

    return overlap_point

def get_distance(src, tgt):

    # print(f"src: {src.position[:2]}, type: {type(src.position[:2])}")
    # dis_of_center = utils.euclideanDistance(src.position[:2], tgt.position[:2], 2)
    # src_w = utils.euclideanDistance(src.position[:2], src.bottom_rect[0][:2], 2)
    # tgt_w = utils.euclideanDistance(tgt.position[:2], tgt.bottom_rect[0][:2], 2)
    
    # faster? 
    dis_of_center2 = np.linalg.norm(np.array(src.position[:2]) - np.array(tgt.position[:2]))
    src_w2 = np.linalg.norm(src.position[:2] - src.bottom_rect[0][:2])
    tgt_w2 = np.linalg.norm(tgt.position[:2] - tgt.bottom_rect[0][:2])
    
    # if src_w != src_w2 or tgt_w != tgt_w2:
    #     print("[ssg_utils.proximity.get_distance]BUG in euclideanDistance is not same as np.linalg.norm")
    #     print(f"slow src_w: {src_w}, fast src_w: {src_w2}")
    #     print(f"slow tgt_w: {tgt_w}, fast tgt_w: {tgt_w2}")
    # if dis_of_center != dis_of_center2:
    #     print("[ssg_utils.proximity.get_distance]BUG in euclideanDistance is not same as np.linalg.norm")
    #     print(f"slow: {dis_of_center}, fast: {dis_of_center2}")
    # print(f"slow: {dis_of_center}, fast: {dis_of_center2}")

    # changing for closeness
    return dis_of_center2 > 4 * (src_w2 + tgt_w2)

def cal_proximity_relationships(neighbor_objs_id, camera_angle, ObjNode_dict, scene_high):
    # t_total_start = time.perf_counter()
    # timings = {
    #     'setup': 0,
    #     'get_space_relations': 0,
    #     'cw_rotate': 0,
    #     'get_direction': 0,
    #     'get_distance': 0,
    #     'generate_relation': 0,
    # }
    
    proximity_relations = []

    relations = ''

    # print('neighbor_objs_id ', neighbor_objs_id)
    # t_start = time.perf_counter()
    neighbor_objs_id_list = [i for i in range(len(neighbor_objs_id))]
    combinations = list(itertools.combinations(neighbor_objs_id_list, 2))
    # timings['setup'] = (time.perf_counter() - t_start) * 1000

    # print('combinations ', combinations)
    for combination in combinations:

        src_idx, tgt_idx = combination
        src = neighbor_objs_id[src_idx]
        tgt = neighbor_objs_id[tgt_idx]

        # is overlap
        # t_start = time.perf_counter()
        overlap_points = get_space_relations(src=ObjNode_dict[src], tgt=ObjNode_dict[tgt])
        # timings['get_space_relations'] += (time.perf_counter() - t_start) * 1000

        if overlap_points > 0 :
            # bulid in
            if overlap_points >=3:
                relations = 'under'
            # close to
            else:
                relations = 'close to'
            # t_start = time.perf_counter()
            proximity_relations.append(utils.generate_relation(ObjNode_dict[src].id, ObjNode_dict[tgt].id, relations))
            proximity_relations.append(utils.generate_relation(ObjNode_dict[tgt].id, ObjNode_dict[src].id, relations))
            # timings['generate_relation'] += (time.perf_counter() - t_start) * 1000

        else:
            # direction
            src_obj_center = ObjNode_dict[src].position
            tgt_obj_center = ObjNode_dict[tgt].position

            # t_start = time.perf_counter()
            src_obj_center_new = utils.cw_rotate(src_obj_center, camera_angle)
            tgt_obj_center_new = utils.cw_rotate(tgt_obj_center, camera_angle)
            # timings['cw_rotate'] += (time.perf_counter() - t_start) * 1000

            if src_obj_center_new == tgt_obj_center_new:
                print ('src_obj_center_new == tgt_obj_center_new ', ObjNode_dict[src].id , ObjNode_dict[tgt].id)
                break
            
            # t_start = time.perf_counter()
            direction = get_direction(src_obj_center_new, tgt_obj_center_new)
            # timings['get_direction'] += (time.perf_counter() - t_start) * 1000

            oppo_direction = get_oppo_direction(direction)
            
            # t_start = time.perf_counter()
            is_far = get_distance(src=ObjNode_dict[src], tgt=ObjNode_dict[tgt])
            # timings['get_distance'] += (time.perf_counter() - t_start) * 1000
            
            if is_far:
                relations = direction + " o'clock direction far from"

            else:
                relations = direction + " o'clock direction near"
            
            # t_start = time.perf_counter()
            proximity_relations.append([ObjNode_dict[tgt].id, ObjNode_dict[src].id, relations])
            if oppo_direction is not None:
                proximity_relations.append([ObjNode_dict[src].id, ObjNode_dict[tgt].id, oppo_direction])
            # timings['generate_relation'] += (time.perf_counter() - t_start) * 1000

    # Calculate total time
    # total_time = (time.perf_counter() - t_total_start) * 1000
    
    # # Print timing breakdown
    # n_objs = len(neighbor_objs_id)
    # n_pairs = len(combinations)
    # print(f"    [proximity] Objects: {n_objs}, Pairs: {n_pairs}, Total: {total_time:.2f} ms")
    # if n_pairs > 0:
    #     print(f"      Setup:              {timings['setup']:6.2f} ms")
    #     print(f"      get_space_relations: {timings['get_space_relations']:6.2f} ms ({timings['get_space_relations']/n_pairs:.3f} ms/pair)")
    #     print(f"      cw_rotate:          {timings['cw_rotate']:6.2f} ms ({timings['cw_rotate']/n_pairs:.3f} ms/pair)")
    #     print(f"      get_direction:      {timings['get_direction']:6.2f} ms ({timings['get_direction']/n_pairs:.3f} ms/pair)")
    #     print(f"      get_distance:       {timings['get_distance']:6.2f} ms ({timings['get_distance']/n_pairs:.3f} ms/pair)")
    #     print(f"      generate_relation:  {timings['generate_relation']:6.2f} ms ({timings['generate_relation']/n_pairs:.3f} ms/pair)")
    
    return proximity_relations
