from ..ssg_data.dictionary import always_supported, hanging
from .. import ssg_utils as utils


def is_supported(target_obj, obj, camera_angle, radius_range = 0.1, threshold_of_z_rate=0.8):

    z_min = obj.z_min
    z_max = obj.z_max
    tz_max = target_obj.z_max
    tz_min = target_obj.z_min

    # overlap of z
    diff_z = z_min - tz_max
    height = z_max - z_min
    z_rate = abs(diff_z) / height

    # must be larger
    # target's bottom area must be larger than obj's bottom area
    if not utils.get_Poly_Area(target_obj.bottom_rect[:, 0:2]) > utils.get_Poly_Area(obj.bottom_rect[:, 0:2]):
        return False

    if target_obj.label == 'floor':
        if not z_min < tz_max:
            return False
    else:
        # must be higher
        # if tz_max > z_max:
        #     return False
        if z_min > (tz_max*0.05 if tz_max > 0 else tz_max*0.95): # floating
            return False
        if z_min < tz_min:
            # Reject if the object bottom is beneath the target’s bottom (object 
            # would be intersecting from below)
            return False
        if not diff_z < height*0.3:
            # Require that the vertical offset between object bottom and target top is less 
            # than 20% of the object height (prevents very large separations).
            return False

    # must be centered
    center = obj.position
    # The object’s horizontal center must lie inside the target’s bottom polygon 
    # (object must be located above the target footprint).
    if not utils.if_inPoly(target_obj.bottom_rect, center):
        return False

    if target_obj.label == 'floor':
        return 'support_express'
    else:
        if z_rate < threshold_of_z_rate :
            # object mostly on top of the target
            return 'support_express'
        elif z_rate >= threshold_of_z_rate and z_rate < 0.95:
            # object is partially embedded/penetrating the target
            return 'embed_express'
        else:
            # object is almost entirely inside/through the target
            return 'inside_express'


def optimaze_support_loops(support_relations_dict):
    # print(f'DEBUG[support]: before optimization: {support_relations_dict} support relationships')
    relationships = []
    for obj_id, tgts in support_relations_dict.items():
        if len(tgts)>1:
            positions = [tgt.position[2] for tgt in tgts]
            hightest_tgt_inedx = positions.index(max(positions))
            hightest_tgt = tgts[hightest_tgt_inedx]
            relationships.append(utils.generate_relation(hightest_tgt.id, obj_id, 'support'))
        else:
            relationships.append(utils.generate_relation(tgts[0].id, obj_id, 'support'))

    # print(f'DEBUG[support]: after optimization: {relationships} support relationships')
    return relationships

def cal_support_relations(ObjNode_list, camera_angle):
    support_relations_dict = {}
    embedded_relationships = []
    hanging_objs = {}

    for target_obj_id in ObjNode_list:
        target_obj = ObjNode_list[target_obj_id]

        for obj_id in ObjNode_list:
            obj = ObjNode_list[obj_id]

            if target_obj.id == obj.id: continue
            if target_obj.label in always_supported or obj.label in always_supported: continue
            if target_obj.label in hanging or obj.label in hanging: continue

            is_support = is_supported(target_obj, obj, camera_angle)

            if is_support:

                if is_support in ['embed_express', 'inside_express']:
                    embedded_relationships.append(utils.generate_relation(target_obj.id, obj.id, is_support))
                else:
                    if obj.id not in support_relations_dict:
                        support_relations_dict[obj.id] = [target_obj]
                    else:
                        support_relations_dict[obj.id].append(target_obj)

                hanging_objs[obj.id] = 1

    return optimaze_support_loops(support_relations_dict), embedded_relationships, hanging_objs
