from os import getenv

import numpy as np
import torch
from pyquaternion import Quaternion
from transforms3d.euler import euler2mat, euler2quat, quat2mat, mat2euler
from math_helper import rot_matrix_between_vectors, angle_between_vectors, quaternion_between_vectors, \
    normalize_batch, rad2deg, normalize, rotation_between_triangles
import quaternion

from utils import torch_tile


class KeyframeIdx:
    root = 0
    spine0 = 1
    spine1 = 2
    spine2 = 3
    spine3 = 4
    neck = 5
    head = 6
    head_end = 7
    rshoulder = 8
    rscap = 9
    rupperarm = 10
    rlowerarm = 11
    rwristtwist = 12
    rwrist = 13
    rindex1 = 14
    rindex2 = 15
    rindex3 = 16
    rindex3_end = 17
    rring1 = 18
    rring2 = 19
    rring3 = 20
    rring3_end = 21
    rmiddle1 = 22
    rmiddle2 = 23
    rmiddle3 = 24
    rmiddle3_end = 25
    rpinky1 = 26
    rpinky2 = 27
    rpinky3 = 28
    rpinky3_end = 29
    rthumb0 = 30
    rthumb1 = 31
    rthumb2 = 32
    rthumb3 = 33
    rthumb3_end = 34
    lshoulder = 35
    lscap = 36
    lupperarm = 37
    llowerarm = 38
    lwristtwist = 39
    lwrist = 40
    lindex1 = 41
    lindex2 = 42
    lindex3 = 43
    lindex3_end = 44
    lring1 = 45
    lring2 = 46
    lring3 = 47
    lring3_end = 48
    lmiddle1 = 49
    lmiddle2 = 50
    lmiddle3 = 51
    lmiddle3_end = 52
    lpinky1 = 53
    lpinky2 = 54
    lpinky3 = 55
    lpinky3_end = 56
    lthumb0 = 57
    lthumb1 = 58
    lthumb2 = 59
    lthumb3 = 60
    lthumb3_end = 61
    rupperleg = 62
    rlowerleg = 63
    rfoot = 64
    rfootball = 65
    rfootball_right = 66
    rfootball_end = 67
    lupperleg = 68
    llowerleg = 69
    lfoot = 70
    lfootball = 71
    lfootball_left = 72
    lfootball_end = 73

    all = {
        'root': root,
        'spine0': spine0,
        'spine1': spine1,
        'spine2': spine2,
        'spine3': spine3,
        'neck': neck,
        'head': head,
        'head_end': head_end,
        'rshoulder': rshoulder,
        'rscap': rscap,
        'rupperarm': rupperarm,
        'rlowerarm': rlowerarm,
        'rwristtwist': rwristtwist,
        'rwrist': rwrist,
        'rindex1': rindex1,
        'rindex2': rindex2,
        'rindex3': rindex3,
        'rindex3_end': rindex3_end,
        'rring1': rring1,
        'rring2': rring2,
        'rring3': rring3,
        'rring3_end': rring3_end,
        'rmiddle1': rmiddle1,
        'rmiddle2': rmiddle2,
        'rmiddle3': rmiddle3,
        'rmiddle3_end': rmiddle3_end,
        'rpinky1': rpinky1,
        'rpinky2': rpinky2,
        'rpinky3': rpinky3,
        'rpinky3_end': rpinky3_end,
        'rthumb0': rthumb0,
        'rthumb1': rthumb1,
        'rthumb2': rthumb2,
        'rthumb3': rthumb3,
        'rthumb3_end': rthumb3_end,
        'lshoulder': lshoulder,
        'lscap': lscap,
        'lupperarm': lupperarm,
        'llowerarm': llowerarm,
        'lwristtwist': lwristtwist,
        'lwrist': lwrist,
        'lindex1': lindex1,
        'lindex2': lindex2,
        'lindex3': lindex3,
        'lindex3_end': lindex3_end,
        'lring1': lring1,
        'lring2': lring2,
        'lring3': lring3,
        'lring3_end': lring3_end,
        'lmiddle1': lmiddle1,
        'lmiddle2': lmiddle2,
        'lmiddle3': lmiddle3,
        'lmiddle3_end': lmiddle3_end,
        'lpinky1': lpinky1,
        'lpinky2': lpinky2,
        'lpinky3': lpinky3,
        'lpinky3_end': lpinky3_end,
        'lthumb0': lthumb0,
        'lthumb1': lthumb1,
        'lthumb2': lthumb2,
        'lthumb3': lthumb3,
        'lthumb3_end': lthumb3_end,
        'rupperleg': rupperleg,
        'rlowerleg': rlowerleg,
        'rfoot': rfoot,
        'rfootball': rfootball,
        'rfootball_right': rfootball_right,
        'rfootball_end': rfootball_end,
        'lupperleg': lupperleg,
        'llowerleg': llowerleg,
        'lfoot': lfoot,
        'lfootball': lfootball,
        'lfootball_left': lfootball_left,
        'lfootball_end': lfootball_end,
    }

    @classmethod
    def exportable_joints(cls):
        result = {}
        for joint_name, joint_idx in cls.all.items():
            if not joint_name.endswith("_end"):
                result[joint_name] = joint_idx

        return result

    @classmethod
    def get_hand_joints(cls):
        joint_l = [cls.lwrist]
        joint_r = [cls.rwrist]
        for j_name, j_idx in cls.all.items():
            if "index" in j_name or "middle" in j_name or "ring" in j_name or "pinky" in j_name or "thumb" in j_name:
                if j_name.startswith("l"):
                    joint_l.append(j_idx)
                else:
                    joint_r.append(j_idx)
        return joint_l, joint_r

    @classmethod
    def get_finger_joints(cls):
        joint_l, joint_r = cls.get_hand_joints()
        return joint_l[1:], joint_r[1:]

    @classmethod
    def get_non_finger_joints(cls):
        l_hands, r_hands = cls.get_hand_joints()
        # keep wrists
        l_hands.remove(l_hands[0])
        r_hands.remove(r_hands[0])

        all = cls.all.copy()
        all_new = {}
        for j_name, j_idx in all.items():
            if not (j_idx in l_hands or j_idx in r_hands):
                all_new[j_name] = j_idx

        return all_new


class Bone:
    Idx = KeyframeIdx
    _joint_names = None

    def __init__(self, end, parent, normalized_length):
        self.end = end
        self.parent = parent
        self.children = []
        self.normalized_length = normalized_length
        if self.parent is not None:
            self.start = parent.end
            self.parent.register_child(self)
        else:
            self.start = None

    @classmethod
    def build_hierarchy(cls):
        root = cls('root', None, 0)
        spine0 = cls('spine0', root, 0.0)
        spine1 = cls('spine1', spine0, 0.0)
        spine2 = cls('spine2', spine1, 0.0)
        spine3 = cls('spine3', spine2, 0.0)
        neck = cls('neck', spine3, 0.0)
        head = cls('head', neck, 0.0)
        head_end = cls('head_end', head, 0.0)
        rshoulder = cls('rshoulder', spine3, 0.0)
        rscap = cls('rscap', rshoulder, 0.0)
        rupperarm = cls('rupperarm', rscap, 0.0)
        rlowerarm = cls('rlowerarm', rupperarm, 0.0)
        rwristtwist = cls('rwristtwist', rlowerarm, 0.0)
        rwrist = cls('rwrist', rwristtwist, 0.0)
        rindex1 = cls('rindex1', rwrist, 0.0)
        rindex2 = cls('rindex2', rindex1, 0.0)
        rindex3 = cls('rindex3', rindex2, 0.0)
        rindex3_end = cls('rindex3_end', rindex3, 0.0)
        rring1 = cls('rring1', rwrist, 0.0)
        rring2 = cls('rring2', rring1, 0.0)
        rring3 = cls('rring3', rring2, 0.0)
        rring3_end = cls('rring3_end', rring3, 0.0)
        rmiddle1 = cls('rmiddle1', rwrist, 0.0)
        rmiddle2 = cls('rmiddle2', rmiddle1, 0.0)
        rmiddle3 = cls('rmiddle3', rmiddle2, 0.0)
        rmiddle3_end = cls('rmiddle3_end', rmiddle3, 0.0)
        rpinky1 = cls('rpinky1', rwrist, 0.0)
        rpinky2 = cls('rpinky2', rpinky1, 0.0)
        rpinky3 = cls('rpinky3', rpinky2, 0.0)
        rpinky3_end = cls('rpinky3_end', rpinky3, 0.0)
        rthumb0 = cls('rthumb0', rwrist, 0.0)
        rthumb1 = cls('rthumb1', rthumb0, 0.0)
        rthumb2 = cls('rthumb2', rthumb1, 0.0)
        rthumb3 = cls('rthumb3', rthumb2, 0.0)
        rthumb3_end = cls('rthumb3_end', rthumb3, 0.0)
        lshoulder = cls('lshoulder', spine3, 0.0)
        lscap = cls('lscap', lshoulder, 0.0)
        lupperarm = cls('lupperarm', lscap, 0.0)
        llowerarm = cls('llowerarm', lupperarm, 0.0)
        lwristtwist = cls('lwristtwist', llowerarm, 0.0)
        lwrist = cls('lwrist', lwristtwist, 0.0)
        lindex1 = cls('lindex1', lwrist, 0.0)
        lindex2 = cls('lindex2', lindex1, 0.0)
        lindex3 = cls('lindex3', lindex2, 0.0)
        lindex3_end = cls('lindex3_end', lindex3, 0.0)
        lring1 = cls('lring1', lwrist, 0.0)
        lring2 = cls('lring2', lring1, 0.0)
        lring3 = cls('lring3', lring2, 0.0)
        lring3_end = cls('lring3_end', lring3, 0.0)
        lmiddle1 = cls('lmiddle1', lwrist, 0.0)
        lmiddle2 = cls('lmiddle2', lmiddle1, 0.0)
        lmiddle3 = cls('lmiddle3', lmiddle2, 0.0)
        lmiddle3_end = cls('lmiddle3_end', lmiddle3, 0.0)
        lpinky1 = cls('lpinky1', lwrist, 0.0)
        lpinky2 = cls('lpinky2', lpinky1, 0.0)
        lpinky3 = cls('lpinky3', lpinky2, 0.0)
        lpinky3_end = cls('lpinky3_end', lpinky3, 0.0)
        lthumb0 = cls('lthumb0', lwrist, 0.0)
        lthumb1 = cls('lthumb1', lthumb0, 0.0)
        lthumb2 = cls('lthumb2', lthumb1, 0.0)
        lthumb3 = cls('lthumb3', lthumb2, 0.0)
        lthumb3_end = cls('lthumb3_end', lthumb3, 0.0)
        rupperleg = cls('rupperleg', root, 0.0)
        rlowerleg = cls('rlowerleg', rupperleg, 0.0)
        rfoot = cls('rfoot', rlowerleg, 0.0)
        rfootball = cls('rfootball', rfoot, 0.0)
        rfootball_right = cls('rfootball_right', rfootball, 0.0)
        rfootball_end = cls('rfootball_end', rfootball, 0.0)
        lupperleg = cls('lupperleg', root, 0.0)
        llowerleg = cls('llowerleg', lupperleg, 0.0)
        lfoot = cls('lfoot', llowerleg, 0.0)
        lfootball = cls('lfootball', lfoot, 0.0)
        lfootball_left = cls('lfootball_left', lfootball, 0.0)
        lfootball_end = cls('lfootball_end', lfootball, 0.0)

        return root

    @classmethod
    def get_shoulder_length(cls):
        return 0.15

    @classmethod
    def get_upper_arm_length(cls):
        return 0.3

    @classmethod
    def get_lower_arm_length(cls):
        return 0.3

    @classmethod
    def get_neck_length(cls):
        return 0.1

    def register_child(self, child):
        self.children.append(child)

    @classmethod
    def __recursive_add_joint(cls, joint, result):
        result.append([joint.start, joint.end])

        for child in joint.children:
            cls.__recursive_add_joint(child, result)

    @classmethod
    def joint_names(cls):
        if cls._joint_names is None:
            result = []
            cls.__recursive_add_joint(cls.build_hierarchy(), result)
            cls._joint_names = []
            for bone in result:
                cls._joint_names.append(bone[1])
        return cls._joint_names

    def __str__(self):
        return f"{self.end} (child of {self.start if self.start is not None else 'None'})"


class Skeleton:
    Idx = KeyframeIdx
    Bone_set = Bone

    def __init__(self, global_positions):
        self.p = global_positions

    def height(self):
        return self.p[self.Idx.head][1]

    def shoulder_width(self):
        return np.linalg.norm(self.p[self.Idx.lshoulder] - self.p[self.Idx.rshoulder])

    def arm_length(self):
        return np.linalg.norm(self.p[self.Idx.rshoulder] - self.p[self.Idx.rwrist])

    def bone_offset_vector(self):
        def __recursive_add_child_offset(_v, bone):
            if bone.start is None:
                _v[self.Idx.all[bone.end]] = self.p[self.Idx.all[bone.end]]
            else:
                distance = self.p[self.Idx.all[bone.end]] - self.p[self.Idx.all[bone.start]]
                _v[self.Idx.all[bone.end]] = distance

            for child in bone.children:
                __recursive_add_child_offset(_v, child)

        v = np.zeros((len(self.Idx.all), 3))
        hierarchy = self.Bone_set.build_hierarchy()
        __recursive_add_child_offset(v, hierarchy)
        return v

    @classmethod
    def parent_idx_vector(cls):
        def __recursive_add_children(bone, parent_index, result):
            result[cls.Idx.all[bone.end]] = parent_index
            bone_idx = cls.Idx.all[bone.end]

            for child in bone.children:
                __recursive_add_children(child, bone_idx, result)

        hierarchy = cls.Bone_set.build_hierarchy()
        parent_indices = np.zeros(len(cls.Idx.all), dtype=np.int32)
        __recursive_add_children(hierarchy, cls.Idx.all['head'], parent_indices)
        parent_indices[cls.Idx.all['head']] = -1
        return parent_indices

    @classmethod
    def child_idx_vector(cls):
        parent_idx = cls.parent_idx_vector()
        child_idx = []
        for i in range(len(parent_idx)):
            child_idx.append([])

        for c_idx, p_idx in enumerate(parent_idx):
            if p_idx > -1 and c_idx > 0:
                child_idx[p_idx].append(c_idx)

        return child_idx

    # remove reference_p
    @classmethod
    def forward_kinematics_torch(cls, bone_offsets, parent_idx, root_offset, qrot):
        """
        performs forward kinematics and returns global positions as vectors and orientations as quaternions
        """
        p = torch.zeros((qrot.shape[0], qrot.shape[1], 3)).to(qrot.device)
        qrot_global = torch.zeros_like(qrot).to(qrot.device)
        _bone_offsets = torch.zeros_like(p).to(qrot.device)
        if type(bone_offsets) is np.ndarray:
            _bone_offsets[:] += torch.from_numpy(bone_offsets).type(torch.float32).to(qrot.device)
        else:
            _bone_offsets[:] += bone_offsets.float().to(qrot.device)

        for joint_idx, parent_idx in enumerate(parent_idx):
            if parent_idx < 0:
                qrot_global[:, joint_idx] = qrot[:, joint_idx]
                p[:, joint_idx] = root_offset
            else:
                qrot_global[:, joint_idx] = quaternion.qmul(qrot_global[:, parent_idx], qrot[:, joint_idx])
                p[:, joint_idx] = p[:, parent_idx] + quaternion.qrot(qrot_global[:, parent_idx],
                                                                     _bone_offsets[:, joint_idx])

        return p, qrot_global

    def positions_to_local_rot(self, p_g):
        r_l = torch.zeros_like(p_g)
        quat_g = torch.zeros((p_g.shape[0], p_g.shape[1], 4), device=p_g.device)
        quat_l = torch.zeros((p_g.shape[0], p_g.shape[1], 4), device=p_g.device)
        quat_g[:, :, 0] += 1.0
        quat_l[:, :, 0] += 1.0

        bone_offsets = torch.from_numpy(self.bone_offset_vector() / 100.0).float().to(p_g.device)
        child_idx_vector = self.child_idx_vector()
        parent_idx_vector = self.parent_idx_vector()

        depend_on_parent_idx = [1, 2, 3, 4]

        for joint_idx, parent_idx in enumerate(parent_idx_vector):
            parent_g = quat_g[:, parent_idx]

            use_par = joint_idx in depend_on_parent_idx

            my_pos = [torch.zeros_like(p_g[:, joint_idx])]
            ref_pos = [torch.zeros_like(p_g[:, joint_idx])]
            for child_idx in child_idx_vector[joint_idx]:
                ref_bone_dir = torch_tile(bone_offsets[child_idx][None, :].clone(), dim=0, n_tile=len(p_g))
                if torch.norm(ref_bone_dir) == 0.0:
                    continue
                my_pos.append(p_g[:, child_idx] - p_g[:, joint_idx])
                if use_par:
                    ref_bone_dir = quaternion.qrot(parent_g, ref_bone_dir)
                ref_pos.append(ref_bone_dir)

            finger_knuckles = []
            # extrimities_start = [Id.rupperarm, Id.lupperarm, Id.rupperleg, Id.lupperleg]
            extrimities_start = []
            if len(my_pos) == 2 and (joint_idx in extrimities_start or joint_idx in finger_knuckles):
                child_idx = child_idx_vector[joint_idx][0]
                grand_child_idx = child_idx_vector[child_idx][0]
                my_pos.append(p_g[:, grand_child_idx] - p_g[:, joint_idx])
                b_off = bone_offsets[grand_child_idx] + bone_offsets[child_idx]
                ref_bone_dir = torch_tile(b_off[None, :], dim=0, n_tile=len(p_g))
                ref_pos.append(ref_bone_dir)

            if len(my_pos) > 2:
                if use_par:
                    quat_l[:, joint_idx] = rotation_between_triangles(ref_pos[:3], my_pos[:3])
                    r_l[:, joint_idx] = quaternion.qeuler(quat_l[:, joint_idx])
                    quat_g[:, joint_idx] = quaternion.qmul(parent_g, quat_l[:, joint_idx])
                else:
                    quat_g[:, joint_idx] = rotation_between_triangles(ref_pos[:3], my_pos[:3])
                    quat_l[:, joint_idx] = quaternion.qmul(quaternion.inverse(parent_g), quat_g[:, joint_idx])
                    r_l[:, joint_idx] = quaternion.qeuler(quat_l[:, joint_idx])
            elif len(my_pos) == 2:
                ref_bone_dir = normalize_batch(ref_pos[1])
                child_idx = child_idx_vector[joint_idx][0]
                anim_bone_dir = normalize_batch(p_g[:, child_idx] - p_g[:, joint_idx])
                if use_par:
                    quat_l[:, joint_idx] = quaternion.quaternion_between_vectors_torch(ref_bone_dir, anim_bone_dir)
                    r_l[:, joint_idx] = quaternion.qeuler(quat_l[:, joint_idx])
                    quat_g[:, joint_idx] = quaternion.qmul(parent_g, quat_l[:, joint_idx])
                else:
                    quat_g[:, joint_idx] = quaternion.quaternion_between_vectors_torch(ref_bone_dir, anim_bone_dir)
                    quat_l[:, joint_idx] = quaternion.qmul(quaternion.inverse(parent_g), quat_g[:, joint_idx])
                    r_l[:, joint_idx] = quaternion.qeuler(quat_l[:, joint_idx])
            else:
                quat_g[:, joint_idx] = parent_g.clone()

        return rad2deg(r_l), quat_l

    @classmethod
    def parent_idx_vector(cls):
        def __recursive_add_children(bone, parent_index, result):
            result[cls.Idx.all[bone.end]] = parent_index
            bone_idx = cls.Idx.all[bone.end]

            for child in bone.children:
                __recursive_add_children(child, bone_idx, result)

        hierarchy = cls.Bone_set.build_hierarchy()
        parent_indices = np.zeros(len(cls.Idx.all), dtype=np.int32)
        __recursive_add_children(hierarchy, cls.Idx.all['root'], parent_indices)
        parent_indices[cls.Idx.all['root']] = -1
        return parent_indices

    @classmethod
    def parent_idx_vector_reduced(cls):
        def __recursive_add_children(bone, parent_index, result):
            if bone.end in indices:
                result[indices[bone.end]] = parent_index
                bone_idx = indices[bone.end]
            else:
                # result[cls.Idx.all[bone.end]] = -1
                bone_idx = parent_index

            for child in bone.children:
                __recursive_add_children(child, bone_idx, result)

        hierarchy = cls.Bone_set.build_hierarchy()
        # indices = cls.Idx.reduced_hands()
        indices = cls.Idx.all
        # parent_indices = np.zeros(len(indices), dtype=np.int32)
        parent_indices = {}
        __recursive_add_children(hierarchy, indices['root'], parent_indices)
        parent_indices[indices['root']] = -1
        return parent_indices


class Keyframe:
    Idx = KeyframeIdx
    __up = None
    __forward = None
    __right = None
    __up_local = None
    __forward_local = None
    __right_local = None

    # TODO remove default l/r_hand_visible params
    def __init__(self, global_positions, object_rotations, r_local, time, l_hand_visible=True, r_hand_visible=True):
        self.p = global_positions
        self.r = object_rotations
        self.r_local = r_local
        self.t = time
        self.l_hand_visible = l_hand_visible
        self.r_hand_visible = r_hand_visible

    @classmethod
    def __batch_torch_recursive_recover_local_rotation(cls, bone, skeleton, r, r_local):
        deg2rad = 3.1415 / 180
        rad2deg = 180 / 3.1415
        joint_idx = cls.Idx.all[bone.end]
        if bone.start is None:
            r_local[:, cls.Idx.all[bone.end]] = r[:, cls.Idx.all[bone.end]]
        else:
            # what is the bone vector - from parent joint to child joint
            parent_idx = cls.Idx.all[bone.start]

            parent_rot = quaternion.euler_to_quaternion_torch(r[:, parent_idx] * deg2rad, 'xyz')
            self_rot = quaternion.euler_to_quaternion_torch(r[:, joint_idx] * deg2rad, 'xyz')
            local_rot = quaternion.qmul(quaternion.inverse(parent_rot), self_rot)
            r_local[:, joint_idx] = rad2deg * quaternion.qeuler(local_rot, order="xyz")

        for child in bone.children:
            cls.__batch_torch_recursive_recover_local_rotation(child, skeleton, r, r_local)

    @classmethod
    def batch_torch_recover_local_rotations(cls, skeleton, r):
        r_local = torch.empty_like(r)
        root = skeleton.Bone_set.build_hierarchy()
        cls.__batch_torch_recursive_recover_local_rotation(root, skeleton, r, r_local)

        return r_local

    def __recursive_recover_local_rotation(self, bone):
        joint_idx = self.Idx.all[bone.end]
        if bone.start is None:
            self.r_local[self.Idx.all[bone.end]] = self.r[self.Idx.all[bone.end]]
        else:
            # what is the bone vector - from parent joint to child joint
            parent_idx = self.Idx.all[bone.start]
            parent_rot = quaternion.euler_to_quaternion(np.deg2rad(self.r[parent_idx]), 'xyz')
            self_rot = quaternion.euler_to_quaternion(np.deg2rad(self.r[joint_idx]), 'xyz')
            local_rot = quaternion.qmul_np(quaternion.inverse_np(parent_rot), self_rot)
            self.r_local[joint_idx] = np.rad2deg(quaternion.qeuler_np(local_rot, order="xyz"))

        for child in bone.children:
            self.__recursive_recover_local_rotation(child)

    def recover_local_rotations(self, skeleton):
        if self.r_local is None:
            self.r_local = np.empty_like(self.r)
        root = skeleton.Bone_set.build_hierarchy()
        self.__recursive_recover_local_rotation(root)

    @classmethod
    def from_numpy(cls, x):
        if x.shape[1] == 7:
            return cls(x[:, :3], x[:, 3:6], np.ones_like(x[:, 3:6]), x[0, 6], x[1, 6] > 0.00001, x[2, 6] > 0.00001)
        else:
            return cls(x[:, :3], x[:, 3:6], x[:, 6:9], x[0, 9], x[1, 9] > 0.00001, x[2, 9] > 0.00001)

    @classmethod
    def from_numpy_arr(cls, x_arr):
        for x in x_arr:
            if x.shape[1] == 7:
                yield cls(x[:, :3], x[:, 3:6], np.ones_like(x[:, 3:6]), x[0, 6], x[1, 6] > 0.00001, x[2, 6] > 0.00001)
            else:
                yield cls(x[:, :3], x[:, 3:6], x[:, 6:9], x[0, 9], x[1, 9] > 0.00001, x[2, 9] > 0.00001)

    def rotate_vector(self, v, local=False):
        rot = self.r_local if local else self.r
        result = np.empty((len(rot), 3))
        for i, r in enumerate(rot):
            f = normalize(euler2mat(*np.deg2rad(r)).dot(v))
            result[i] = f
        return result

    def reset_cache(self):
        self.__right = None
        self.__forward = None
        self.__up = None

    def forward(self, local=False):
        if local:
            if self.__forward_local is None:
                self.__forward_local = self.rotate_vector(np.array([0, 0, 1]), local)
            return self.__forward_local
        else:
            if self.__forward is None:
                self.__forward = self.rotate_vector(np.array([0, 0, 1]), local)
            return self.__forward

    def up(self, local=False):
        if local:
            if self.__up_local is None:
                self.__up_local = self.rotate_vector(np.array([0, 1, 0]), local)
            return self.__up_local
        else:
            if self.__up is None:
                self.__up = self.rotate_vector(np.array([0, 1, 0]), local)
            return self.__up

    def right(self, local=False):
        if local:
            if self.__right_local is None:
                self.__right_local = self.rotate_vector(np.array([1, 0, 0]), local)
            return self.__right_local
        else:
            if self.__right is None:
                self.__right = self.rotate_vector(np.array([1, 0, 0]), local)
            return self.__right

    # joint velocities
    def velocity(self, other):
        delta_t = max(0.007, self.t - other.t)
        delta_p = self.p - other.p
        return delta_p / delta_t

    # joint angular velocity quaternions
    def angular_speed(self, other):
        delta_t = max(0.0001, self.t - other.t)
        result = np.empty((len(self.r), 3))
        for i, r in enumerate(self.r):
            quat_r = Quaternion(*euler2quat(*(np.deg2rad(r))))
            quat_other = Quaternion(*euler2quat(*np.deg2rad(other.r[i])))
            delta_r = quat_r.inverse * quat_other
            result[i] = delta_r.vector / delta_t

        return result

    def to_numpy(self):
        result = np.empty((len(self.p), 10), dtype=np.float32)
        result[:, :3] = self.p
        result[:, 3:6] = self.r
        result[:, 6:9] = self.r_local
        result[0, 9] = self.t
        result[1, 9] = 1 if self.l_hand_visible else 0
        result[2, 9] = 1 if self.r_hand_visible else 0
        return result

    def bone_p_diff(self, joints):
        return self.p[self.Idx.all[joints[1]]] - self.p[self.Idx.all[joints[0]]]

    def bone_direction(self, joints):
        p_diff = self.bone_p_diff(joints)
        if np.linalg.norm(p_diff) == 0:
            return p_diff
        return p_diff / np.linalg.norm(p_diff)

    def bone_length(self, joints):
        return np.linalg.norm(self.bone_p_diff(joints))

    def position(self, joint_name):
        return self.p[self.Idx.all[joint_name]]

    def elbow_angle(self, shoulderIdx, elbowIdx, wristIdx):
        # shoulder hand distance
        s_h_diff = self.p[wristIdx] - self.p[shoulderIdx]
        s_h_axis = normalize(s_h_diff)

        inverse_s_h_rot = np.linalg.inv(rot_matrix_between_vectors(np.array([0, 0, 1]), s_h_axis))

        s_h_mid = s_h_diff * 0.5 + self.p[shoulderIdx]
        elbow_dir = self.p[elbowIdx] - s_h_mid
        elbow_dir = normalize(inverse_s_h_rot.dot(elbow_dir))

        elbow_ref_down = np.array([0, -1, 0])

        angle = angle_between_vectors(elbow_ref_down, elbow_dir)

        elbow_ref_right = np.array([1, 0, 0])

        if elbow_ref_right.dot(elbow_dir) < 0:
            angle *= -1.0

        return angle

    def elbow_angle_r(self):
        angle = self.elbow_angle(self.Idx.rshoulder, self.Idx.relbow, self.Idx.rwrist)
        return angle

    def elbow_angle_l(self):
        angle = self.elbow_angle(self.Idx.lshoulder, self.Idx.lelbow, self.Idx.lwrist)
        return angle

    def _local_shoulder_neck_offset(self, shoulder_idx):
        inverse_neck_rot = quat2mat(Quaternion(*euler2quat(*np.deg2rad(self.r[self.Idx.neck]))).inverse)
        neck_shoulder = normalize(self.p[shoulder_idx] - self.p[self.Idx.neck])
        neck_shoulder = inverse_neck_rot.dot(neck_shoulder)
        return neck_shoulder

    def local_shoulder_neck_offset_l(self):
        return self._local_shoulder_neck_offset(self.Idx.lshoulder)

    def local_shoulder_neck_offset_r(self):
        return self._local_shoulder_neck_offset(self.Idx.rshoulder)


def get_path_data():
    return getenv("PATH_DATA")


path_data = get_path_data()
