# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import math_helper
import quaternion
from utils import torch_tile
from math_helper import normalize_batch


# deprecated
def bone_dir_to_pose(joint_mask, correct_positions, estimated_dirs, parent_indices, bone_lengths, norm_dir=True):
    estimated_pose = torch.zeros_like(correct_positions)
    joint_mask_3 = torch_tile(joint_mask, 1, 3)
    inv_joint_mask_3 = joint_mask_3 == False
    estimated_pose[inv_joint_mask_3] += correct_positions[inv_joint_mask_3]

    for idx, par_idx in enumerate(parent_indices):
        if par_idx < 0:
            continue
        i0 = 3 * idx
        i1 = 3 * (idx + 1)
        vec = estimated_dirs[:, i0:i1]
        if norm_dir:
            vec = normalize_batch(vec, 1)
        vec *= bone_lengths[idx]
        off = estimated_pose[:, 3 * par_idx:3 * (par_idx + 1)]
        mask = joint_mask_3[:, i0:i1]
        estimated_pose[:, i0:i1][mask] = (off + vec)[mask]

    return estimated_pose


# deprecated
def combine_known_estimated_joint_pose(joint_mask, correct_positions, estimated_positions):
    estimated_pose = torch.zeros_like(correct_positions)
    joint_mask_3 = torch_tile(joint_mask, 1, 3)
    inv_joint_mask_3 = joint_mask_3 == False
    estimated_pose[inv_joint_mask_3] += correct_positions[inv_joint_mask_3]
    estimated_pose[joint_mask_3] += estimated_positions[joint_mask_3]

    return estimated_pose


class Solver:
    outputs = 3

    def __init__(self, joint_idx, parent_joint_idx=None, input_idx=None, parent_input_idx=None, output_idx=None,
                 parent_output_idx=None):
        self.joint_idx = joint_idx
        self.parent_index = parent_joint_idx if parent_joint_idx is not None else -1
        self.input_idx = input_idx if input_idx is not None else joint_idx * self.outputs
        self.parent_input_idx = parent_input_idx if parent_input_idx is not None else self.parent_index * self.outputs
        self.output_idx = output_idx if output_idx is not None else joint_idx * self.outputs
        self.parent_output_idx = parent_output_idx if parent_output_idx is not None else self.parent_index * self.outputs

    def to_pos_batch(self, prediction, **kwargs):
        raise NotImplemented()


class PositionSolver(Solver):
    outputs = 3


class PosDirect(PositionSolver):
    def to_pos_batch(self, prediction, **kwargs):
        return prediction[:, self.input_idx: self.input_idx + 3]


class PosDirectBody(PositionSolver):
    def to_pos_batch(self, prediction, body, **kwargs):
        return body[:, self.input_idx: self.input_idx + 3]


class PosDirectReplaceOccluded(PositionSolver):
    def to_pos_batch(self, prediction, ref_positions, mask, **kwargs):
        if ref_positions is not None:
            output = torch.clone(ref_positions[:, self.input_idx: self.input_idx + 3])
        else:
            return torch.clone(prediction[:, self.input_idx: self.input_idx + 3])

        if mask is not None:
            joint_mask = mask[:, self.joint_idx]
            output[joint_mask] = torch.clone(prediction[:, self.input_idx: self.input_idx + 3][joint_mask])
        return output


class PosDoubleLocalReplaceOccluded(PositionSolver):

    def __init__(self, joint_idx, parent_joint_idx=None, input_idx=None, parent_input_idx=None, output_idx=None,
                 parent_output_idx=None, grand_parent_idx=None, grand_parent_input_idx=None, grand_use_body=False):
        super().__init__(joint_idx, parent_joint_idx, input_idx, parent_input_idx, output_idx, parent_output_idx)
        self.grand_parent_idx = grand_parent_idx if grand_parent_idx is not None else 0
        self.grand_parent_input_idx = grand_parent_input_idx if grand_parent_input_idx is not None else self.grand_parent_idx * 3
        self.grand_use_body = grand_use_body

    def to_pos_batch(self, prediction, ref_positions, mask, body, **kwargs):
        if ref_positions is not None:
            output = ref_positions[:, self.input_idx: self.input_idx + 3] \
                     + ref_positions[:, self.parent_input_idx: self.parent_input_idx + 3]
            output += body[:, self.grand_parent_input_idx: self.grand_parent_input_idx + 3] if self.grand_use_body else ref_positions[:,
                                                                                                                        self.grand_parent_input_idx: self.grand_parent_input_idx + 3]
        else:
            output = prediction[:, self.input_idx: self.input_idx + 3] \
                     + prediction[:, self.parent_input_idx: self.parent_input_idx + 3]
            output += body[:, self.grand_parent_input_idx: self.grand_parent_input_idx + 3] if self.grand_use_body else prediction[:,
                                                                                                                        self.grand_parent_input_idx: self.grand_parent_input_idx + 3]
            return output

        if mask is not None:
            joint_mask = mask[:, self.joint_idx]
            parent_mask = mask[:, self.parent_index]
            grand_parent_mask = mask[:, self.grand_parent_idx]
            if self.grand_use_body:
                grand_parent_mask *= False

            parent_pos = torch.zeros_like(ref_positions[:, self.parent_input_idx: self.parent_input_idx + 3])
            parent_pos[~parent_mask] = ref_positions[~parent_mask, self.parent_input_idx: self.parent_input_idx + 3]
            parent_pos[parent_mask] = prediction[parent_mask, self.parent_input_idx: self.parent_input_idx + 3]
            if self.grand_use_body:
                grand_parent_pos = body[:, self.grand_parent_input_idx: self.grand_parent_input_idx + 3]
            else:
                grand_parent_pos = torch.zeros_like(ref_positions[:, self.grand_parent_input_idx: self.grand_parent_input_idx + 3])
                grand_parent_pos[~grand_parent_mask] = ref_positions[~grand_parent_mask, self.grand_parent_input_idx: self.grand_parent_input_idx + 3]
                grand_parent_pos[grand_parent_mask] = prediction[grand_parent_mask, self.grand_parent_input_idx: self.grand_parent_input_idx + 3]
            output[joint_mask] = (prediction[:, self.input_idx: self.input_idx + 3] + parent_pos + grand_parent_pos)[joint_mask]
        return output


class PosLocalBodyReplaceOccluded(PositionSolver):
    def to_pos_batch(self, prediction, ref_positions, mask, body, **kwargs):
        if ref_positions is not None:
            output = ref_positions[:, self.input_idx: self.input_idx + 3] + body[:, self.parent_input_idx: self.parent_input_idx + 3]
        else:
            return prediction[:, self.input_idx: self.input_idx + 3] + body[:, self.parent_input_idx: self.parent_input_idx + 3]

        if mask is not None:
            joint_mask = mask[:, self.joint_idx]
            output[joint_mask] = (prediction[:, self.input_idx: self.input_idx + 3] + body[:, self.parent_input_idx: self.parent_input_idx + 3])[joint_mask]
        return output


class PosZeroRotationLocalBody(PositionSolver):
    def __init__(self, joint_idx, parent_joint_idx=None, input_idx=None, parent_input_idx=None, output_idx=None,
                 parent_output_idx=None, parent_up_idx=None, parent_forward_idx=None):
        super().__init__(joint_idx, parent_joint_idx, input_idx, parent_input_idx, output_idx, parent_output_idx)

        self.joint_idx = joint_idx
        self.parent_index = parent_joint_idx
        self.input_idx = input_idx if input_idx is not None else joint_idx * self.outputs
        self.parent_input_idx = parent_input_idx if parent_input_idx is not None else joint_idx * self.outputs
        self.output_idx = output_idx if output_idx is not None else joint_idx * self.outputs
        self.parent_output_idx = parent_output_idx if parent_output_idx is not None else joint_idx * self.outputs
        self.parent_up_idx = parent_up_idx
        self.parent_forward_idx = parent_forward_idx

    def to_pos_batch(self, prediction, body, **kwargs):
        parent_pos = body[:, self.parent_input_idx:self.parent_input_idx + 3]
        output = prediction[:, self.input_idx:self.input_idx + 3]
        parent_forward = body[:, self.parent_forward_idx:self.parent_forward_idx + 3]
        parent_up = body[:, self.parent_up_idx:self.parent_up_idx + 3]

        key = "PosZeroRotationLocalBody_par_quat" + str(self.parent_index)
        if key not in kwargs["temp_mem"]:
            kwargs["temp_mem"][key] = quaternion.up_forward_to_quat(parent_up, parent_forward, normalize=True)
        parent_quat = kwargs["temp_mem"][key]
        output = quaternion.qrot(parent_quat, output) + parent_pos
        return output


class SolverPacker:
    def __init__(self, solvers):
        self.solvers = solvers
        self.outputs = sum([x.outputs for x in solvers])

    def solve_batch(self, prediction, **kwargs):
        result = torch.empty((len(prediction), self.outputs), dtype=torch.float32).to(prediction.device)
        temp_mem = {}
        idx = 0

        for s in self.solvers:
            result[:, idx:idx + s.outputs] = s.to_pos_batch(prediction, solved_p=result, temp_mem=temp_mem, **kwargs)
            idx += s.outputs

        return result
