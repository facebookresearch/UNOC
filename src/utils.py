# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from math_helper import torch_bdot, distance_to_triangle
import torch.nn.functional as F


class Scaler():
    def __init__(self, tensor: torch.tensor, group_size=1, idx_end=-1, **kwargs):
        self.group_size = group_size
        self.idx_end = idx_end

    def _t(self, tensor) -> torch.tensor:
        return tensor[:, :self.idx_end].view(len(tensor), -1, 3), tensor[:, self.idx_end:]

    def _inv_t(self, t, remainder) -> torch.tensor:
        tensor = t.view(len(t), -1)
        return torch.cat([tensor, remainder], dim=1)

    def reset(self):
        pass

    def transform_recurrent(self, tensor: torch.tensor) -> torch.tensor:
        for i in range(tensor.shape[1]):
            tensor[:, i] = self.transform(tensor[:, i])
        return tensor

    def transform(self, tensor: torch.tensor) -> torch.tensor:
        return tensor

    def inv_transform(self, tensor: torch.tensor) -> torch.tensor:
        return tensor


class PytNonScaler(Scaler):
    pass


class PyTMinMaxScaler(Scaler):
    """
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, tensor: torch.tensor, range=[0, 1], **kwargs):
        super().__init__(tensor, **kwargs)
        t, _ = self._t(tensor)
        self.range = range
        self.min = t.min(dim=0, keepdim=True)[0]
        self.max = t.max(dim=0, keepdim=True)[0]
        self.inv_s = (self.max - self.min) / (range[1] - range[0])
        self.inv_s[self.inv_s == 0.0] = 1.0
        self.s = 1.0 / self.inv_s

    def reset(self):
        self.s = torch.ones_like(self.s)
        self.inv_s = torch.ones_like(self.s)
        self.min = torch.zeros_like(self.s)

    def transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t.sub_(self.min).mul_(self.s).add_(self.range[0])
        return self._inv_t(t, remainder)

    def inv_transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t.sub_(self.range[0]).mul_(self.inv_s.to(t.device)).add_(self.min.to(t.device))
        return self._inv_t(t, remainder)


class PyTMinMaxGlobalScaler(Scaler):
    """
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, tensor: torch.tensor, range=[0, 1], **kwargs):
        super().__init__(tensor, **kwargs)
        t, _ = self._t(tensor)
        self.range = range
        self.min = t.min(dim=0, keepdim=True)[0]
        self.max = t.max(dim=0, keepdim=True)[0]
        tmp_min = self.min.min(dim=1, keepdim=True)[0].clone()
        tmp_max = self.max.max(dim=1, keepdim=True)[0].clone()
        self.min *= 0
        self.max *= 0
        self.min += tmp_min
        self.max += tmp_max
        self.inv_s = (self.max - self.min) / (range[1] - range[0])
        self.inv_s[self.inv_s == 0.0] = 1.0
        self.s = 1.0 / self.inv_s

    def reset(self):
        self.s = torch.ones_like(self.s)
        self.inv_s = torch.ones_like(self.s)
        self.min = torch.zeros_like(self.s)

    def transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t.sub_(self.min).mul_(self.s).add_(self.range[0])
        return self._inv_t(t, remainder)

    def inv_transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t.sub_(self.range[0]).mul_(self.inv_s.to(t.device)).add_(self.min.to(t.device))
        return self._inv_t(t, remainder)


class PytNormalization(Scaler):
    """
    Subtracts mean and divides by STD
    """

    def __init__(self, tensor: torch.tensor, **kwargs):
        super().__init__(tensor, **kwargs)
        t, _ = self._t(tensor)
        self.mean = torch.mean(t, dim=(2, 0))
        self.mean = torch_tile(self.mean[:, None], dim=1, n_tile=self.group_size)
        self.std = torch.std(t, dim=(2, 0))
        self.std = torch_tile(self.std[:, None], dim=1, n_tile=self.group_size)

    def reset(self):
        self.mean = torch.zeros_like(self.mean)
        self.std = torch.ones_like(self.std)

    def transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t = (t - self.mean.to(t.device)) / self.std.to(t.device)
        result = self._inv_t(t, remainder)
        return result

    def inv_transform(self, tensor: torch.tensor):
        t, remainder = self._t(tensor)
        t = (t * self.std.to(t.device)) + self.mean.to(t.device)
        result = self._inv_t(t, remainder)
        return result


def torch_tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
    return torch.index_select(a, dim, order_index)


def pos_loss(input, target, config, body, mask, **kwargs):
    target_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(target, ref_positions=None, mask=None, body=body, **kwargs) \
        .reshape((input.shape[0], -1, 3))

    out_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(input, ref_positions=None, mask=None, body=body, **kwargs) \
        .reshape((input.shape[0], -1, 3))

    return target_pos - out_pos


def mse_pos_loss(**kwargs):
    return torch.mean(pos_loss(**kwargs) ** 2)


def mse_loss(input, target, **kwargs):
    return torch.mean((input - target) ** 2)


def tmp_mse_speed_loss(input, target, prev_out, **kwargs):
    movement = input - prev_out[:, -1]
    smoothen_frames = 4
    sum_frames = smoothen_frames
    prev_movement = (prev_out[:, -1] - prev_out[:, -2]) * smoothen_frames
    for i in range(smoothen_frames - 1):
        sum_frames += smoothen_frames - i
        prev_movement += (prev_out[:, -2 - i] - prev_out[:, -3 - i]) * (smoothen_frames - 1 - i)

    prev_movement /= sum_frames

    return torch.mean((movement - prev_movement) ** 2)


def tmp_mse_movement_loss(input, target, prev_out, **kwargs):
    movement = input - prev_out[:, -1]
    return torch.mean((movement) ** 2)


def l1_loss(input, target, **kwargs):
    return torch.mean(torch.abs(input - target))


def weighted_mse_loss(input, target, mask, weight, **kwargs):
    loss = (input - target) ** 2
    weights = torch.ones(mask.shape, dtype=torch.float32).to(input.device)
    weights[mask] *= weight
    loss *= weights
    return torch.mean(loss)


def weighted_mse_hand_local_loss(input, target, mask, weight, finger_weight, config, **kwargs):
    input3 = input.reshape((len(input), -1, 3))
    target3 = target.reshape((len(target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_wrist, r_wrist = l_hand_joints[0], r_hand_joints[0]
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    target3[:, l_hand_wo_wrist] -= torch_tile((target3[:, l_wrist:l_wrist + 1] - input3[:, l_wrist:l_wrist + 1]), dim=1, n_tile=len(l_hand_wo_wrist))
    target3[:, r_hand_wo_wrist] -= torch_tile((target3[:, r_wrist:r_wrist + 1] - input3[:, r_wrist:r_wrist + 1]), dim=1, n_tile=len(r_hand_wo_wrist))
    loss = (input3 - target3) ** 2
    weights = torch.ones(mask.shape, dtype=torch.float32).to(input3.device)
    weights[mask] *= weight
    weights = weights.reshape_as(target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    loss *= weights
    loss *= finger_weights
    return torch.mean(loss)


def weighted_mse_finger_weight_loss(input, target, mask, weight, finger_weight, config, **kwargs):
    input3 = input.reshape((len(input), -1, 3))
    target3 = target.reshape((len(target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    loss = (input3 - target3) ** 2
    weights = torch.ones(mask.shape, dtype=torch.float32).to(input3.device)
    weights[mask] *= weight
    weights = weights.reshape_as(target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    loss *= weights
    loss *= finger_weights
    return torch.mean(loss)


def weighted_mean_finger_weight_loss(input, target, mask, weight, finger_weight, config, **kwargs):
    input3 = input.view((len(input), -1, 3))
    target3 = target.reshape((len(target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    weights = torch.ones(mask.shape, dtype=torch.float32).to(input3.device)
    weights[mask] *= weight
    weights = weights.reshape_as(target3)[:, :, 0]
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    loss = torch.norm(input3 - target3, dim=2) * finger_weights * weights
    return torch.mean(loss)


def weighted_mse_finger_weight_occ_learning_loss(input, target, mask, weight, finger_weight, config, mask_weight=1.0, use_predicted_mask=True, **kwargs):
    pos_input = input[:, :config.out_features.features_wo_occlusions()]
    mask_input = input[:, config.out_features.features_wo_occlusions():]
    pos_target = target[:, :config.out_features.features_wo_occlusions()]
    mask_target = target[:, config.out_features.features_wo_occlusions():]
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (pos_input3 - target3) ** 2
    weights = torch.ones(mask_input.shape, dtype=torch.float32).to(pos_input3.device)
    if use_predicted_mask:
        weights[mask_input > 0.5] *= weight
    else:
        weights[mask[:, ::3]] *= weight
    weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    pos_loss *= weights
    pos_loss *= finger_weights
    occ_mask_loss = (mask_target - mask_input) ** 2
    return torch.mean(pos_loss) + torch.mean(occ_mask_loss) * mask_weight


def weighted_mse_finger_classifier_occ_mask_occ_learning_loss(input, target, mask, weight, finger_weight, config, mask_weight=1.0, use_predicted_mask=True, **kwargs):
    pos_input = input[:, :config.out_features.features_wo_occlusions()]
    mask_input = input[:, config.out_features.features_wo_occlusions():]
    pos_target = target[:, :config.out_features.features_wo_occlusions()]
    # mask_target = target[:, config.out_features.features_wo_occlusions():]
    mask_target = torch.zeros_like(target[:, config.out_features.features_wo_occlusions():])
    mask_target[mask[:, ::3]] += 1
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (pos_input3 - target3) ** 2
    weights = torch.ones(mask_input.shape, dtype=torch.float32).to(pos_input3.device)

    if use_predicted_mask:
        weights[mask_input > 0] *= weight
    else:
        weights[mask[:, ::3]] *= weight

    weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    pos_loss *= weights
    pos_loss *= finger_weights
    # occ_mask_loss = (mask_target - mask_input) ** 2
    occ_mask_loss = torch.nn.BCEWithLogitsLoss()(mask_input, mask_target)
    return torch.mean(pos_loss) + torch.abs(occ_mask_loss) * mask_weight


def weighted_mse_parent_local_occ_learning_loss(input, target, mask, weight, config, use_predicted_mask=False, mask_threshold=0.5, **kwargs):
    pos_input = input[:, :config.out_features.features_wo_occlusions()]
    mask_input = input[:, config.out_features.features_wo_occlusions():]
    pos_target = target[:, :config.out_features.features_wo_occlusions()]
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()

    par_idx = config.skeleton.parent_idx_vector()
    par_idx[0] = 0
    par_pos_input3 = pos_input3[:, par_idx]
    par_pos_target3 = target3[:, par_idx]

    diff_in = pos_input3 - par_pos_input3
    diff_target = target3 - par_pos_target3

    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (diff_in - diff_target) ** 2
    weights = torch.ones(mask_input.shape, dtype=torch.float32).to(pos_input3.device)
    if use_predicted_mask:
        weights[mask_input > mask_threshold] *= weight
    else:
        weights[mask[:, ::3]] *= weight
    weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(target3)
    pos_loss *= weights
    return torch.mean(pos_loss)


def weighted_mse_parent_local_loss(input, target, mask, weight, config, **kwargs):
    pos_input = input[:, :config.out_features.features_wo_occlusions()]
    pos_target = target[:, :config.out_features.features_wo_occlusions()]
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()

    par_idx = config.skeleton.parent_idx_vector()
    par_idx[0] = 0
    par_pos_input3 = pos_input3[:, par_idx]
    par_pos_target3 = target3[:, par_idx]

    diff_in = pos_input3 - par_pos_input3
    diff_target = target3 - par_pos_target3

    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (diff_in - diff_target) ** 2
    weights = torch.ones(mask.shape, dtype=torch.float32).to(pos_input3.device)
    weights[mask] *= weight
    # weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(target3)
    pos_loss *= weights
    return torch.mean(pos_loss)


def weighted_mse_finger_weight_occ_learning_speed_learning_loss(input, target, mask, weight, finger_weight, config, mask_weight=1.0, speed_weight=1.0,
                                                                use_predicted_mask=True, **kwargs):
    n_f_wo_occ = config.out_features.features_wo_occlusions()
    p_i0, p_in, v_i0, v_in, occ_i0, occ_in = 0, n_f_wo_occ // 2, n_f_wo_occ // 2, n_f_wo_occ, n_f_wo_occ, input.shape[-1]
    pos_input = input[:, p_i0:p_in]
    speed_input = input[:, v_i0:v_in]
    mask_input = input[:, occ_i0:occ_in]
    pos_target = target[:, p_i0:p_in]
    speed_target = target[:, v_i0:v_in]
    mask_target = target[:, occ_i0:occ_in]
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    pos_target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()
    speed_input3 = speed_input.reshape((len(speed_input), -1, 3))
    speed_target3 = speed_target.reshape((len(speed_target), -1, 3)).clone()

    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (pos_input3 - pos_target3) ** 2
    speed_loss = (speed_input3 - speed_target3) ** 2

    weights = torch.ones(mask_input.shape, dtype=torch.float32).to(pos_input3.device)
    if use_predicted_mask:
        weights[mask_input > 0.5] *= weight
    else:
        weights[mask[:, ::3]] *= weight
    weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(pos_target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    pos_loss *= weights
    speed_loss *= weights
    pos_loss *= finger_weights
    speed_loss *= weights
    occ_mask_loss = (mask_target - mask_input) ** 2
    return torch.mean(pos_loss) + torch.mean(occ_mask_loss) * mask_weight + torch.mean(speed_loss) * speed_weight


def weighted_mean_finger_weight_occ_learning_loss(input, target, mask, weight, finger_weight, config, mask_weight=1.0, use_predicted_mask=True, **kwargs):
    pos_input = input[:, :config.out_features.features_wo_occlusions()]
    mask_input = input[:, config.out_features.features_wo_occlusions():]
    pos_target = target[:, :config.out_features.features_wo_occlusions()]
    mask_target = target[:, config.out_features.features_wo_occlusions():]
    pos_input3 = pos_input.reshape((len(pos_input), -1, 3))
    target3 = pos_target.reshape((len(pos_target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    pos_loss = (pos_input3 - target3)
    weights = torch.ones(mask_input.shape, dtype=torch.float32).to(pos_input3.device)
    if use_predicted_mask:
        weights[mask_input > 0.5] *= weight
    else:
        weights[mask[:, ::3]] *= weight
    weights = torch_tile(weights, dim=1, n_tile=3)
    weights = weights.reshape_as(target3)
    finger_weights = torch.ones_like(weights)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    pos_loss *= weights
    pos_loss *= finger_weights
    occ_mask_loss = (mask_target - mask_input)
    return torch.mean(pos_loss) + torch.mean(occ_mask_loss) * mask_weight


def mse_finger_weight_loss(input, target, finger_weight, config, **kwargs):
    input3 = input.reshape((len(input), -1, 3))
    target3 = target.reshape((len(target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_hand_wo_wrist, r_hand_wo_wrist = l_hand_joints.copy(), r_hand_joints.copy()
    l_hand_wo_wrist.remove(l_hand_wo_wrist[0])
    r_hand_wo_wrist.remove(r_hand_wo_wrist[0])
    loss = (input3 - target3) ** 2
    finger_weights = torch.ones_like(input3)
    finger_weights[:, l_hand_wo_wrist] *= finger_weight
    finger_weights[:, r_hand_wo_wrist] *= finger_weight
    loss *= finger_weights
    return torch.mean(loss)


def wrist_distance_loss(input, target, mask, weight, config, **kwargs):
    input3 = input.reshape((len(input), -1, 3))
    target3 = target.reshape((len(target), -1, 3)).clone()
    l_hand_joints, r_hand_joints = config.skeleton.Idx.get_hand_joints()
    l_wrist, r_wrist = l_hand_joints[0], r_hand_joints[0]
    d_target = torch.norm(target3[:, l_wrist] - target3[:, r_wrist], dim=1)
    d_input = torch.norm(input3[:, l_wrist] - input3[:, r_wrist], dim=1)
    loss = (d_target - d_input) ** 2
    w_mask = mask[:, l_wrist * 3] | mask[:, r_wrist * 3]
    weights = torch.ones(w_mask.shape, dtype=torch.float32).to(input3.device)
    weights[w_mask] *= weight
    weights = weights.reshape_as(d_target)
    loss *= weights
    return torch.mean(loss)


def angular_loss(input, target, **kwargs):
    input = input.reshape(-1, 3)
    target = target.reshape(-1, 3)
    return torch.mean((1 - torch_bdot(input, target)) ** 2)


def bone_length_loss(input, target, mask, config, body, bone_lengths, use_combined_pose, **kwargs):
    if use_combined_pose:
        _mask = mask
        _ref_positions = body
    else:
        _mask = None
        _ref_positions = None

    target_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(target, ref_positions=_ref_positions, mask=_mask, bone_lengths=bone_lengths, body=body) \
        .reshape((input.shape[0], -1, 3))

    out_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(input, ref_positions=None, mask=None, bone_lengths=bone_lengths, body=body) \
        .reshape((input.shape[0], -1, 3))

    target_lengths = torch.empty((input.shape[0], len(config.skeleton.Idx.all)), device=target.device)
    out_lengths = torch.empty((input.shape[0], len(config.skeleton.Idx.all)), device=target.device)
    parent_idx = config.skeleton.parent_idx_vector()
    for bone0, bone1 in enumerate(parent_idx):
        if bone1 >= 0:
            target_lengths[:, bone0] = torch.norm(target_pos[:, bone0] - target_pos[:, bone1], dim=1)
            out_lengths[:, bone0] = torch.norm(out_pos[:, bone0] - out_pos[:, bone1], dim=1)
        else:
            target_lengths[:, bone0] = torch.norm(target_pos[:, bone0], dim=1)
            out_lengths[:, bone0] = torch.norm(out_pos[:, bone0], dim=1)

    loss = target_lengths - out_lengths
    return loss


def finger_weighted_bone_length_loss_direct(input, target, mask, config, body, use_combined_pose, finger_weight=1.0, ignore_joint=-1, wrist_local_fingers=True,
                                            **kwargs):
    if use_combined_pose:
        _mask = mask
        _ref_positions = body
    else:
        _mask = None
        _ref_positions = None

    out_pos = input[:, :config.out_features.features_wo_occlusions()].reshape((input.shape[0], -1, 3))
    target_pos = target[:, :config.out_features.features_wo_occlusions()].reshape((target.shape[0], -1, 3))

    target_lengths = torch.empty((input.shape[0], len(config.skeleton.Idx.all)), device=target.device)
    out_lengths = torch.empty((input.shape[0], len(config.skeleton.Idx.all)), device=target.device)
    parent_idx = config.skeleton.parent_idx_vector()

    l_wrist_idx, r_wrist_idx = config.skeleton.Idx.lwrist, config.skeleton.Idx.rwrist

    for bone0, bone1 in enumerate(parent_idx):
        if bone0 == ignore_joint:
            target_lengths[:, bone0] = 0.0
            out_lengths[:, bone0] = 0.0

        if wrist_local_fingers and bone1 in [l_wrist_idx, r_wrist_idx]:
            target_lengths[:, bone0] = torch.norm(target_pos[:, bone0], dim=1)
            out_lengths[:, bone0] = torch.norm(out_pos[:, bone0], dim=1)

        else:
            if bone1 >= 0:
                target_lengths[:, bone0] = torch.norm(target_pos[:, bone0] - target_pos[:, bone1], dim=1)
                out_lengths[:, bone0] = torch.norm(out_pos[:, bone0] - out_pos[:, bone1], dim=1)
            else:
                target_lengths[:, bone0] = torch.norm(target_pos[:, bone0], dim=1)
                out_lengths[:, bone0] = torch.norm(out_pos[:, bone0], dim=1)

    loss = target_lengths - out_lengths
    l_finger_idx, r_finger_idx = config.skeleton.Idx.get_finger_joints()
    all_finger_idx = [*r_finger_idx, *l_finger_idx]
    loss[:, all_finger_idx] *= finger_weight
    return loss


def mse_finger_weighted_bone_length_loss_direct(input, target, mask, config, body, use_combined_pose, finger_weight=1.0, ignore_joint=-1,
                                                wrist_local_fingers=True,
                                                **kwargs):
    from math import sqrt
    return torch.mean(
        finger_weighted_bone_length_loss_direct(input, target, mask, config, body, use_combined_pose,
                                                sqrt(finger_weight),
                                                ignore_joint,
                                                wrist_local_fingers,
                                                **kwargs) ** 2)


def finger_length_loss(input, target, config, body, **kwargs):
    target_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(target, ref_positions=None, mask=None, bone_lengths=kwargs["bone_lengths"], body=body) \
        .reshape((input.shape[0], -1, 3))

    out_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(input, ref_positions=None, mask=None, bone_lengths=kwargs["bone_lengths"], body=body) \
        .reshape((input.shape[0], -1, 3))

    l_finger_idx, r_finger_idx = config.skeleton.Idx.get_finger_joints()
    all_finger_idx = [*l_finger_idx, *r_finger_idx]
    target_lengths = torch.empty((input.shape[0], len(all_finger_idx)), device=target.device)
    out_lengths = torch.empty((input.shape[0], len(all_finger_idx)), device=target.device)
    parent_idx = config.skeleton.parent_idx_vector()
    for bone0, bone1 in enumerate(parent_idx):
        if bone0 not in all_finger_idx:
            continue
        arr_idx = all_finger_idx.index(bone0)
        if bone1 >= 0:
            target_lengths[:, arr_idx] = torch.norm(target_pos[:, bone0] - target_pos[:, bone1], dim=1)
            out_lengths[:, arr_idx] = torch.norm(out_pos[:, bone0] - out_pos[:, bone1], dim=1)
        else:
            target_lengths[:, arr_idx] = torch.norm(target_pos[:, bone0], dim=1)
            out_lengths[:, arr_idx] = torch.norm(out_pos[:, bone0], dim=1)

    loss = target_lengths - out_lengths
    return loss


def finger_length_loss_finger_pos_only_input(input, target, config, body, **kwargs):
    target_pos = target.reshape((input.shape[0], -1, 3))

    out_pos = input.reshape((input.shape[0], -1, 3))

    l_finger_idx, r_finger_idx = config.skeleton.Idx.get_finger_joints()
    # all_finger_idx = [*l_finger_idx, *r_finger_idx]
    all_finger_idx = [*r_finger_idx, *l_finger_idx]
    target_lengths = torch.empty((input.shape[0], len(all_finger_idx)), device=target.device)
    out_lengths = torch.empty((input.shape[0], len(all_finger_idx)), device=target.device)
    v_parent_idx = config.skeleton.parent_idx_vector()
    for _child_idx, _parent_idx in enumerate(v_parent_idx):
        if _child_idx not in all_finger_idx:
            continue

        idx_off = min(l_finger_idx) if _child_idx in l_finger_idx else min(r_finger_idx)
        child_idx = _child_idx - idx_off
        parent_idx = _parent_idx - idx_off
        arr_idx = all_finger_idx.index(_child_idx)
        if parent_idx >= 0:
            target_lengths[:, arr_idx] = torch.norm(target_pos[:, child_idx] - target_pos[:, parent_idx], dim=1)
            out_lengths[:, arr_idx] = torch.norm(out_pos[:, child_idx] - out_pos[:, parent_idx], dim=1)
        else:
            target_lengths[:, arr_idx] = torch.norm(target_pos[:, child_idx], dim=1)
            out_lengths[:, arr_idx] = torch.norm(out_pos[:, child_idx], dim=1)

    loss = target_lengths - out_lengths
    return loss


def finger_direction_loss(input, target, config, body, **kwargs):
    target_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(target, ref_positions=None, mask=None, bone_lengths=kwargs["bone_lengths"], body=body) \
        .reshape((input.shape[0], -1, 3))

    out_pos = config \
        .out_features.features_to_pose() \
        .solve_batch(input, ref_positions=None, mask=None, bone_lengths=kwargs["bone_lengths"], body=body) \
        .reshape((input.shape[0], -1, 3))

    l_finger_idx, r_finger_idx = config.skeleton.Idx.get_finger_joints()
    # all_finger_idx = [*l_finger_idx, *r_finger_idx]
    all_finger_idx = [*r_finger_idx, *l_finger_idx]

    Idx = config.skeleton.Idx
    l_finger_tri_idx = [
        [Idx.lindex1, Idx.lindex2, Idx.lindex3],
        [Idx.lring1, Idx.lring2, Idx.lring3],
        [Idx.lmiddle1, Idx.lmiddle2, Idx.lmiddle3],
        [Idx.lpinky1, Idx.lpinky2, Idx.lpinky3],
        [Idx.lthumb1, Idx.lthumb2, Idx.lthumb3],
    ]
    r_finger_tri_idx = [
        [Idx.rindex1, Idx.rindex2, Idx.rindex3],
        [Idx.rring1, Idx.rring2, Idx.rring3],
        [Idx.rmiddle1, Idx.rmiddle2, Idx.rmiddle3],
        [Idx.rpinky1, Idx.rpinky2, Idx.rpinky3],
        [Idx.rthumb1, Idx.rthumb2, Idx.rthumb3],
    ]
    # finger_tri_idx = [*l_finger_tri_idx, *r_finger_tri_idx]
    finger_tri_idx = [*r_finger_tri_idx, *l_finger_tri_idx]
    finger_tri_idx = [y for x in finger_tri_idx for y in x]
    target_finger_tris = target_pos[:, finger_tri_idx].reshape((len(target_pos), len(finger_tri_idx) // 3, 3, 3))
    actual_finger_pos = out_pos[:, all_finger_idx]

    # first_finger_joint_idx = [Idx.lindex1, Idx.lring1, Idx.lmiddle1, Idx.lpinky1, Idx.lthumb0, Idx.rindex1, Idx.rring1, Idx.rmiddle1, Idx.rpinky1, Idx.rthumb0]
    first_finger_joint_idx = [Idx.rindex1, Idx.rring1, Idx.rmiddle1, Idx.rpinky1, Idx.rthumb0, Idx.lindex1, Idx.lring1, Idx.lmiddle1, Idx.lpinky1, Idx.lthumb0]

    distances = torch.zeros((len(actual_finger_pos), len(all_finger_idx)), device=input.device)
    for i, idx in enumerate(all_finger_idx):
        # tri_idx = i // 4
        tri_idx = -1
        for first_finger_idx in first_finger_joint_idx:
            if idx >= first_finger_idx:
                tri_idx += 1
            else:
                break
        distance = distance_to_triangle(target_finger_tris[:, tri_idx], actual_finger_pos[:, i])
        distances[:, i] = distance

    return distances


def finger_direction_loss_finger_pos_input_only(input, target, config, body, **kwargs):
    target_pos = target.reshape((input.shape[0], -1, 3))
    out_pos = input.reshape((input.shape[0], -1, 3))

    l_finger_idx, r_finger_idx = config.skeleton.Idx.get_finger_joints()
    # all_finger_idx = [*l_finger_idx, *r_finger_idx]

    Idx = config.skeleton.Idx
    l_finger_tri_idx = [
        [Idx.lindex1, Idx.lindex2, Idx.lindex3],
        [Idx.lring1, Idx.lring2, Idx.lring3],
        [Idx.lmiddle1, Idx.lmiddle2, Idx.lmiddle3],
        [Idx.lpinky1, Idx.lpinky2, Idx.lpinky3],
        [Idx.lthumb1, Idx.lthumb2, Idx.lthumb3],
    ]
    r_finger_tri_idx = [
        [Idx.rindex1, Idx.rindex2, Idx.rindex3],
        [Idx.rring1, Idx.rring2, Idx.rring3],
        [Idx.rmiddle1, Idx.rmiddle2, Idx.rmiddle3],
        [Idx.rpinky1, Idx.rpinky2, Idx.rpinky3],
        [Idx.rthumb1, Idx.rthumb2, Idx.rthumb3],
    ]

    first_finger_joint_idx = [Idx.rindex1, Idx.rring1, Idx.rmiddle1, Idx.rpinky1, Idx.rthumb0, Idx.lindex1, Idx.lring1, Idx.lmiddle1, Idx.lpinky1, Idx.lthumb0]

    l_finger_idx_off = min(l_finger_idx)
    r_finger_idx_off = min(r_finger_idx)
    for i in range(len(l_finger_idx)):
        r_finger_idx[i] -= r_finger_idx_off
        l_finger_idx[i] += len(l_finger_idx) - l_finger_idx_off

    for i in range(len(l_finger_tri_idx)):
        for j in range(len(l_finger_tri_idx[i])):
            r_finger_tri_idx[i][j] -= r_finger_idx_off
            l_finger_tri_idx[i][j] += len(l_finger_idx) - l_finger_idx_off

    for i in range(len(first_finger_joint_idx)):
        first_finger_joint_idx[i] -= (r_finger_idx_off if i < 5 else (l_finger_idx_off - len(l_finger_idx)))

    all_finger_idx = [*r_finger_idx, *l_finger_idx]
    # finger_tri_idx = [*l_finger_tri_idx, *r_finger_tri_idx]
    finger_tri_idx = [*r_finger_tri_idx, *l_finger_tri_idx]
    finger_tri_idx = [y for x in finger_tri_idx for y in x]
    target_finger_tris = target_pos[:, finger_tri_idx].reshape((len(target_pos), len(finger_tri_idx) // 3, 3, 3))
    actual_finger_pos = out_pos[:, all_finger_idx]

    # first_finger_joint_idx = [Idx.lindex1, Idx.lring1, Idx.lmiddle1, Idx.lpinky1, Idx.lthumb0, Idx.rindex1, Idx.rring1, Idx.rmiddle1, Idx.rpinky1, Idx.rthumb0]

    distances = torch.zeros((len(actual_finger_pos), len(all_finger_idx)), device=input.device)
    for i, idx in enumerate(all_finger_idx):
        # tri_idx = i // 4
        tri_idx = -1
        for first_finger_idx in first_finger_joint_idx:
            if idx >= first_finger_idx:
                tri_idx += 1
            else:
                break
        distance = distance_to_triangle(target_finger_tris[:, tri_idx], actual_finger_pos[:, i])
        distances[:, i] = distance

    return distances


def mse_prev_out_distance_loss(input, input1, config, **kwargs):
    n_in_non_occ = config.in_features.features_wo_occlusions()
    in_wo_occ = input[:, :n_in_non_occ]
    in1_wo_occ = input1[:, :n_in_non_occ]
    return torch.mean((in_wo_occ - in1_wo_occ) ** 2)


def mse_finger_direction_loss(input, target, config, body, **kwargs):
    return torch.mean(finger_direction_loss(input, target, config, body, **kwargs) ** 2)


def mse_finger_direction_loss_finger_pos_input_only(input, target, config, body, **kwargs):
    return torch.mean(finger_direction_loss_finger_pos_input_only(input, target, config, body, **kwargs) ** 2)


def mean_bone_length_loss(input, target, mask, config, body, bone_lengths, **kwargs):
    return torch.mean(torch.abs(bone_length_loss(input, target, mask, config, body, bone_lengths, use_combined_pose=False, **kwargs)))


def mean_bone_length_to_gt_loss(input, target, mask, config, body, bone_lengths, **kwargs):
    return torch.mean(torch.abs(bone_length_loss(input, target, mask, config, body, bone_lengths, use_combined_pose=True, **kwargs)))


def mean_finger_length_loss(input, target, config, body, **kwargs):
    return torch.mean(finger_length_loss(input, target, config, body, **kwargs))


def mse_bone_length_loss(input, target, mask, config, body, bone_lengths, **kwargs):
    return torch.mean(bone_length_loss(input, target, mask, config, body, bone_lengths, use_combined_pose=False, **kwargs) ** 2)


def mse_bone_length_to_gt_loss(input, target, mask, config, body, bone_lengths, **kwargs):
    return torch.mean(bone_length_loss(input, target, mask, config, body, bone_lengths, use_combined_pose=True, **kwargs) ** 2)


def mse_finger_length_loss(input, target, config, body, **kwargs):
    return torch.mean(finger_length_loss(input, target, config, body, **kwargs) ** 2)


def mse_finger_length_loss_finger_pos_only_input(input, target, config, body, **kwargs):
    return torch.mean(finger_length_loss_finger_pos_only_input(input, target, config, body, **kwargs) ** 2)


def combined_loss(losses, **kwargs):
    sum = 0
    verbose = "verbose" in kwargs and kwargs["verbose"] is True

    for loss, weight in losses:
        val = loss(**kwargs) * weight
        sum += val
        if verbose:
            print(f" {str(loss)}: {val:.3f} ", end=" ")

    if verbose:
        print(f" sum: {sum:.3f}", end="")
    return sum


def bone_lengths(positions, parent_indices):
    bone_lengths = torch.zeros(len(parent_indices))
    for idx, par_idx in enumerate(parent_indices):
        if par_idx >= 0:
            bone_lengths[idx] = torch.norm(positions[idx] - positions[par_idx])
        else:
            bone_lengths[idx] = torch.norm(positions[idx])

    return bone_lengths


def bone_lengths_batch(positions, parent_indices):
    bone_lengths = torch.zeros((len(positions), len(parent_indices)))
    for idx, par_idx in enumerate(parent_indices):
        if par_idx >= 0:
            bone_lengths[:, idx] = torch.norm(positions[:, idx] - positions[:, par_idx], dim=1)
        else:
            bone_lengths[:, idx] = torch.norm(positions[:, idx], dim=1)

    return bone_lengths


def bone_lengths_np(positions, parent_indices):
    bone_lengths = np.zeros(len(parent_indices))
    for idx, par_idx in enumerate(parent_indices):
        if par_idx >= 0:
            bone_lengths[idx] = np.linalg.norm(positions[idx] - positions[par_idx])
        else:
            bone_lengths[idx] = np.linalg.norm(positions[idx])

    return bone_lengths


def shuffle_X_y(X, y):
    indices = np.array(range(len(X)))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def swish(x):
    res = x / (1.0 - torch.exp(-x))
    res[torch.isnan(res)] = 0.0
    return res


def direct(x):
    return x


def fix_bone_lengths(positions, mask, c, target_lengths, it=10, k=0.95):
    def get_mean_target_position(joint_idx):
        # calculates the target position of the joint using the direction to parent and children and the correct bone length
        # returns the mean
        # note -> because some bones have multiple children, the children also need to have a constant distance between each other
        pos_arr = []
        if parent_idx[joint_idx] >= 0:
            diff = target[:, joint_idx] / actual_lengths[:, joint_idx]
            p_parent = p[:, parent_idx[joint_idx]] if parent_idx[joint_idx] >= 0 else torch.zeros((len(p), 3), device=p.device)
            off = p[:, joint_idx] - p_parent
            pos_arr.append(p_parent + (off * torch_tile(diff[:, None], dim=1, n_tile=3)))

            # calculate the target distance between this bone and the siblings
            for child in child_idx[parent_idx[joint_idx]]:
                if child == joint_idx:
                    continue
                diff = target[:, child] / actual_lengths[:, child]
                p_child = p[:, child]
                off = p[:, joint_idx] - p_child
                pos_arr.append(p_child + (off * torch_tile(diff[:, None], dim=1, n_tile=3)))

        for child in child_idx[joint_idx]:
            diff = target[:, child] / actual_lengths[:, child]
            p_child = p[:, child]
            off = p[:, joint_idx] - p_child
            pos_arr.append(p_child + (off * torch_tile(diff[:, None], dim=1, n_tile=3)))

        pos_arr = torch.stack(pos_arr)
        return torch.mean(pos_arr, dim=0)

    p = torch.clone(positions).to(positions.device)
    new_p = torch.empty_like(p)
    parent_idx = c.skeleton.parent_idx_vector()
    child_idx = c.skeleton.child_idx_vector()
    target = torch_tile(torch.from_numpy(target_lengths[None, :]).to(p.device), dim=0, n_tile=len(p)).float().to(p.device)
    _k = k
    for i in range(it):
        actual_lengths = bone_lengths_batch(p, parent_idx).to(p.device)
        err = torch.abs(actual_lengths / target)
        err[torch.isnan(err)] = 0
        err[torch.isinf(err)] = 0
        print(torch.mean(err))
        for joint_idx in range(len(c.skeleton.Idx.all)):
            mean_p = get_mean_target_position(joint_idx)
            diff = mean_p - p[:, joint_idx]
            joint_mask = mask[:, joint_idx]
            new_p[joint_mask, joint_idx] = (p[:, joint_idx] + diff * _k)[joint_mask]
            new_p[~joint_mask, joint_idx] = p[:, joint_idx][~joint_mask]
        # _k *= k
        p = torch.clone(new_p)

    return p


def combine_body_and_fingers(body, fingers, c):
    n_frames = len(body)
    body_pose = c.in_features.features_to_pose().solve_batch(body, ref_positions=None, mask=None).reshape((n_frames, -1, 3))
    finger_pose = c.out_features.features_to_pose().solve_batch(fingers.detach(), ref_positions=None, mask=None).reshape((n_frames, -1, 3))
    l_finger_idx, r_finger_idx = c.skeleton.Idx.get_hand_joints()
    l_finger_idx = l_finger_idx[1:]
    r_finger_idx = r_finger_idx[1:]

    lwrist_idx = c.skeleton.Idx.lwrist
    rwrist_idx = c.skeleton.Idx.rwrist

    l_wrist_pos = torch_tile(body_pose[:, lwrist_idx:lwrist_idx + 1, :], dim=1, n_tile=len(l_finger_idx))
    r_wrist_pos = torch_tile(body_pose[:, rwrist_idx:rwrist_idx + 1, :], dim=1, n_tile=len(r_finger_idx))

    body_pose[:, l_finger_idx] = finger_pose[:, len(l_finger_idx):] + l_wrist_pos
    body_pose[:, r_finger_idx] = finger_pose[:, :len(l_finger_idx)] + r_wrist_pos

    return body_pose


def get_float_mask(X, c):
    n_wo_occ = c.in_features.features_wo_occlusions()
    if n_wo_occ != X.shape[-1]:
        mask = X[:, n_wo_occ:]
    else:
        mask = torch.zeros((X.shape[0], len(c.skeleton.Idx.all)))
    return mask


def get_joint_mask(X, c):
    return get_float_mask(X, c) > 0.0001


def blend_poses(a, b, mask, blend_frames=9):
    conv_filter = torch.ones((1, blend_frames), dtype=torch.float32, device=mask.device) / blend_frames
    blend_mask = torch.zeros_like(mask, dtype=torch.float32)
    blend_mask[mask] = 1
    if blend_frames > 1:
        blend_mask = F.conv1d(blend_mask.t()[:, None], conv_filter[:, None, :], padding=(blend_frames - 1) // 2)[:, 0].t()
    blend_mask = blend_mask.reshape(-1, 1).repeat(1, 3).view(blend_mask.shape[0], blend_mask.shape[1] * 3)

    c = b * blend_mask
    c += a * (1 - blend_mask)
    return c


def blend_poses_past(a, b, mask, past_frames=4):
    blend_frames = past_frames * 2 + 1
    conv_filter = torch.ones((1, blend_frames), dtype=torch.float32, device=mask.device) / (past_frames + 1)
    conv_filter[0, past_frames + 1:] = 0
    blend_mask = torch.zeros_like(mask, dtype=torch.float32)
    blend_mask[mask] = 1
    if blend_frames > 1:
        blend_mask = F.conv1d(blend_mask.t()[:, None], conv_filter[:, None, :], padding=(blend_frames - 1) // 2)[:, 0].t()
    blend_mask = blend_mask.reshape(-1, 1).repeat(1, 3).view(blend_mask.shape[0], blend_mask.shape[1] * 3)

    c = b * blend_mask
    c += a * (1 - blend_mask)
    return c


def pos_smooth(p, n=4):
    p_tmp = torch.zeros_like(p)
    count_tmp = torch.zeros_like(p)
    for i in range(-n // 2, n // 2 + 1, 1):
        i0 = max(i, 0)
        i1 = min(i + len(p_tmp), len(p_tmp))
        p_tmp[i0 - i:i1 - i] += p[i0: i1]
        count_tmp[i0 - i:i1 - i] += 1
    return p_tmp / count_tmp


def pos_smooth_past(p, n=4):
    p_tmp = torch.zeros_like(p)
    count_tmp = torch.zeros_like(p)
    for i in range(-n + 1, 1):
        i0 = max(i, 0)
        i1 = min(i + len(p_tmp), len(p_tmp))
        p_tmp[i0 - i:i1 - i] += p[i0: i1] * i
        count_tmp[i0 - i:i1 - i] += i
    return p_tmp / count_tmp


# source https://answers.unity.com/questions/24756/formula-behind-smoothdamp.html
def pos_smooth_past_momentum(p, smooth_frames=5, max_speed=10):
    p_tmp = torch.zeros_like(p)
    speed = p[1] - p[0]
    p_tmp[0] = p[0]
    delta_time = 0.033
    smooth_time = delta_time * smooth_frames
    for i in range(1, len(p)):
        current = p_tmp[i - 1]
        target = p[i]
        num = 2.0 / smooth_time
        num2 = num * delta_time
        num3 = 1.0 / (1.0 + num2 + 0.48 * num2 * num2 + 0.235 * num2 * num2 * num2)
        num4 = current - target
        num5 = target
        num6 = max_speed * smooth_time
        num4 = torch.clamp(num4, -num6, num6)
        target = current - num4
        num7 = (speed + num * num4) * delta_time
        speed = (speed - num * num7) * num3
        num8 = target + (num4 + num7) * num3
        is_zero = (num5 - current > 0.0) * (num8 > num5) * (num8 == 0.0)
        num8[is_zero] = num5[is_zero]
        speed[is_zero] = (num8 - num5)[is_zero] / delta_time
        p_tmp[i] = num8

    return p_tmp


def force_correct_finger_lengths(p, config, bone_lengths):
    parent_idx = config.skeleton.parent_idx_vector()
    finger_idx = config.skeleton.Idx.get_finger_joints()

    p3 = p.reshape((len(p), -1, 3))
    new_p3 = p3.clone()

    for hand in finger_idx:
        for idx in hand:
            gt_len = torch_tile(bone_lengths[idx][None], n_tile=len(p3), dim=0)
            p_par = p3[:, parent_idx[idx]]
            dir = p3[:, idx] - p_par
            cur_len = torch.norm(dir, dim=1)
            len_fac = gt_len / cur_len
            new_p_par = new_p3[:, parent_idx[idx]]
            new_p3[:, idx] = new_p_par + dir * len_fac[:, None]

    return new_p3.reshape(len(p), -1)


# source: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
def percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
