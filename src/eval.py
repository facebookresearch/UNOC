# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

import definitions
import featuresets
from data_creation_torch import create_training_data_config
from parsers.unoc_parser import UnocParser
from plots import Vispy3DScatter
from training_config import Train_Config
import training_config_instance as configs
from utils import torch_tile, bone_lengths, get_joint_mask, pos_smooth_past_momentum, get_float_mask, percentile

def print_pose_accuracy(joint_mask, ref_pos, pred_pos, c: Train_Config):
    distances = torch.norm((pred_pos - ref_pos).reshape((len(pred_pos), -1, 3)), dim=2)
    distances_flat = distances.reshape(-1)
    mean_acc = torch.sqrt(torch.mean(distances_flat ** 2))

    non_finger_j = list(c.skeleton.Idx.get_non_finger_joints().values())
    non_finger_distances = distances[:, non_finger_j]
    non_finger_distances_flat = non_finger_distances.reshape(-1)
    non_finger_mean_acc = torch.sqrt(torch.mean(non_finger_distances_flat ** 2))

    if c.eval_occluded_joints:
        untracked_acc = torch.sqrt(torch.mean(distances_flat[joint_mask.reshape(-1)] ** 2))
        non_finger_untracked_acc = torch.sqrt(torch.mean(non_finger_distances_flat[joint_mask[:, non_finger_j].reshape(-1)] ** 2))
    else:
        untracked_acc = 0.0
        non_finger_untracked_acc = 0.0

    print(f"rmse acc = {mean_acc:.3f}m, rmse untracked acc = {untracked_acc:.3f}m, rmse acc non finger = {non_finger_mean_acc:.3f}m, rmse untracked acc "
          f"non_finger = {non_finger_untracked_acc:.3f}m")
    if c.eval_occluded_joints:
        return distances_flat, distances_flat[joint_mask.reshape(-1)], distances_flat[~joint_mask.reshape(-1)]
    else:
        return distances_flat, distances_flat, distances_flat


def print_val_loss(epoch, c: configs.Train_Config, criterion, X, y, train_loss, bone_lengths_, bone_offsets):
    if c.model.recurrent:
        c.model.init_hidden(X.shape[0])
    with torch.no_grad():
        out = c.model(X)

    out = c.y_scaler.inv_transform(out)
    y_scaled = c.y_scaler.inv_transform(torch.clone(y))

    if c.model.recurrent:
        X = X[:, -1]

    joint_mask = get_joint_mask(X, c) > 0.00001 if c.in_features.features_wo_occlusions() != X.shape[-1] else torch.zeros_like(y) > 0.00001
    float_mask = get_float_mask(X, c)
    mask = torch_tile(joint_mask, 1, c.out_features.features_wo_occlusions() // (joint_mask.shape[-1]))
    loss = criterion(input=out, target=y_scaled, mask=mask, config=c, bone_lengths=bone_lengths_, body=X, bone_offsets=bone_offsets, float_mask=float_mask)

    if c.eval_occluded_joints:
        joint_mask = get_joint_mask(X, c) > 0.00001 if c.in_features.features_wo_occlusions() != X.shape[-1] else torch.zeros_like(y) > 0.00001
        mask = torch_tile(joint_mask, 1, c.out_features.features_wo_occlusions() // (joint_mask.shape[-1]))
        loss = criterion(input=out, target=y_scaled, mask=mask, config=c, bone_lengths=bone_lengths_, body=X, bone_offsets=bone_offsets, float_mask=float_mask)

        p_out = out[:, :c.out_features.features_wo_occlusions()]
        p_y_scaled = y_scaled[:, :c.out_features.features_wo_occlusions()]
        tracked_inputs = mask == False
        untracked_loss = p_out[mask] - p_y_scaled[mask]
        tracked_loss = p_out[tracked_inputs] - p_y_scaled[tracked_inputs]
        untracked_rmse = torch.sqrt(torch.mean(torch.pow(untracked_loss, 2)))
        tracked_rmse = torch.sqrt(torch.mean(torch.pow(tracked_loss, 2)))
    else:
        untracked_rmse = 0.0
        tracked_rmse = 0.0

    print(
        f"\repoch: {epoch}, train loss: {train_loss:.5f} val loss: {loss:.5f}, untracked val rmse = {untracked_rmse:.4f}, tracked val rmse = {tracked_rmse:.4f}",
        end="")


def print_val_loss_tmp(epoch, c: configs.Train_Config, criterion, X, y, train_loss, bone_lengths_, c2: configs.Train_Config = None, train_loss2=None):
    if c.model.recurrent:
        c.model.init_hidden(X.shape[0])

    X = torch.clone(X[:c.max_batch_size]).to(c.device)
    y = torch.clone(y[:c.max_batch_size]).to(c.device)
    prev_out = torch.clone(y).to(c.device)

    with torch.no_grad():
        for i in range(c.sequence_length, len(X)):
            X_cat = torch.cat([X[i:i + 1], prev_out[None, i - c.sequence_length:i]], dim=2)
            _out = c.y_scaler.inv_transform(c.model(X_cat))
            prev_out[i] = torch.clone(_out)

            if c2 is not None:
                X_cat2 = torch.cat([X[i:i + 1], prev_out[None, i - c.sequence_length + 1:i + 1]], dim=2)
                if not c2.model.recurrent:
                    X_cat2 = torch.cat([X[i:i + 1, -1], prev_out[None, i], prev_out[None, i - 1]], dim=1)

                _out2 = c2.y_scaler.inv_transform(c2.model(X_cat2))
                prev_out[i] = torch.clone(_out2)

    X = X[c.sequence_length:]
    y = y[c.sequence_length:]
    out = prev_out[c.sequence_length:]

    out = c.y_scaler.inv_transform(out)
    y_scaled = c.y_scaler.inv_transform(torch.clone(y))

    if c.model.recurrent:
        X = X[:, -1]

    joint_mask = get_joint_mask(X, c) > 0.00001
    mask = torch_tile(joint_mask, 1, c.in_features.features_wo_occlusions() // (joint_mask.shape[-1]))

    prev_out_multi_frame = prev_out[c.sequence_length - 1:-1, None, :]
    for i in range(1, c.sequence_length):
        if c.sequence_length - i - 1 >= 0:
            prev_out_multi_frame = torch.cat((prev_out[c.sequence_length - i - 1:-i - 1, None, :], prev_out_multi_frame), dim=1)
        else:
            prev_out_multi_frame = torch.cat((prev_out[0:-c.sequence_length, None, :], prev_out_multi_frame), dim=1)
    loss = criterion(input=out, target=y_scaled, mask=mask, config=c, bone_lengths=bone_lengths_, prev_out=prev_out_multi_frame, body=X)

    if c2 is not None:
        loss2 = criterion(input=out, target=y_scaled, mask=mask, config=c2, bone_lengths=bone_lengths_, prev_out=prev_out_multi_frame, body=X)

    y_scaled_wo_mask = y_scaled[:, :c.out_features.features_wo_occlusions()]
    out_wo_mask = out[:, :c.out_features.features_wo_occlusions()]
    tracked_inputs = mask == False
    untracked_loss = out_wo_mask[mask] - y_scaled_wo_mask[mask]
    tracked_loss = out_wo_mask[tracked_inputs] - y_scaled_wo_mask[tracked_inputs]
    untracked_rmse = torch.sqrt(torch.mean(torch.pow(untracked_loss, 2)))
    tracked_rmse = torch.sqrt(torch.mean(torch.pow(tracked_loss, 2)))

    print(
        f"\repoch: {epoch}, train loss: {train_loss:.5f} {f'c2 train loss {train_loss2:.5f}' if c2 is not None else ''} "
        f"val loss: {loss:.5f} {f'c2 val loss {loss2:.5f}' if c2 is not None else ''}, "
        f"untracked val rmse = {untracked_rmse:.4f}, tracked val rmse = {tracked_rmse:.4f}",
        end="")


def plot_evaluation(X, y, out, data, config: Train_Config, untracked_only=True, parent_indices=None, vispy=None, mask=None, **kwargs):
    p_X = config.in_features.features_to_pose().solve_batch(X, ref_positions=None, mask=None, body=X, **kwargs)
    p_out = config.out_features.features_to_pose().solve_batch(out, ref_positions=None, mask=None, body=X, **kwargs)
    p_y = config.out_features.features_to_pose().solve_batch(y, ref_positions=None, mask=None, body=X, **kwargs)

    plot_smoothed = False
    if plot_smoothed:
        p_X = pos_smooth_past_momentum(p_out, smooth_frames=5, max_speed=5)
        p_y = pos_smooth_past_momentum(p_out, smooth_frames=3, max_speed=5)
        p_out = pos_smooth_past_momentum(p_out, smooth_frames=2, max_speed=5)

    X = X.cpu().detach().numpy()
    p_X = p_X.detach().cpu().numpy()
    p_out = p_out.detach().cpu().numpy()
    p_y = p_y.detach().cpu().numpy()

    p_out = p_out.reshape((len(X), -1, 3))
    p_y = p_y.reshape((len(X), -1, 3))
    p_X = p_X.reshape((len(X), -1, 3))

    p_out[:, :, [0, 2]] -= p_out[:, None, config.skeleton.Idx.root, [0, 2]]
    p_y[:, :, [0, 2]] -= p_y[:, None, config.skeleton.Idx.root, [0, 2]]
    p_X[:, :, [0, 2]] -= p_X[:, None, config.skeleton.Idx.root, [0, 2]]

    p_out[:, :, 0] -= 1
    p_X[:, :, 0] += 1

    if vispy is None:
        vispy = Vispy3DScatter()

    X_col = np.array([.2, .2, .2, 1])

    out_colors = np.zeros((X.shape[0], len(config.skeleton.Idx.all), 4), dtype=np.float32)
    out_colors[:, :] = np.array([0, 0, 1, 1])
    Y_colors = np.zeros(out_colors.shape, dtype=np.float32)
    Y_colors[:, :] = np.array([0, 1, 0, 1])
    if config.eval_occluded_joints:
        Y_colors[X[:, config.in_features.features_wo_occlusions():] > 0.00001] = np.array([1, 0, 0, 1])

    out_colors = Y_colors

    X_colors = np.zeros(out_colors.shape, dtype=np.float32)
    X_colors[:, :] = X_col
    if config.eval_occluded_joints:
        X_colors[X[:, config.in_features.features_wo_occlusions():] > 0.00001] = np.array([1, 0, 0, 1])

    points = np.concatenate((p_out, p_y, p_X), axis=1)
    colors = np.concatenate((out_colors, Y_colors, X_colors), axis=1)

    bones = None
    if parent_indices is not None:
        if not isinstance(parent_indices, dict):
            new_par_idx = {}
            for i, par in enumerate(parent_indices):
                new_par_idx[i] = par
            parent_indices = new_par_idx

        bones = np.zeros((y.shape[0], p_y.shape[1] * 2, 3))
        keys = list(parent_indices.keys())
        for self_idx, par_idx in parent_indices.items():
            if par_idx < 0:
                continue
            bones[:, keys.index(self_idx) * 2] = p_y[:, keys.index(par_idx), [0, 2, 1]]
            bones[:, keys.index(self_idx) * 2 + 1] = p_y[:, keys.index(self_idx), [0, 2, 1]]

    if untracked_only:
        untracked_indices = np.max(X[:, config.in_features.features_wo_occlusions():], axis=1) > 0.00001
        points = points[untracked_indices]
        if bones is not None:
            bones = bones[untracked_indices]
        colors = colors[untracked_indices]

    occ_mask = get_joint_mask(X, config)

    vispy.plot_skeleton_with_bones(
        points[:, :, [0, 2, 1]],
        config.skeleton,
        p_colors=colors,
        l_colors=np.array([[0., 0., 1., 1.], [0., 1., 0., 1.], list(X_col)]),
        speed=2,
        fps=data.dataset_fps * 2 // config.sequence_distance,
        n_skeletons=3,
        red_bones_for_skeleton=[0, 1, 2] if config.eval_occluded_joints else None,
        occlusions=occ_mask,
        center=np.array([0, 0, 1])
    )

    return vispy


def get_bone_length(par_indices):
    features = featuresets.global_pos_zeroXY(replace_occluded=None)
    X = torch.load(definitions.path_data + features.name + "0.dat")
    return bone_lengths(X[0].reshape(-1, 3), par_indices)


def get_bone_length_dataset(par_indices, data):
    return data.get_bone_lengths(par_indices, normalized=True)


def get_bone_offsets_dataset_normalized(_parser: UnocParser):
    parser = _parser.__class__()
    skeleton = parser.get_normalized_skeleton()
    return torch.from_numpy(skeleton.bone_offset_vector() / parser.scale).float()


def get_empty_joint_acc_dict(c: Train_Config):
    if c.skeleton == definitions.Skeleton:
        return {
            "neck": [],
            "shoulder": [],
            "elbow": [],
        }
    return {
        "shoulder": [],
        "elbow": [],
        "wrist": [],
        "finger": [],
        "finger_wo_thumb": [],
        "thumb": [],
        "wrist_local_finger": [],
        "hip": [],
        "knee": [],
        "foot": [],
    }


def get_joint_group_idx(c: Train_Config):
    I = c.skeleton.Idx
    if c.skeleton == definitions.Skeleton:
        return {
            "neck": [definitions.Skeleton.Idx.neck],
            "shoulder": [definitions.Skeleton.Idx.lupperarm, definitions.Skeleton.Idx.rupperarm],
            "elbow": [definitions.Skeleton.Idx.llowerarm, definitions.Skeleton.Idx.rlowerarm],
        }
    return {
        "shoulder": [I.lupperarm, I.lscap, I.lshoulder, I.rupperarm, I.rscap, I.rshoulder],
        "elbow": [I.llowerarm, I.rlowerarm],
        "wrist": [I.lwrist, I.rwrist],
        "finger": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "finger_wo_thumb": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
        ],
        "thumb": [
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "wrist_local_finger": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "hip": [I.lupperleg, I.rupperleg],
        "knee": [I.llowerleg, I.rlowerleg],
        "foot": [I.lfoot, I.lfootball, I.rfoot, I.rfootball]
    }


def get_reduced_joint_group_idx(c: Train_Config):
    I = c.skeleton.Idx
    if c.skeleton == definitions.Skeleton:
        return {
            "neck": [definitions.Skeleton.Idx.neck],
            "shoulder": [definitions.Skeleton.Idx.lupperarm, definitions.Skeleton.Idx.rupperarm],
            "elbow": [definitions.Skeleton.Idx.llowerarm, definitions.Skeleton.Idx.rlowerarm],
        }
    return {
        "shoulder": [I.lupperarm, I.rupperarm],
        "elbow": [I.llowerarm, I.rlowerarm],
        "wrist": [I.lwrist, I.rwrist],
        "finger": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "finger_wo_thumb": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
        ],
        "thumb": [
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "wrist_local_finger": [
            I.lindex1, I.lindex2, I.lindex3,
            I.lmiddle1, I.lmiddle2, I.lmiddle3,
            I.lring1, I.lring2, I.lring3,
            I.lpinky1, I.lpinky2, I.lpinky3,
            I.lthumb0, I.lthumb1, I.lthumb2, I.lthumb3,
            I.rindex1, I.rindex2, I.rindex3,
            I.rmiddle1, I.rmiddle2, I.rmiddle3,
            I.rring1, I.rring2, I.rring3,
            I.rpinky1, I.rpinky2, I.rpinky3,
            I.rthumb0, I.rthumb1, I.rthumb2, I.rthumb3,
        ],
        "hip": [I.lupperleg, I.rupperleg],
        "knee": [I.llowerleg, I.rlowerleg],
        "foot": [I.lfoot, I.rfoot]
    }


def append_batch_occ_acc(joint_occ, joint_acc_all, joint_acc_untracked, joint_acc_tracked, mask, y_batch, out, c: Train_Config, bone_lengths, X_batch,
                         bone_offsets):
    ref_pose = c.out_features.features_to_pose().solve_batch(y_batch, ref_positions=None, mask=None, bone_lengths=bone_lengths, body=X_batch,
                                                             bone_offsets=bone_offsets)
    ref_pose = ref_pose.reshape([out.shape[0], -1, 3])
    pred_pose = c.out_features.features_to_pose().solve_batch(out, ref_positions=None, mask=None, bone_lengths=bone_lengths, body=X_batch,
                                                              bone_offsets=bone_offsets)
    pred_pose = pred_pose.reshape([out.shape[0], -1, 3])

    Idx = get_reduced_joint_group_idx(c)

    for k, v in Idx.items():
        err = None
        joint_mask = None
        for group_idx, joint_idx in enumerate(v):
            if k == "wrist_local_finger":
                wrist_idx = Idx['wrist'][0] if group_idx < len(v) / 2 else Idx['wrist'][1]
                ref_wrist = ref_pose[:, wrist_idx]
                pred_wrist = pred_pose[:, wrist_idx]
                ref_p = ref_pose[:, joint_idx] - ref_wrist
                pred_p = pred_pose[:, joint_idx] - pred_wrist
                _err = torch.norm(ref_p - pred_p, dim=1) ** 2
            else:
                _err = torch.norm(ref_pose[:, joint_idx] - pred_pose[:, joint_idx], dim=1) ** 2
            if err is None:
                err = _err
                joint_mask = torch.clone(mask[:, joint_idx])
            else:
                err = torch.cat([err, _err], dim=0)
                joint_mask = torch.cat([joint_mask, torch.clone(mask[:, joint_idx])], dim=0)

        mask_float = torch.zeros(joint_mask.shape, dtype=torch.float32)
        mask_float[joint_mask] += 1.0
        occ = mask_float
        err_all = err
        if torch.mean(occ) > 0.000001:
            err_untracked = err[joint_mask]
        else:
            err_untracked = err_all * 0.0
        if torch.mean(occ) < 0.999999:
            err_tracked = err[~joint_mask]
        else:
            err_tracked = err_all * 0.0
        joint_occ[k].append(occ)
        joint_acc_all[k].append(err_all)
        joint_acc_tracked[k].append(err_tracked)
        joint_acc_untracked[k].append(err_untracked)


def append_bone_length_acc(bone_length_acc_ratio, bone_length_acc_distance, mask, out, c: Train_Config, bone_lengths, X_batch, bone_offsets):
    pred_pose = c.out_features.features_to_pose().solve_batch(out, ref_positions=None, mask=None, bone_lengths=bone_lengths, body=X_batch,
                                                              bone_offsets=bone_offsets)

    pred_pose = pred_pose.reshape([out.shape[0], -1, 3])

    Idx = get_joint_group_idx(c)
    parent_idx = c.skeleton.parent_idx_vector()

    for k, v in Idx.items():
        err_ratio = None
        err_distance = None
        joint_mask = None
        for joint_idx in v:
            p_parent = pred_pose[:, parent_idx[joint_idx]] if parent_idx[joint_idx] >= 0 else torch.zeros_like(pred_pose[:, joint_idx])
            bone_len = torch.norm(p_parent - pred_pose[:, joint_idx], dim=1)
            _err_ratio = bone_len[:] / bone_lengths[joint_idx]
            _err_ratio[_err_ratio < 1] = 1.0 / _err_ratio[_err_ratio < 1]
            _err_ratio -= 1
            _err_distance = (bone_len[:] - bone_lengths[joint_idx])
            if err_ratio is None:
                err_ratio = _err_ratio
                err_distance = _err_distance
                joint_mask = torch.clone(mask[:, joint_idx])
            else:
                err_ratio = torch.cat([err_ratio, _err_ratio], dim=0)
                err_distance = torch.cat([err_distance, _err_distance], dim=0)
                joint_mask = torch.cat([joint_mask, torch.clone(mask[:, joint_idx])], dim=0)

        mask_float = torch.zeros(joint_mask.shape, dtype=torch.float32)
        mask_float[joint_mask] += 1.0
        bone_length_acc_ratio[k].append(err_ratio)
        bone_length_acc_distance[k].append(err_distance)


def print_joint_occ_acc(joint_occ, joint_acc_all, joint_acc_untracked, joint_acc_tracked):
    body_occ = None
    body_acc_all = None
    body_acc_untracked = None
    body_acc_tracked = None

    for k in joint_occ.keys():
        mean_occ = torch.mean(torch.cat(joint_occ[k], dim=0)).item()
        rmse_acc_all = torch.sqrt(torch.mean(torch.cat(joint_acc_all[k], dim=0))).item()
        rmse_acc_untracked = torch.sqrt(torch.mean(torch.cat(joint_acc_untracked[k], dim=0))).item()
        rmse_acc_tracked = torch.sqrt(torch.mean(torch.cat(joint_acc_tracked[k], dim=0))).item()

        mean_acc_all = torch.mean(torch.sqrt(torch.cat(joint_acc_all[k], dim=0))).item()
        mean_acc_untracked = torch.mean(torch.sqrt(torch.cat(joint_acc_untracked[k], dim=0))).item()
        mean_acc_tracked = torch.mean(torch.sqrt(torch.cat(joint_acc_tracked[k], dim=0))).item()

        if not k.startswith("finger") and k != "thumb" and k != "wrist_local_finger":
            if body_occ is None:
                body_occ = torch.cat(joint_occ[k], dim=0)
                body_acc_all = torch.cat(joint_acc_all[k], dim=0)
                body_acc_untracked = torch.cat(joint_acc_untracked[k], dim=0)
                body_acc_tracked = torch.cat(joint_acc_tracked[k], dim=0)
            else:
                body_occ = torch.cat([body_occ, *joint_occ[k]], dim=0)
                body_acc_all = torch.cat([body_acc_all, *joint_acc_all[k]], dim=0)
                body_acc_untracked = torch.cat([body_acc_untracked, *joint_acc_untracked[k]], dim=0)
                body_acc_tracked = torch.cat([body_acc_tracked, *joint_acc_tracked[k]], dim=0)

        print(f"{k:<20} occ: {mean_occ:.2f}   "
              f"acc_all: {rmse_acc_all:.3f} ({mean_acc_all:.3f})    "
              f"acc_untracked: {rmse_acc_untracked:.3f} ({mean_acc_untracked:.3f})    "
              f"acc_tracked: {rmse_acc_tracked:.3f} ({mean_acc_tracked:.3f})")

    mean_body_occ = torch.mean(body_occ).item()
    rmse_body_acc_all = torch.sqrt(torch.mean(body_acc_all)).item()
    rmse_body_acc_untracked = torch.sqrt(torch.mean(body_acc_untracked)).item()
    rmse_body_acc_tracked = torch.sqrt(torch.mean(body_acc_tracked)).item()
    mean_body_acc_all = torch.mean(torch.sqrt(body_acc_all)).item()
    mean_body_acc_untracked = torch.mean(torch.sqrt(body_acc_untracked)).item()
    mean_body_acc_tracked = torch.mean(torch.sqrt(body_acc_tracked)).item()

    print(f"{'body':<20} occ: {mean_body_occ:.2f}   "
          f"acc_all: {rmse_body_acc_all:.3f} ({mean_body_acc_all:.3f})    "
          f"acc_untracked: {rmse_body_acc_untracked:.3f} ({mean_body_acc_untracked:.3f})    "
          f"acc_tracked: {rmse_body_acc_tracked:.3f} ({mean_body_acc_tracked:.3f})")


def print_bone_length_acc(bone_length_acc_ratio, bone_length_acc_distance, c: Train_Config, bone_lengths):
    body_acc_ratio = None
    body_acc_distance = None
    body_lengths = None
    all_ratio = None
    all_distance = None
    all_lengths = None

    joint_groups = get_joint_group_idx(c)
    group_bone_lengths = get_empty_joint_acc_dict(c)
    for group_name, joints in joint_groups.items():
        for j in joints:
            group_bone_lengths[group_name].append(bone_lengths[j].item())

    for k in bone_length_acc_ratio.keys():
        acc_ratio = torch.mean(torch.cat(bone_length_acc_ratio[k], dim=0)).item()
        rmse_acc_distance = torch.sqrt(torch.mean(torch.cat(bone_length_acc_distance[k], dim=0) ** 2)).item()
        mean_acc_distance = torch.mean(torch.abs(torch.cat(bone_length_acc_distance[k], dim=0))).item()
        avg_bone_length = torch.mean(torch.tensor(group_bone_lengths[k])).item()

        if all_ratio is None:
            all_ratio = torch.cat(bone_length_acc_ratio[k], dim=0)
            all_distance = torch.cat(bone_length_acc_distance[k], dim=0)
            all_lengths = torch.tensor(group_bone_lengths[k])
        else:
            all_ratio = torch.cat([all_ratio, *bone_length_acc_ratio[k]], dim=0)
            all_distance = torch.cat([all_distance, *bone_length_acc_distance[k]], dim=0)
            all_lengths = torch.cat([all_lengths, torch.tensor(group_bone_lengths[k])], dim=0)

        if not k.startswith("finger") and k != "thumb" and k != "wrist_local_finger":
            if body_acc_ratio is None:
                body_acc_ratio = torch.cat(bone_length_acc_ratio[k], dim=0)
                body_acc_distance = torch.cat(bone_length_acc_distance[k], dim=0)
                body_length = torch.tensor(group_bone_lengths[k])
            else:
                body_acc_ratio = torch.cat([body_acc_ratio, *bone_length_acc_ratio[k]], dim=0)
                body_acc_distance = torch.cat([body_acc_distance, *bone_length_acc_distance[k]], dim=0)
                body_length = torch.cat([body_length, torch.tensor(group_bone_lengths[k])], dim=0)

        print(f"{k:<20}mean ratio: {acc_ratio:.3f}   distance: {rmse_acc_distance:.4f}/{mean_acc_distance:.4f}   avg length: {avg_bone_length:.4f}")

    body_acc_ratio = torch.mean(body_acc_ratio).item()
    rmse_body_acc_distance = torch.sqrt(torch.mean(body_acc_distance ** 2)).item()
    mean_body_acc_distance = torch.mean(torch.abs(body_acc_distance)).item()
    body_length = torch.mean(body_length).item()
    all_acc_ratio = torch.mean(all_ratio).item()
    rmse_all_acc_distance = torch.sqrt(torch.mean(all_distance ** 2)).item()
    mean_all_acc_distance = torch.mean(torch.abs(all_distance)).item()
    all_length = torch.mean(all_lengths).item()
    print(f"{'body':<20}mean ratio: {body_acc_ratio:.3f}   distance: {rmse_body_acc_distance:.4f}/{mean_body_acc_distance:.4f}   avg length: {body_length:.4f}")
    print(f"{'all':<20}mean ratio: {all_acc_ratio:.3f}   distance: {rmse_all_acc_distance:.4f}/{mean_all_acc_distance:.4f}   avg length: {all_length:.4f}")


def print_err(title, err_all, err_untracked, err_tracked):
    rmse_acc = torch.sqrt(torch.mean(torch.pow(torch.cat(err_all, dim=0), 2))).item()
    rmse_acc_untracked = torch.sqrt(torch.mean(torch.pow(torch.cat(err_untracked, dim=0), 2))).item()
    rmse_acc_tracked = torch.sqrt(torch.mean(torch.pow(torch.cat(err_tracked, dim=0), 2))).item()
    mean_acc = torch.mean(torch.cat(err_all, dim=0)).item()
    mean_acc_untracked = torch.mean(torch.cat(err_untracked, dim=0)).item()
    mean_acc_tracked = torch.mean(torch.cat(err_tracked, dim=0)).item()
    print(f"{title:<20}rmse acc = {rmse_acc:.3f}    rmse untracked acc = {rmse_acc_untracked:.3f}    rmse tracked acc = {rmse_acc_tracked:.3f}")
    print(f"{title:<20}mean acc = {mean_acc:.3f}    mean untracked acc = {mean_acc_untracked:.3f}    mean tracked acc = {mean_acc_tracked:.3f}")


def eval_joint_acceleration(X, y, pred, c: Train_Config, ref_pose, pred_pose, mask):
    v_ref = torch.norm((ref_pose[1:] - ref_pose[:-1]).reshape(len(ref_pose) - 1, -1, 3), dim=2)
    v_pred = torch.norm((pred_pose[1:] - pred_pose[:1]).reshape(len(ref_pose) - 1, -1, 3), dim=2)
    a_ref = v_ref[1:] - v_ref[:-1]
    a_pred = v_pred[1:] - v_pred[:-1]

    keep_frames = torch.max(a_ref, dim=1)[0] < percentile(torch.abs(a_ref), 99)

    a_ref = a_ref[keep_frames]
    a_pred = a_pred[keep_frames]
    _mask = mask[2:][keep_frames]
    a_ref = a_ref[_mask]
    a_pred = a_pred[_mask]

    a_diff = (torch.abs(a_ref) - torch.abs(a_pred))

    ref_90th_percentile = 0.03
    pred_90th_percentile = 0.03
    diff_90th_percentile = 0.03

    n_inputs = len(a_diff.reshape(-1))

    a_diff = a_diff[a_diff >= diff_90th_percentile]
    a_ref = a_ref[a_ref >= ref_90th_percentile]
    a_pred = a_pred[a_pred >= pred_90th_percentile]

    # try to remove the frames where joints snap into correct position when suddenly not occluded

    a_diff_mean = torch.mean(a_diff).item()
    a_diff_rmse = torch.sqrt(torch.mean(a_diff ** 2))
    a_ref_mean = torch.mean(torch.abs(a_ref)).item()
    a_ref_rmse = torch.sqrt(torch.mean(a_ref ** 2))
    a_pred_mean = torch.mean(torch.abs(a_pred)).item()
    a_pred_rmse = torch.sqrt(torch.mean(a_pred ** 2))
    print(f"diff rmse (mean) = {a_diff_rmse:.3f} ({a_diff_mean:.3f}), pred rmse (mean) = {a_pred_rmse:.3f} ({a_pred_mean:.3f}), "
          f"ref rmse (mean) = {a_ref_rmse:.3f} ({a_ref_mean:.3f})")
    print(f"n_frames > 0.1: ref {len(a_ref.reshape(-1)) / n_inputs:.5f}  pred  {len(a_pred.reshape(-1)) / n_inputs:.5f}")
    print("")


def eval(parser, c, update, plot=False):
    par_indices = c.skeleton.parent_idx_vector()
    bone_lengths_ = torch.from_numpy(get_bone_length_dataset(par_indices, parser)).to(c.device)
    bone_offsets = get_bone_offsets_dataset_normalized(parser).to(c.device)

    # initialize various data that we are evaluating
    acc = []
    untracked_acc = []
    tracked_acc = []
    acc_repeat = []
    untracked_acc_repeat = []
    tracked_acc_repeat = []
    joint_occ = get_empty_joint_acc_dict(c)
    joint_acc_all = get_empty_joint_acc_dict(c)
    joint_acc_untracked = get_empty_joint_acc_dict(c)
    joint_acc_tracked = get_empty_joint_acc_dict(c)
    bone_length_acc_ratio = get_empty_joint_acc_dict(c)
    bone_length_acc_distance = get_empty_joint_acc_dict(c)

    joint_occ_input = get_empty_joint_acc_dict(c)
    joint_acc_all_input = get_empty_joint_acc_dict(c)
    joint_acc_untracked_input = get_empty_joint_acc_dict(c)
    joint_acc_tracked_input = get_empty_joint_acc_dict(c)

    compare_input_acc = False
    eval_joint_acc = True
    eval_bone_length = False
    eval_jitter = False
    enable_timing = False

    vispy_plot = None

    for X_val, y_val in create_training_data_config(parser, c, update, save_all_to_mem=True, shuffle_all=False, dataset_name=parser.name(), test_set=True):
        max_samples = 200000

        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]

        if c.y_scaler is None:
            X_init = X_val[:, -1] if c.model.recurrent else X_val
            c.x_scaler = c.scale(X_init, group_size=3, idx_end=c.in_features.features_wo_occlusions())
            c.y_scaler = c.scale(y_val, group_size=3, idx_end=c.out_features.features_wo_occlusions())

        y_val = c.y_scaler.transform(y_val)
        X_val = c.x_scaler.transform_recurrent(X_val) if c.model.recurrent else c.x_scaler.transform(X_val)

        n_occ = 0
        n_sum = 0

        batch_size = X_val.shape[0] if c.batch_size is None else c.batch_size

        for batch in range(max(X_val.shape[0] // batch_size, 1)):
            X_batch = X_val[batch * batch_size: (batch + 1) * batch_size].to(c.device)
            y_batch = y_val[batch * batch_size: (batch + 1) * batch_size].to(c.device)
            with torch.no_grad():
                if enable_timing:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                out = c.model(X_batch)
                out = c.model.post_loss_activation(out)
                if enable_timing:
                    end.record()
                    torch.cuda.synchronize()
                    print(f"inference duration: {start.elapsed_time(end)}")

            if c.model.recurrent:
                mask = get_joint_mask(X_batch[:, -1], c) > 0.00001
                X_batch[:, -1] = c.x_scaler.inv_transform(X_batch[:, -1])
            else:
                mask = get_joint_mask(X_batch, c) > 0.00001
                X_batch = c.x_scaler.inv_transform(X_batch)

            n_occ += torch.sum(mask).cpu().item()
            n_sum += mask.shape[0] * mask.shape[1]

            y_batch = c.y_scaler.inv_transform(y_batch)
            out = c.y_scaler.inv_transform(out)

            X_batch_0 = X_batch[:, -1] if c.model.recurrent else X_batch

            ref_pose = c.out_features.features_to_pose().solve_batch(y_batch, ref_positions=None, mask=mask, bone_lengths=bone_lengths_, body=X_batch_0,
                                                                     bone_offsets=bone_offsets)
            pred_pose = c.out_features.features_to_pose().solve_batch(out, ref_positions=None, mask=None, bone_lengths=bone_lengths_, body=X_batch_0,
                                                                      bone_offsets=bone_offsets)

            if eval_jitter:
                eval_joint_acceleration(X_batch, y_batch, out, c, ref_pose, pred_pose, mask)

            print("use predicted pos:    ", end="")
            _acc, _untracked_acc, _tracked_acc = print_pose_accuracy(mask, ref_pose, pred_pose, c)
            if compare_input_acc:
                print("reuse last known pos: ", end="")
                in_pose = c.in_features.features_to_pose().solve_batch(X_batch_0, ref_positions=None, mask=mask, bone_lengths=bone_lengths, body=X_batch_0,
                                                                       bone_offsets=bone_offsets)
                _acc_repeat, _untracked_acc_repeat, _tracked_acc_repeat = print_pose_accuracy(mask, ref_pose, in_pose, c)
                acc_repeat.append(_acc_repeat)
                untracked_acc_repeat.append(_untracked_acc_repeat)
                tracked_acc_repeat.append(_tracked_acc_repeat)
                X_batch_wo_mask = X_batch[:, -1] if c.model.recurrent else X_batch
                X_batch_wo_mask = X_batch_wo_mask[:, :c.in_features.features_wo_occlusions()]
                if eval_joint_acc:
                    append_batch_occ_acc(joint_occ_input, joint_acc_all_input, joint_acc_untracked_input, joint_acc_tracked_input, mask, y_batch,
                                         X_batch_wo_mask, c, bone_lengths_, X_batch_0, bone_offsets=bone_offsets)

            acc.append(_acc)
            untracked_acc.append(_untracked_acc)
            tracked_acc.append(_tracked_acc)

            if eval_joint_acc:
                append_batch_occ_acc(joint_occ, joint_acc_all, joint_acc_untracked, joint_acc_tracked, mask, y_batch, out, c, bone_lengths_, X_batch_0,
                                     bone_offsets)

            if eval_bone_length:
                append_bone_length_acc(bone_length_acc_ratio, bone_length_acc_distance, mask, out, c, bone_lengths_, X_batch_0, bone_offsets)

            if plot is True or plot == 1:
                vispy_plot = plot_evaluation(X_batch_0, y_batch, out, parser, c, untracked_only=False, parent_indices=par_indices, vispy=vispy_plot,
                                             bone_lengths=bone_lengths_, bone_offsets=bone_offsets, mask=mask)
            if plot is not False and plot is not False:
                plot -= 1

    print("\n\n-------- predicted ---------")
    print_err("prediction ", acc, untracked_acc, tracked_acc)
    if eval_joint_acc:
        print_joint_occ_acc(joint_occ, joint_acc_all, joint_acc_untracked, joint_acc_tracked)

    if compare_input_acc:
        print("\n\n-------- input ---------")
        print_err("input ", acc_repeat, untracked_acc_repeat, tracked_acc_repeat)
        if eval_joint_acc:
            print_joint_occ_acc(joint_occ_input, joint_acc_all_input, joint_acc_untracked_input, joint_acc_tracked_input)

    if eval_bone_length:
        print("\n\n-------- bone lengths ---------")
        print_bone_length_acc(bone_length_acc_ratio, bone_length_acc_distance, c, bone_lengths_)


if __name__ == "__main__":
    update = False
    data = UnocParser()

    import sys

    if len(sys.argv) < 2:
        print("No argument provided. Please run using 'python eval.py <mode> <plot?>' where <mode> can be one of [body, finger]")
        exit(-1)
    mode = sys.argv[1]
    plot = sys.argv[2] == "plot" if len(sys.argv) > 2 else False

    if mode == "body":
        config = configs.Single_GRU_head_local_body_hand_local_finger_pos_learn_occ_parent_local_loss_little_rand_rot_class_loss()
    elif mode == "finger":
        config = configs.single_gru_hip_forward_local_pos_wrist_up_forward_input_wrist_local_finger_prediction()
    else:
        print(f"Invalid argument {mode} provided. Please run using 'python eval.py <mode>' where <mode> can be one of [body, finger]")

    config.sequence_distance = data.dataset_fps // 30

    config.sequence_skip = config.sequence_distance

    config.model.load()
    config.model.eval()
    config.set_batch_size(2048 * 4)
    print(config.summary())
    eval(data, config, update, plot=plot)
