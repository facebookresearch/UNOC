# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import quaternion
from parsers.bvh_converter import BvhConverter
from data_creation_torch import create_training_data_config
from eval import print_pose_accuracy, print_val_loss
from parsers.unoc_parser import UnocParser
import featuresets
import definitions

import training_config_instance as configs
from training_config import Train_Config
from utils import torch_tile, bone_lengths, shuffle_X_y, get_joint_mask, get_float_mask


def get_bone_length(par_indices):
    features = featuresets.global_pos_zeroXY(replace_occluded=None)
    X = torch.load(definitions.path_data + features.name + "0.dat")
    return bone_lengths(X[0].reshape(-1, 3), par_indices)


def get_bone_length_dataset(par_indices, data):
    return data.get_bone_lengths(par_indices, normalized=True)


def get_bone_length_from_input(par_indices, y, features: featuresets.Featureset):
    pos = features.features_to_pose().solve_batch(y[0:1], ref_positions=y[0:1], mask=None)[0]
    return bone_lengths(pos.reshape(-1, 3), par_indices)


def get_bone_offsets_dataset_normalized(_parser: UnocParser):
    parser = _parser.__class__()
    skeleton = parser.get_normalized_skeleton()
    return torch.from_numpy(skeleton.bone_offset_vector() / parser.scale).float()


def rotate_features_random(X_train_batch, c, y_train_batch):
    n_pos = c.out_features.features_wo_occlusions()
    random_rot3 = torch.zeros((len(X_train_batch), 3), device=X_train_batch.device)
    random_rot3[:, 1] += torch.rand(len(random_rot3), device=random_rot3.device) * c.rotate_random - c.rotate_random / 2
    random_quat = quaternion.euler_to_quaternion_torch(random_rot3)

    X = X_train_batch[:, :, :n_pos]
    y = y_train_batch[:, :n_pos]
    random_quat_X = torch_tile(random_quat, dim=0, n_tile=X.shape[1] * X.shape[2] // 3)
    random_quat_y = torch_tile(random_quat, dim=0, n_tile=y.shape[1] // 3)
    X_train_batch2 = quaternion.qrot(random_quat_X, X.reshape(-1, 3)).reshape_as(X)
    y_train_batch2 = quaternion.qrot(random_quat_y, y.reshape(-1, 3)).reshape_as(y)

    X_train_batch[:, :, :n_pos] = X_train_batch2
    y_train_batch[:, :n_pos] = y_train_batch2
    return X_train_batch, y_train_batch


def translate_features_random(X_train_batch, c, y_train_batch):
    n_pos = c.out_features.features_wo_occlusions()
    random_trans3 = torch.zeros((len(X_train_batch), 3), device=X_train_batch.device)
    random_trans3[:, 0] += torch.rand(len(random_trans3), device=random_trans3.device) * c.translate_random - c.translate_random / 2
    random_trans3[:, 2] += torch.rand(len(random_trans3), device=random_trans3.device) * c.translate_random - c.translate_random / 2
    for i in range(n_pos // 3):
        for j in range(c.sequence_length):
            X_train_batch[:, j, i * 3:(i + 1) * 3] += random_trans3
        y_train_batch[:, i * 3:(i + 1) * 3] += random_trans3
    return X_train_batch, y_train_batch


def add_mask_noise(X_train_batch, c, y_train_batch):
    if c.model.recurrent:
        for i in range(X_train_batch.shape[1]):
            joint_mask = get_float_mask(X_train_batch[:, i], c).clone()
            noise_mask = torch.rand(joint_mask.shape, device=X_train_batch.device) < c.mask_noise['share']
            noise = torch.rand(joint_mask.shape, device=X_train_batch.device) * c.mask_noise['intensity']
            joint_mask[noise_mask] += noise[noise_mask]
            X_train_batch[:, i, -joint_mask.shape[1]:] = joint_mask

    else:
        joint_mask = get_float_mask(X_train_batch, c).clone()
        noise_mask = torch.rand(joint_mask.shape) < c.mask_noise['share']
        noise = torch.rand(joint_mask.shape) * c.mask_noise['intensity']
        joint_mask[noise_mask] += noise[noise_mask]
        X_train_batch[:, -joint_mask.shape[1]:] = joint_mask

    return X_train_batch, y_train_batch


def eval_pose_accuracy(_X_val, c, _y_val, bone_lengths_, bone_offsets):
    print("")
    c.model.eval()
    X_val = torch.clone(_X_val[:c.max_batch_size])
    y_val = torch.clone(_y_val[:c.max_batch_size])
    with torch.no_grad():
        out = c.y_scaler.inv_transform(c.model(X_val))
    y_val = c.y_scaler.inv_transform(y_val)

    if c.model.recurrent:
        X_val = X_val[:, -1]

    mask = get_joint_mask(X_val, c) > 0.000001
    out_to_pose = c.out_features.features_to_pose()
    ref_pose = out_to_pose.solve_batch(y_val, ref_positions=None, mask=mask, bone_lengths=bone_lengths_, body=X_val, bone_offsets=bone_offsets)
    pred_pose = out_to_pose.solve_batch(out, ref_positions=None, mask=mask, bone_lengths=bone_lengths_, body=X_val, bone_offsets=bone_offsets)

    print_pose_accuracy(mask, ref_pose, pred_pose, c)
    return X_val, y_val, out


def train_config(parser: BvhConverter, c: Train_Config, update: bool):
    """
    :param parser: a parser that implements "load_numpy"
    :param c: Definition of model, skeleton, training params and feature sets
    :param update: renew training features - if False, last features will be loaded from disk
    :param plot: plot prediction after training
    :return:
    """

    criterion = c.get_loss_fun()
    par_indices = c.skeleton.parent_idx_vector()
    bone_lengths_ = torch.from_numpy(get_bone_length_dataset(par_indices, parser)).to(c.device)
    bone_offsets = get_bone_offsets_dataset_normalized(parser).to(c.device)

    for epoch in range(c.epochs):
        print("")
        for _X, _y in create_training_data_config(parser, c, update, save_all_to_mem=True, shuffle_all=True,
                                                  dataset_name=parser.name(), test_set=False):
            X = _X.to(c.device)
            y = _y.to(c.device)

            if c.y_scaler is None:
                X_init = X[:, -1] if c.model.recurrent else X
                c.x_scaler = c.scale(X_init, group_size=3, idx_end=c.in_features.features_wo_occlusions())
                c.y_scaler = c.scale(y, group_size=3, idx_end=c.out_features.features_wo_occlusions())

            y = c.y_scaler.transform(y)
            X = c.x_scaler.transform_recurrent(X) if c.model.recurrent else c.x_scaler.transform(X)

            if c.random_noise > 0:
                X += torch.rand_like(X, device=X.device) * c.random_noise - torch.ones_like(X, device=X.device) * (c.random_noise * 0.5)
                y += torch.rand_like(y, device=y.device) * c.random_noise - torch.ones_like(y, device=y.device) * (c.random_noise * 0.5)

            X_train = X[:X.shape[0] * 8 // 10]
            y_train = y[:y.shape[0] * 8 // 10]
            X_val = X[X_train.shape[0]:]
            y_val = y[y_train.shape[0]:]

            batch_size = X_train.shape[0] if c.batch_size is None else c.batch_size
            X_train, y_train = shuffle_X_y(X_train, y_train)
            batches = max(X_train.shape[0] // batch_size, 1)
            end_idx = X_train.shape[0] - (X_train.shape[0] % batch_size)

            for batch in range(batches):
                c.model.train()
                c.optimizer.zero_grad()
                X_train_batch = X_train[batch:end_idx:batches]
                y_train_batch = y_train[batch:end_idx:batches]

                if c.rotate_random > 0.0001:
                    X_train_batch, y_train_batch = rotate_features_random(X_train_batch, c, y_train_batch)

                if c.translate_random > 0.0001:
                    X_train_batch, y_train_batch = translate_features_random(X_train_batch, c, y_train_batch)

                if c.mask_noise is not None:
                    X_train_batch, y_train_batch = add_mask_noise(X_train_batch, c, y_train_batch)

                if c.model.recurrent:
                    c.model.init_hidden(batch_size)

                out = c.model(X_train_batch)

                # how many features before the occlusion mask at the end
                X_current_frame = X_train_batch[:, -1] if c.model.recurrent else X_train_batch
                joint_mask = get_joint_mask(X_current_frame, c)
                float_mask = get_float_mask(X_current_frame, c)
                loss_mask = torch_tile(joint_mask, 1, c.in_features.features_wo_occlusions() // (joint_mask.shape[-1]))

                loss = criterion(input=out, target=y_train_batch, mask=loss_mask, config=c, bone_lengths=bone_lengths_, body=X_current_frame,
                                 bone_offsets=bone_offsets, float_mask=float_mask)
                loss.backward()
                c.optimizer.step()
                out = c.model.post_loss_activation(out)

                print(f"\rloss={loss.item()}", end="")

                combine_out_finger_in_body = False

                if combine_out_finger_in_body:
                    pred_pos = c.out_features.features_to_pose().solve_batch(out.detach(), body=X_current_frame, mask=joint_mask, ref_positions=y_train_batch,
                                                                             bone_offsets=bone_offsets)
                    pred_pos = pred_pos.reshape((len(out), -1, 3))

                    from plots import Vispy3DScatter
                    vispy = Vispy3DScatter()
                    vispy.plot_skeleton_with_bones(pred_pos.cpu().numpy()[:, :, [0, 2, 1]], c.skeleton, fps=parser.dataset_fps / 30, speed=0.2)

                c.model.eval()
                if batch == batches - 1:
                    print_val_loss(epoch, c, criterion, X_val[:c.max_batch_size], y_val[:c.max_batch_size], loss, bone_lengths_, bone_offsets)
                    eval_pose_accuracy(X_val, c, y_val, bone_lengths_, bone_offsets)

        update = False

    c.model.save()


def main():
    update_training_data = False
    data = UnocParser()

    import sys

    if len(sys.argv) < 2:
        print("No argument provided. Please run using 'python eval.py <mode>' where <mode> can be one of [body, finger]")
        exit(-1)
    mode = sys.argv[1]

    if mode == "body":
        config = configs.Single_GRU_head_local_body_hand_local_finger_pos_learn_occ_parent_local_loss_little_rand_rot_class_loss()
    elif mode == "finger":
        config = configs.single_gru_hip_forward_local_pos_wrist_up_forward_input_wrist_local_finger_prediction()
    else:
        print(f"Invalid argument {mode} provided. Please run using 'python eval.py <mode>' where <mode> can be one of [body, finger]")

    # override starting batch size
    config.set_batch_size(256)
    print(config.summary())

    iterations = 5

    for i in range(iterations):
        train_config(data, config, update_training_data)
        config.set_batch_size(min(config.batch_size * 2, config.max_batch_size))

        # reuse training data after first epoch
        update_training_data = False
        if config.epoch_update is not None:
            config.epoch_update(config, i)


if __name__ == "__main__":
    main()
