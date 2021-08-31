# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import featuresets
from definitions import Skeleton
from training_config import Train_Config
import models
import torch
import utils
from functools import partial


def Single_GRU_head_local_body_hand_local_finger_pos_learn_occ_parent_local_loss_little_rand_rot_class_loss():
    head_idx = Skeleton.Idx.head
    config = Train_Config(
        in_features=featuresets.HeadLocalPosHandLocalFingersGlobalPositionPreserving(replace_occluded="last_known"),
        out_features=featuresets.HeadLocalPosHandLocalFingersGlobalPositionPreservingBinaryOcc(replace_occluded="keep"),
        model=models.GRU_Single_Dense_Single_Skip_Dropout_Learn_Occ,
        model_kwargs={"n_hidden": 512, "activation": torch.relu, "suffix": "_head_local_pos_learn_occ"},
        batch_size=512,
        sequence_length=27,
        sequence_distance=4,  # only use every 4th frame, leading to 30 FPS
        sequence_skip=1,
        input_size=int(2e4),
        skeleton=Skeleton,
        loss=partial(utils.combined_loss,
                     losses=[
                         (partial(utils.weighted_mse_finger_classifier_occ_mask_occ_learning_loss, weight=1.2, finger_weight=10.0, mask_weight=0.0025),
                          1100),
                         (partial(utils.weighted_mse_parent_local_occ_learning_loss, weight=1.2, mask_threshold=0.0), 300),
                         (partial(utils.mse_finger_weighted_bone_length_loss_direct, use_combined_pose=False, finger_weight=5.0, ignore_joint=head_idx,
                                  wrist_local_fingers=True), 700)
                     ]),
        epochs=1,
        learning_rate=0.001,
        weight_decay=0,
        use_amsgrad=True,
        scale=utils.PytNonScaler,
        normalized_bones=True,
        max_batch_size=1024,
        random_noise=0.00,
        rotate_random=0.3,
        random_occlusion=None,
    )

    return config


def single_gru_hip_forward_local_pos_wrist_up_forward_input_wrist_local_finger_prediction():
    config = Train_Config(
        in_features=featuresets.HipForwardLocalPosWristUpForwardNoFinger(replace_occluded=None),
        out_features=featuresets.HandZeroRotationLocalFingersOnly(replace_occluded=None),
        model=models.GRU_Single_Dense_Single_Skip_Dropout,
        model_kwargs={"n_hidden": 512, "gru_layers": 1, "activation": torch.relu,
                      "out_activation": utils.direct, "suffix": "_hip_forward_local_pos_wrist_up_forward_input_wrist_local_finger"},
        batch_size=512,
        sequence_length=27,
        sequence_distance=4,  # only use every 4th frame, leading to 30 FPS
        sequence_skip=1,
        input_size=int(2e4),
        skeleton=Skeleton,
        loss=partial(utils.combined_loss,
                     losses=[
                         (utils.mse_loss, 2000),
                         (utils.mse_finger_length_loss_finger_pos_only_input, 1000),
                     ]),
        epochs=1,
        learning_rate=0.001,
        scale=utils.PytNonScaler,
        normalized_bones=True,
        max_batch_size=2048,
        random_noise=0.00,
        rotate_random=0.0,
        random_occlusion=None
    )

    return config
