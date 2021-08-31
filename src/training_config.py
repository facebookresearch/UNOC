# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import models
import definitions
import torch.optim as optim
import featuresets
from utils import angular_loss, weighted_mse_loss, mse_loss, l1_loss
import utils
from os import getenv


class Train_Config:
    def __init__(self,
                 in_features: featuresets.Featureset,
                 out_features: featuresets.Featureset,
                 model: models.MyModule,
                 model_kwargs,
                 device=None,
                 batch_size=None,
                 sequence_length=0,
                 sequence_distance=1,
                 sequence_skip=0,
                 input_size=1e4,
                 skeleton=definitions.Skeleton,
                 loss="mse",
                 epochs=10,
                 learning_rate=1e-3,
                 weight_decay=0,
                 use_amsgrad=False,
                 max_batch_size=None,
                 scale=utils.PytNonScaler,
                 normalized_bones=False,
                 random_noise=0,
                 rotate_random=0,
                 translate_random=0,
                 random_occlusion=0,
                 eval_occluded_joints=True,
                 epoch_update=None,
                 mask_noise=None
                 ):
        """
        :param in_features: X feature set
        :param out_features: y feature set
        :param model: Network
        :param model_kwargs: Network additional params
        :param device: "cuda" or "cpu"
        :param batch_size: for training
        :param sequence_length: how many frames of history are seen by recurrent nets
        :param sequence_distance: num of frames between each recurrent sample (if 10, we see sequence_length * 10 frames into past)
        :param sequence_skip: num of frames between two samples (if 10, we only have 1/10 of the training data)
        :param input_size: num of samples we load in each iteration
        :param skeleton: Body Skeleton definition
        :param loss: loss function, can be a combined one if utils.combined_loss is used
        :param epochs: how many epochs do we want to train
        :param max_batch_size: how far can we increase the batch size during training - mainly a hardware limit
        :param scale: what scaling method should we use?
        :param normalized_bones: if all animations should use the same skeleton
        :param random_noise: amount of random noise applied during training
        :param rotate_random: number of radians the input positions are rotated around the y-axis during training
        """
        self.in_features = in_features
        self.out_features = out_features
        self.model_class = model
        self.device = 'cuda:' + getenv("CUDA_DEVICE", '0') if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model = model(in_features.data_to_features().entries,
                           out_features.data_to_features().entries,
                           device=self.device, sequence_length=sequence_length, **model_kwargs)
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sequence_distance = sequence_distance
        self.sequence_skip = sequence_skip
        self.input_size = input_size
        self.skeleton = skeleton
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=use_amsgrad
        )
        self.max_batch_size = max_batch_size if max_batch_size is not None else batch_size
        self.scale = scale
        self.normalized_bones = normalized_bones
        self.x_scaler = None
        self.y_scaler = None
        self.random_noise = random_noise
        self.rotate_random = rotate_random
        self.random_occlusion = random_occlusion
        self.translate_random = translate_random
        self.eval_occluded_joints = eval_occluded_joints
        self.epoch_update = epoch_update
        self.training = True
        self.mask_noise = mask_noise

    def get_loss_fun(self):
        if self.loss == "mse":
            return mse_loss
        if self.loss == "l1":
            return l1_loss
        if self.loss == "angular":
            return angular_loss
        if self.loss == "weighted_mse":
            return weighted_mse_loss
        if callable(self.loss):
            return self.loss

    def set_batch_size(self, size):
        self.batch_size = size
        if self.model.recurrent:
            self.model.init_hidden(size)

    def reset_model(self):
        self.model = self.model_class(self.in_features.data_to_features().entries,
                                      self.out_features.data_to_features().entries,
                                      device=self.device, sequence_length=self.sequence_length, **self.model_kwargs)
        self.optimizer = optim.Adam(self.model.parameters())

    def summary(self):
        text = "========== Configuration summary =========="
        text += f"\nin_features_name = {self.in_features.base_name} replace_occluded = {self.in_features.replace_occluded}"
        text += f"\nout_features_name = {self.out_features.base_name} replace_occluded = {self.out_features.replace_occluded}"
        text += f"\nmodel = {self.model.name}, \n   args = {self.model_kwargs}, \n   recurrent = {'True' if self.model.recurrent is True else 'False'}"
        text += f"\nbatch = {self.batch_size}, input_size = {self.input_size}, max_batch = {self.max_batch_size}"
        text += f"\nsequence length = {self.sequence_length}, distance = {self.sequence_distance}, skip = {self.sequence_skip}"
        text += f"\nreplace occluded = {self.in_features.replace_occluded}"
        text += f"\nscale = {self.scale},  normalized_bones = {self.normalized_bones}"
        text += f"\nrandom: rot = {self.rotate_random}   trans = {self.translate_random}\n   noise = {self.random_noise}  occ = {self.random_occlusion}"
        text += f"\ntraining = {self.training}"
        text += f"\noptimizer = {self.optimizer}"

        return text
