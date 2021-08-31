# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from definitions import path_data


class MyModule(nn.Module):
    name = ""
    suffix = ""
    n_hidden = 0
    recurrent = False

    def __init__(self, suffix="", **kwargs):
        super(MyModule, self).__init__()
        self.suffix = suffix

    def _path(self):
        return os.path.join(path_data, self.name + self.suffix + f"_{self.n_hidden}.dat")

    def save(self):
        torch.save(self.state_dict(), self._path())

    def load(self):
        model = torch.load(self._path())
        # no idea why, but sometimes torch.load returns an ordered_dict...
        if type(model) == type(OrderedDict()):
            self.load_state_dict(model)
        else:
            self.load_state_dict(model.state_dict())

    def post_loss_activation(self, x):
        return x


class GRU_Single_Dense_Single_Skip_Dropout_Learn_Occ(MyModule):
    name = "GRU_Single_Dense_Single_Skip_Dropout_Learn_Occ"
    recurrent = True

    def __init__(self, n_in, n_out, device, n_hidden=512, sequence_length=20, activation=F.relu, n_occ_mask=74, **kwargs):
        super(GRU_Single_Dense_Single_Skip_Dropout_Learn_Occ, self).__init__(**kwargs)
        self.activation = activation
        self.n_in_wo_occ = n_in - n_occ_mask
        self.n_occ_mask = n_occ_mask
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.hidden = None
        self.device = device
        self.gru_layers = 1
        self.hidden = None
        self.gru = nn.GRU(input_size=self.n_in_wo_occ, hidden_size=n_hidden, num_layers=self.gru_layers, batch_first=True).to(device)
        self.dropout = nn.Dropout(0.2)
        self.fc1_pose = nn.Linear(n_hidden + self.n_in_wo_occ, n_hidden).to(device)
        self.fc1_occ = nn.Linear(n_hidden + self.n_in_wo_occ, n_hidden).to(device)
        self.fc2_pose = nn.Linear(n_hidden, self.n_in_wo_occ).to(device)
        self.fc2_occ = nn.Linear(n_hidden, n_occ_mask).to(device)

    def forward(self, x):
        X_wo_occ_mask = x[:, :, :self.n_in_wo_occ]
        if self.hidden is None or self.hidden.shape[1] != X_wo_occ_mask.shape[0]:
            self.init_hidden(X_wo_occ_mask.shape[0])
        out, self.hidden = self.gru(X_wo_occ_mask, self.hidden)
        out = torch.cat([out[:, -1], X_wo_occ_mask[:, -1]], dim=1)
        out_pose = self.activation(self.fc1_pose(out))
        out_pose = self.fc2_pose(out_pose)
        out_occ = self.activation(self.fc1_occ(out))
        out_occ = self.fc2_occ(out_occ)
        out = torch.cat([out_pose, out_occ], dim=1)
        return out

    def post_loss_activation(self, x):
        x[:, -self.n_occ_mask] = torch.sigmoid(x[:, -self.n_occ_mask])
        return x

    def init_hidden(self, batch_size):
        self.hidden = torch.autograd.Variable(torch.randn(self.gru_layers, batch_size, self.n_hidden)).to(self.device)


class GRU_Single_Dense_Single_Skip_Dropout(MyModule):
    name = "GRU_Single_Dense_Single_Skip_Dropout"
    recurrent = True

    def __init__(self, n_in, n_out, device, n_hidden=512, sequence_length=20, activation=F.relu, **kwargs):
        super(GRU_Single_Dense_Single_Skip_Dropout, self).__init__(**kwargs)
        self.activation = activation
        self.sequence_length = sequence_length
        self.n_hidden = n_hidden
        self.hidden = None
        self.device = device
        self.gru_layers = 1
        self.hidden = None
        self.gru = nn.GRU(input_size=n_in, hidden_size=n_hidden, num_layers=self.gru_layers, batch_first=True, dropout=0.25).to(
            device)
        self.fc1_pose = nn.Linear(n_hidden + n_in, n_hidden).to(device)
        self.fc2_pose = nn.Linear(n_hidden, n_out).to(device)

    def forward(self, x):
        if self.hidden.shape[1] != len(x):
            self.init_hidden(len(x))

        out, self.hidden = self.gru(x, self.hidden)
        out = torch.cat([out[:, -1], x[:, -1]], dim=1)
        out_pose = self.activation(self.fc1_pose(out))
        out_pose = self.fc2_pose(out_pose)
        out = out_pose
        return out

    def init_hidden(self, batch_size):
        self.hidden = torch.autograd.Variable(torch.randn(self.gru_layers, batch_size, self.n_hidden)).to(self.device)
