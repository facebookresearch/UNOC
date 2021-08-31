# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is based on the source published under:
# SOURCE: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import torch
import numpy as np
import math_helper


# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3

    if len(v.shape) == len(q.shape) - 1:
        _v = torch.zeros((q.shape[0], 3), device=q.device)
        _v[:] += v
    else:
        _v = v

    original_shape = list(_v.shape)
    q = q.view(-1, 4)
    _v = _v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, _v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (_v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order="xyz", epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


# Numpy-backed implementations

def qmul_np(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = np.matmul(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return np.stack((w, x, y, z), axis=1).reshape(original_shape)

def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects an array of shape (*, 4) for q and a array of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a array of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    # assert q.shape[:-1] == v.shape[:-1]

    if len(v.shape) == len(q.shape) - 1:
        _v = np.zeros((q.shape[0], 3))
        _v[:] += v
    else:
        _v = v

    original_shape = list(_v.shape)
    q = q.reshape(-1, 4)
    _v = _v.reshape(-1, 3)

    qvec = q[:, 1:]
    uv = np.cross(qvec, _v, axis=1)
    uuv = np.cross(qvec, uv, axis=1)
    return (_v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


def qeuler_np(q, order="xyz", epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.reshape(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = np.arctan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = np.arcsin(np.clip(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = np.arcsin(np.clip(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = np.arctan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = np.arctan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = np.arcsin(np.clip(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = np.arcsin(np.clip(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = np.arctan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.arctan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.arcsin(np.clip(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return np.stack((x, y, z), axis=1).reshape(original_shape)


# def qeuler_np(q, order="xyz", epsilon=0, use_gpu=False):
#     if use_gpu:
#         q = torch.from_numpy(q).cuda()
#         return qeuler(q, order, epsilon).cpu().numpy()
#     else:
#         q = torch.from_numpy(q).contiguous()
#         return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def inverse(q):
    _q = torch.clone(q).view(-1, 4)
    _q[:, 1:] *= -1
    _q[:, :] /= math_helper.torch_bdot(_q, _q)[:, None]
    return _q


def inverse_np(q):
    original_shape = q.shape
    _q = np.copy(q).reshape((-1, 4))
    _q[:, 1:] *= -1
    _q[:, :] /= math_helper.np_bdot(_q, _q)[:, None]
    return _q.reshape(original_shape)


def identity(n):
    q = torch.zeros((n, 4))
    q[:, 0] = 1
    return q


def euler_to_quaternion(e, order="xyz"):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def euler_to_quaternion_torch(e, order="xyz"):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    _e = e.reshape(-1, 3)

    x = _e[:, 0]
    y = _e[:, 1]
    z = _e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), axis=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), axis=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.view(original_shape)


def quaternion_between_vectors(v_from, v_to):
    """
    calculates quaternions that simulate a rotation from v_from to v_to. no rotation along the vector axes
    """
    a = np.cross(v_from, v_to)
    l_from = np.linalg.norm(v_from, axis=1)
    l_to = np.linalg.norm(v_to, axis=1)
    w = l_from * l_to + math_helper.np_bdot(v_from, v_to)

    quaternions = np.zeros((v_from.shape[0], 4), dtype=np.float32)
    quaternions[:, 0] = w
    quaternions[:, 1:] = a
    return math_helper.normalize_batch_np(quaternions, dim=1)


def quaternion_between_vectors_torch(v_from, v_to):
    """
    calculates quaternions that simulate a rotation from v_from to v_to. no rotation along the vector axes
    """
    a = torch.cross(v_from, v_to)
    l_from = torch.norm(v_from, dim=1)
    l_to = torch.norm(v_to, dim=1)
    w = l_from * l_to + math_helper.torch_bdot(v_from, v_to)

    quaternions = torch.zeros((v_from.shape[0], 4), dtype=torch.float32, device=v_from.device)
    quaternions[:, 0] = w
    quaternions[:, 1:] = a
    return math_helper.normalize_batch(quaternions, dim=1)


def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.tensor) -> torch.Tensor:
    if not axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}"
                         .format(axis.shape))

    from math_helper import normalize_batch
    _axis_normalized = normalize_batch(axis, dim=1)
    # unpack input and compute conversion
    quaternion: torch.Tensor = torch.zeros((len(angle), 4), device=axis.device)
    sin_angle = torch.sin(angle * 0.5)
    quaternion[..., 0] += torch.cos(angle * 0.5)
    quaternion[..., 1] += _axis_normalized[:, 0] * sin_angle
    quaternion[..., 2] += _axis_normalized[:, 1] * sin_angle
    quaternion[..., 3] += _axis_normalized[:, 2] * sin_angle
    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        input = torch.rand(4, 3, 4)  # Nx3x4
        output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 3  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


# source: https://answers.unity.com/questions/467614/what-is-the-source-code-of-quaternionlookrotation.html
def up_forward_to_quat(up: torch.tensor, forward: torch.tensor, normalize=True):
    # TODO check if that really works
    vector = math_helper.normalize_batch(forward, dim=1) if normalize else forward
    vector2 = math_helper.normalize_batch(torch.cross(up, vector), dim=1) if normalize else torch.cross(up, vector)
    vector3 = torch.cross(vector, vector2)
    m00 = vector2[:, 0]
    m01 = vector2[:, 1]
    m02 = vector2[:, 2]
    m10 = vector3[:, 0]
    m11 = vector3[:, 1]
    m12 = vector3[:, 2]
    m20 = vector[:, 0]
    m21 = vector[:, 1]
    m22 = vector[:, 2]

    num8 = (m00 + m11) + m22
    q = torch.zeros((len(up), 4), device=up.device)

    type1 = num8 > 0.0
    num = torch.sqrt(num8[type1] + 1.0)
    q[type1, 0] = (num * 0.5)
    num = 0.5 / num
    q[type1, 1] = ((m12 - m21)[type1] * num)
    q[type1, 2] = ((m20 - m02)[type1] * num)
    q[type1, 3] = ((m01 - m10)[type1] * num)

    type2 = ((m00 >= m11) * (m00 >= m22)) & ~type1
    num7 = torch.sqrt((((1.0 + m00) - m11) - m22)[type2])
    num4 = 0.5 / num7
    q[type2, 1] = (0.5 * num7)
    q[type2, 2] = ((m01 + m10)[type2] * num4)
    q[type2, 3] = ((m02 + m20)[type2] * num4)
    q[type2, 0] = ((m12 - m21)[type2] * num4)

    type3 = (m11 > m22) & ~(type1 + type2)
    num6 = torch.sqrt((((1.0 + m11) - m00) - m22)[type3])
    num3 = 0.5 / num6
    q[type3, 2] = (0.5 * num6)
    q[type3, 1] = ((m10 + m01)[type3] * num3)
    q[type3, 3] = ((m21 + m12)[type3] * num3)
    q[type3, 0] = ((m20 - m02)[type3] * num3)

    type4 = ~(type1 + type2 + type3)
    num5 = torch.sqrt((((1.0 + m22) - m00) - m11)[type4])
    num2 = 0.5 / num5
    q[type4, 3] = (0.5 * num5)
    q[type4, 1] = ((m20 + m02)[type4] * num2)
    q[type4, 2] = ((m21 + m12)[type4] * num2)
    q[type4, 0] = ((m01 - m10)[type4] * num2)

    return math_helper.normalize_batch(q) if normalize else q
