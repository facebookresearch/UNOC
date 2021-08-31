# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from transforms3d.euler import euler2mat, quat2mat, mat2euler
from pyquaternion import Quaternion
import math
import torch
import quaternion


def up(r):
    return euler2mat(*r).dot(np.array([0., 1., 0.]))


def right(r):
    return euler2mat(*r).dot(np.array([1., 0., 0.]))


def forward(r):
    return euler2mat(*r).dot(np.array([0., 0., 1.]))


def up_deg(r):
    return up(np.deg2rad(r))


def right_deg(r):
    return right(np.deg2rad(r))


def forward_deg(r):
    return forward(np.deg2rad(r))


def horizontal_forward(r):
    forward_vec = euler2mat(*r).dot(np.array([0., 0., 1.]))
    forward_vec[1] *= 0.

    return normalize(forward_vec)


def quaternion_between_vectors(v_from, v_to):
    a = np.cross(v_from, v_to)
    l_from = np.linalg.norm(v_from)
    l_to = np.linalg.norm(v_to)
    w = l_from * l_to + v_from.dot(v_to)

    q = Quaternion(w, *a)
    return q.normalised


def rot_matrix_between_vectors(v_from, v_to):
    return quat2mat(quaternion_between_vectors(v_from, v_to))


def angle_between_vectors(v_from, v_to):
    return math.acos(v_from.dot(v_to))


def isosceles_triangle_height(a, b):
    return math.sqrt(max(1e-6, a * a - (b / 2) * (b / 2)))


def elbow_axis_distance(upper_arm_length, shoulder_hand_distance):
    return isosceles_triangle_height(upper_arm_length, shoulder_hand_distance)


def up_forward_to_rot_matrix(up, forward):
    right = np.cross(up, forward)
    return np.array([right, up, forward]).T


def up_forward_to_euler(up, forward):
    return mat2euler(up_forward_to_rot_matrix(up, forward))


def up_forward_to_euler_batch(up, forward):
    # TODO check if the axis order is correct!
    raise NotImplementedError()
    R = up_forward_to_rot_matrix_batch_np(up, forward)

    sy = np.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])

    singular = sy < 1e-6
    not_singular = ~singular

    result = np.zeros_like(up, dtype=np.float32)
    result[not_singular, 0] = np.arctan2(R[not_singular, 2, 1], R[not_singular, 2, 2])
    result[not_singular, 1] = np.arctan2(R[not_singular, 2, 0], sy[not_singular])
    result[not_singular, 2] = np.arctan2(R[not_singular, 1, 0], R[not_singular, 0, 0])
    result[singular, 0] = np.arctan2(-R[singular, 1, 2], R[singular, 1, 1])
    result[singular, 1] = np.arctan2(-R[singular, 2, 0], sy[singular])
    result[singular, 2] = 0

    # note this is now in Z-X-Y order --> need to convert to X-Y-Z order to be compatible with remaining framework
    q = quaternion.euler_to_quaternion(result, order="zxy")
    result = quaternion.qeuler_np(q, order="xyz")

    return result


def up_forward_to_rot_matrix_batch_np(up, forward):
    right = np.cross(up, forward)
    return np.concatenate((right[:, :, None], up[:, :, None], forward[:, :, None]), axis=2).transpose((0, 2, 1))
    # return np.concatenate((right[:, :, None], up[:, :, None], forward[:, :, None]), axis=2)


def torch_bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)


def np_bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return np.matmul(a.reshape((B, 1, S)), b.reshape((B, S, 1))).reshape(-1)


def torch_repeat(a, times):
    return torch.clone(a.unsqueeze(0).expand(times, *a.shape))


def batch_rot_vectors_quat(quat, vector):
    if len(vector.shape) < len(quat.shape):
        _vector = vector.unsqueeze(0).expand(quat.shape[0], 3)
    else:
        _vector = vector

    return quaternion.qrot(quat, _vector)


def batch_rot_vectors_euler(euler_angles, vector):
    _euler_angles = torch.from_numpy(euler_angles) if type(euler_angles) is np.ndarray else euler_angles
    quat = quaternion.euler_to_quaternion_torch(_euler_angles)
    return batch_rot_vectors_quat(quat, vector)


def batch_up(euler_angles):
    return batch_rot_vectors_euler(euler_angles, torch.tensor([0., 1., 0.]))


def batch_forward(euler_angles):
    return batch_rot_vectors_euler(euler_angles, torch.tensor([0., 0., 1.]))


def batch_right(euler_angles):
    return batch_rot_vectors_euler(euler_angles, torch.tensor([1., 0., 0.]))


def batch_up_quat(quat):
    return batch_up(quaternion.qeuler(quat))


def batch_forward_quat(quat):
    return batch_forward(quaternion.qeuler(quat))


def batch_right_quat(quat):
    return batch_right(quaternion.qeuler(quat))


def batch_horizontal_forward(euler_angles):
    forward_vec = batch_rot_vectors_euler(euler_angles, torch.tensor([0., 0., 1.]))
    forward_vec[:, 1] *= 0
    return normalize_batch(forward_vec, dim=1)


def batch_horizontal_right(euler_angles):
    forward_vec = batch_rot_vectors_euler(euler_angles, torch.tensor([1., 0., 0.]))
    forward_vec[:, 1] *= 0
    return normalize_batch(forward_vec, dim=1)


def batch_rot_vectors_quat_np(quat, vector):
    if len(vector.shape) < len(quat.shape):
        _vector = np.repeat(vector[None, :], len(quat), axis=0)
    else:
        _vector = vector

    return quaternion.qrot_np(quat, _vector)


def batch_rot_vectors_euler_np(euler_angles, vector):
    _euler_angles = euler_angles.numpy() if type(euler_angles) is torch.tensor else euler_angles
    quat = quaternion.euler_to_quaternion(_euler_angles)
    return batch_rot_vectors_quat_np(quat, vector)


def batch_up_np(euler_angles):
    return batch_rot_vectors_euler_np(euler_angles, np.array([0., 1., 0.]))


def batch_forward_np(euler_angles):
    return batch_rot_vectors_euler_np(euler_angles, np.array([0., 0., 1.]))


def batch_right_np(euler_angles):
    return batch_rot_vectors_euler_np(euler_angles, np.array([1., 0., 0.]))


def batch_horizontal_forward_np(euler_angles):
    forward_vec = batch_rot_vectors_euler_np(euler_angles, np.array([0., 0., 1.]))
    forward_vec[:, 1] *= 0
    return normalize_batch_np(forward_vec, dim=1)


def normalize(v):
    length = max(1e-9, np.linalg.norm(v))
    return v / length


def normalize_batch(v, dim=1, require_differentiable=False):
    length = torch.norm(v, dim=dim, keepdim=True)
    if not require_differentiable:
        length[length < 1.e-7] = 1.e-7
    return v / length


def normalize_batch_np(v, dim=1):
    length = np.linalg.norm(v, axis=dim)
    length[length < 1.e-7] = 1.e-7
    length = np.repeat(length, v.shape[dim], axis=0).reshape((-1, v.shape[dim]))
    return v / length


def surface_normal_np(p):
    v = p[1] - p[0]
    w = p[2] - p[0]
    return normalize(np.cross(v, w))


def surface_normal_batch_np(p):
    v = p[:, 1] - p[:, 0]
    w = p[:, 2] - p[:, 0]
    return normalize(np.cross(v, w))


def surface_normal_batch(p):
    v = p[:, 1] - p[:, 0]
    w = p[:, 2] - p[:, 0]
    return normalize_batch(torch.cross(v, w))


def distance_to_triangle(tri, p):
    n = surface_normal_batch(tri)
    t = n * tri[:, 0] - n * p
    p_0 = p + t * n
    return torch.norm(p - p_0, dim=1)


def translation_matrix(t):
    M = torch.eye(4)
    M[:3, 3] = t
    return M


def translation_matrix_batch(t):
    M = torch_repeat(torch.eye(4), len(t))
    M[:, :3, 3] = t[:]
    return M


def rotation_matrix(r):
    from math import cos, sin
    M = torch.eye(4)
    a, b, c = r

    cosA = cos(a)
    sinA = sin(a)
    MA = torch.eye(4)
    MA[1, 1] = cosA
    MA[1, 2] = -sinA
    MA[2, 1] = sinA
    MA[2, 2] = cosA

    cosB = cos(b)
    sinB = sin(b)
    MB = torch.eye(4)
    MB[0, 0] = cosB
    MB[0, 2] = sinB
    MB[2, 0] = -sinB
    MB[2, 2] = cosB

    cosC = cos(c)
    sinC = sin(c)
    MC = torch.eye(4)
    MC[0, 0] = cosC
    MC[0, 1] = -sinC
    MC[1, 0] = sinC
    MC[1, 1] = cosC

    M = torch.mm(M, MC)
    M = torch.mm(M, MB)
    M = torch.mm(M, MA)
    return M


def rotation_matrix_batch(r):
    from torch import cos, sin
    M = torch_repeat(torch.eye(4), len(r))
    MA = torch_repeat(torch.eye(4), len(r))
    MB = torch_repeat(torch.eye(4), len(r))
    MC = torch_repeat(torch.eye(4), len(r))
    a, b, c = r[:, 0], r[:, 1], r[:, 2]

    cosA = cos(a)
    sinA = sin(a)
    MA[:, 1, 1] = cosA
    MA[:, 1, 2] = -sinA
    MA[:, 2, 1] = sinA
    MA[:, 2, 2] = cosA

    cosB = cos(b)
    sinB = sin(b)
    MB[:, 0, 0] = cosB
    MB[:, 0, 2] = sinB
    MB[:, 2, 0] = -sinB
    MB[:, 2, 2] = cosB

    cosC = cos(c)
    sinC = sin(c)
    MC[:, 0, 0] = cosC
    MC[:, 0, 1] = -sinC
    MC[:, 1, 0] = sinC
    MC[:, 1, 1] = cosC

    M = torch.bmm(M, MC)
    M = torch.bmm(M, MB)
    M = torch.bmm(M, MA)
    return M


def scale_matrix(t):
    M = torch.eye(4)
    for i in range(3):
        M[i, i] = t[i]
    return M


def scale_matrix_batch(s):
    M = torch_repeat(torch.eye(4), len(s))
    for i in range(3):
        M[:, i, i] = s[:, i]
    return M


def deg2rad(x):
    if type(x) is np.ndarray:
        return x.copy() * 0.0174532925199432
    if type(x) is torch.Tensor:
        return x.clone() * 0.0174532925199432
    return x * 0.0174532925199432


def rad2deg(x):
    if type(x) is np.ndarray:
        return x.copy() * 57.29577951308232087
    if type(x) is torch.Tensor:
        return x.clone() * 57.29577951308232087
    return x * 57.29577951308232087


def transformation_get_scale(M):
    s_x = torch.norm(M[:, :, 0], dim=1)[:, None]
    s_y = torch.norm(M[:, :, 1], dim=1)[:, None]
    s_z = torch.norm(M[:, :, 2], dim=1)[:, None]

    return torch.cat((s_x, s_y, s_z), dim=1)


def transformation_unscaled_batch(M):
    s_x = torch_repeat(torch.norm(M[:, :, 0], dim=1), 4).t()
    s_y = torch_repeat(torch.norm(M[:, :, 1], dim=1), 4).t()
    s_z = torch_repeat(torch.norm(M[:, :, 2], dim=1), 4).t()

    _M = torch.clone(M)
    _M[:, :, 0] /= s_x
    _M[:, :, 1] /= s_y
    _M[:, :, 2] /= s_z

    _M[:] /= torch_repeat(torch_repeat(M[:, 3, 3], 4), 4).transpose(0, 2)

    return _M


def rotation_matrix_to_euler(M):
    x = torch.atan2(M[2, 1], M[2, 2])
    y = torch.atan2(-M[2, 0], torch.sqrt(torch.pow(M[2, 1], 2) + torch.pow(M[2, 2], 2)))
    z = torch.atan2(M[1, 0], M[0, 0])

    return torch.cat((x, y, z), dim=0)


def rotation_matrix_to_euler_batch(M):
    # x = torch.atan2(M[:, 2, 1], M[:, 2, 2])
    # y = torch.atan2(-M[:, 2, 0], torch.sqrt(torch.pow(M[:, 2, 1], 2) + torch.pow(M[:, 2, 2], 2)))
    # z = torch.atan2(M[:, 1, 0], M[:, 0, 0])
    #
    # return torch.cat((x[:, None], y[:, None], z[:, None]), dim=1)

    caseA = M[:, 0, 2] < 1
    caseB = M[:, 0, 2] > -1

    case1 = caseA * caseB
    case2 = caseA * (caseB == False)
    case3 = caseA == False

    res = torch.zeros((len(M), 3), dtype=torch.float32)
    res[case1, 1] = torch.asin(M[case1, 0, 2])
    res[case1, 0] = torch.atan2(-M[case1, 1, 2], M[case1, 2, 2])
    res[case1, 2] = torch.atan2(-M[case1, 0, 1], M[case1, 0, 0])

    res[case2, 1] = -np.pi * 0.5
    res[case2, 0] = -torch.atan2(M[case2, 1, 0], M[case2, 1, 1])
    res[case2, 2] = 0

    res[case3, 1] = np.pi * 0.5
    res[case3, 0] = torch.atan2(M[case3, 1, 0], M[case3, 1, 1])
    res[case3, 2] = 0

    return res


def rotation_between_triangles(a, b):
    _a = torch.stack(a).clone()
    _b = torch.stack(b).clone()
    quat1 = quaternion.quaternion_between_vectors_torch(normalize_batch(_a[1]), normalize_batch(b[1]))
    _b[1] = _a[1].clone()
    _b[2] = quaternion.qrot(quaternion.inverse(quat1.clone()), _b[2])

    n_a = surface_normal_batch(torch.transpose(_a, 0, 1))
    n_b = surface_normal_batch(torch.transpose(_b, 0, 1))

    quat2 = quaternion.quaternion_between_vectors_torch(n_a, n_b)
    quat_sum = quaternion.qmul(quat1, quat2)

    check = torch.stack(a).clone()
    check[1] = quaternion.qrot(quat_sum, check[1])
    check[2] = quaternion.qrot(quat_sum, check[2])

    return quat_sum


def quat_smooth(q, n=4):
    q_tmp = torch.zeros_like(q)
    for i in range(-n // 2, n // 2 + 1, 1):
        i0 = max(i, 0)
        i1 = min(i + len(q_tmp), len(q_tmp))
        q_tmp[i0 - i:i1 - i] += q[i0: i1]
    return normalize_batch(q_tmp, dim=len(q_tmp.shape) - 1)


def angle_between_vectors(a, b):
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5)
    b_norm = b.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product / (a_norm * b_norm)
    cos[torch.isnan(cos)] = 1
    angle = torch.acos(cos)
    return angle
