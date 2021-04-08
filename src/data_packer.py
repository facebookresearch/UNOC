import numpy as np

from definitions import KeyframeIdx as Idx
import math_helper
import torch
import quaternion


class Feature:
    entries = 1
    allow_occlusion = True

    def __init__(self, idx=0):
        self.idx = idx

    def extract(self, keyframe, **kwargs):
        raise NotImplementedError()

    def extract_batch(self, keyframes, **kwargs):
        raise NotImplementedError()


class PositionNoOcc(Feature):
    entries = 3
    allow_occlusion = False

    def extract(self, keyframe, **kwargs):
        return keyframe.p[self.idx]

    def extract_batch(self, keyframes, **kwargs):
        return keyframes[:, self.idx, :3]


class Position_ZeroXY(Feature):
    entries = 3

    def __init__(self, idx=0, head_idx=0):
        super().__init__(idx)
        self.head_idx = head_idx

    def extract(self, keyframe, **kwargs):
        p_head = keyframe.p[self.head_idx]
        p_head[1] = 0
        return keyframe.p[self.idx] - p_head

    def extract_batch(self, keyframes, **kwargs):
        p_head = keyframes[:, self.head_idx, :3] * np.array([1., 0., 1.])
        p_head[:, 1] = 0
        return keyframes[:, self.idx, :3] - p_head


class JointLocalPosition(Feature):
    entries = 3

    def __init__(self, idx=0, ref_idx=0):
        super().__init__(idx)
        self.ref_idx = ref_idx

    def extract(self, keyframe, **kwargs):
        return keyframe.p[self.idx] - keyframe.p[self.ref_idx]

    def extract_batch(self, keyframes, **kwargs):
        return keyframes[:, self.idx, :3] - keyframes[:, self.ref_idx, :3]


class JointZeroRotationLocalPosition(Feature):
    entries = 3

    def __init__(self, idx=0, ref_idx=0):
        super().__init__(idx)
        self.ref_idx = ref_idx

    def extract(self, keyframe, **kwargs):
        raise NotImplementedError()

    def extract_batch(self, keyframes, **kwargs):
        ref_quat = quaternion.euler_to_quaternion(np.deg2rad(keyframes[:, self.ref_idx, 3:6]))
        local_pos = keyframes[:, self.idx, :3] - keyframes[:, self.ref_idx, :3]
        return quaternion.qrot_np(quaternion.inverse_np(ref_quat), local_pos)


class RootForwardLocalPosition(Feature):
    entries = 3
    """
    sets root pos to (0,Y,0) and rotates skeleton so that head is always looking forward horizontally
    """

    def __init__(self, idx=0, ref_idx=0):
        super().__init__(idx)
        self.ref_idx = ref_idx

    def extract(self, keyframe, **kwargs):
        head_joint_distance = keyframe.p[self.idx]
        head_joint_distance[[0, 2]] -= keyframe.p[self.ref_idx][[0, 2]]
        forward_vec = math_helper.forward_deg(keyframe.r[self.ref_idx])
        ref_dir = np.array([0., 0., 1.])
        rot_around_y = math_helper.quaternion_between_vectors(ref_dir, forward_vec)
        result = rot_around_y.rotate(head_joint_distance)
        return result

    def extract_batch(self, keyframes, **kwargs):
        ref_self_distance = np.copy(keyframes[:, self.idx, :3])
        ref_self_distance[:, [0, 2]] -= keyframes[:, self.ref_idx, [0, 2]]
        rot_around_y = math_helper.deg2rad(keyframes[:, self.ref_idx, 6:9]).copy()
        rot_around_y[:] *= np.array([0.0, 1.0, 0.0])
        rot_around_y = quaternion.euler_to_quaternion(rot_around_y)
        result = quaternion.qrot_np(quaternion.inverse_np(rot_around_y), ref_self_distance)
        return result


class RootForwardLocalUp(Feature):
    entries = 3

    def __init__(self, idx=0, ref_idx=0):
        super().__init__(idx)
        self.ref_idx = ref_idx

    def extract(self, keyframe, **kwargs):
        raise NotImplementedError()

    def extract_batch(self, keyframes, **kwargs):
        rot_around_y = math_helper.deg2rad(keyframes[:, self.ref_idx, 6:9]).copy()
        rot_around_y[:] *= np.array([0.0, 1.0, 0.0])
        rot_around_y = quaternion.euler_to_quaternion(rot_around_y)
        q = quaternion.euler_to_quaternion(math_helper.deg2rad(keyframes[:, self.idx, 3:6]))
        q = quaternion.qmul_np(quaternion.inverse_np(rot_around_y), q)
        up = math_helper.batch_up_quat(torch.from_numpy(q))
        return up.numpy()


class RootForwardLocalForward(Feature):
    entries = 3

    def __init__(self, idx=0, ref_idx=0):
        super().__init__(idx)
        self.ref_idx = ref_idx

    def extract(self, keyframe, **kwargs):
        raise NotImplementedError()

    def extract_batch(self, keyframes, **kwargs):
        rot_around_y = math_helper.deg2rad(keyframes[:, self.ref_idx, 6:9]).copy()
        rot_around_y[:] *= np.array([0.0, 1.0, 0.0])
        rot_around_y = quaternion.euler_to_quaternion(rot_around_y)
        q = quaternion.euler_to_quaternion(math_helper.deg2rad(keyframes[:, self.idx, 3:6]))
        q = quaternion.qmul_np(quaternion.inverse_np(rot_around_y), q)
        forward = math_helper.batch_forward_quat(torch.from_numpy(q))
        return forward.numpy()


class IsOccluded(Feature):
    entries = 1

    def extract(self, keyframe, **kwargs):
        raise NotImplemented()

    def extract_batch(self, keyframes, **kwargs):
        return keyframes[:, self.idx, 9]


def _accumulate_occlusions(keyframes, idx, occlusion_offset=1):
    acc = 0
    arr = keyframes[:, idx, 9]
    for i in range(keyframes.shape[0]):
        a = arr[i]
        acc *= a
        acc += a

        if np.isnan(acc):
            print("")

        arr[i] = 0 if acc < 0.5 else occlusion_offset + np.log10(acc)
    return arr


class IsOccludedSince(Feature):
    entries = 1

    def extract(self, keyframe, **kwargs):
        raise NotImplemented()

    def extract_batch(self, keyframes, **kwargs):
        return _accumulate_occlusions(keyframes, self.idx)


class IsOccludedSinceZeroOffset(Feature):
    entries = 1

    def extract(self, keyframe, **kwargs):
        raise NotImplemented()

    def extract_batch(self, keyframes, **kwargs):
        return _accumulate_occlusions(keyframes, self.idx, occlusion_offset=0)


class NeverOccluded(Feature):
    entries = 1

    def extract(self, keyframe, **kwargs):
        raise 0

    def extract_batch(self, keyframes, **kwargs):
        return np.zeros((len(keyframes), 1))


class FeaturePacker:
    def __init__(self, features, replace_occluded, name):
        self.features = features
        self.entries = sum([x.entries for x in features])
        self.replace_occluded = replace_occluded
        self.name = name

    def extract(self, keyframe, **kwargs):
        result = np.empty(self.entries, dtype=np.float32)
        idx = 0

        for f in self.features:
            if self.replace_occluded is not None and (
                    f.idx == Idx.lwrist and not keyframe.l_hand_visible or f.idx == Idx.rwrist and not keyframe.r_hand_visible):
                if isinstance(self.replace_occluded, float):
                    result[idx: idx + f.entries] = self.replace_occluded
                elif callable(self.replace_occluded):
                    result[idx: idx + f.entries] = self.replace_occluded(result[idx: idx + f.entries])
                else:
                    raise AttributeError("invalid replacement")

            else:
                result[idx: idx + f.entries] = f.extract(keyframe, **kwargs)
            idx += f.entries
        return result

    def extract_batch(self, frames, **kwargs):
        result = np.empty((len(frames), self.entries), dtype=np.float32)
        idx = 0

        if "random_occlusion" in kwargs and kwargs["random_occlusion"] is not None:
            random_occlusion = float(kwargs["random_occlusion"])
            occ = np.random.rand(frames.shape[0], frames.shape[1])
            if random_occlusion > 0.0001:
                frames[:, :, 9][occ > (1 - random_occlusion)] = 1
                frames[:, :, 9][occ <= (1 - random_occlusion)] = 0
            elif random_occlusion < 0.0001:
                frames[:, :, 9][occ > (1 + random_occlusion)] = 1

        # all joints with an id less that min_occluded_index cannot be occluded
        min_occluded_index = -1
        for f in self.features:
            result[:, idx: idx + f.entries] = f.extract_batch(frames, **kwargs).reshape((result.shape[0], f.entries))
            if self.replace_occluded is not None and f.idx >= min_occluded_index and \
                    (type(f) is not IsOccluded and type(f) is not IsOccludedSince and type(f) is not IsOccludedSinceZeroOffset):
                occ_mask = frames[:, f.idx, 9] > 0.00001
                occluded_frames = result[occ_mask, idx:idx + f.entries]
                if isinstance(self.replace_occluded, float):
                    occluded_frames = self.replace_occluded

                elif callable(self.replace_occluded):
                    occluded_frames = self.replace_occluded(occluded_frames)

                elif self.replace_occluded == "last_known" or self.replace_occluded == "repeat":
                    if len(occluded_frames) > 0:
                        non_nan = np.where(~occ_mask, np.arange(occ_mask.shape[0]), 0)
                        np.maximum.accumulate(non_nan, axis=0, out=non_nan)
                        tmp = result[non_nan, idx:idx + f.entries]
                        occluded_frames = tmp[occ_mask]

                elif self.replace_occluded == "keep":
                    pass

                elif self.replace_occluded == "random":
                    occluded_frames = np.random.random(occluded_frames.shape)

                else:
                    raise AttributeError("invalid replacement")
                result[occ_mask, idx:idx + f.entries] = occluded_frames
            idx += f.entries
        return result
