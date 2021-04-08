import data_packer as d
import features_to_pose as inv_d
from definitions import KeyframeIdx, Skeleton


def global_pos_zeroXY(replace_occluded):
    features = []
    for idx in KeyframeIdx.all.values():
        features.append(d.Position_ZeroXY(idx, KeyframeIdx.head))

    if replace_occluded is not None:
        for idx in KeyframeIdx.all.values():
            features.append(d.IsOccluded(idx))
    return d.FeaturePacker(features, replace_occluded,
                           name=f"global_pos_zeroXY{'_occ' if replace_occluded is not None else ''}")


class Featureset:
    base_name = "feature_set"

    def __init__(self, replace_occluded):
        self.replace_occluded = replace_occluded

    def file_name(self):
        return f"{self.base_name}{'_occ_' + str(self.replace_occluded) if self.replace_occluded is not None else ''}"

    def data_to_features(self):
        raise NotImplemented()

    def features_to_pose(self):
        raise NotImplemented()

    def features_wo_occlusions(self):
        tmp_replace_occluded = self.replace_occluded
        self.replace_occluded = None
        features = self.data_to_features().entries
        self.replace_occluded = tmp_replace_occluded
        return features


class HipLocalPosHandLocalFingersGlobalPositionPreserving(Featureset):
    head_idx = KeyframeIdx.root
    l_wrist_idx = KeyframeIdx.lwrist
    r_wrist_idx = KeyframeIdx.rwrist
    base_name = "hip_local_pos_hand_local_fingers_global_position_preserving"
    use_binary_occ = False

    def _get_joint_idx(self):
        return KeyframeIdx.all.values()

    def _get_lfinger_idx(self):
        l_hand, _ = KeyframeIdx.get_hand_joints()
        return l_hand[1:]

    def _get_rfinger_idx(self):
        _, r_hand = KeyframeIdx.get_hand_joints()
        return r_hand[1:]

    def data_to_features(self):
        features = []
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        for idx in self._get_joint_idx():
            if idx in l_finger_idx:
                features.append(d.JointLocalPosition(idx, ref_idx=self.l_wrist_idx))
            elif idx in r_finger_idx:
                features.append(d.JointLocalPosition(idx, ref_idx=self.r_wrist_idx))
            elif idx == self.head_idx:
                features.append(d.PositionNoOcc(idx))
            else:
                features.append(d.JointLocalPosition(idx, ref_idx=self.head_idx))

        if self.replace_occluded is not None:
            for idx in self._get_joint_idx():
                if idx == self.head_idx:
                    features.append(d.NeverOccluded(idx))
                elif self.use_binary_occ:
                    features.append(d.IsOccluded(idx))
                else:
                    features.append(d.IsOccludedSinceZeroOffset(idx))

        return d.FeaturePacker(features, self.replace_occluded, name=self.file_name())

    def features_to_pose(self):
        solvers = []
        indices = self._get_joint_idx()
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        for idx in range(len(indices)):
            if idx in l_finger_idx:
                solvers.append(inv_d.PosDoubleLocalReplaceOccluded(idx, self.l_wrist_idx, grand_parent_idx=self.head_idx, grand_use_body=True))
            elif idx in r_finger_idx:
                solvers.append(inv_d.PosDoubleLocalReplaceOccluded(idx, self.r_wrist_idx, grand_parent_idx=self.head_idx, grand_use_body=True))
            elif idx == self.head_idx:
                solvers.append(inv_d.PosDirectBody(idx))
            else:
                solvers.append(inv_d.PosLocalBodyReplaceOccluded(idx, self.head_idx))

        return inv_d.SolverPacker(solvers)


class HeadLocalPosHandLocalFingersGlobalPositionPreserving(HipLocalPosHandLocalFingersGlobalPositionPreserving):
    head_idx = KeyframeIdx.head
    base_name = "head_local_pos_hand_local_fingers_global_position_preserving"


class HeadLocalPosHandLocalFingersGlobalPositionPreservingBinaryOcc(HeadLocalPosHandLocalFingersGlobalPositionPreserving):
    use_binary_occ = True
    base_name = "head_local_pos_hand_local_fingers_global_position_preserving_binary_occ"


class HipLocalPosHandLocalFingersGlobalPositionPreservingBinaryOcc(HipLocalPosHandLocalFingersGlobalPositionPreserving):
    use_binary_occ = True
    base_name = "hip_local_pos_hand_local_fingers_global_position_preserving_binary_occ"


class HipForwardLocalPosWristUpForwardNoFinger(Featureset):
    base_name = "hip_forward_local_pos_wrist_up_forward_no_finger"
    hip_idx = KeyframeIdx.root
    l_wrist_idx = KeyframeIdx.lwrist
    r_wrist_idx = KeyframeIdx.rwrist

    def _get_joint_idx(self):
        return KeyframeIdx.all.values()

    def _get_lfinger_idx(self):
        l_hand, _ = KeyframeIdx.get_hand_joints()
        return l_hand[1:]

    def _get_rfinger_idx(self):
        _, r_hand = KeyframeIdx.get_hand_joints()
        return r_hand[1:]

    def data_to_features(self):
        features = []
        append_features = []
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        for idx in self._get_joint_idx():
            if idx in l_finger_idx or idx in r_finger_idx:
                pass
            elif idx == self.l_wrist_idx or idx == self.r_wrist_idx:
                features.append(d.RootForwardLocalPosition(idx, ref_idx=self.hip_idx))
                append_features.append(d.RootForwardLocalUp(idx, ref_idx=self.hip_idx))
                append_features.append(d.RootForwardLocalForward(idx, ref_idx=self.hip_idx))
            else:
                features.append(d.RootForwardLocalPosition(idx, ref_idx=self.hip_idx))

        features.extend(append_features)

        if self.replace_occluded is not None:
            for idx in self._get_joint_idx():
                features.append(d.IsOccludedSince(idx))

        return d.FeaturePacker(features, self.replace_occluded, name=self.file_name())

    def features_to_pose(self):
        solvers = []
        indices = self._get_joint_idx()
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        in_idx = 0
        for idx in range(len(indices)):
            if idx in l_finger_idx or idx in r_finger_idx:
                par_idx = self.l_wrist_idx if idx in l_finger_idx else self.r_wrist_idx
                solvers.append(inv_d.PosDirect(par_idx, input_idx=in_idx - 3))
            else:
                solvers.append(inv_d.PosDirectReplaceOccluded(idx, input_idx=in_idx))
                in_idx += 3

        return inv_d.SolverPacker(solvers)


class HandZeroRotationLocalFingersOnly(Featureset):
    base_name = "hand_zero_rotation_local_fingers_only"
    l_wrist_idx = KeyframeIdx.lwrist
    r_wrist_idx = KeyframeIdx.rwrist
    hip_idx = KeyframeIdx.root

    def _get_joint_idx(self):
        return KeyframeIdx.all.values()

    def _get_lfinger_idx(self):
        l_hand, _ = KeyframeIdx.get_hand_joints()
        return l_hand[1:]

    def _get_rfinger_idx(self):
        _, r_hand = KeyframeIdx.get_hand_joints()
        return r_hand[1:]

    def data_to_features(self):
        features = []
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        for idx in self._get_joint_idx():
            if idx in l_finger_idx:
                features.append(d.JointZeroRotationLocalPosition(idx, ref_idx=self.l_wrist_idx))
            elif idx in r_finger_idx:
                features.append(d.JointZeroRotationLocalPosition(idx, ref_idx=self.r_wrist_idx))

        if self.replace_occluded is not None:
            for idx in self._get_joint_idx():
                if idx in l_finger_idx or idx in r_finger_idx:
                    features.append(d.IsOccludedSince(idx))

        return d.FeaturePacker(features, self.replace_occluded, name=self.file_name())

    def features_to_pose(self):
        solvers = []
        indices = self._get_joint_idx()
        l_finger_idx = self._get_lfinger_idx()
        r_finger_idx = self._get_rfinger_idx()
        out_idx = 0
        par_idx = 0

        par_pos_end_idx = (len(indices) - len(l_finger_idx) - len(r_finger_idx)) * 3
        par_r_up_idx = par_pos_end_idx
        par_r_forward_idx = par_r_up_idx + 3
        par_l_up_idx = par_r_forward_idx + 3
        par_l_forward_idx = par_l_up_idx + 3

        for idx in range(len(indices)):
            if idx in l_finger_idx or idx in r_finger_idx:
                solvers.append(inv_d.PosZeroRotationLocalBody(
                    joint_idx=out_idx,
                    parent_joint_idx=par_idx - 1,
                    parent_input_idx=(par_idx - 1) * 3,
                    parent_up_idx=par_l_up_idx if idx in l_finger_idx else par_r_up_idx,
                    parent_forward_idx=par_l_forward_idx if idx in l_finger_idx else par_r_forward_idx
                ))
                out_idx += 1
            else:
                solvers.append(inv_d.PosDirectBody(idx, input_idx=par_idx * 3))
                par_idx += 1

        return inv_d.SolverPacker(solvers)
