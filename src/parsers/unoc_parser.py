import torch

import quaternion
from .bvh_converter import BvhConverter
import os
import numpy as np
import json


class UnocParser(BvhConverter):
    root_path = os.getenv("PATH_UNOC")
    dataset_fps = 120
    scale = 100
    save_separately = True

    joint_mappings = {
        'root': 'b_root',
        'spine0': 'b_spine0',
        'spine1': 'b_spine1',
        'spine2': 'b_spine2',
        'spine3': 'b_spine3',
        'neck': 'b_neck0',
        'head': 'b_head',
        'head_end': 'b_head_end',
        'rshoulder': 'b_r_shoulder',
        'rscap': 'p_r_scap',
        'rupperarm': 'b_r_arm',
        'rlowerarm': 'b_r_forearm',
        'rwristtwist': 'b_r_wrist_twist',
        'rwrist': 'b_r_wrist',
        'rindex1': 'b_r_index1',
        'rindex2': 'b_r_index2',
        'rindex3': 'b_r_index3',
        'rindex3_end': 'b_r_index3_end',
        'rring1': 'b_r_ring1',
        'rring2': 'b_r_ring2',
        'rring3': 'b_r_ring3',
        'rring3_end': 'b_r_ring3_end',
        'rmiddle1': 'b_r_middle1',
        'rmiddle2': 'b_r_middle2',
        'rmiddle3': 'b_r_middle3',
        'rmiddle3_end': 'b_r_middle3_end',
        'rpinky1': 'b_r_pinky1',
        'rpinky2': 'b_r_pinky2',
        'rpinky3': 'b_r_pinky3',
        'rpinky3_end': 'b_r_pinky3_end',
        'rthumb0': 'b_r_thumb0',
        'rthumb1': 'b_r_thumb1',
        'rthumb2': 'b_r_thumb2',
        'rthumb3': 'b_r_thumb3',
        'rthumb3_end': 'b_r_thumb3_end',
        'lshoulder': 'b_l_shoulder',
        'lscap': 'p_l_scap',
        'lupperarm': 'b_l_arm',
        'llowerarm': 'b_l_forearm',
        'lwristtwist': 'b_l_wrist_twist',
        'lwrist': 'b_l_wrist',
        'lindex1': 'b_l_index1',
        'lindex2': 'b_l_index2',
        'lindex3': 'b_l_index3',
        'lindex3_end': 'b_l_index3_end',
        'lring1': 'b_l_ring1',
        'lring2': 'b_l_ring2',
        'lring3': 'b_l_ring3',
        'lring3_end': 'b_l_ring3_end',
        'lmiddle1': 'b_l_middle1',
        'lmiddle2': 'b_l_middle2',
        'lmiddle3': 'b_l_middle3',
        'lmiddle3_end': 'b_l_middle3_end',
        'lpinky1': 'b_l_pinky1',
        'lpinky2': 'b_l_pinky2',
        'lpinky3': 'b_l_pinky3',
        'lpinky3_end': 'b_l_pinky3_end',
        'lthumb0': 'b_l_thumb0',
        'lthumb1': 'b_l_thumb1',
        'lthumb2': 'b_l_thumb2',
        'lthumb3': 'b_l_thumb3',
        'lthumb3_end': 'b_l_thumb3_end',
        'rupperleg': 'b_r_upleg',
        'rlowerleg': 'b_r_leg',
        # 'rfoottwist1': 'b_r_talocrural',
        'rfoot': 'b_r_subtalar',
        # 'rfoottwist2': 'b_r_transversetarsal',
        'rfootball': 'b_r_ball',
        'rfootball_right': 'b_r_ball_right',
        'rfootball_end': 'b_r_ball_end',
        'lupperleg': 'b_l_upleg',
        'llowerleg': 'b_l_leg',
        # 'lfoottwist1': 'b_l_talocrural',
        'lfoot': 'b_l_subtalar',
        # 'lfoottwist2': 'b_l_transversetarsal',
        'lfootball': 'b_l_ball',
        'lfootball_left': "b_l_ball_left",
        'lfootball_end': 'b_l_ball_end'
    }

    added_joint_offsets = {
        'b_r_ball_right': np.array([0.05, 0., 0.]),
        'b_l_ball_left': np.array([-0.05, 0., 0.])
    }

    @classmethod
    def get_all_files_of_folder(cls, folder_path):
        file_paths = []
        count_frames = False
        frames = 0
        if len(list(os.walk(folder_path))) == 0:
            return []

        for file_name in list(os.walk(folder_path))[0][2]:
            if file_name in cls.ignore_files:
                continue
            if file_name.endswith(".bvh") and "merged" in file_name:
                file_paths.append(os.path.join(folder_path, file_name))

                if count_frames:
                    with open(file_paths[-1]) as f:
                        line = ""
                        while "Frames: " not in line:
                            line = f.readline()
                        frames += int(line.split(":    ")[1])

        if count_frames:
            print(f"frames: {frames}")

        return file_paths

    def skeleton(self):
        return self.Skeleton(self.skeleton_pos())

    def skeleton_pos(self):
        offset = np.zeros(3)
        poses = []
        self._add_pose_recursive(self.root, offset, poses)

        root_height = -poses[list(self.joints.keys()).index(self.toe_joint)][1]
        offset_pos = np.array([0, root_height, 0])

        parent_idx = self.Skeleton.parent_idx_vector()
        positions = np.zeros((len(self.joint_mappings), 3))
        for key in self.joint_mappings.keys():
            positions_index = self.KeyframeIdx.all[key]
            joint_mapping = self.joint_mappings[key]
            if joint_mapping in self.added_joint_offsets:
                positions[positions_index] = positions[parent_idx[positions_index]] + self.added_joint_offsets[joint_mapping] * self.scale
            else:
                bvh_index = list(self.joints.keys()).index(joint_mapping)
                positions[positions_index] = poses[bvh_index].reshape(3) + offset_pos

        return positions

    @classmethod
    def get_all_folders(cls):
        return list(os.walk(cls.root_path))[0][1]

    @classmethod
    def get_all_keyframe_folders(cls):
        return {cls.root_path}

    @classmethod
    def get_all_file_paths(cls):
        result = []
        for folder in cls.get_all_folders():
            result.extend(cls.get_all_files_of_folder(cls.root_path + folder + "/merged bvh/"))
        return result

    def get_occlusions_and_out_of_views(self, folder):
        with open(os.path.join(folder, "occlusions.json")) as f:
            occlusions_json = json.load(f)
            self.file_occlusions = {}
            self.file_out_of_views = {}
            self.file_too_close = {}
            self.file_occlusions_joints = occlusions_json['joints']
            for anim in occlusions_json['clipStats']:
                self.file_occlusions[anim['name']] = np.array(anim['occlusionMask'])
                self.file_out_of_views[anim['name']] = np.array(anim['outOfViewMask'])
                self.file_too_close[anim['name']] = np.array(anim['tooCloseMask'])

        # file = ""
        # if file not in self.file_occlusions:
        #     return [], []
        #
        # return self.file_occlusions[file], self.file_out_of_views[file], self.file_too_close[file]

    def convert_all(self, file="", folder=""):
        # _p, _r = self.all_frame_poses()
        p, r = self.batch_all_frame_poses()
        p = p.numpy()
        r = r.numpy()
        file_short = file[:63]
        print(f"converting {folder}{file}")
        if file_short not in self.file_occlusions:
            file_short = file + ".bvh"
        occlusions, out_of_views, too_close = self.file_occlusions[file_short], self.file_out_of_views[file_short], \
                                              self.file_too_close[file_short]

        occlusions, out_of_views, too_close = self._resize_occlusions(len(p), occlusions, out_of_views, too_close)

        keyframes = []
        keyframes.extend(self.batch_convert_frames(p, r, occlusions, out_of_views, too_close))

        # import plots
        # vispy = plots.Vispy3DScatter()
        # vispy.plot(np.array(keyframes)[:, :, [0, 2, 1]])
        return np.array(keyframes)

    def batch_convert_frames(self, p, r, occlusions, out_of_views, too_close):
        p_converted = np.empty((len(p), len(self.KeyframeIdx.all), 3))
        r_converted = np.empty((len(p), len(self.KeyframeIdx.all), 3))

        parent_idx = self.Skeleton.parent_idx_vector()
        for our_joint_name, bvh_joint_name in self.joint_mappings.items():
            id_keyframe = self.KeyframeIdx.all[our_joint_name]
            if bvh_joint_name in self.added_joint_offsets:
                p_idx = parent_idx[id_keyframe]
                offset = np.repeat(self.added_joint_offsets[bvh_joint_name][None, :], len(p_converted), axis=0)
                q = quaternion.euler_to_quaternion(np.deg2rad(r_converted[:, p_idx]))
                p_converted[:, id_keyframe] = p_converted[:, p_idx] + quaternion.qrot_np(q, offset)
                r_converted[:, id_keyframe] = r_converted[:, p_idx].copy()
            else:
                id_bvh = list(self.joints.keys()).index(bvh_joint_name)
                p_converted[:, id_keyframe] = p[:, id_bvh] / self.scale
                r_converted[:, id_keyframe] = r[:, id_bvh]

        r_local = self.Keyframe.batch_torch_recover_local_rotations(self.Skeleton,
                                                                    torch.from_numpy(r_converted)).numpy()

        result = np.zeros((len(p), len(self.KeyframeIdx.all), 10), dtype=np.float32)
        result[:, :, :3] = p_converted
        result[:, :, 3:6] = r_converted
        result[:, :, 6:9] = r_local
        result[:, 0, 9] = np.linspace(0.0, len(p) / float(self.fps), len(p))

        use_out_of_view = True
        use_too_close = True
        occ_mask = np.array(occlusions)
        if use_out_of_view and len(occlusions) > 0:
            occ_mask |= np.array(out_of_views)
        if use_too_close and len(occlusions) > 0:
            occ_mask |= np.array(too_close)

        if len(occlusions) > 0:
            for occlusion_joint_idx, joint_name in enumerate(self.file_occlusions_joints):
                if joint_name not in self.KeyframeIdx.all:
                    continue

                joint_bit_mask = 1 << occlusion_joint_idx
                result_joint_index = self.KeyframeIdx.all[joint_name]
                result[:, result_joint_index, 9][occ_mask & joint_bit_mask == joint_bit_mask] += 1.
        return result

    def _get_occlusion_source_index(self):
        return np.array([
            62, 62, 62, 62, -1, -1, -1, -1,  # torso and head no occlusion, root uses occlusion of left hip
            10, 10, 10,  # rshoulder
            11,  # relbow
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,  # rwrist
            37, 37, 37,  # lshoulder
            38,  # lelbow
            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,  # lwrist
            62,  # rupperleg
            63,  # rlowerleg
            64, 64, 64, 64,  # rfoot
            68,  # lupperleg
            69,  # llowerleg
            70, 70, 70, 70  # lfoot
        ])

    @classmethod
    def get_normalized_path(cls):
        return os.path.join(os.getenv("PATH_UNOC"), "reference_skeleton.bvh")

    @classmethod
    def get_normalized_object(cls):
        parser = cls()
        parser.parse_file(cls.get_normalized_path())
        return parser

    @classmethod
    def get_normalized_skeleton(cls):
        return cls.get_normalized_object().skeleton()

    def _inherit_occlusions(self, keyframes):
        frames = np.copy(keyframes)
        occlusion_indices = self._get_occlusion_source_index()
        for i in range(len(occlusion_indices)):
            if occlusion_indices[i] >= 0:
                frames[:, i, 9] = frames[:, occlusion_indices[i], 9]
            else:
                frames[:, i, 9] = 0
        return frames

    def _apply_postprocessing(self, keyframes):
        frames = super()._apply_postprocessing(keyframes)
        frames = self._inherit_occlusions(frames)
        return frames

    @classmethod
    def name(cls):
        return "unoc"


if __name__ == "__main__":
    parser = UnocParser()

    # convert all files with normalized bone lengths
    keyframes = UnocParser.convert_all_files(normalize_skeleton=True)

    # convert all files with actual bone lengths
    keyframes = UnocParser.convert_all_files(normalize_skeleton=False)

    # import plots
    #
    # plot = plots.Vispy3DScatter()
    # for skeletons, keyframes_arrs in UnocParser().load_numpy():
    #     for take_idx in range(len(keyframes_arrs)):
    #         keyframes_arr = keyframes_arrs[take_idx]
    #         skeleton = skeletons[take_idx]
    #         keyframes = list(UnocParser.Keyframe.from_numpy_arr(keyframes_arr))
    #         plot.plot_skeleton_with_bones(keyframes_arr[:, :, [0, 2, 1]], skeleton=UnocParser.Skeleton, scale=1,
    #                                       center=np.array([0, 0, 1]), speed=20)
    #         print(len(keyframes))
