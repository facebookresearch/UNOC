# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import numpy as np
from tqdm import tqdm

from npybvh import bvh
import definitions
import torch


class BvhConverter(bvh.Bvh):
    root_path = ""
    joint_mappings = {}
    toe_joint = "b_r_ball"
    scale = 1
    ignore_files = []

    Keyframe = definitions.Keyframe
    KeyframeIdx = definitions.KeyframeIdx
    Skeleton = definitions.Skeleton
    save_separately = False
    file_occlusions = None
    file_out_of_views = None
    file_occlusions_joints = []

    def _resize_occlusions(self, target_size, occlusions, out_of_views, too_close):
        if len(occlusions) > target_size:
            print(f"occlusions are longer than animation ({len(occlusions)} vs {target_size})")
            occlusions = occlusions[len(occlusions) - target_size:]
            out_of_views = out_of_views[len(out_of_views) - target_size:]
            too_close = too_close[len(too_close) - target_size:]

        if len(occlusions) < target_size:
            print(f"occlusions are shorter than animation ({len(occlusions)} vs {target_size})")
            new_occ = np.zeros(target_size, dtype=np.int)
            new_oov = np.zeros(target_size, dtype=np.int)
            new_too_close = np.zeros(target_size, dtype=np.int)
            new_occ[:len(occlusions)] = occlusions
            new_oov[:len(occlusions)] = out_of_views
            new_too_close[:len(occlusions)] = too_close
            new_occ[len(occlusions):] = new_occ[len(occlusions) - 1]
            new_oov[len(occlusions):] = new_oov[len(occlusions) - 1]
            new_too_close[len(occlusions):] = too_close[len(occlusions) - 1]
            occlusions = new_occ
            out_of_views = new_oov
            too_close = new_too_close

        return occlusions, out_of_views, too_close

    def convert_all(self, file="", folder=""):
        p, r = self.batch_all_frame_poses()
        p = p.numpy()
        r = r.numpy()
        occlusions, out_of_views, too_close = self.get_occlusions_and_out_of_views(file)

        occlusions, out_of_views, too_close = self._resize_occlusions(len(p), occlusions, out_of_views)

        keyframes = []
        keyframes.extend(self.batch_convert_frames(p, r, occlusions, out_of_views, too_close))

        return np.array(keyframes)

    def batch_convert_frames(self, p, r, occlusions, out_of_views, too_close):
        p_converted = np.empty((len(p), len(self.KeyframeIdx.all), 3))
        r_converted = np.empty((len(p), len(self.KeyframeIdx.all), 3))

        for our_joint_name, bvh_joint_name in self.joint_mappings.items():
            id_keyframe = self.KeyframeIdx.all[our_joint_name]
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

    def get_occlusions_and_out_of_views(self, file=""):
        return [], [], []

    def skeleton(self):
        offset = np.zeros(3)
        poses = []
        self._add_pose_recursive(self.root, offset, poses)
        root_height = -poses[list(self.joints.keys()).index(self.toe_joint)][1]
        offset_pos = np.array([0, root_height, 0])

        positions = np.zeros((len(self.joint_mappings), 3))
        for key in self.joint_mappings.keys():
            positions_index = self.KeyframeIdx.all[key]
            bvh_index = list(self.joints.keys()).index(self.joint_mappings[key])
            positions[positions_index] = poses[bvh_index].reshape(3) + offset_pos

        return self.Skeleton(positions)

    @classmethod
    def get_all_file_paths(cls):
        result = []
        for folder in cls.get_all_folders():
            result.extend(cls.get_all_files_of_folder(folder))
        return

    @classmethod
    def get_normalized_path(cls):
        raise NotImplemented()

    @classmethod
    def convert_all_files(cls, normalize_skeleton=False):
        keyframes = []
        skeletons = []
        file_count = 0

        for file in tqdm(cls.get_all_file_paths(), f"Converting bvh to numpy"):
            parser = cls()
            # anim_name = file.split('/')[-1][:-4]
            anim_name = os.path.split(file)[-1][:-4]
            folder = file.split(anim_name)[0]
            parser.get_occlusions_and_out_of_views(folder)
            parser.parse_file(file)

            if normalize_skeleton:
                override_parser = cls()
                override_parser.parse_file(cls.get_normalized_path())
                for key, val in parser.joints.items():
                    val.offset = override_parser.joints[key].offset

            skeleton = parser.skeleton()
            if skeleton is None:
                continue
            skeletons.append(skeleton)
            keyframes.append(parser.convert_all(anim_name, folder))

            if cls.save_separately:
                suffix = f"{cls.Keyframe.__name__}{file_count}{'_normalized' if normalize_skeleton else ''}"
                with open(os.path.join(cls.root_path, f"skeletons_{suffix}.pickle"), "wb+") as outfile:
                    pickle.dump(skeletons, outfile)

                np.save(os.path.join(cls.root_path, f"keyframes_{suffix}.npy"), keyframes, allow_pickle=True)
                file_count += 1
                skeletons.clear()
                keyframes.clear()

        if not cls.save_separately:
            suffix = f"{cls.Keyframe.__name__}{'_normalized' if normalize_skeleton else ''}"
            with open(os.path.join(cls.root_path, f"skeletons{suffix}.pickle"), "wb+") as outfile:
                pickle.dump(skeletons, outfile)

            np.save(os.path.join(cls.root_path, f"keyframes{suffix}.npy"), keyframes, allow_pickle=True)

    @classmethod
    def get_all_folders(cls):
        return list(os.walk(cls.root_path))[0][1]

    @classmethod
    def get_all_files_of_folder(cls, folder_path):
        file_paths = []
        count_frames = False
        frames = 0
        for file_name in list(os.walk(folder_path))[0][2]:
            if file_name in cls.ignore_files:
                continue
            if file_name.endswith(".bvh"):
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

    def _apply_postprocessing(self, keyframes):
        return keyframes

    @classmethod
    def get_all_keyframe_folders(cls):
        return cls.get_all_folders()

    def _check_and_convert(self, normalized):
        folders = self.__class__.get_all_keyframe_folders()
        suffix = f"{self.Keyframe.__name__}{0}{'_normalized' if normalized else ''}"
        if not os.path.exists(os.path.join(list(folders)[0], f"skeletons_{suffix}.pickle")):
            self.__class__.convert_all_files(normalize_skeleton=normalized)

    def load_numpy(self, normalized=False):
        # check if dataset was already converted. Otherwise, do now
        self._check_and_convert(normalized)

        folders = self.__class__.get_all_keyframe_folders()
        skeletons = []
        keyframes = []
        for folder in folders:
            file_count = 0
            _suffix = ("_normalized" if normalized else "")
            if self.save_separately:
                suffix = f"{self.Keyframe.__name__}{file_count}{_suffix}"
                while os.path.exists(os.path.join(folder, f"skeletons_{suffix}.pickle")):
                    try:
                        with open(os.path.join(folder, f"skeletons_{suffix}.pickle"), "rb+") as infile:
                            _skeletons = pickle.load(infile)
                        _keyframes = np.load(os.path.join(folder, f"keyframes_{suffix}.npy"), allow_pickle=True)
                    except:
                        print(f"could not import {folder}{suffix}")
                    if _keyframes.shape[0] == 1:
                        _keyframes = _keyframes.reshape(_keyframes.shape[1:])
                    _keyframes = self._apply_postprocessing(_keyframes)
                    _keyframes = list(_keyframes.reshape((1, *_keyframes.shape)))
                    yield _skeletons, _keyframes
                    file_count += 1
                    suffix = f"{self.Keyframe.__name__}{file_count}{_suffix}"
                return
            else:
                suffix = _suffix
                try:
                    with open(os.path.join(folder, f"skeletons{suffix}.pickle"), "rb+") as infile:
                        _skeletons = pickle.load(infile)
                    _keyframes = np.load(os.path.join(folder, f"keyframes{suffix}.npy"), allow_pickle=True)
                except:
                    print(f"could not import {folder}{suffix}")
                _keyframes = self._apply_postprocessing(_keyframes)
                skeletons.extend(_skeletons)
                keyframes.extend(list(_keyframes))

        if not self.save_separately:
            return zip(skeletons, keyframes)

    def get_bone_lengths(self, parent_idx, normalized=True):
        from utils import bone_lengths_np

        for skeleton, tracks in self.load_numpy(normalized=normalized):
            for _track in tracks:
                track = _track[0] if len(_track.shape) == 4 else _track
                return bone_lengths_np(track[0, :, :3], parent_idx)

    @classmethod
    def name(cls):
        return "none"
