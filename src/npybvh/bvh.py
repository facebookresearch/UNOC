# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler
from multiprocessing import Pool
import torch
import quaternion


class BvhJoint:
    def __init__(self, name, parent, index_offset=0):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []
        self.index_offset = index_offset
        self._rotation_order = ""

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])

    def get_rotation_order(self):
        if self._rotation_order == "":
            for channel in self.channels:
                if channel.endswith("rotation"):
                    self._rotation_order += channel.split("rotation")[0].lower()
        return self._rotation_order

    def hierarchy_string(self, indent):
        _i = indent
        if self.parent is None:
            s = f"{_i}ROOT {self.name}\n"
        elif len(self.children) == 0:
            s = f"{_i}End Site\n"
        else:
            s = f"{_i}JOINT {self.name}\n"
        s += f"{_i}{{\n"
        _i = indent + "    "
        s += f"{_i}OFFSET {np.array2string(self.offset, precision=4, separator=' ')[1:-1]}\n"
        if len(self.channels) > 0:
            s += f"{_i}CHANNELS {len(self.channels)} {' '.join(self.channels)}\n"

        for child in self.children:
            s += child.hierarchy_string(_i)

        _i = indent
        s += f"{_i}}}\n"

        return s


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root: BvhJoint = None
        self.keyframes: np.ndarray = None
        self.keyframes_torch: torch.tensor = None
        self.frames = 0
        self.fps = 0
        self.header = ""

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        index_offset = 0
        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent, index_offset)
                self.joints[joint.name] = joint
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                index_offset += int(words[1])
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1], index_offset)
                # joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def get_skeleton(self):
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)
        return pos

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        pos = self.get_skeleton()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        scale = np.max(pos) / 2
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(0, scale * 2)
        plt.show()

    def get_bone_offsets(self, include_end=True):
        off = []
        for joint in self.joints.values():
            if include_end or not joint.name.endswith("_end"):
                off.append(joint.offset)
        return off

    def get_parent_idx(self, include_end=True):
        p_idx = []
        joints = []
        for joint in self.joints.values():
            if include_end or not joint.name.endswith("_end"):
                joints.append(joint)
                s = joint.name + ";"
                s += (joint.parent.name if joint.parent is not None else "None") + ";"
                s += str(joint.offset[0]) + ";"
                s += str(joint.offset[1]) + ";"
                s += str(joint.offset[2]) + ";"
                print(s)

        for joint in joints:
            if joint.parent is not None:
                p_idx.append(joints.index(joint.parent))
            else:
                p_idx.append(-1)
        return p_idx

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)
            _line = re.sub('\\s+', ' ', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            # for angle_index in range(len(words)):
            #     self.keyframes[frame, angle_index] = float(words[angle_index])
            self.keyframes[frame] = np.fromstring(_line, dtype=np.float32, sep=" ")

            frame += 1

        self.keyframes_torch = torch.from_numpy(self.keyframes)

    def save_header(self, hierarchy, motion):
        self.header = hierarchy
        # frame_info = motion.split('\n', 4)
        # self.header += "MOTION\n"
        # self.header += frame_info[1] + "\n" + frame_info[2] + "\n"

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self.save_header(hierarchy, motion)
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            local_rotation *= 0.0
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

            M_channel = euler2mat(*np.deg2rad(local_rotation))
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def extract_joint_rotation(self, frame_pose, joint):
        return self._extract_rotation(frame_pose, joint.index_offset, joint)[0]

    def set_joint_rotation(self, frame_pose, joint, M):
        axes = 's' + joint.get_rotation_order()
        rot = mat2euler(M, axes=axes)
        frame_pose[joint.index_offset: joint.index_offset + 3] = np.rad2deg(rot)

    def clear(self, n_frames):
        self.keyframes = np.zeros((n_frames, self.keyframes.shape[1]))
        self.keyframes_torch = torch.from_numpy(self.keyframes)
        self.frames = n_frames

    def set_pos(self, p):
        joint = self.root
        self.keyframes[:, joint.index_offset: joint.index_offset + 3] = p
        self.keyframes_torch = torch.from_numpy(self.keyframes)

    def override_joint_rotation(self, rot, joint_mappings):
        for joint_idx, joint_name in enumerate(joint_mappings.values()):
            joint = self.joints[joint_name]
            off = 3 if len(joint.channels) > 3 else 0
            self.keyframes[:, joint.index_offset + off: joint.index_offset + off + 3] = rot[:, joint_idx]
        self.keyframes_torch = torch.from_numpy(self.keyframes)

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = None

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + (offset_position if offset_position is not None else M_parent.dot(joint.offset))

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.eye(3)
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))

        return p, r

    def _batch_extract_position(self, joint, index_offset):
        offset_position = torch.zeros((self.frames, 3))
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[:, 0] = self.keyframes_torch[:, index_offset]
            elif channel == "Yposition":
                offset_position[:, 1] = self.keyframes_torch[:, index_offset]
            elif channel == "Zposition":
                offset_position[:, 2] = self.keyframes_torch[:, index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def batch_get_joint_rotation(self, joint: BvhJoint):
        local_rotation = torch.zeros((self.frames, 3))
        q_rotation = quaternion.identity(self.frames)
        deg2rad = 0.0174533
        index_offset = joint.index_offset
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            local_rotation *= 0.0
            if channel == "Xrotation":
                local_rotation[:, 0] = self.keyframes_torch[:, index_offset]
            elif channel == "Yrotation":
                local_rotation[:, 1] = self.keyframes_torch[:, index_offset]
            elif channel == "Zrotation":
                local_rotation[:, 2] = self.keyframes_torch[:, index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

            q_channel = quaternion.euler_to_quaternion_torch(deg2rad * local_rotation)
            q_rotation = quaternion.qmul(q_rotation, q_channel)

        return q_rotation

    def _batch_extract_rotation(self, index_offset, joint):
        local_rotation = torch.zeros((self.frames, 3))
        q_rotation = quaternion.identity(self.frames)
        deg2rad = 0.0174533
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            local_rotation *= 0.0
            if channel == "Xrotation":
                local_rotation[:, 0] = self.keyframes_torch[:, index_offset]
            elif channel == "Yrotation":
                local_rotation[:, 1] = self.keyframes_torch[:, index_offset]
            elif channel == "Zrotation":
                local_rotation[:, 2] = self.keyframes_torch[:, index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

            q_channel = quaternion.euler_to_quaternion_torch(deg2rad * local_rotation)
            q_rotation = quaternion.qmul(q_rotation, q_channel)

        return q_rotation, index_offset

    def _batch_recursive_apply_pose(self, joint, index_offset, p, r, q_parent, p_parent):
        rad2deg = 57.2958
        if joint.position_animated():
            offset_position, index_offset = self._batch_extract_position(joint, index_offset)
        else:
            offset_position = None

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            position = p_parent + quaternion.qrot(q_parent, torch.from_numpy(joint.offset).float())
            p[:, joint_index] = position
            r[:, joint_index] = quaternion.qeuler(q_parent) * rad2deg
            q = q_parent
        else:
            if joint.rotation_animated():
                q_rotation, index_offset = self._batch_extract_rotation(index_offset, joint)
            else:
                q_rotation = quaternion.identity(self.frames)

            q = quaternion.qmul(q_parent, q_rotation)
            position = p_parent + (offset_position if offset_position is not None else quaternion.qrot(q_parent, torch.from_numpy(joint.offset).float()))
            rotation = rad2deg * quaternion.qeuler(q)
            joint_index = list(self.joints.keys()).index(joint.name)
            p[:, joint_index] = position
            r[:, joint_index] = rotation

        for c in joint.children:
            index_offset = self._batch_recursive_apply_pose(c, index_offset, p, r, q, position)

        return index_offset

    def batch_all_frame_poses(self):
        p = torch.empty((self.frames, len(self.joints), 3))
        r = torch.empty((self.frames, len(self.joints), 3))
        p_parent = torch.zeros((self.frames, 3))
        q_parent = quaternion.identity(self.frames)

        self._batch_recursive_apply_pose(self.root, 0, p, r, q_parent, p_parent)

        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def all_frame_poses_parallel(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        with Pool(8) as pool:
            p_r = pool.map(self.frame_pose, list(range(len(self.keyframes))))
            p, r = map(list, zip(*p_r))

        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None, draw=True, scale=30):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(0, 2 * scale)

        if draw:
            plt.draw()
            plt.pause(0.001)

    def plot_frame(self, frame, fig=None, ax=None, draw=True, scale=30):
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax, draw=draw, scale=scale)

    def joint_names(self):
        return list(self.joints.keys())

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def export_file(self, path):
        with open(path, "w+") as f:
            f.write(self.header)
            f.write("MOTION\n")
            f.write(f"Frames: {self.frames}\n")
            f.write(f"Frame Time: {1.0 / self.fps:0.5f}\n")
            for line in self.keyframes:
                # f.write('    '.join(map(str, line)) + "\n")
                f.write('    '.join([f"{x:.5f}" for x in line]) + "\n")

    def plot_all_frames(self, scale=30):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.frames):
            self.plot_frame(i, fig, ax, scale=scale)

    def plot_all_frame_vispy(self, scale=30, flip_x=True, debug_markers=[], speed=3, vispy_window=None, cam_speed=0, skip=0):
        p, r = self.batch_all_frame_poses()
        if skip > 0:
            p = p[skip:]
            r = r[skip:]
        if flip_x:
            p[:, :, 0] *= -1
            if len(debug_markers) > 0:
                debug_markers[:, 0] *= -1
        import plots
        if vispy_window is None:
            vispy_window = plots.Vispy3DScatter()
        bones = np.zeros((p.shape[0], p.shape[1] * 2 - 2, 3))
        p = p.numpy()
        joint_names = list(self.joints.keys())
        for idx, joint in enumerate(self.joints.values()):
            if idx == 0:
                continue
            else:
                bones[:, (idx - 1) * 2] = p[:, joint_names.index(joint.parent.name)]
                bones[:, (idx - 1) * 2 + 1] = p[:, idx]

        p = p[:, :, [0, 2, 1]]
        bones = bones[:, :, [0, 2, 1]]
        markers = debug_markers[:, [0, 2, 1]] if len(debug_markers) > 0 else []

        center = np.array([0, 0, np.mean(p[:, :, 2])])

        vispy_window.plot(p,
                          bones,
                          scale=scale,
                          speed=speed,
                          fps=speed * self.fps,
                          center=center,
                          debug_markers=markers,
                          camera_rotation_speed=cam_speed)

        return vispy_window

    def save(self, path):
        s = "HIERARCHY\n" + self.root.hierarchy_string("")
        s += "MOTION\n"
        s += f"Frames: {self.frames}\n"
        s += f"Frame Time: {1. / self.fps:.5f}\n"
        for line in self.keyframes:
            s += '  '.join(f"{item: 9.4f}" for item in line) + "\n"
        # print(s)
        with open(path, "w+") as f:
            f.write(s)

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"


def plot_folder(path):
    import os
    vispy_window = None
    for file in list(os.walk(path))[0][2]:
        if not file.endswith(".bvh"):
            continue
        print(f"plotting {path + file}")
        anim = Bvh()
        anim.parse_file(path + file)
        # anim.plot_hierarchy()
        vispy_window = anim.plot_all_frame_vispy(scale=100, speed=10, vispy_window=vispy_window, cam_speed=0)


def plot_file(path):
    # create Bvh parser
    anim = Bvh()
    # parse file
    anim.parse_file(path)

    # anim.plot_hierarchy()
    anim.plot_all_frame_vispy(scale=100, speed=1.0, cam_speed=0.0)
    # draw the skeleton in T-pose

    # extract single frame pose: axis0=joint, axis1=positionXYZ/rotationXYZ
    p, r = anim.frame_pose(0)

    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    # all_p, all_r = anim.all_frame_poses()
    #
    # # print all joints, their positions and orientations
    # for _p, _r, _j in zip(p, r, anim.joint_names()):
    #     print(f"{_j}: p={_p}, r={_r}")

    # draw the skeleton for the given frame
    # anim.plot_frame(22)

    # show full animation
    # anim.plot_all_frames(scale=100)


def plot_skeleton(path, skeleton=None):
    anim = Bvh()
    anim.parse_file(path)
    p = np.repeat(anim.get_skeleton()[None, :], 1000, axis=0)
    import plots
    vispy_window = plots.Vispy3DScatter()
    if skeleton is not None:
        vispy_window.plot_skeleton_with_bones(p, fps=0, skeleton=skeleton)
    else:
        v = np.empty((p.shape[0], p.shape[1] * 2, p.shape[2]))
        for joint_idx, parent_idx in enumerate(anim.get_parent_idx(include_end=True)):
            i0 = parent_idx if parent_idx >= 0 else joint_idx
            i1 = joint_idx
            v[:, i1 * 2] = p[:, i0]
            v[:, i1 * 2 + 1] = p[:, i1]

        vispy_window.plot(p, fps=0, v=v)
