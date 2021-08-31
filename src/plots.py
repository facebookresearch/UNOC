# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from vispy import visuals, scene
import time


class Vispy3DScatter:
    def __init__(self):
        self.Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.Line3D = scene.visuals.create_visual_node(visuals.LineVisual)
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor="w")
        self.view = self.canvas.central_widget.add_view()
        self.scale = 1.0
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 5
        self.view.camera.elevation = 12.
        self.view.camera.distance = self.scale * 5
        self.view.camera.azimuth = 180
        self.center = np.zeros(3)
        c_axis = "b"
        self.axis_config = dict(font_size=48, axis_color=c_axis, tick_color=c_axis, text_color=c_axis,
                                parent=self.view.scene)
        self.x_axis = None
        self.y_axis = None
        self.update_axis()
        self.scatter_plot = self.Scatter3D(parent=self.view.scene)
        self.scatter_debug_plot = self.Scatter3D(parent=self.view.scene)
        self.line_plot = None

    def update_axis(self):
        if self.x_axis is None:
            self.x_axis = scene.Axis(pos=[[-self.scale, -self.scale], [self.scale, -self.scale]],
                                     domain=(-self.scale + self.center[0], self.scale + self.center[0]),
                                     tick_direction=(0, -1), **self.axis_config)
        else:
            self.x_axis.pos = [[-self.scale, -self.scale], [self.scale, -self.scale]]
            self.x_axis.domain = (-self.scale + self.center[0], self.scale + self.center[0])

        if self.y_axis is None:
            self.y_axis = scene.Axis(pos=[[-self.scale, -self.scale], [-self.scale, self.scale]],
                                     domain=(-self.scale + self.center[1], self.scale + self.center[1]),
                                     tick_direction=(-1, 0), **self.axis_config)
        else:
            self.y_axis.pos = [[-self.scale, -self.scale], [-self.scale, self.scale]]
            self.y_axis.domain = (-self.scale + self.center[1], self.scale + self.center[1])

        self.x_axis.transform = scene.STTransform(translate=(0, 0, -self.scale))
        self.y_axis.transform = scene.STTransform(translate=(0, 0, -self.scale))

    def plot_skeleton_with_bones(self, p, skeleton, scale=1, p_colors="blue", l_colors="blue", center=np.zeros(3),
                                 speed=1, debug_markers=[], camera_rotation_speed=0, n_skeletons=1,
                                 red_bones_for_skeleton=None, occlusions=None, fps=None):
        v = np.empty((p.shape[0], p.shape[1] * 2, p.shape[2]))

        n_joints = len(skeleton.Idx.all)
        for skeleton_idx in range(n_skeletons):
            off = skeleton_idx * n_joints
            for joint_idx, parent_idx in enumerate(skeleton.parent_idx_vector()):
                i0 = off + (parent_idx if parent_idx >= 0 else joint_idx)
                i1 = off + joint_idx
                v[:, i1 * 2] = p[:, i0]
                v[:, i1 * 2 + 1] = p[:, i1]

        if type(l_colors) is not type("") and len(l_colors) > 1 and red_bones_for_skeleton is None:
            color_arr = []
            for color in l_colors:
                for i in range(n_joints * 2):
                    color_arr.append(color)
            l_colors = np.array(color_arr)

        if fps is None:
            fps = speed * 60

        if red_bones_for_skeleton is not None and type(l_colors) == np.ndarray:
            color_arr = np.empty((v.shape[0], v.shape[1], 4))
            red_color = np.array([1., 0., 0., 1.])[None, None, :]
            red_color = np.repeat(np.repeat(red_color, v.shape[0], axis=0), v.shape[1] // n_skeletons, axis=1)
            for skeleton_idx in range(n_skeletons):
                color_arr[:, (n_joints * 2) * skeleton_idx: (n_joints * 2) * (skeleton_idx + 1)] = l_colors[
                    skeleton_idx]
                if skeleton_idx in red_bones_for_skeleton:
                    c = color_arr[:, (n_joints * 2) * skeleton_idx: (n_joints * 2) * (skeleton_idx + 1)]
                    occ_mask = occlusions > 0.00001
                    c[:, ::2][occ_mask] = red_color[:, ::2][occ_mask]
                    c[:, 1::2][occ_mask] = red_color[:, 1::2][occ_mask]
                    color_arr[:, (n_joints * 2) * skeleton_idx: (n_joints * 2) * (skeleton_idx + 1)] = c

            l_colors = color_arr

        self.plot(p, v, scale, p_colors, l_colors, center, speed, debug_markers, camera_rotation_speed, fps=fps)

    def plot(self, p, v=None, scale=1, p_colors="blue", l_colors="blue", center=np.zeros(3), speed=1,
             debug_markers=[], camera_rotation_speed=0, fps=None):
        if scale != self.scale:
            self.scale = scale
            self.view.camera.distance = self.scale * 5
        self.center = center
        self.update_axis()
        _p = p if len(p.shape) == 3 else np.array([p])
        if v is not None:
            _v = v if len(v.shape) == 3 else np.array([v])
        else:
            _v = None
        _p = _p[:] - center
        _v = _v[:] - center if _v is not None else np.array([])
        debug_markers = debug_markers[:] - center if len(debug_markers) > 0 else debug_markers

        config = dict(face_color=p_colors, symbol='o', size=3, edge_width=1, edge_color=p_colors)
        config_debug = dict(face_color="grey", symbol='x', size=3, edge_width=1, edge_color="grey")
        config_line = dict(width=5, color=l_colors, connect="segments")
        if len(_v) > 0 and self.line_plot is None:
            self.line_plot = self.Line3D(parent=self.view.scene, antialias=True)
            self.line_plot.set_gl_state('translucent', blend=True, depth_test=True)

        if len(debug_markers) > 0:
            self.scatter_debug_plot.set_data(debug_markers, **config_debug)

        if fps is None:
            fps = speed * 60

        p_idx = 0
        start_time = time.time()
        while p_idx < len(_p) - 1:
            delta_time = (time.time() - start_time)
            p_idx = min(int(delta_time * fps), len(_p) - 1)
            if camera_rotation_speed != 0:
                self.view.camera.azimuth = p_idx / speed / 3 * camera_rotation_speed + 130
            if isinstance(p_colors, np.ndarray) and len(p_colors.shape) >= 2 and p_colors.shape[0] == len(_p):
                config["face_color"] = p_colors[p_idx]
                config["edge_color"] = p_colors[p_idx]
            self.scatter_plot.set_data(_p[p_idx], **config)
            if len(_v) > p_idx:
                if isinstance(l_colors, np.ndarray) and len(l_colors.shape) >= 2 and l_colors.shape[0] == len(_p):
                    config_line["color"] = l_colors[p_idx]
                if isinstance(l_colors, np.ndarray) and len(l_colors.shape) == 2:
                    config_line["color"] = l_colors
                self.line_plot.set_data(_v[p_idx], **config_line)
            self.canvas.update()
            self.canvas.app.process_events()
            self.canvas.app.process_events()
            print(f"\r{p_idx}", end="")
            time.sleep(0.016)

    print("")


if __name__ == "__main__":
    samples = 60
    data = np.random.random((samples, 100, 3)) * 2 + np.array([-1, -1, 0])
    lines = np.random.random((samples, 100, 3)) * 2 + np.array([-1, -1, 0])
    p_color = np.random.random((samples, 100, 4))
    l_color = np.random.random((samples, 100, 4))
    plot = Vispy3DScatter()
    plot.plot(data, lines, scale=1, p_colors=p_color, l_colors=l_color, center=np.array([0, 0, 1]))
    time.sleep(1)
    plot.plot(data, lines, scale=2, p_colors=p_color, l_colors=l_color, center=np.array([0, 0, 2]))
