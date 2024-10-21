import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from .divers import move_figure


class Anim3d:
    def __init__(
        self, xlim=[-0.02, 0.33], ylim=[-0.04, 0.31], zlim=[0.0, 0.21], viewpoint="side"
    ):

        self.fig = plt.figure(figsize=(6, 6))
        # title = self.fig._suptitle.get_text()
        if len(plt.get_fignums()) == 1:
            move_figure(self.fig, 700, 50)
        if len(plt.get_fignums()) == 2:
            move_figure(self.fig, 1300, 50)
        self.ax = self.fig.add_subplot(projection="3d")
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.viewpoint = viewpoint

    def init_3d_anim(self, T__W_C):
        self.init_ax()

        self.cam_pyramid = self.get_camera_pyramid(T__W_C, "c", 0.02, 1)
        self.ax.add_collection3d(self.cam_pyramid)
        # self.ax.plot(0.1,0.1,-0.2, linestyle="", marker="o")

        self.fig.show()
        # We need to draw the canvas before we start animating...
        self.fig.canvas.draw()

        self.B__W = np.zeros((3))
        self.maze_corners__W = np.zeros((4, 3))

        (self.ball_plot,) = self.ax.plot(
            0, 0, 0, linestyle="", marker="o", animated=True
        )
        self.plate_collection = Poly3DCollection(
            [[np.zeros(3)]], facecolors="blue", linewidths=1, edgecolors="r", alpha=0.1
        )
        self.ax.add_collection3d(self.plate_collection)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def init_ax(self):
        # self.ax.set_aspect("auto")
        self.ax.set_aspect("auto")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        if self.viewpoint == "top":
            self.ax.view_init(90, -90, 0)
        if self.viewpoint == "side":
            self.ax.view_init(30, -60, 0)

    def update_anim(self):

        self.fig.canvas.restore_region(self.background)

        # vertices = np.random.rand(4,3)/10
        vertices = [list(self.maze_corners__W)]
        self.plate_collection.set_verts(vertices)
        self.plate_collection.do_3d_projection()

        self.ball_plot.set_data(self.B__W[0], self.B__W[1])
        self.ball_plot.set_3d_properties(self.B__W[2])

        self.ax.draw_artist(self.ball_plot)
        self.ax.draw_artist(self.plate_collection)
        self.fig.canvas.blit(self.ax.bbox)

    def get_camera_pyramid(
        self, extrinsic, color="r", focal_len_scaled=5, aspect_ratio=1
    ):
        vertex_std = np.array(
            [
                [0, 0, 0, 1],
                [
                    focal_len_scaled * aspect_ratio,
                    -focal_len_scaled * aspect_ratio,
                    focal_len_scaled,
                    1,
                ],
                [
                    focal_len_scaled * aspect_ratio,
                    focal_len_scaled * aspect_ratio,
                    focal_len_scaled,
                    1,
                ],
                [
                    -focal_len_scaled * aspect_ratio,
                    focal_len_scaled * aspect_ratio,
                    focal_len_scaled,
                    1,
                ],
                [
                    -focal_len_scaled * aspect_ratio,
                    -focal_len_scaled * aspect_ratio,
                    focal_len_scaled,
                    1,
                ],
            ]
        )
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [
            [
                vertex_transformed[0, :-1],
                vertex_transformed[1][:-1],
                vertex_transformed[2, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[4, :-1],
                vertex_transformed[1, :-1],
            ],
            [
                vertex_transformed[1, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
        ]
        return Poly3DCollection(
            meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35
        )


if __name__ == "__main__":
    tstart = time.time()

    anim = Anim3d(np.eye(4))
    for i in range(2000):
        anim.update_anim()
    print(2000 / (time.time() - tstart))
