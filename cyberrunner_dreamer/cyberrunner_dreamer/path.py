import numpy as np
import pickle


class LinearPath:
    def __init__(
        self,
        waypoints,
        distance=0.0002,
        walls_h=None,
        walls_v=None,
        holes=None,
        board_width=0.276,
        board_height=0.231,
        wall_r=0.0025,
        waypoints_sections=None,
        sections=None,
    ):

        waypoints = np.asarray(waypoints, dtype=np.float32)
        self.orig_waypoints = waypoints

        points = np.empty((0, 2), dtype=np.float32)
        self.sections = sections
        if sections is not None:
            dimensions = np.array([board_width, board_height]) + 2.0 * wall_r
            num_cells = np.array(sections.shape[::-1], dtype=int)
            self.cell_size = dimensions / num_cells.astype(np.float64)
            self.section_p_range = np.zeros((sections.max() + 1, 2), dtype=int)
            prev_section = waypoints_sections[0]

        for i in range(1, waypoints.shape[0]):  # TODO: vectorize
            v = waypoints[i] - waypoints[i - 1]
            num_points = int(np.floor(np.linalg.norm(v) / distance)) + 1

            l = np.linspace(
                0.0, 1.0, num_points, endpoint=False, dtype=np.float32
            ).reshape(num_points, 1)
            p = waypoints[i - 1] + l * v

            points = np.vstack((points, p))

            if waypoints_sections is not None:
                section = waypoints_sections[i]
                if section == prev_section:
                    self.section_p_range[section, 1] += num_points
                else:
                    self.section_p_range[prev_section, 1] += num_points // 2
                    prev_end = self.section_p_range[prev_section, 1]
                    self.section_p_range[section, 0] = prev_end
                    self.section_p_range[section, 1] = prev_end + (
                        num_points - num_points // 2
                    )
                    prev_section = section

        # Create map
        self.distance = distance
        self.points = points
        self.num_points = points.shape[0]
        self.width = board_width
        self.height = board_height
        self.wall_r = wall_r
        self.indices = np.arange(0, self.points.shape[0])
        self.pos = 0

        # If either sections or walls and holes are specified precompute closest
        # point (-1 if blocked by walls or holes)  TODO move to function
        self.closest_idx = self.closest_dim_x = self.closest_dim_y = None
        if walls_h is not None and walls_v is not None and holes is not None:
            self.closest_dim_x = int(board_width / distance) + 1
            self.closest_dim_y = int(board_height / distance) + 1
            self.closest_idx = -1 * np.ones(
                (self.closest_dim_y, self.closest_dim_x),
                dtype=int,
            )
            for x in range(self.closest_dim_x):
                for y in range(self.closest_dim_y):
                    pos = np.array([x * distance, y * distance])

                    # TODO remove this
                    # self.closest_idx[y, x] = np.argmin(
                    #     np.linalg.norm(points - pos, axis=1)
                    # )
                    # continue

                    # Filter out points that are blocked by walls or holes
                    ball_r = wall_r # 0.0#0.0075
                    hy_dist = walls_h[:, 2] - pos[1]
                    hx_mask = np.logical_and(
                        pos[0] >= (walls_h[:, 0] - ball_r),
                        pos[0] <= (walls_h[:, 1] + ball_r),
                    )
                    hy_pos_mask = np.logical_and(
                        hy_dist >= 0,
                        hx_mask,
                    )
                    hy_neg_mask = np.logical_and(
                        hy_dist < 0,
                        hx_mask,
                    )
                    if np.any(hy_pos_mask):
                        y_dist_w_pos = hy_dist[hy_pos_mask].min()
                    else:
                        y_dist_w_pos = np.inf
                    if np.any(hy_neg_mask):
                        y_dist_w_neg = hy_dist[hy_neg_mask].max()
                    else:
                        y_dist_w_neg = -np.inf

                    vx_dist = walls_v[:, 2] - pos[0]
                    vy_mask = np.logical_and(
                        pos[1] >= (walls_v[:, 0] - ball_r),
                        pos[1] <= (walls_v[:, 1] + ball_r),
                    )
                    vx_pos_mask = np.logical_and(
                        vx_dist >= 0,
                        vy_mask,
                    )
                    vx_neg_mask = np.logical_and(
                        vx_dist < 0,
                        vy_mask,
                    )
                    if np.any(vx_pos_mask):
                        x_dist_w_pos = vx_dist[vx_pos_mask].min()
                    else:
                        x_dist_w_pos = np.inf
                    if np.any(vx_neg_mask):
                        x_dist_w_neg = vx_dist[vx_neg_mask].max()
                    else:
                        x_dist_w_neg = -np.inf

                    hole_r = 0.0075
                    hole_y_dist = holes[:, 1] - pos[1]
                    hole_x_mask = np.logical_and(
                        pos[0] >= (holes[:, 0] - hole_r),
                        pos[0] <= (holes[:, 0] + hole_r),
                    )
                    hole_y_pos_mask = np.logical_and(
                        hole_y_dist >= 0,
                        hole_x_mask,
                    )
                    hole_y_neg_mask = np.logical_and(
                        hole_y_dist < 0,
                        hole_x_mask,
                    )
                    if np.any(hole_y_pos_mask):
                        y_dist_h_pos = max(
                            0,
                            hole_y_dist[hole_y_pos_mask].min() - hole_r,
                        )
                    else:
                        y_dist_h_pos = np.inf
                    if np.any(hole_y_neg_mask):
                        y_dist_h_neg = min(
                            0,
                            hole_y_dist[hole_y_neg_mask].max() + hole_r,
                        )
                    else:
                        y_dist_h_neg = -np.inf

                    hole_x_dist = holes[:, 0] - pos[0]
                    hole_y_mask = np.logical_and(
                        pos[1] >= (holes[:, 1] - hole_r),
                        pos[1] <= (holes[:, 1] + hole_r),
                    )
                    hole_x_pos_mask = np.logical_and(
                        hole_x_dist >= 0,
                        hole_y_mask,
                    )
                    hole_x_neg_mask = np.logical_and(
                        hole_x_dist < 0,
                        hole_y_mask,
                    )
                    if np.any(hole_x_pos_mask):
                        x_dist_h_pos = max(
                            0,
                            hole_x_dist[hole_x_pos_mask].min() - hole_r,
                        )
                    else:
                        x_dist_h_pos = np.inf
                    if np.any(hole_x_neg_mask):
                        x_dist_h_neg = min(
                            0,
                            hole_x_dist[hole_x_neg_mask].max() + hole_r,
                        )
                    else:
                        x_dist_h_neg = -np.inf

                    x1 = pos[0] + max(x_dist_w_neg, x_dist_h_neg)
                    x2 = pos[0] + min(x_dist_w_pos, x_dist_h_pos)
                    y1 = pos[1] + max(y_dist_w_neg, y_dist_h_neg)
                    y2 = pos[1] + min(y_dist_w_pos, y_dist_h_pos)

                    # x1 = pos[0] + x_dist_h_neg
                    # x2 = pos[0] + x_dist_h_pos
                    # y1 = pos[1] + y_dist_h_neg
                    # y2 = pos[1] + y_dist_h_pos

                    p_mask = (
                        (points[:, 1] >= y1)
                        & (points[:, 1] <= y2)
                        & (points[:, 0] >= x1)
                        & (points[:, 0] <= x2)
                    )

                    if np.any(p_mask):
                        idx = np.argmin(np.linalg.norm(points[p_mask] - pos, axis=1))
                        self.closest_idx[y, x] = self.indices[p_mask][idx]

    def step(self):
        p = self.points[self.pos]
        self.pos = min(self.points.shape[0] - 1, self.pos + 1)

        return p

    def reset(self):
        self.pos = 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def closest_point(self, point):
        if self.closest_idx is None:
            if self.sections is None:
                idx = np.argmin(np.linalg.norm(self.points - point, axis=1))
            else:
                s = ((point + self.wall_r) / self.cell_size).astype(int)
                section = self.sections[s[1], s[0]]
                i1, i2 = self.section_p_range[section]
                idx = -1
                if i2 > i1:
                    idx = i1 + np.argmin(
                        np.linalg.norm(self.points[i1:i2] - point, axis=1)
                    )
        else:  # TODO handle out of bounds
            i, j = int(round(point[1] / self.distance)), int(
                round(point[0] / self.distance)
            )
            if (not 0 <= i < self.closest_idx.shape[0]) or (
                not 0 <= j < self.closest_idx.shape[1]
            ):
                idx = -1
            else:
                idx = self.closest_idx[i, j]

        p = None
        if idx != -1:
            p = self.points[idx]

        return idx, p

    def get_rel_path(self, point, num, step=50):
        idx = self.closest_point(point)[0]
        if idx == -1:
            rel_path = np.zeros((num, 2))
        else:
            indices = np.clip(
                np.arange(idx, idx + num * step, step), 0, self.points.shape[0] - 1
            )
            rel_path = self.points[indices] - point

        return rel_path

    def set_pos(self, pos):
        self.pos = np.clip(pos, 0, self.points.shape[0] - 1)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            p = pickle.load(f)

        return p


if __name__ == "__main__":
    from layout import cyberrunner_hard_layout
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layout = cyberrunner_hard_layout
    p = LinearPath(
        np.array(layout["waypoints"]),
        walls_h=np.array(layout["walls_h"]),
        walls_v=np.array(layout["walls_v"]),
        holes=np.empty((0, 3)), #np.array(layout["holes"]),
        distance=0.0002
    )

    # Vectorized jump-detection: mark neighbor cells with jump > threshold as -1
    threshold = 0.057
    idx_map = p.closest_idx.copy()

    # Vertical neighbors (downward)
    idx1 = p.closest_idx[:-1, :]
    idx2 = p.closest_idx[1:, :]
    valid = (idx1 != -1) & (idx2 != -1)
    # Compute jump distances based on index difference
    jump_dists = np.abs(idx1[valid] - idx2[valid]) * p.distance
    jump_mask = np.zeros_like(valid, dtype=bool)
    jump_mask[valid] = jump_dists > threshold
    # Mark both cells of each jump
    idx_map[:-1, :][jump_mask] = -1
    idx_map[1:, :][jump_mask] = -1

    # Horizontal neighbors (rightward)
    idx1 = p.closest_idx[:, :-1]
    idx2 = p.closest_idx[:, 1:]
    valid = (idx1 != -1) & (idx2 != -1)
    jump_dists = np.abs(idx1[valid] - idx2[valid]) * p.distance
    jump_mask = np.zeros_like(valid, dtype=bool)
    jump_mask[valid] = jump_dists > threshold
    idx_map[:, :-1][jump_mask] = -1
    idx_map[:, 1:][jump_mask] = -1

    # Visualize the closest_idx matrix with -1 values in red
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
    cmap.set_under('red')
    im = ax.imshow(idx_map, interpolation='none', cmap=cmap, vmin=0)
    ax.invert_yaxis()

    # Wall thickness (meters) to pixels
    thickness_px = 0.003 / p.distance

    # Overlay horizontal walls as black rectangles
    for x_start, x_end, y in layout["walls_h"]:
        x0 = x_start / p.distance
        width = (x_end - x_start) / p.distance
        y0 = y / p.distance - thickness_px / 2
        rect = Rectangle((x0, y0), width, thickness_px, facecolor='black', edgecolor='none')
        ax.add_patch(rect)

    # Overlay vertical walls as black rectangles
    for y_start, y_end, x_v in layout["walls_v"]:
        y0 = y_start / p.distance
        height = (y_end - y_start) / p.distance
        x0 = x_v / p.distance - thickness_px / 2
        rect = Rectangle((x0, y0), thickness_px, height, facecolor='black', edgecolor='none')
        ax.add_patch(rect)

    # Overlay holes
    s = (0.015 / p.distance) ** 2.0
    s = 100
    print(s)
    xs = [x / p.distance for x, y in layout["holes"]]
    ys = [y / p.distance for x, y in layout["holes"]]
    ax.scatter(xs, ys, marker='o', s=s)

    # Overlay path of all sample points as black line
    xs_path = p.points[:, 0] / p.distance
    ys_path = p.points[:, 1] / p.distance
    ax.plot(xs_path, ys_path, linestyle='-', linewidth=1, color='black')

    plt.show()
