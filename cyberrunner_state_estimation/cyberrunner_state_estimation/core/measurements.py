# import sys
# sys.path.insert(0, 'G:/RA_IDSC_current/01_Source/cyberrunner')

import cv2
import numpy as np


from cyberrunner_state_estimation.core.detection import Detector, DetectorFixedPts
from cyberrunner_state_estimation.core.plate_pose import PlatePoseEstimator
from cyberrunner_state_estimation.utils.anim_3d import Anim3d
from cyberrunner_state_estimation.utils.divers import init_win_subimages


class Measurements:
    def __init__(
        self, markers, do_anim_3d=True, viewpoint="side", show_subimages_detector=False
    ):
        self.detector = Detector(markers[4:], show_subimages=show_subimages_detector)
        self.detector_fixed_points = DetectorFixedPts(
            markers[:4], show_subimages=show_subimages_detector
        )
        self.plate_pose = PlatePoseEstimator()

        self.plate_angles = (None, None)
        self.ball_pos = None
        self.ball_img_coords = None
        self.ball_subimg = None

        if do_anim_3d:
            if viewpoint == "side":
                self.anim_3d_side = Anim3d(viewpoint="side")
                self.anim_3d_top = None
            if viewpoint == "top":
                self.anim_3d_top = Anim3d(viewpoint="top")
                self.anim_3d_side = None
            if viewpoint == "topandside":
                self.anim_3d_side = Anim3d(viewpoint="side")
                self.anim_3d_top = Anim3d(viewpoint="top")
        else:
            self.anim_3d_top = None
            self.anim_3d_side = None

        if show_subimages_detector:
            init_win_subimages()

    def process_frame(
        self,
        frame,
        get_ball_subimg=False,
    ):
        """
        Process the frame to compute the angles of the plate and the position of the ball in the maze frame {m}.

        Args :
            frame: np.ndarray, dim: (400, 640)
        """
        if self.plate_pose.T__W_C is None:
            self.camera_localization(frame)
            if self.anim_3d_top is not None:
                self.anim_3d_top.init_3d_anim(self.plate_pose.T__W_C)
            if self.anim_3d_side is not None:
                self.anim_3d_side.init_3d_anim(self.plate_pose.T__W_C)

        frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        if get_ball_subimg:
            frame_copy = frame.copy()

        corners_img_coords, ball_img_coords = self.detector.process_frame(
            frame
        )  # (x,y)

        raw_pts = np.zeros((5, 2))
        raw_pts[:4, :] = corners_img_coords
        raw_pts[4, :] = ball_img_coords
        undist_pts = self.plate_pose.undistort_points(raw_pts)  # (x,y)
        corners_undist = undist_pts[:4, :]
        ball_undist = undist_pts[4, :]

        alpha, beta = self.plate_pose.estimate_anglesXY(corners_undist)
        self.plate_angles = (alpha, beta)
        self.ball_pos = self.ball_pos_backproject(
            ball_undist, self.plate_pose.K, self.plate_pose.T__C_M
        )
        self.ball_img_coords = ball_img_coords
        if get_ball_subimg:  # TODO: make function
            if np.isnan(self.ball_pos[0]):
                self.ball_subimg = np.zeros((64, 64, 3), dtype=np.uint8)
            else:  # TODO clean up and optimize
                points_board = np.zeros((64 * 64, 4))
                points_board[:, -1] = 1.0
                points_board[:, :2] = (
                    1.0e-3 * np.mgrid[-32:32, -32:32][::-1].reshape(2, -1).transpose()
                )
                points_board[:, 1] *= -1
                points_board[:, :3] += self.ball_pos
                points_cam = (self.plate_pose.T__C_M @ points_board.T).T[:, :3]
                points_cam[:, :2] = points_cam[:, [1, 0]]
                points_cam[:, 2] *= -1
                points_cam = self.plate_pose.o.world2cam(points_cam)
                points_cam = points_cam.reshape(64, 64, 2).astype(np.float32)
                self.ball_subimg = cv2.remap(
                    frame_copy, points_cam[..., 1], points_cam[..., 0], cv2.INTER_LINEAR
                )

        if self.anim_3d_top is not None:
            self.update_3d_anim_top()
        if self.anim_3d_side is not None:
            self.update_3d_anim_side()

    def get_ball_subimg(self):
        return self.ball_subimg

    def get_ball_coordinates(self):
        """
        Return the pixel coordinates of the ball in the image frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (2,)
                    2d position of the ball in the image frame.

        """
        return self.ball_img_coords

    def get_ball_position_in_maze(self):
        """
        Return the position of the ball in the maze frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (3,)
                    3d position of the ball in the maze frame {m}.
                    note: the z-coordinate of the ball in maze frame is fixed and known
                    by assumption of constant contact with the maze: z__m_b = ball_radius.

        """
        return self.ball_pos

    def get_plate_pose(self):
        """
        Return the angles (Euler YXZ) that describe the orientation of the maze frame {m} wrt the world frame {w}.

        Returns:
            ball_pos: Tuple(float, float)
                      (alpha, beta) around X and Y respectively
        """
        return self.plate_angles

    def camera_localization(self, frame):
        """
        Compute the pose of the camera {c} wrt to the world frame {w} : T__W_C.
        """
        fix_pts = self.detector_fixed_points.detect_corners(frame)
        self.detector.fixed_corners = fix_pts
        self.plate_pose.camera_localization(fix_pts)
        self.create_mask(frame)

    def create_mask(self, frame):
        h, w = frame.shape[:2]
        coords = np.mgrid[0:h, 0:w].transpose(1, 2, 0).reshape(-1, 2)
        camera_points = self.plate_pose.o.cam2world(coords)[:, [1, 0, 2]]
        camera_points[:, 2] *= -1
        world_vec = (self.plate_pose.T__W_C[:3, :3] @ camera_points.T).T
        world_vec = world_vec / world_vec[:, 2:]
        world_points = (
            world_vec * (-self.plate_pose.T__W_C[2, -1])
            + self.plate_pose.T__W_C[:3, -1]
        )
        mask = (
            (world_points[:, 0] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 0]
                <= self.plate_pose.L_EXT_INT_X + 2.0 * self.plate_pose.r
            )
            & (world_points[:, 1] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 1]
                <= self.plate_pose.L_EXT_INT_Y + 2.0 * self.plate_pose.r
            )
        )
        self.mask = 255 * mask.reshape(h, w, 1).astype(np.uint8)

    def ball_pos_backproject(self, ball_undist, K, T__C_M):
        """
        Compute the 3d position of the ball in the maze frame {m}.
        Returns:
            x_M: np.ndarray, dim: (3,)
                 3d position of the ball in the maze frame {m}.
                 note: the z-coordinate of the ball in maze frame is fixed and known
                 by assumption of constant contact with the maze: z__m_b = ball_radius.
        """
        if np.any(ball_undist == np.nan):
            return np.array([np.nan, np.nan, np.nan])
        d = PlatePoseEstimator.R_BALL
        v, u = ball_undist
        H = K @ T__C_M[:3, :]
        h_11, h_12, h_13, h_14 = H[0, :]
        h_21, h_22, h_23, h_24 = H[1, :]
        h_31, h_32, h_33, h_34 = H[2, :]
        A = np.array(
            [[u * h_31 - h_11, u * h_32 - h_12], [v * h_31 - h_21, v * h_32 - h_22]]
        )
        b = np.array(
            [
                d * h_13 + h_14 - d * u * h_33 - u * h_34,
                d * h_23 + h_24 - d * v * h_33 - v * h_34,
            ]
        )
        x = np.linalg.solve(A, b)
        x_M = np.array([x[0], x[1], PlatePoseEstimator.R_BALL])
        return x_M

    def update_3d_anim_top(self):
        self.anim_3d_top.B__W = (
            self.plate_pose.T__W_M @ np.hstack((self.ball_pos, np.array([1])))
        )[:-1]
        self.anim_3d_top.maze_corners__W = self.plate_pose.estimate_maze_corners__W()
        self.anim_3d_top.update_anim()

    def update_3d_anim_side(self):
        self.anim_3d_side.B__W = (
            self.plate_pose.T__W_M @ np.hstack((self.ball_pos, np.array([1])))
        )[:-1]
        self.anim_3d_side.maze_corners__W = self.plate_pose.estimate_maze_corners__W()
        self.anim_3d_side.update_anim()
