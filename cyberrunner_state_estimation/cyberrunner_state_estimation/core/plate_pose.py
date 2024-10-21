import os
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from cyberrunner_state_estimation.utils.ocam_model import OcamModel

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
c_name = ["blue", "green", "red", "yellow"]


class PlatePoseEstimator:

    # constants
    L_EXT_INT_X = 0.317
    L_EXT_INT_Y = 0.2715
    C2C_X = 0.2835  # distance btw circles centers along x direction
    C2C_Y = 0.2385  # distance btw circles centers along y direction
    H_BORDERS = 0.022  # height of the maze borders default = 0.22

    r = 0.008 / 2
    R_BALL = 0.012 / 2
    edge_width = 7.5e-3  # to check

    MODEL_POINTS_FIXED_CORNERS = np.array(
        [
            (-r, 0.05, 0),  # Corner 1
            (L_EXT_INT_X + r, 0.05, 0),  # Corner 2
            (L_EXT_INT_X + r, 0.222, 0),  # Corner 3
            (-r, 0.222, 0),  # Corner 4
        ],
        dtype=np.float32,
    )

    # 3D model points of corners (center of circles) in maze frame {M}
    MODEL_POINTS_CORNERS = np.array(
        [
            [-C2C_X / 2, -C2C_Y / 2, H_BORDERS],  # Corner 1 (dl)
            [+C2C_X / 2, -C2C_Y / 2, H_BORDERS],  # Corner 2 (dr)
            [+C2C_X / 2, +C2C_Y / 2, H_BORDERS],  # Corner 3 (ur)
            [-C2C_X / 2, +C2C_Y / 2, H_BORDERS],  # Corner 4 (ul)
        ],
        dtype=np.float32,
    )

    MAZE_CORNERS__M = np.array(
        [  # only use for 3d ploting
            [-C2C_X / 2 + edge_width / 2, -C2C_Y / 2 + edge_width / 2, 0],
            [+C2C_X / 2 - edge_width / 2, -C2C_Y / 2 + edge_width / 2, 0],
            [+C2C_X / 2 - edge_width / 2, +C2C_Y / 2 - edge_width / 2, 0],
            [-C2C_X / 2 + edge_width / 2, +C2C_Y / 2 - edge_width / 2, 0],
        ]
    )

    def __init__(self, print_details: bool = False):
        self.print_details = print_details

        share = get_package_share_directory("cyberrunner_state_estimation")
        o = OcamModel(os.path.join(share, "calib_results_cyberrunner.txt"))
        o.scale(3)  # From 1920 to 640 res
        self.o = o
        xc, yc = o.xc, o.yc
        self.f = 300
        self.K_ocam = np.array([[-self.f, 0, xc], [0, -self.f, yc], [0, 0, 1]])
        self.K = np.array([[+self.f, 0, yc], [0, +self.f, xc], [0, 0, 1]])

        self.T__W_M = None
        self.T__W_C = None
        self.img_points_corners_undist = None
        self.img_points_fixed_corners_undist = None

    def estimate_anglesXY(self, corners_undist):  # (x,y)
        """
        Compute the angles (Euler XYZ) that describe the orientation of the maze frame {m} wrt to the world frame {w}.

        Args :
            corners_undist: np.ndarray, dim: (4,2)
                            undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.
        Returns :
            alpha: float
                    angle around +X axis
            beta: float
                    angle around +Y axis
        """
        self.img_points_corners_undist = corners_undist
        self.T__W_M = self.get_maze_pose_in_world(self.img_points_corners_undist)
        alpha, beta = self.getXYAnglesFrom_R__W_M(self.T__W_M[:3, :3], deg=False)
        return alpha, beta

    def get_pose_T__C_P(
        self, model_points: np.ndarray, img_points: np.ndarray, print_=False
    ):
        """
        Compute the pose of the frame {p} in which model points are expressed wrt to the camera frame {c}.

        Args :
            model_points: np.ndarray, dim: (4,3)
                           3d coordinates of the points in their frame {p}.
            img_points: np.ndarray, dim: (4,2)
                        undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.

        Returns :
            T__C_P: np.ndarray, dim: (4,4)
                  pose in SE(3) of the frame {p} in which model points are expressed wrt to the camera frame {c}.
            R__C_P: np.ndarray, dim: (3,3)
                  rotation matrix of T__C_P.
            P__C: np.ndarray, dim: (3,)
                  translation vector of T__C_P.

        """
        img_points = np.flip(
            img_points, axis=1
        )  # conversion to opencv convention: (u,v) = (column, line)
        _, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, img_points, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        if self.print_details:
            print("rot vec [deg]:")
            print(180 / np.pi * rotation_vec)
            print("tr vec:")
            print(translation_vec)

        R__C_P = rotation_mat
        P__C = translation_vec
        T__C_P = np.hstack((R__C_P, P__C))  # $$$
        T__C_P = np.vstack((T__C_P, np.array([0, 0, 0, 1])))
        return T__C_P, R__C_P, P__C

    def get_maze_pose_in_world(self, image_points):
        """
        Compute the pose of the maze {m} wrt to the world frame {w}.

        Args :
            image_points: np.ndarray, dim: (4,2)
                           undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.

        Returns :
           T_inv: np.ndarray, dim: (4,4)
                  inverse of the matrix T in SE(3)
        """
        T__C_M, _, _ = self.get_pose_T__C_P(
            PlatePoseEstimator.MODEL_POINTS_CORNERS, image_points
        )
        self.T__C_M = T__C_M
        T__W_M = self.T__W_C @ T__C_M
        return T__W_M

    def invert_pose(self, T):
        """
        Compute the inverse of the matrix T [4x4] in SE(3).

        Args :
            T: np.ndarray, dim: (4,4)
               pose matrix in SE(3).
        Returns :
           T_inv: np.ndarray, dim: (4,4)
                  inverse of the matrix T in SE(3).
        """
        R = T[:3, :3]
        t = np.expand_dims(T[:3, -1], axis=1)
        T_inv = np.hstack((R.T, -R.T @ t))
        T_inv = np.vstack((T_inv, np.array([0, 0, 0, 1])))
        return T_inv

    def getXYAnglesFrom_R__W_M(self, R, deg=False):
        """
        Compute the angles (Euler YXZ) that describe the orientation of the given rotation matrix R__W_M.

        Args :
            R: np.ndarray, dim: (3,3)
               rotation matrix that describe the orientation of the maze {m} wrt the world frame {w}.
        Returns :
            alpha: float
                    angle around +X axis
            beta: float
                    angle around +Y axis
        """
        beta = np.arctan(R[0, 2] / R[2, 2])  # around +y
        alpha = np.arcsin(-R[1, 2])  # around +x
        if deg:
            alpha = alpha * 180 / np.pi
            beta = beta * 180 / np.pi
        return alpha, beta

    def undistort_points(self, img_points_raw: np.ndarray):  # (x,y)
        """
        Undistort the points using the camera calibration data via cam2world.

        Args :
            img_points_raw:    np.ndarray, dim: (N,2)
                               image coordinates of the raw points in (x,y) = (line, column) convention.
        Returns :
            img_point_undist:  np.ndarray, dim: (N,2)
                               image coordinates of the undistorted points in (x,y) = (line, column) convention.

        """
        P_w = self.o.cam2world(img_points_raw).T  # dim: (3, N)
        pt_undist = self.K_ocam @ P_w  # dim: (3, N)
        pt_undist = pt_undist / pt_undist[2, :]
        img_point_undist = pt_undist.T[:, :2]
        return img_point_undist

    def estimate_maze_corners__W(self):  # used only for plotting
        """
        Estimate the coordinates of the corners of the maze (not the detection dots but the real corners of maze,
        i.e. the limits maze) in the world frame {w}. Useful for plotting only.

        Returns :
            Ps__W:  np.ndarray, dim: (4,3)
                    coordinates of the corners of the maze in the world frame {w}.

        """
        Ps__M = PlatePoseEstimator.MAZE_CORNERS__M.T
        Ps__W = (self.T__W_M @ np.vstack((Ps__M, np.ones(4))))[:-1, :]
        Ps__W = Ps__W.T
        return Ps__W

    def camera_localization(self, img_fix_pts):
        """
        Estimate the pose of camera {c} wrt the world frame {w}: T__W_C (also noted T^W_C).

        Args:
           img_fix_pts: np.ndarray, dim: (4,2)
                        the raw image coordinates of the four fixed reference dots of the external frame of the labyrinth
                        in (x,y) = (line, column) convention.

        """
        self.img_points_fixed_corners_undist = self.undistort_points(
            img_fix_pts
        )  # (x,y)
        print("camera localization: getting T__C_W")
        T__C_W, _, _ = self.get_pose_T__C_P(
            PlatePoseEstimator.MODEL_POINTS_FIXED_CORNERS,
            self.img_points_fixed_corners_undist,
        )
        self.T__C_W = T__C_W
        self.T__W_C = self.invert_pose(T__C_W)
        np.set_printoptions(precision=3)
        print("T^W_C:")
        print(self.T__W_C)
        print(("\n"))
