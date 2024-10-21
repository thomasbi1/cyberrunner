import numpy as np
import time
import cv2
import os
from cyberrunner_state_estimation.core.measurements import Measurements
from cyberrunner_state_estimation.core.estimator import KF, KFBias, FiniteDiff
from ament_index_python.packages import get_package_share_directory


class EstimationPipeline:
    def __init__(
        self,
        fps,
        estimator="KF",
        FiniteDiff_mean_steps=0,
        print_measurements: bool = False,
        show_image: bool = False,
        do_anim_3d=False,  # CHANGE HERE TO TURN ON/OFF THE 3D PLOT
        viewpoint="side",
        show_subimages_detector=False,
    ):

        share = get_package_share_directory("cyberrunner_state_estimation")
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")
        self.measurements = Measurements(
            markers=markers,
            do_anim_3d=do_anim_3d,
            viewpoint=viewpoint,
            show_subimages_detector=show_subimages_detector,
        )
        if estimator == "FiniteDiff":
            self.estimator = FiniteDiff(fps, FiniteDiff_mean_steps)
        if estimator == "KF":
            self.estimator = KF(fps)
        if estimator == "KFBias":
            self.estimator = KFBias(fps)

        self.print_measurements = print_measurements
        self.show_image = show_image

    def estimate(self, frame, return_ball_subimg=False):
        """
        Compute the measurements and estimate the state.

        Args:
            frame: np.ndarray, dim: (400,640)
            return_ball_subimg: bool
        Returns:
            x_hat: np.ndarray, dim: (n_states,)
            P: np.ndarray dim: (n_states, n_states)
                covariance matrix
            inputs: np.ndarray dim: (2,)
                [alpha, beta]
            ball_subimg: optional, np.ndarray, dim: (64, 64, 3)
        """
        t0 = time.time()
        self.measurements.process_frame(frame, return_ball_subimg)
        xb, yb, _ = self.measurements.get_ball_position_in_maze()
        if return_ball_subimg:
            ball_subimg = self.measurements.get_ball_subimg()
        inputs = self.measurements.get_plate_pose()  # alpha, beta
        tmeas = time.time() - t0
        t0 = time.time()
        x_hat, P = self.estimator.estimate(
            inputs=inputs, measurement=np.array([xb, yb])
        )

        if type(self.estimator).__name__ == "KFBias":
            alpha_est = inputs[0] + x_hat[4]
            beta_est = inputs[1] + x_hat[5]
        else:  # type(self.estimator).__name__ == "KF":
            alpha_est = inputs[0]
            beta_est = inputs[1]
        alpha_est *= 180 / np.pi
        beta_est *= 180 / np.pi
        testimator = time.time() - t0
        # np.set_printoptions(precision=3)
        np.set_printoptions(formatter={"float": "{: 0.3f}".format}, precision=3)

        if self.print_measurements:
            print(
                f"ball: ({xb:6.3f}, {yb:>6.3f}) | (a, b): ({inputs[0]*180/np.pi:>5.2f}, {inputs[1]*180/np.pi:>5.2f}) [deg] | tmeas:{1000*tmeas:5.2f} [ms] | x_hat:{x_hat} | ab_est:({alpha_est:5.2f}, {beta_est:5.2f}) [deg]"
            )
        if self.show_image:
            self.measurements.detector.draw_corners(frame)
            self.measurements.detector.draw_ball(frame)
            cv2.imshow("ori", frame)
            cv2.waitKey(1)
        if return_ball_subimg:
            return x_hat, P, inputs, ball_subimg, xb, yb
        else:
            return x_hat, P, inputs, xb, yb
