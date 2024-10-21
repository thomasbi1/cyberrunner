#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from cyberrunner_state_estimation.core.estimation_pipeline import EstimationPipeline

from cyberrunner_interfaces.msg import StateEstimate


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("cyberrunner_state_estimation")
        self.subscription = self.create_subscription(
            Image, "cyberrunner_camera/image", self.listener_callback, 10
        )
        self.subscription
        self.publisher_ = self.create_publisher(StateEstimate, "stateEstimation", 10)

        self.get_logger().info("Image subscriber has been initialized.")
        self.br = CvBridge()
        self.estimation_pipeline = EstimationPipeline(
            fps=55.0 / 1.0,  # $$
            estimator="FiniteDiff",  #  "FiniteDiff",  "KF", "KFBias"
            FiniteDiff_mean_steps=4,
            print_measurements=True,
            show_image=False,
            do_anim_3d=False,
            viewpoint="top",  # 'top', 'side', 'topandside'
            show_subimages_detector=False,
        )

    def listener_callback(self, data):
        # self.get_logger().info('Receiving image frame')
        frame = self.br.imgmsg_to_cv2(data)

        # cv2.imshow("before", frame)
        b, g, r = np.mean(np.mean(frame, axis=0), axis=0)
        # print(b,g,r)
        if g > 100 and b < 40 and r < 40:
            print("SKIP THIS FRAME")
            # cv2.waitKey(1)
            return
        # cv2.imshow("before", frame)
        x_hat, P, angles = self.estimation_pipeline.estimate(frame)

        msg = StateEstimate()
        msg.x_b = x_hat[0]
        msg.y_b = x_hat[1]
        msg.x_b_dot = x_hat[2]
        msg.y_b_dot = x_hat[3]
        msg.alpha = -angles[1]
        msg.beta = angles[0]
        self.publisher_.publish(msg)
        # self.get_logger().info(f"Publishing: {x_hat}")

        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
