#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cyberrunner_state_estimation.core.estimation_pipeline import EstimationPipeline
from cyberrunner_interfaces.msg import StateEstimateSub


class ImageSubscriber(Node):
    def __init__(self, skip=1):
        super().__init__("cyberrunner_state_estimation")
        self.subscription = self.create_subscription(
            Image, "cyberrunner_camera/image", self.listener_callback, 10
        )
        self.publisher_ = self.create_publisher(
            StateEstimateSub, "cyberrunner_state_estimation/estimate_subimg", 10
        )
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Image subscriber has been initialized.")
        self.br = CvBridge()
        self.estimation_pipeline = EstimationPipeline(
            fps=55.0 / 1.0,  # $$
            estimator="KF",  #  "FiniteDiff",  "KF", "KFBias"
            print_measurements=True,
            show_image=False,
            do_anim_3d=False,
            viewpoint="top",  # 'top', 'side', 'topandside'
            show_subimages_detector=False,
        )

        self.skip = skip
        self.count = 0
        # self.prev_a = self.prev_b = 0.0
        self.a = np.zeros(15, dtype=float)
        self.b = np.zeros(15, dtype=float)

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
        x_hat, P, angles, subimg, xb, yb = self.estimation_pipeline.estimate(
            frame, return_ball_subimg=True
        )

        if self.count % self.skip == 0:
            msg = StateEstimateSub()
            msg.state.x_b = xb
            msg.state.y_b = yb
            msg.state.x_b_dot = x_hat[2]
            msg.state.y_b_dot = x_hat[3]
            msg.state.alpha = -angles[1]
            msg.state.beta = angles[0]
            msg.subimg = self.br.cv2_to_imgmsg(subimg)
            self.publisher_.publish(msg)
            # self.get_logger().info(f"Publishing: {x_hat}")

        # Broadcast transforms
        if self.count == 0:
            t = self.get_tf_msg(
                self.estimation_pipeline.measurements.plate_pose.T__W_C,
                'camera',
                'world',
            )
            self.tf_static_broadcaster.sendTransform(t)
        t_maze = self.get_tf_msg(
            self.estimation_pipeline.measurements.plate_pose.T__W_M,
            'maze',
            'world',
        )
        T__B_M = np.eye(4)
        T__B_M[:3, -1] = self.estimation_pipeline.measurements.get_ball_position_in_maze()
        t_ball = self.get_tf_msg(
            T__B_M,
            'maze',
            'ball'
        )
        self.tf_broadcaster.sendTransform([t_maze, t_ball])

        # self.a[:-1] = self.a[1:]
        # self.a[-1] = msg.state.alpha
        # self.b[:-1] = self.b[1:]
        # self.b[-1] = msg.state.beta
        # print("a_dot: {:.4f}, b_dot: {:.4f}".format((self.a[-1] - self.a[0]) * 55.0 / 14.0, (self.b[-1] - self.b[0]) * 55.0 / 14.0))
        # #self.prev_a = msg.state.alpha
        # self.prev_b = msg.state.beta
        # cv2.imshow("sub", subimg)
        # cv2.waitKey(1)
        self.count += 1

    def get_tf_msg(self, se3, frame_id, child_frame_id):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = se3[0, 3]
        t.transform.translation.y = se3[1, 3]
        t.transform.translation.z = se3[2, 3]
        q = Rotation.from_matrix(se3[:3, :3]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        return t


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
