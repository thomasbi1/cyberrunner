import numpy as np
import cv2
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory


class MarkerSelector(Node):
    def __init__(self):
        super().__init__("marker_selector")
        self.sub = self.create_subscription(
            Image, "cyberrunner_camera/image", self.select_marker, 1
        )
        self.br = CvBridge()
        cv2.namedWindow("Select markers")
        cv2.setMouseCallback("Select markers", self.draw_x)

        self.markers = []
        self.clicked = True
        self.number_str = ["lower left", "lower right", "upper right", "upper left"]
        self.marker_str = ["outer", "inner"]

        self.get_logger().info(
            "Make sure the labyrinth playing board is approximately even (Inclination angles close to 0)"
        )

    def select_marker(self, data):
        frame = self.br.imgmsg_to_cv2(data)
        for i, c in enumerate(self.markers):
            cv2.drawMarker(frame, c, (255, 255, 255), cv2.MARKER_CROSS, 15, 3)
            cv2.putText(
                frame,
                str(i + 1),
                (c[0] + 10, c[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
            )
        if self.clicked and len(self.markers) != 8:
            self.get_logger().info(
                "Please select the {} marker of the {} markers".format(
                    self.number_str[len(self.markers) % 4],
                    self.marker_str[len(self.markers) // 4],
                )
            )
            self.clicked = False
        if len(self.markers) == 8:
            shared = get_package_share_directory("cyberrunner_state_estimation")
            np.savetxt(
                os.path.join(shared, "markers.csv"),
                np.asarray(self.markers),
                delimiter=",",
            )
            self.get_logger().info(
                "Successfully selected and saved all markers. Click any key to exit..."
            )
            cv2.imshow("Select markers", frame)
            cv2.waitKey(0)
            exit()

        cv2.imshow("Select markers", frame)
        cv2.waitKey(1)

    def draw_x(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.markers.append([x, y])
            self.clicked = True


def main(args=None):
    rclpy.init(args=args)
    ms = MarkerSelector()
    rclpy.spin(ms)
    rclpy.shutdown()
