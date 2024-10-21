#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("cyberrunner_camera")
        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)
        self.rate = self.create_rate(55.0)
        self.br = CvBridge()
        img = cv2.imread("mask_cr1.png")
        cap = cv2.VideoCapture("brio_eval_corl.mp4")
        for _ in range(15):
            ret, frame = cap.read()
        while True:
            #ret, frame = cap.read()
            frame = img
            msg = self.br.cv2_to_imgmsg(frame)
            self.publisher.publish(msg)
            time.sleep(0.01)
            # print("hi")
            cv2.imshow("a", frame)
            cv2.waitKey(1)
            # self.rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    vid = VideoPublisher()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
