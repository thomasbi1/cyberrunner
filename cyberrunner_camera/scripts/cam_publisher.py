#!/usr/bin/env python3
import time
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CamPublisher(Node):
    def __init__(self, device):
        super().__init__("cyberrunner_camera")
        self.publisher = self.create_publisher(Image, "cyberrunner_camera/image", 1)
        self.br = CvBridge()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("U", "Y", "V", "Y"))
        # self.cap.set(cv2.CAP_PROP_MODE, cv2.CAP_FFMPEG)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # 60
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1280
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        for _ in range(3):
            self.cap.read()

        previous = time.time()
        count = 0
        while True:
            ret, frame = self.cap.read()
            t0 = time.time()
            if not ret:
                print("Failed to obtain camera image")
                exit()

            frame = cv2.resize(frame, (640, 360))
            frame = cv2.copyMakeBorder(frame, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)
            now = time.time()
            dur = now - previous
            if dur > 0.02:
                print("publish image: {}".format(now - previous))
            previous = now
            msg = self.br.cv2_to_imgmsg(frame)
            self.publisher.publish(msg)

            debug = True
            if debug:

                cv2.imshow("img", frame)  # cv2.resize(frame, (160, 100)))
                cv2.waitKey(1)
            t1 = time.time()
            if dur > 0.02:
                print("Processing: {}".format(t1 - t0))

            count += 1

    def cap(self):
        pass


def main(args=None):
    args = sys.argv[1:]
    device = "/dev/video0" if not args else args[0]
    print("Using device: {}. To use a different device use the command-line argument, e.g.,\n"
          "ros2 run cyberrunner_camera cam_publisher.py /dev/video*".format(device))
    rclpy.init(args=args)
    vid = CamPublisher(device)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
