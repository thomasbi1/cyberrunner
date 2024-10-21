#!/usr/bin/env python3

import pygame
import numpy as np
import rclpy
from rclpy.node import Node
from cyberrunner_interfaces.msg import DynamixelInc


# define a main function
def main(args=None):

    rclpy.init(args=args)

    # ROS Node/Pub
    node = Node("pantilt")
    pub = node.create_publisher(DynamixelInc, "cyberrunner_dynamixel/cmd", 10)
    msg = DynamixelInc()

    # prototype values
    center_pos = np.array([0, 0.0])
    fac = np.array([450.0, -450.0])

    # initialize the pygame module
    pygame.init()
    # load and set the logo
    pygame.display.set_caption("CYBERRUNNER control")

    # create a surface on screen that has the size of 640 x 480
    window_size = (2 * 640, 2 * 480)
    screen = pygame.display.set_mode(window_size)

    # clock
    clock = pygame.time.Clock()

    # main loop
    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            if e == pygame.QUIT:
                running = False

        norm_const = np.array(window_size) / 2.0
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float)
        mouse_pos -= norm_const
        mouse_pos /= norm_const
        print(mouse_pos)

        goal_pos = center_pos + mouse_pos * fac
        msg.inc_1 = int(goal_pos[0])
        msg.inc_2 = int(goal_pos[1])
        pub.publish(msg)


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()
