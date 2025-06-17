import sys

from cyberrunner_dreamer import cyberrunner_layout
from cyberrunner_dreamer.path import LinearPath

from cyberrunner_interfaces.msg import DynamixelVel, StateEstimateSub
from cyberrunner_interfaces.srv import DynamixelReset

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import gym
import numpy as np
import time
import cv2
import os
from ament_index_python.packages import get_package_share_directory


class CyberrunnerGym(gym.Env):
    def __init__(
        self,
        repeat=1,
        layout=cyberrunner_layout.cyberrunner_hard_layout,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=0.0,
        reward_on_goal=0.5,
    ):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()

        self.future = None

        self.cheat = False
        self.observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
            states=gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            goal=gym.spaces.Box(-np.inf, np.inf, (num_rel_path * 2,), np.float32),
            progress=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            log_reward=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
        self.obs = dict(self.observation_space.sample())
        self.num_rel_path = num_rel_path
        # self.norm_min = np.array(
        #     [0.0, 0.0, -10 * np.pi / 180.0, -10 * np.pi / 180.0] +
        #     [-0.01 * (k + 1) for k in range(self.num_rel_path) for _ in
        #      range(2)])
        self.norm_max = np.array([10 * np.pi / 180.0, 10 * np.pi / 180.0, 0.276, 0.231])
        self.goal_norm_max = np.array(
            [0.0002 * 60 * k for k in range(1, self.num_rel_path + 1) for _ in range(2)]
        )

        self.node = Node("cyberrunner_gym")
        self.publisher = self.node.create_publisher(
            DynamixelVel,
            "cyberrunner_dynamixel/cmd",
            1,
        )
        self.subscription = self.node.create_subscription(
            StateEstimateSub,
            "cyberrunner_state_estimation/estimate_subimg",
            self._msg_to_obs,
            1,
        )
        self.client = self.node.create_client(
            DynamixelReset, "cyberrunner_dynamixel/reset"
        )

        self.br = CvBridge()

        self.repeat = repeat

        self.offset = np.array([0.276, 0.231]) / 2.0
        shared = get_package_share_directory("cyberrunner_dreamer")
        self.p = LinearPath.load(os.path.join(shared, "path_0002_hard.pkl"))
        # if not self.cheat:
        #     self.p = LinearPath.load("path_0002_hard.pkl")
        # else:
        #     self.p = LinearPath(np.array(layout["waypoints"]))
        # self.p = LinearPath(
        #    np.array(layout["waypoints"]),
        #     walls_h=np.array(layout["walls_h"]),
        #     walls_v=np.array(layout["walls_v"]),
        #     holes=np.array(layout["holes"]),
        # )
        self.prev_pos_path = 0
        self.num_wait_steps = num_wait_steps
        self.reward_on_fail = reward_on_fail
        self.reward_on_goal = reward_on_goal

        self.ball_detected = False

        # Dynamixel
        self.max_angle_vel = 200  # 40 * (np.pi / 180.0)  # 20
        self.alpha_fac = -1.0  # -6.0 / 0.051
        self.beta_fac = -1.0  # -12 / 0.088

        self.last_time = 0
        self.progress = 0
        self.accum_reward = 0.0
        self.steps = 0
        self.off_path = False

        self.episodes = 0

        self.new_obs = False

    def step(self, action):

        self.steps += 1

        # Send action to dynamixel
        if self.cheat and (self.p.num_points - self.prev_pos_path <= 200):
            action[1] = 1.0
        self._send_action(action)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward = self._get_reward(obs)

        # Get done
        done = self._get_done(obs)
        if done and (not self.success):
            reward = self.reward_on_fail
        if self.success:
            reward += self.reward_on_goal

        if done or self.steps == 3000:
            if self.success:
                # reward += 0.5
                time.sleep(2)
            # action = np.zeros((2,))
            # self._send_action(action)
            print("Reset board")
            self._reset_board()

        if self.success:
            info = {"is_terminal": False}
        else:
            info = {}

        now = time.time()
        if (now - self.last_time) > (1.0 / 35.0):
            print("Slower than 35fps")
        # print("[Step]: {:.4f}s".format(now - self.last_time))
        self.last_time = now
        # print(self.prev_pos_path)
        self.accum_reward += reward if not done else 0
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([reward if not done else 0], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        print("Resetting ...")
        self.episodes += 1
        print("Previous reward: {}".format(self.accum_reward))
        print("Previous episode length: {}".format(self.steps / 60.0))
        print("Episodes: {}".format(self.episodes))
        self.accum_reward = 0.0
        self.steps = 0
        self.success = False
        self.ball_detected = False

        # Wait for board reset to be done
        if self.future is not None:
            rclpy.spin_until_future_complete(self.node, self.future, timeout_sec=10)

        # Set velocities to 0  TODO: use action/service to set board to 0 state
        action = np.zeros((2,))
        self._send_action(action)

        kb_reset = False
        if kb_reset:
            # Wait for keyboard press
            cv2.imshow("reset", np.zeros((200, 200, 3)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Done resetting ...")

            # Get observation
            for _ in range(2):
                obs = self._get_obs()
        else:
            count = 0
            obs = self._get_obs()
            while count < self.num_wait_steps:
                obs = self._get_obs()
                count = count + 1 if self.ball_detected else 0

        self.prev_pos_path = self.p.closest_point(obs["states"][2:4])[0]
        self.progress = 0
        self.last_time = time.time()
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([0], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        pass

    def _send_action(self, action):
        # Scale action
        action = action.copy()
        action *= self.max_angle_vel
        vel_1 = self.alpha_fac * action[0]  # TODO define these as parameters
        vel_2 = self.beta_fac * action[1]

        # To message and publish
        msg = DynamixelVel()
        msg.vel_1 = vel_1
        msg.vel_2 = vel_2
        self.publisher.publish(msg)

    def _get_obs(self):
        # Spin repeat times to get the next observation
        while not self.new_obs:
            for _ in range(self.repeat):
                rclpy.spin_once(self.node)
        self.new_obs = False

        return self.obs.copy()

    def _get_reward(self, obs):
        # If no ball is detected, reward is 0
        if not self.ball_detected:
            reward = 0.0
        else:
            curr_pos_path, p = self.p.closest_point(obs["states"][2:4])
            self.off_path = curr_pos_path == -1
            if self.off_path and self.cheat:
                curr_pos_path = self.prev_pos_path
            self.progress = curr_pos_path - self.prev_pos_path
            reward = float(curr_pos_path - self.prev_pos_path) * 0.004 / 16.0
            # reward = np.clip(reward, -5.0, 5.0)
            self.prev_pos_path = curr_pos_path
            # self.curr_dist = np.linalg.norm(p - obs["states"][:2])
        return reward

    def _get_done(self, obs):
        # Done if ball is not detected
        done = not self.ball_detected

        # Done if off path and not cheating
        if (not self.cheat) and self.off_path:
            done = True
            print("[Done]: OFFPATH")

        # Done if reached goal
        if self.p.num_points - self.prev_pos_path <= 1:
            self.success = True
            done = True
            print("[Done]: SUCCESS")
            # self._send_action(np.array([0.0, 0.75]))
            # time.sleep(0.05)
            self._send_action(np.array([0.0, 0.0]))

        if (not self.cheat) and self.progress > 300:
            done = True
            print("[Done]: Too high progress")

        # if self.curr_dist > 0.02:
        #     done = True

        return done

    def _msg_to_obs(self, msg):
        # ROS message to gym observation
        if np.isnan(msg.state.x_b):
            self.ball_detected = False
        else:
            self.ball_detected = True
            states = np.array(
                [msg.state.alpha, msg.state.beta, msg.state.x_b, msg.state.y_b]
            )
            states[2:] += self.offset
            rel_path = self.p.get_rel_path(states[2:], self.num_rel_path, 60).flat
            # states = np.concatenate((states, rel_path), axis=0)
            img = self.br.imgmsg_to_cv2(msg.subimg)
            self.obs = {"states": states, "goal": rel_path, "image": img}

        self.new_obs = True

    def _normalize_states(self, states):
        return states / self.norm_max

    def _reset_board(self):
        # Reset torque
        req = DynamixelReset.Request()
        self.future = self.client.call_async(req)
