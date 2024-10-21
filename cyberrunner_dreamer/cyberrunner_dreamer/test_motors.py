import sys
from datetime import datetime
import rclpy
import gym
import numpy as np
from datetime import datetime

# import cyberrunner_dreamer


def main(args=None):
    rclpy.init(args=args)
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    # date_str = '2023-08-23:04-17-46'
    argv = [
        "--configs",
        "cyberrunner",
        "small",  # TODO add config file here!
        "--logdir",
        "~/cyberrunner_logs/" + date_str,
        "--run.script",
        "parallel",
        "--run.train_ratio",
        "-1",
    ]
    # train(argv)
    import time

    env = gym.make("cyberrunner-ros-v0")
    obs = env.reset()
    prev_pos = obs["states"][0] * 10.0 * 3.14159 / 180.0
    last = time.time()
    N = 25
    avg = np.zeros((N,))
    for i in range(N):
        obs, reward, done, info = env.step(np.array([1.0, 0.0]))
        pos = obs["states"][0] * 10.0 * 3.14159 / 180.0
        now = time.time()
        print("FPS: {}".format(1.0 / (now - last)))
        degs = (pos - prev_pos) * 55.0 * 180.0 / 3.14159
        avg[i] = degs
        print("Degrees per second: {}".format(degs))
        prev_pos = pos

        last = now

    env.step(np.array([0.0, 0.0]))
    print("Mean: {}".format(avg.mean()))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
