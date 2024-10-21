import sys
from datetime import datetime
import rclpy
from dreamerv3.train import main as train
from datetime import datetime


def main(args=None):
    rclpy.init(args=args)
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    date_str = now.strftime("%Y%m%d-%H%M%S")
    logdir = "robust_2"  #'meetyourlab' # 'cyberrunner_clean_1100k' #'wef'#'cyberrunner_clean_10k' #  'meetyourlab'
    checkpoint = "~/cyberrunner_logs/{}/checkpoint.ckpt".format(logdir)
    # date_str = '2023-08-23:04-17-46'
    argv = [
        "--configs",
        "cyberrunner",
        "large",  # TODO add config file here!
        "--task",
        "gym_cyberrunner_dreamer:cyberrunner-ros-v0",
        "--logdir",
        "~/cyberrunner_logs/eval_" + date_str,
        "--run.from_checkpoint",
        checkpoint,
        "--run.steps",
        "10000",
        "--run.script",
        "eval_only",
        "--jax.policy_devices",
        "0",
        "--jax.train_devices",
        "0",
    ]
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
