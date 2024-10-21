# CyberRunner

[![CyberRunner](https://i.imgur.com/voNwSro.gif)](https://youtu.be/zQMKfuWZRdA)

CyberRunner is an AI robot whose task is to learn how to play the popular and widely accessible labyrinth marble game. It is able to beat the best human player with only 6 hours of practice.

This repository contains all necessary code and documentation to build your own CyberRunner robot, and let the robot learn to solve the maze!

**Author: Thomas Bi <br />
With contributions by: Ethan Marot, Tim Flückiger, Cara Koepele, Aswin Ramachandran**

To learn more:
* [Project website](https://cyberrunner.ai)
* [Video](https://youtu.be/zQMKfuWZRdA)
* [Paper](https://arxiv.org/abs/2312.09906)

## Overview

![Method](https://i.imgur.com/a9JeV7V.png)

CyberRunner exploits recent advances in model-based reinforcement learning and its ability to make informed decisions about potentially successful behaviors by planning into the future. The robot learns by collecting experience. While playing the game, it captures observations and receives rewards based on its performance, all through the “eyes” of a camera looking down at the labyrinth. A memory is kept of the collected experience. Using this memory, the model-based reinforcement learning algorithm learns how the system behaves, and based on its understanding of the game it recognizes which strategies and behaviors are more promising. Consequently, the way the robot uses the two motors – its “hands” – to play the game is continuously improved. Importantly, the robot does not stop playing to learn; the algorithm runs concurrently with the robot playing the game. As a result, the robot keeps getting better, run after run.


## Documentation

To get started with CyberRunner, please refer to the [Docs](http://cyberrunner.readthedocs.io/).

## Citing
If you use this work in an academic context, please cite the following publication:

* T. Bi, R. D'Andrea,
**"Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based Reinforcement Learning"**, 2023. ([PDF](https://arxiv.org/abs/2312.09906))

        @article{bi2023sample,
          title={Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based Reinforcement Learning},
          author={Bi, Thomas and D'Andrea, Raffaello},
          journal={arXiv preprint arXiv:2312.09906},
          year={2023}
        }

## License
The source code is released under an [AGPL-3.0 license](LICENSE).
