from . import Uniform
from . import generic
from . import selectors
from . import limiters
import numpy as np


class Cyberrunner(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )

    def add(self, step, worker=0, load=False):
        # Do transformations
        # print(step)
        # print(worker)
        i = 0
        if load:
            super().add(step, str(i), load)
            return
        for dir in [0]:
            for flip_h in [0, 1]:
                for flip_v in [0, 1]:
                    step = step.copy()
                    image = step["image"].copy()
                    states = step["states"].copy()
                    goal = (
                        step["goal"].copy() if dir == 0 else step["goal_reverse"].copy()
                    )
                    reward = (
                        step["reward"].copy() if dir == 0 else -step["reward"].copy()
                    )
                    action = step["action"].copy()
                    if flip_h:
                        image = np.flip(image, axis=1)
                        action[0] *= -1
                        states[0] *= -1
                        states[2] = 1 - states[2]
                        goal[0::2] *= -1
                        # states[np.arange(4, 14, 2)] *= -1

                    if flip_v:
                        image = np.flip(image, axis=0)
                        action[1] *= -1
                        states[1] *= -1
                        states[3] = 1 - states[3]
                        goal[1::2] *= -1
                        # states[np.arange(5, 14, 2)] *= -1
                    step["image"] = image
                    step["states"] = states
                    step["goal"] = goal
                    step["action"] = action
                    step["reward"] = reward

                    super().add(step, str(i), load)
                    i += 1
