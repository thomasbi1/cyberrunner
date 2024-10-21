from collections import deque

import numpy as np


class Fifo:
    def __init__(self):
        self.queue = deque()

    def __call__(self):
        return self.queue[0]

    def __setitem__(self, key, steps):
        self.queue.append(key)

    def __delitem__(self, key):
        if self.queue[0] == key:
            self.queue.popleft()
        else:
            # TODO: This branch is unused but very slow.
            self.queue.remove(key)


class Uniform:
    def __init__(self, seed=0):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        index = self.rng.integers(0, len(self.keys)).item()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class Prio:
    def __init__(self, seed=0):
        self.indices = {}
        self.sampled_indices = []
        self.keys = []
        self.prios = []
        self.max_progress = 0
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        # index = self.rng.integers(0, len(self.keys)).item()
        if not self.sampled_indices:  # TODO this is wrong when replay buffer is full
            probs = (0.4 + np.asarray(self.prios) / self.max_progress) ** 0.8
            self.sampled_indices = self.rng.choice(
                len(probs), 512, p=probs / probs.sum()
            ).tolist()
        # probs = (0.4 + np.asarray(self.prios) / self.max_progress) ** 0.8
        # index = self.rng.choice(len(self.keys), p=probs/probs.sum())
        index = self.sampled_indices.pop()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        progress = np.asarray([step["progress"] for step in steps])
        max_progress = progress.max()
        self.max_progress = max(self.max_progress, max_progress)
        self.prios.append(max_progress)
        self.keys.append(key)

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        last_prio = self.prios.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.prios[index] = last_prio
            self.indices[last] = index
        print("deleted {} {}".format(len(self.keys), len(self.prios)))
