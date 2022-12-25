from collections import deque
import random
import numpy as np


class Replay:
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)

    def add_experience(self, transition):
        self.buffer.append(transition)


    def sample(self):
        arr = np.random.default_rng().choice(self.buffer, size=self.batch_size, replace=False)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = np.array(list(arr[:, 1]))
        rewards_batch = np.vstack(arr[:, 2])
        states_next = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4])
        return states_batch, actions_batch, rewards_batch, states_next, done_batch
