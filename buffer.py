from collections import deque
import random
import numpy as np
import math
from itertools import repeat
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

def normalize(val, min, max):
    return (val - min)/(max - min)

class Record:
    def __init__(self, max_buffer_size, max_tape_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.max_tape_size = max_tape_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.tape = deque(maxlen=max_tape_size)
        self.priorites = deque(maxlen=max_tape_size)
        self.indexes = deque(maxlen=max_tape_size)
        self.stack_priorities = []

    def add_roll_outs(self, state, R):
        self.tape.append([state, R])
        self.priorites.append(1.0)
        ln = len(self.tape)
        if ln <= self.max_tape_size: self.indexes.append(ln-1)

    def add_priorities(self,indices,priorites):
        for idx,priority in zip(indices,priorites):
            self.priorites[idx]=priority

    def sample(self):
        arr = np.array(random.sample(self.buffer, self.batch_size))
        states_batch = np.vstack(arr[:, 0])
        actions_batch =np.array(list(arr[:, 1]))
        rewards_batch = np.vstack(arr[:, 2])
        next_states_batch = np.vstack(arr[:, 3])
        st_dev_batch = np.vstack(arr[:, 4])
        done_batch = np.vstack(arr[:, 5])
        return states_batch, actions_batch, rewards_batch, next_states_batch, st_dev_batch, done_batch

    def restore(self):
        arr, indices = np.array(random.sample(self.buffer, self.batch_size)), []
        #indices = random.choices(self.indexes, weights=self.priorites, k = self.batch_size)
        #arr = np.array([self.tape[indx-1] for indx in indices])
        states_batch = np.vstack(arr[:, 0])
        return_batch = np.vstack(arr[:, 1])
        return states_batch, return_batch, indices
