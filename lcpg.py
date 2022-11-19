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
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.priorites = deque(maxlen=max_buffer_size)
        self.indexes = deque(maxlen=max_buffer_size)
        self.stack_priorities = []

    def add_experience(self, state, action, reward, Q, next_state, done):
        self.buffer.append([state, action, reward, Q, next_state, done])
        self.priorites.append(1.0)
        ln = len(self.buffer)
        if ln <= self.max_buffer_size: self.indexes.append(ln-1)

    def add_priorities(self,indices,priorites):
        for idx,priority in zip(indices,priorites):
            self.priorites[idx]=priority

    def sample(self, t):
        if t=="ER":
            indices = []
            arr = np.array(random.sample(self.buffer, self.batch_size))
        elif t=="PER":
            indices = random.choices(self.indexes, weights=self.priorites, k = self.batch_size)
            arr = np.array([self.buffer[indx-1] for indx in indices])
        elif type=="FULL":
            indices = []
            arr = np.array(self.buffer)
        states_batch = np.vstack(arr[:, 0])
        actions_batch =np.array(list(arr[:, 1]))
        rewards_batch = np.vstack(arr[:, 2])
        Q_batch = np.vstack(arr[:, 3])
        next_states_batch = np.vstack(arr[:, 4])
        done_batch = np.vstack(arr[:, 5])
        return states_batch, actions_batch, rewards_batch, Q_batch, next_states_batch, done_batch, indices

    def generate_indeces(self):
        if len(self.buffer)>self.batch_size:
            for indices, priorites in self.stack_priorities:
                for idx,priority in zip(indices,priorites):
                    self.priorites[idx]=priority
            self.stack_priorities = []
