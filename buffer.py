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

    def add_experience(self, state, action, reward, Q, next_state):
        self.buffer.append([state, action, reward, Q, next_state])

    def sample(self):
        #arr = np.array(random.sample(self.buffer, self.batch_size)+[self.buffer[indx-1] for indx in range(-4,0)])
        arr = np.array(random.sample(self.buffer, self.batch_size))
        states_batch = np.vstack(arr[:, 0])
        actions_batch =np.array(list(arr[:, 1]))
        rewards_batch = np.vstack(arr[:, 2])
        Q_batch = np.vstack(arr[:, 3])
        next_states_batch = np.vstack(arr[:, 4])
        return states_batch, actions_batch, rewards_batch, Q_batch, next_states_batch
