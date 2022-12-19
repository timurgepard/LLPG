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

class Replay:
    def __init__(self, max_record_size, batch_size):
        self.max_record_size = max_record_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_record_size)
        self.cache = []


    def add_roll_outs(self, roll_out):
        self.buffer.append(roll_out)


    def restore(self, n_steps, gamma):
        if len(self.buffer)>=self.batch_size:
            arr = np.random.default_rng().choice(self.buffer, size=self.batch_size, replace=False)

            Sts =  np.vstack(arr[:, 0, :])
            Ats = np.vstack(arr[:, 1, :])
            rts = np.vstack(arr[:, 2, :])
            Sts_ = np.vstack(arr[:, 3, :])

            Ql = Qt = np.zeros((self.batch_size,1))
            for t in range(n_steps):
                Qt += gamma**t*np.vstack(rts[:,t]) # here Q is calcualted
                if t<n_steps-1: Ql += 0.1*0.95**t*Qt
            Ql += 0.95**(n_steps-1)*Qt
            St = np.vstack(Sts[:,0])
            At = np.vstack(Ats[:,0])
            St_ = np.vstack(Sts_[:, n_steps-1])

        return St, At, Ql, St_
