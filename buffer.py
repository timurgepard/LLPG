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
    def __init__(self, max_buffer_size, max_record_size, batch_size):
        self.max_record_size = max_record_size
        self.batch_size = batch_size
        self.record = deque(maxlen=max_record_size)
        self.cache = []
        self.priorites = deque(maxlen=max_buffer_size)
        self.indexes = deque(maxlen=max_buffer_size)

    #transitions are saved in temporary memory cache, as soon as record is filled, cached is emptified
    def add_transition(self, transition):
        self.cache.append(transition)

    def add_priorities(self,indices,priorites):
        for idx,priority in zip(indices,priorites):
            self.priorites[idx]=priority

    #instead of one transition roll-out is saved for current time step for future steps with length = batch_size
    def add_roll_outs(self, roll_out):
        self.record.append(roll_out)
        self.priorites.append(1.0)
        ln = len(self.record)
        if ln <= self.max_record_size: self.indexes.append(ln-1)

    def restore(self, n_step, gamma):
        #combined exp replay: add last n steps to batch:
        #cer = random.sample(self.record, self.batch_size-n_step)+[self.record[indx-1] for indx in range(-n_step,0)]
        #roll-outs are retrieved here
        sample = random.sample(self.indexes, 20*self.batch_size)
        indices = random.choices(sample, weights= [self.priorites[indx-1] for indx in sample], k = self.batch_size)
        arr = np.array([self.record[indx-1] for indx in indices])

        Sts =  np.vstack(arr[:, 0, :])
        Ats = np.vstack(arr[:, 1, :])
        rts = np.vstack(arr[:, 2, :])
        Sts_ = np.vstack(arr[:, 3, :])
        Ts = np.vstack(arr[:, 4])

        # n-step related batch transition
        Qt = np.zeros((self.batch_size,1))
        for t in range(n_step):
            Qt += gamma**t*np.vstack(rts[:,t]) # here Q is calcualted
        St0 = np.vstack(Sts[:,0])
        At0 = np.vstack(Ats[:,0])
        Stn = np.vstack(Sts[:,n_step-1])
        Atn = np.vstack(Ats[:,n_step-1])
        Stn_ = np.vstack(Sts_[:,n_step-1])
        Tn = np.vstack(Ts[:,n_step-1])

        return St0, At0, Stn, Atn, Stn_, Tn, Qt, indices
