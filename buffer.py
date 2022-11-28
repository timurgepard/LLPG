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
        self.max_buffer_size = max_buffer_size
        self.max_record_size = max_record_size
        self.batch_size = batch_size
        self.record = deque(maxlen=max_record_size)
        self.cache = []

    #transitions are saved in temporary memory cache, as soon as record is filled, cached is emptified
    def add_transition(self, transition):
        self.cache.append(transition)

    #instead of one transition roll-out is saved for current time step for future steps with length = batch_size
    def add_roll_outs(self, roll_out):
        self.record.append(roll_out)

    def restore(self, n_step, gamma):
        #combined exp replay: add last n steps to batch: Exp Replay < Combined Exp Replay < Prioritized Exp Replay
        cer = random.sample(self.record, self.batch_size-n_step)+[self.record[indx-1] for indx in range(-n_step,0)]
        #roll-outs are retrieved here
        arr = np.array(cer)
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

        return St0, At0, Stn, Atn, Stn_, Tn, Qt
