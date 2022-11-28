import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, SimpleRNN
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

def atanh(x):
    return K.abs(x)*K.tanh(x)

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        state = Input(shape=(self.state_dim), dtype='float64')
        x = Dense(400, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = Dense(300, activation='relu', kernel_initializer=RU(-1/np.sqrt(40),1/np.sqrt(40)))(x)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)


class _q_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        state_i = Dense(400, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        action = Input(shape=(self.action_dim,), name='action_input', dtype='float64')
        x = concatenate([state_i, action])
        x = Dense(300, activation='relu', kernel_initializer=RU(-1/np.sqrt(40),1/np.sqrt(40)))(x)
        out = Dense(1, activation='linear', kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=[state, action], outputs=out)
