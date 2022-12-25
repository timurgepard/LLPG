import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow_addons.layers import NoisyDense
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
        state = Input(shape=self.state_dim, dtype='float64')
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = concatenate([x, state])
        x = Dense(80, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim),1/np.sqrt(128+self.state_dim)))(x)
        x = concatenate([x, state])
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        action = Input(shape=(self.action_dim,), name='action_input')
        x = concatenate([x, state, action])
        x = Dense(80, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim+self.action_dim),1/np.sqrt(128+self.state_dim+self.action_dim)))(x)
        x = concatenate([x, state, action])
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action], outputs=out)
