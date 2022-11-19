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
        state = Input(shape=(self.state_dim), dtype='float32')
        x = Dense(75, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = Dense(50, activation=atanh,kernel_initializer=RU(-1/np.sqrt(75),1/np.sqrt(75)))(x)
        out = Dense(self.action_dim, activation='tanh', kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)

class _dist_network():
    def __init__(self, state_dim, action_bound_range=1):
        self.state_dim = state_dim

    def model(self):
        state = Input(shape=(self.state_dim), dtype='float32')
        std = Dense(1, activation='sigmoid', kernel_initializer=RU(-0.003,0.003))(state)
        return Model(inputs=state, outputs=std+0.01)

class _q_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float32')
        state_i = Dense(75, activation=atanh, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        action = Input(shape=(self.action_dim,), name='action_input', dtype='float32')
        std = Input(shape=(1,), name='std_input', dtype='float32')
        x = concatenate([state_i, action, std])
        x = Dense(50, activation=atanh, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-1/np.sqrt(75),1/np.sqrt(75)))(x)
        out = Dense(1, activation='linear', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=[state, action, std], outputs=out)

class _v_network():
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float32')
        x = Dense(75, activation=atanh, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = Dense(50, activation=atanh, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-1/np.sqrt(75),1/np.sqrt(75)))(x)
        out = Dense(1, activation='linear', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)
