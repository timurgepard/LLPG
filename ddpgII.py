
import multiprocessing as mp
import ctypes
from copy import deepcopy
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp


import random
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from buffer import Record
from actor_critic import _actor_network,_critic_network
import math
from collections import deque

import gym
#import gym_vrep
import pybulletgym
import time

def normalize(val, min, max):
    return (val - min)/(max - min)

class DDPG():
    def __init__(self,
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 clip = 25,
                 discount_factor  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.gamma = discount_factor  ## discount factor
        self.explore_time = explore_time
        self.act_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_episodes = n_episodes

        self.env = env
        self.observ_min = self.env.observation_space.low
        self.observ_max = self.env.observation_space.high
        self.action_dim = action_dim = env.action_space.shape[0]

        self.state_cache = []
        self.reward_cache = []
        self.TSt = None
        self.At = None
        self.Rt = None
        self.stack = []
        self.s_x = 0.0
        self.epsilon = 1.0
        self.Q = [-9999.0]

        self.N_stm = 10
        observation_dim = len(env.reset())
        self.state_dim = state_dim = observation_dim

        self.exp_weights = np.ones((self.N_stm,1,observation_dim), dtype='float32')*np.exp(-(2/self.N_stm)*np.arange(0, self.N_stm, 1, dtype='float32')).reshape((self.N_stm, 1, 1))

        self.Q_log = []
        self.dq_da_history = []
        self.clip = clip
        self.T = max_time_steps + self.clip  ## Time limit for a episode
        self.stack_steps = max_time_steps


        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)

        self.record = Record(self.max_buffer_size, self.batch_size)

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()

        self.QNN_pred = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_pred.compile(loss='mse', optimizer=self.QNN_Adam)

        self.QNN_target = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_target.set_weights(self.QNN_pred.get_weights())
        self.QNN_target.compile(loss='mse', optimizer=self.QNN_Adam)

        self.agents_running = False

        self.training = False

        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def forward(self, state):
        action = self.ANN(state)
        epsilon = max(self.epsilon, 0.1)
        if random.uniform(0.0, 1.0)>self.epsilon:
            action = action[0]
        else:
            action = action[0] + tf.random.normal([self.action_dim], 0.0, 3*epsilon)
        return np.clip(action, -1.0, 1.0)

    def update_buffer(self):
        t_last = len(self.stack)
        t_last_clipped = t_last - self.clip
        if t_last>self.clip:
            for t, (TSt,At,Rt,TSt_) in enumerate(self.stack):
                if t<t_last_clipped:
                    Qt = 0.0
                    discount = 1.0
                    for k in range(t, t+self.clip):
                        Rk = self.stack[k][2]
                        Qt = Qt + discount*Rk
                        discount *= self.gamma
                    Qt_ = (Qt - Rt)/self.gamma + self.stack[t+self.clip][2]*discount
                    self.record.add_experience(TSt,At,Rt,Qt,TSt_,Qt_)
            self.stack = self.stack[-self.clip:]


    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ddpg_backprop(self, actor, critic1, critic2, optimizer, tstates_batch, dq_da_history, N):
        with tf.GradientTape(persistent=True) as tape:
            a = actor(tstates_batch)
            tape.watch(a)
            q1 = critic1([tstates_batch, a])
            q2 = critic2([tstates_batch, a])
            q = tf.math.minimum(q1,q2)
        dq_da = tape.gradient(q, a)
        dq_da_history.append(dq_da)

        if len(dq_da_history)>N:
            dq_da_history = dq_da_history[-N:]
            dq_da = np.mean(dq_da_history, axis=0)


        with tf.GradientTape(persistent=True) as tape:
            a = actor(tstates_batch)
            theta = actor.trainable_variables
        dq_da = np.abs(dq_da)*np.tanh(dq_da)
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
        optimizer.apply_gradients(zip(da_dtheta, actor.trainable_variables))


    def train_on_batch(self,QNN,St,At,Q):
        with tf.GradientTape() as tape:
            e = Q-QNN([St, At])
            atanh2 = tf.math.abs(e)*tf.math.tanh(e**2)
        gradient = tape.gradient(atanh2, QNN.trainable_variables)
        self.QNN_Adam.apply_gradients(zip(gradient, QNN.trainable_variables))

    def QNN_update(self,St,At,Rt,Qt,St_,Qt_):
        self.train_on_batch(self.QNN_target, St, At, Qt)
        At_ = self.ANN(St_)
        Q_ = self.QNN_target([St_, At_])
        Q = Rt + self.gamma*(Q_+Qt_)/2
        self.train_on_batch(self.QNN_pred, St, At, (Q+Qt)/2)

    def sync_target(self):
        self.QNN_target.set_weights(self.QNN_pred.get_weights())

    def clear_stack(self):

        self.state_cache = []
        self.reward_cache = []
        self.stack = []


    def save(self):
        result = 0
        while result<10:
            time.sleep(0.01)
            try:
                result += 1
                self.ANN.save('./models/actor.h5')
                self.QNN_pred.save('./models/critic_pred.h5')
                self.QNN_target.save('./models/critic_target.h5')
                return
            except:
                pass



    def epsilon_dt(self):
        self.s_x += 0.01
        self.epsilon = math.exp(-1.0*self.s_x)*math.cos(self.s_x)


    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')

        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []
        self.rec = False
        for episode in range(self.n_episodes):

            self.episode = episode

            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            self.epsilon_dt()

            t = 0
            done_cnt = 0

            DONE = False
            while not DONE:
                t = 0
                done_cnt = 0
                distribute = False
                reward = 0.0

                DONE = False
                for _ in range(self.T+self.clip):
                    if reward: last_reward = reward

                    action = np.array(self.forward(state))
                    state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                    state_next = np.array(state_next).reshape(1, self.state_dim)

                    if t>=self.T: done = True

                    if done:
                        if DONE==False: DONE=True

                    if DONE:
                        if done_cnt == 0:
                            score += reward
                            if (abs(reward)>10*abs(last_reward)): distribute = True

                        if distribute: reward = reward/self.clip

                        if done_cnt>self.clip:
                            done_cnt = 0
                            break
                        else:
                            done_cnt += 1
                    else:
                        score += reward

                        self.env.render()

                        if len(self.record.buffer)>3*self.batch_size:
                            if cnt%(1+self.explore_time//cnt)==0:
                                self.St, self.At, self.Rt, self.Qt, self.St_, self.Qt_ = self.record.sample_batch()
                                self.QNN_update(self.St,self.At,self.Rt,self.Qt,self.St_,self.Qt_)
                                self.ddpg_backprop(self.ANN, self.QNN_target, self.QNN_pred, self.ANN_Adam, self.St, self.dq_da_history, 4)

                        cnt += 1
                        t += 1

                        if len(self.stack)>=(20 + self.clip) and cnt%20 == 0:
                            self.update_buffer()

                    self.stack.append([state, action, reward, state_next])
                    state = state_next

                self.update_buffer()
                self.stack = []


                if episode>=10 and episode%10==0:
                    self.save()


            self.action_noise.reset()
            score_history.append(score)
            avg_score = np.mean(score_history[-10:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f, | %f | record size %d' % (episode, score, avg_score, self.epsilon, len(self.record.buffer)))



    def test(self):

        with open('Scores.txt', 'w+') as f:
            f.write('')

        self.epsilon = 0.0
        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []

        for episode in range(self.n_episodes):

            self.episode = episode

            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)

            t = 0
            done_cnt = 0

            for t in range(self.T):

                self.env.render()

                action = self.forward(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)


                if done:
                    if reward <=-100 or reward >=100:
                        reward = reward/self.clip

                    if done_cnt>self.clip:
                        done_cnt = 0
                        break
                    else:
                        done_cnt += 1

                score += reward
                state = state_next

                cnt += 1


            score_history.append(score)
            avg_score = np.mean(score_history[-10:])

            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f ' % (episode, score, avg_score))


#env = gym.make('Pendulum-v1').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('HumanoidMuJoCoEnv-v0').env
#env = gym.make('BipedalWalkerHardcore-v3').env
env = gym.make('BipedalWalker-v3').env
#env = gym.make('HalfCheetahMuJoCoEnv-v0').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =100000, # maximum transitions to be stored in buffer
                 batch_size = 100, # batch size for training actor and critic networks
                 max_time_steps = 5000,# no of time steps per epoch
                 clip = 300,
                 discount_factor  = 0.98,
                 explore_time = 5000,
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
