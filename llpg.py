import os
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop

import random
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from buffer import Replay
from actor_critic import _actor_network,_q_network
import math
from collections import deque

import gym
import pybulletgym
import time


def normalize(val, min, max):
    return (val - min)/(max - min)


class DDPG():
    def __init__(self,
                 env_name, # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 n_steps=64,
                 normalize_Q_by = 1,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 cross_fire=True,
                 temperature=0.0,
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 gamma  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.max_record_size = max_buffer_size
        self.batch_size = batch_size
        self.explore_time = explore_time
        self.act_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_episodes = n_episodes
        self.env = gym.make(env_name).env
        self.observ_min = self.env.observation_space.low
        self.observ_max = self.env.observation_space.high
        self.action_dim = action_dim = self.env.action_space.shape[0]

        self.x = -3.0
        self.eps = 1.0
        self.gamma = gamma
        self.norm = normalize_Q_by
        self.n_steps = n_steps

        observation_dim = len(self.env.reset())
        self.state_dim = state_dim = observation_dim

        self.n_step = 4
        self.T = max_time_steps  ## Time limit for a episode
        self.replay = Replay(self.max_buffer_size, self.max_record_size, self.batch_size)

        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)


        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _q_network(self.state_dim, self.action_dim).model()

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN = _q_network(self.state_dim, self.action_dim).model()

        self.ANN_t.set_weights(self.ANN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())

        self.tr = 0
        self.n_step = 4

        print("Env:", env_name)
        #############################################
        #----Action based on exploration policy-----#
        #############################################


    def chose_action(self, state):
        action = self.ANN(state)[0]
        if random.uniform(0.0,1.0)<self.eps:
            action += tf.random.normal([self.action_dim], 0.0, self.eps+0.2)
        return np.clip(action, -1.0, 1.0)

    def update_buffer(self):
        active_steps = (len(self.replay.cache) - self.n_steps)
        if active_steps>0:
            for t in range(len(self.replay.cache)):
                if t<active_steps:
                    arr = np.array(self.replay.cache[t:t+self.n_steps])
                    Sts = arr[:,0]
                    Ats = arr[:,1]
                    rts = arr[:,2]
                    Sts_ = arr[:,3]
                    Ts = arr[:,4]
                    self.replay.add_roll_outs([Sts,Ats,rts,Sts_,Ts])
        self.replay.cache = self.replay.cache[-self.n_steps:]

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def eps_step(self):
        self.eps =  (1.0-self.sigmoid(self.x))
        self.n_step = 2*round(1/self.eps)

        if self.n_step<self.n_steps:
            self.x += self.act_learning_rate
        self.tr += 1


    def ANN_update(self, ANN, QNN, opt_a, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            tape.watch(A)
            Q = QNN([St,A])
            Q = tf.math.reduce_mean(Q)
        dQ_dA = tape.gradient(Q, A) #first take gradient of dQ/dA
        dQ_dA = tf.math.abs(dQ_dA)*tf.math.tanh(dQ_dA) #then smooth it
        dA_dW = tape.gradient(A, ANN.trainable_variables, output_gradients=-dQ_dA) #then apply to action network
        opt_a.apply_gradients(zip(dA_dW, ANN.trainable_variables))


    def NN_update(self,NN,opt,input,output):
        with tf.GradientTape() as tape:
            e = output-NN(input)
            e = e*tf.math.tanh(e)   #differetiable abs(x): xtanh
            L = tf.math.reduce_mean(e)
        dL_dw = tape.gradient(L, NN.trainable_variables)
        opt.apply_gradients(zip(dL_dw, NN.trainable_variables))


    def TD_secure(self):
        St, At, Stn, Atn, Stn_, Tn, Qt = self.replay.restore(self.n_step, self.gamma)
        An_ = self.ANN_t(Stn_)
        Qn_ = self.QNN_t([Stn_, An_])
        Q = Qt + (1-Tn)*self.gamma**self.n_step*Qn_
        #Q += 0.01*tf.math.log(self.eps)/self.norm

        self.NN_update(self.QNN, self.QNN_Adam, [St, At], Q)
        self.ANN_update(self.ANN, self.QNN, self.ANN_Adam, St)
        self.update_target()


    def update_target(self):
        self.tow_update(self.ANN_t, self.ANN, 0.001)
        self.tow_update(self.QNN_t, self.QNN, 0.001)



    def tow_update(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target


    def save(self):
        self.ANN.save('./models/actor_pred.h5')
        self.QNN.save('./models/critic_pred.h5')
        self.ANN_t.save('./models/actor_target.h5')
        self.QNN_t.save('./models/critic_target1.h5')


    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def gradual_start(self, t, start_t):
        if t<start_t:
            return t%(1+start_t//(t+1))==0
        return True

    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        state_dim = len(self.env.reset())
        self.cnt, self.cs = 0, -1
        score_history = []
        print('ep: score, avg, | y | std | record size ')
        r_mean = 1.0

        for episode in range(self.n_episodes):
            score = 0.0
            state = np.array(self.env.reset(), dtype='float64').reshape(1, state_dim)
            done, T = False, False
            rewards = []
            for t in range(self.T):
                #self.env.render(mode="human")
                action = self.chose_action(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, self.state_dim)
                rewards.append(reward)
                self.cnt += 1
                if done or t>=(self.T-1):
                    r_max = np.max(np.abs(rewards[:-1]))
                    if abs(reward)>10*r_max: T = True
                    self.replay.add_transition([state, action, reward/self.norm, state_next, T])
                    #here we collect correct Q
                    for t in range(self.n_steps):
                        if T:
                            self.replay.add_transition([state, action, reward/self.norm, state_next, T])
                            for t in range(self.n_steps):
                                self.replay.add_transition([state, action, 0.0, state_next, T])
                            break
                        else:
                            state = np.array(state_next).reshape(1, self.state_dim)
                            action = self.chose_action(state)
                            state_next, reward, _, _ = self.env.step(action)  # step returns obs+1, reward, done
                            self.replay.add_transition([state, action, reward/self.norm, state_next, T])
                            state = state_next
                            if abs(reward)>10*r_max: T = True
                    break
                self.replay.add_transition([state, action, reward/self.norm, state_next, T])
                state = state_next
                if len(self.replay.cache)>=self.n_steps and self.cnt%self.n_step == 0: # replay buffer is populated each n steps, after steps is enough for Qt.
                    self.update_buffer()


                if len(self.replay.record)>self.batch_size:
                    if self.cnt%(self.n_step)==0:
                        if self.gradual_start(self.cnt, self.explore_time): # starts training gradualy globally
                            if self.gradual_start(t, self.n_steps): # starts training gradually within episode
                                self.eps_step()
                                self.TD_secure()


            self.update_buffer()
            self.replay.cache = []

            if episode>=10 and episode%10==0:
                self.save()

            score = sum(rewards)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f, | once in %d steps, Q for %d, eps %f| record %d| step %d| tr step %d' % (episode, score, avg_score, self.n_step, self.n_step, self.eps, len(self.replay.record), self.cnt, self.tr))

    def test(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        state_dim = len(self.env.reset())
        rewards = []
        score_history = []
        r_mean = 1.0
        for episode in range(1):
            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float64').reshape(1, state_dim)
            for t in range(self.T):
                self.env.render()
                action = np.array(self.chose_action(state))
                state_next, reward, done, info = self.env.step(action)
                state_next = np.array(state_next).reshape(1, state_dim)
                state = state_next
                rewards.append(reward)
                if done: break
            score = sum(rewards)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

option = 2

if option == 1:
    env = 'Pendulum-v0'
    max_time_steps = 200
    actor_learning_rate = 0.001
    critic_learning_rate = 0.01
elif option == 2:
    env = 'LunarLanderContinuous-v2'
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 3:
    env = 'BipedalWalker-v3'
    max_time_steps = 1000
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 4:
    env = 'HumanoidPyBulletEnv-v0'
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 5:
    env = 'HalfCheetahPyBulletEnv-v0'
    max_time_steps = 200
    actor_learning_rate = 0.00001
    critic_learning_rate = 0.0001
elif option == 6:
    env = 'MountainCarContinuous-v0'
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001


ddpg = DDPG(     env_name=env, # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 n_steps = 32,
                 normalize_Q_by = 1, #1 no normalization, 10-1000 possible values
                 max_buffer_size =100000, # maximum transitions to be stored in buffer
                 batch_size = 64, # batch size for training actor and critic networks
                 max_time_steps = max_time_steps,# no of time steps per epoch
                 gamma  = 0.98,
                 explore_time = 10000,
                 actor_learning_rate = actor_learning_rate,
                 critic_learning_rate = critic_learning_rate,
                 n_episodes = 1000000) # no of episodes to run

ddpg.train()
