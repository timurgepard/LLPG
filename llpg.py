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
                 n_steps = 8,
                 divide_rewards_by = 1,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 cross_fire=True,
                 temperature=0.0,
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 gamma  = 0.99,
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.max_record_size = max_buffer_size
        self.batch_size = batch_size
        self.act_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_episodes = n_episodes
        self.env = gym.make(env_name).env
        self.action_dim = action_dim = self.env.action_space.shape[0]
        observation_dim = len(self.env.reset())
        self.state_dim = state_dim = observation_dim

        self.x = 0.0
        self.eps =  math.exp(-self.x)
        self.gamma = gamma
        self.rewards_norm = divide_rewards_by

        self.n_steps = n_steps
        self.tr_steps = 2
        self.horizon = int(batch_size/8)
        self.max_steps = max_time_steps  ## Time limit for a episode
        self.replay = Replay(self.max_record_size, self.batch_size)

        self.ANN_opt = Adam(self.act_learning_rate)
        self.QNN_opt = Adam(self.critic_learning_rate)

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN = _q_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _q_network(self.state_dim, self.action_dim).model()
        self.ANN_t.set_weights(self.ANN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())



        print("Env:", env_name)
        #############################################
        #----Action based on exploration policy-----#
        #############################################


    def forward(self, state, cnt):
        eps = max(self.eps, 0.2)
        if random.uniform(0.0, 1.0)>eps:
            action = self.ANN(state)[0]
        else:
            action = self.ANN(state)[0] + tf.random.normal([self.action_dim], 0.0, 2*eps)
        return np.clip(action, -1.0, 1.0)

    def update_buffer(self):
        active_steps = (len(self.replay.cache) - self.horizon)
        if active_steps>0:
            for t in range(len(self.replay.cache)):
                if t<active_steps:
                    arr = np.array(self.replay.cache[t:t+self.horizon])
                    Sts = arr[:,0]
                    Ats = arr[:,1]
                    rts = arr[:,2]
                    Sts_ = arr[:,3]
                    self.replay.add_roll_outs([Sts,Ats,rts,Sts_])
        self.replay.cache = self.replay.cache[-self.horizon:]



    #############################################
    # --------------Update Networks--------------#
    #############################################

    def eps_step(self):
        self.eps = math.exp(-self.x)
        self.tr_steps = 2**round(1/self.eps)
        self.n_steps = 2*self.tr_steps
        if self.n_steps<self.horizon:
            self.x += 0.2*self.act_learning_rate


    def ANN_update(self, ANN, QNN, opt_a, St, norm=1):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            Q = QNN([St,A])
            R = tf.math.reduce_mean(Q)
        dR_dA = tape.gradient(R, A) #first take gradient of dQ/dA
        dR_dA = tf.math.abs(dR_dA)*tf.math.tanh(dR_dA/norm) #then smooth it
        dA_dW = tape.gradient(A, ANN.trainable_variables, output_gradients=-dR_dA)
        opt_a.apply_gradients(zip(dA_dW , ANN.trainable_variables))


    def NN_update(self,NN,opt,input,output, norm=1):
        with tf.GradientTape() as tape:
            e  = (1/2)*(output-NN(input))**2
            L = tf.math.reduce_mean(e)
        dL_dw = tape.gradient(L, NN.trainable_variables)
        opt.apply_gradients(zip(dL_dw, NN.trainable_variables))


    def TD_n(self):
        self.eps_step()
        St, At, Ql, Stn_ = self.replay.restore(self.n_steps, self.gamma)
        A_ = self.ANN_t(Stn_)
        Q_ = self.QNN_t([Stn_, A_])
        Q = Ql + self.gamma**self.n_steps*Q_
        self.NN_update(self.QNN, self.QNN_opt, [St, At], Q, norm=1)
        self.ANN_update(self.ANN, self.QNN, self.ANN_opt, St, norm=1)
        self.update_target(self.ANN_t, self.QNN_t, self.ANN, self.QNN)


    def update_target(self, ANN_t, QNN_t, ANN, QNN):
        self.tow_update(ANN_t, ANN, 0.001)
        self.tow_update(QNN_t, QNN, 0.001)

    def tow_update(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target

    def gradual_start(self, t, steps, start_t):
        return t%(steps+start_t//(t+1))==0


    def save(self):
        self.ANN.save('./models/actor_pred.h5')
        self.ANN_t.save('./models/actor_target1.h5')
        self.QNN.save('./models/critic_pred.h5')
        self.QNN_t.save('./models/critic_target1.h5')



    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        cnt, self.td, score_history = 1, 0, []

        for episode in range(self.n_episodes):
            score, done_cnt, Rt, end = 0, 0, 0.0, False
            state = np.array(self.env.reset(), dtype='float32').reshape(1, self.state_dim)

            for t in range(self.max_steps+self.horizon):
                action = self.forward(state, cnt)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, self.state_dim)

                if done: end = True
                if end or t>=(self.max_steps):
                    if Rt == 0.0: Rt = reward #Rt - terminal reward
                    if abs(Rt)>=50*abs(score/t): reward = Rt/25
                    if done_cnt<self.horizon:
                        done_cnt += 1
                    else:
                        break
                else:
                    cnt += 1
                    score += reward
                    #self.env.render(mode="human")
                    if cnt%self.n_steps == 0: self.update_buffer()
                    if len(self.replay.buffer)>20*self.batch_size:
                        #if self.gradual_start(t, self.n_steps, self.horizon) == 0:
                        if cnt%self.tr_steps==0: self.TD_n()


                self.replay.cache.append([state, action, reward/self.rewards_norm, state_next])
                state = state_next

            self.update_buffer()
            self.replay.cache = []

            score += Rt
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            if episode>=0 and episode%10==0:
                print('%d: %f, %f, | eps %f | record size %d' % (episode, score, avg_score, self.eps, len(self.replay.buffer)))
                self.save()






option = 6

if option == 1:
    env = 'Pendulum-v0'
    max_time_steps = 400
    actor_learning_rate = 0.001
    critic_learning_rate = 0.01
elif option == 2:
    env = 'LunarLanderContinuous-v2'
    max_time_steps = 400
    actor_learning_rate = 0.0004
    critic_learning_rate = 0.004
elif option == 3:
    env = 'HalfCheetahPyBulletEnv-v0'
    max_time_steps = 400
    actor_learning_rate = 0.0002
    critic_learning_rate = 0.002
elif option == 4:
    env = 'MountainCarContinuous-v0'
    max_time_steps = 400
    actor_learning_rate = 0.0004
    critic_learning_rate = 0.004
elif option == 5:
    env = 'BipedalWalker-v3'
    max_time_steps = 400
    actor_learning_rate = 0.0004
    critic_learning_rate = 0.004
elif option == 6:
    env = 'HumanoidPyBulletEnv-v0'
    max_time_steps = 400
    actor_learning_rate = 0.0002
    critic_learning_rate = 0.002
elif option == 7:
    env = 'Walker2DPyBulletEnv-v0'
    max_time_steps = 400
    actor_learning_rate = 0.0002
    critic_learning_rate = 0.002


ddpg = DDPG(     env_name=env, # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 n_steps = 4,
                 divide_rewards_by = 1000, #This brings Q to r range
                 max_buffer_size =100000, # maximum transitions to be stored in buffer
                 batch_size = 128, # batch size for training actor and critic networks
                 max_time_steps = max_time_steps,# no of time steps per epoch
                 gamma  = 0.99,
                 actor_learning_rate = actor_learning_rate,
                 critic_learning_rate = critic_learning_rate,
                 n_episodes = 1000000) # no of episodes to run

ddpg.train()
