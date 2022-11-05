
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
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


        self.s_x = 0.0
        self.epsilon = 1.0

        observation_dim = len(env.reset())
        self.state_dim = state_dim = observation_dim

        self.clip = clip
        self.T = max_time_steps  ## Time limit for a episode
        self.stack_steps = max_time_steps

        self.ANN_Adagrad = Adagrad(self.act_learning_rate)
        self.QNN_Adagrad = Adagrad(self.critic_learning_rate)

        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)

        self.stack = []
        self.dq_da_history = []
        self.record = Record(self.max_buffer_size, self.batch_size)

        self.ANN_target = _actor_network(self.state_dim, self.action_dim).model()
        self.ANN_pred = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN_target = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_pred = _critic_network(self.state_dim, self.action_dim).model()


        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def forward(self, state):
        action = self.ANN_pred(state)
        epsilon = max(self.epsilon, 0.1)
        if random.uniform(0.0, 2.0)>epsilon:
            action = action[0]
        else:
            action = action[0] + tf.random.normal([self.action_dim], 0.0, epsilon)
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
                        Qt = Qt + discount*Rk # here Q is calcualted
                        discount *= self.gamma
                    self.record.add_experience(TSt,At,Rt,Qt,TSt_)
            self.stack = self.stack[-self.clip:]


    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ANN_update(self, ANN, QNN, opt, tstates_batch):
        with tf.GradientTape(persistent=True) as tape:
            a = ANN(tstates_batch)
            q = -QNN([tstates_batch, a])        #minus sign makes Q increase
            q = tf.math.abs(q)*tf.math.tanh(q)  #smothes learning, prevents convergence to local optimum, prediction errors, etc.
        dq_dw = tape.gradient(q, ANN.trainable_variables)
        opt.apply_gradients(zip(dq_dw, ANN.trainable_variables))

    def QNN_update(self,QNN,opt,St,At,Q):
        with tf.GradientTape() as tape:
            e = (Q-QNN([St, At]))**2
            atanh2 = tf.math.sqrt(e)*tf.math.tanh(e)   #secures against outliers, quadratic between ~ 0 to 1, linear after
        de_dw = tape.gradient(atanh2, QNN.trainable_variables)
        opt.apply_gradients(zip(de_dw, QNN.trainable_variables))

    def TD_secure(self):
        St, At, Rt, Qt, St_ = self.record.sample_batch() #samples replay memory, contains correct Q value (Qt)
        self.update_target(self.QNN_target, self.QNN_pred, 0.001)   #tau update critic
        self.update_target(self.ANN_target, self.ANN_pred, 0.001)   #tau update actor
        At_ = self.ANN_target(St_)
        Q_ = self.QNN_target([St_, At_])
        Q = Rt + self.gamma*Q_
        self.QNN_update(self.QNN_pred, self.QNN_Adam, St, At, 0.5*(Q+Qt)) #trains on TD (Q) and Monte-Carlo rollout (Qt)
        self.ANN_update(self.ANN_pred, self.QNN_pred, self.ANN_Adam, St)


    def update_target(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target

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
        self.epsilon = 2.0*math.exp(-1.0*self.s_x)*math.cos(self.s_x)


    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')

        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []
        print('ep: score, avg, | eps | record size ')
        for episode in range(self.n_episodes):

            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            self.epsilon_dt()

            t, done_cnt, done = 0, 0, False

            while not done:
                t = 0
                done_cnt = 0
                distribute = False
                reward = 0.0

                for _ in range(self.T+self.clip):
                    if reward: last_reward = reward

                    action = np.array(self.forward(state))
                    state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                    state_next = np.array(state_next).reshape(1, self.state_dim)

                    if t>=self.T: done = True

                    if done:
                        if done_cnt == 0:
                            score += reward
                            if (abs(reward)>10*abs(last_reward)): distribute = True

                        if distribute: reward /= self.clip

                        if done_cnt>self.clip:
                            break
                        else:
                            done_cnt += 1
                    else:
                        score += reward
                        self.env.render()

                        if len(self.record.buffer)>3*self.batch_size:
                            if cnt%(1+self.explore_time//cnt)==0:  #this just makes training starting gradually
                                self.TD_secure()

                        cnt += 1
                        t += 1

                        if len(self.stack)>=(20 + self.clip) and cnt%20 == 0: # replay buffer is populated each 20 steps, after steps is enough for Qt.
                            self.update_buffer()

                    self.stack.append([state, action, reward, state_next])
                    state = state_next

                self.update_buffer()
                self.stack = []

                if episode>=10 and episode%10==0:
                    self.save()

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

#env = gym.make('Pendulum-v0').env
#env = gym.make('LunarLanderContinuous-v2').env
env = gym.make('BipedalWalker-v3').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =100000, # maximum transitions to be stored in buffer
                 batch_size = 100, # batch size for training actor and critic networks
                 max_time_steps = 2000,# no of time steps per epoch
                 clip = 100, # Q is calculated till n-steps, even after termination for correctness
                 discount_factor  = 0.97,
                 explore_time = 5000,
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
