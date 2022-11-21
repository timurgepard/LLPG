import os
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta

import random
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from buffer import Replay
from actor_critic import _actor_network,_q_network, _v_network, _dist_network
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
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 max_record_size = 1000000,
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 n_steps = 25,
                 gamma  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.max_record_size = max_record_size
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
        self.y = 1.0
        self.gamma = gamma
        self.type = "DDPG"

        observation_dim = len(self.env.reset())
        self.state_dim = state_dim = observation_dim

        self.n_steps = n_steps
        self.gamma_sum=sum([self.gamma**i for i in range(self.n_steps)])
        self.T = max_time_steps  ## Time limit for a episode
        self.stack_steps = max_time_steps
        self.stack = []
        self.replay = Replay(self.max_buffer_size, self.max_record_size, self.batch_size)


        self.ANN_Adam = Adam(self.act_learning_rate)
        self.sNN_Adadelta = Adadelta(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)
        self.VNN_Adam = Adam(self.critic_learning_rate)


        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.sNN = _dist_network(self.state_dim).model()
        self.QNN = _q_network(self.state_dim, self.action_dim).model()
        self.VNN = _v_network(self.state_dim).model()
        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.sNN_t = _dist_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _q_network(self.state_dim, self.action_dim).model()
        self.ANN_t.set_weights(self.ANN.get_weights())
        self.sNN_t.set_weights(self.sNN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())
        self.gauss_const = math.log(1/math.sqrt(2*math.pi))

        print("Env:", env_name)
        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def chose_action(self, state):
        std = self.sNN(state)
        action = self.ANN(state)[0] + tf.random.normal([self.action_dim], 0.0, std[0])
        return np.clip(action, -1.0, 1.0), std

    def update_buffer(self):
        active_steps = (len(self.stack) - self.n_steps)
        if active_steps>0:
            for t, (St,rt) in enumerate(self.stack):
                if t<active_steps:
                    Qt, Ql, l = 0.0, 0.0, 0.5
                    l_ = 1-l
                    for k in range(t, t+self.n_steps):
                        i = k-t
                        Qt += self.gamma**i*self.stack[k][1] # here Q is calcualted
                        Ql += l_*(l**i)*Qt if i<self.n_steps-1 else l**self.n_steps*Qt #TD lambda decaying weights sum(05*05^(t-1)*Qt)+0.5^T*Qt
                    self.replay.add_roll_outs(St,Ql)
        self.stack = self.stack[-self.n_steps:]

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def def_algorithm(self):
        self.y = max(1.0-self.sigmoid(self.x), 0.05)
        self.x += 0.1*self.critic_learning_rate
        div = round(1/self.y)
        if div<=1:
            self.type = "DDPG"
        elif 1<div<=2:
            self.type = "TD3"
        elif 2<div<=4:
            self.type = "SAC"
        elif div>4:
            self.type = "GAE"

    def ANN_update(self, ANN, sNN, QNN, VNN, opt_a, opt_std, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            std = sNN(St)
            Q = QNN([St, A, std])
            if self.type!="DDPG":
                V = VNN(St)
                Q = 2*Q-V #(Q+(Q-V)) if Q=V then Q, else Q+A
                if self.type=="SAC" or self.type =="GAE": #soft
                    At = tf.random.normal(A.shape, 0.0, std)
                    log_prob = self.gauss_const-tf.math.log(std)-(tf.math.reduce_mean(A-At)/std)**2
                    log_prob -= tf.math.reduce_sum(tf.math.log(1-tf.math.tanh(At)**2))
                    Q = Q+0.1*np.abs(np.mean(Q))*log_prob #10% of R => log_prob entropy
                    if self.type=="GAE":
                        Q = log_prob*Q # log_prob now directs sign of gradient
            Q = tf.math.abs(Q)*tf.math.tanh(Q)  #exponential linear x: atanh, to smooth gradient
            R = -tf.math.reduce_mean(Q) #for gradient increase
        dR_dW = tape.gradient(R, ANN.trainable_variables)
        opt_a.apply_gradients(zip(dR_dW, ANN.trainable_variables))
        dR_dw = tape.gradient(R, sNN.trainable_variables)
        opt_std.apply_gradients(zip(dR_dw, sNN.trainable_variables))

    def QNN_update(self,QNN,opt,St,At,st,Q):
        with tf.GradientTape() as tape:
            e = Q-QNN([St, At, st])
            e = e*tf.math.tanh(e)   #differetiable abs(x): xtanh
            L = tf.math.reduce_mean(e)
        dL_dw = tape.gradient(L, QNN.trainable_variables)
        opt.apply_gradients(zip(dL_dw, QNN.trainable_variables))

    def VNN_update(self):
        St, Vt, idx = self.replay.restore()
        with tf.GradientTape() as tape:
            e = Vt-self.VNN(St)
            e = e*tf.math.tanh(e)   #differetiable abs(x): xtanh
            L = tf.math.reduce_mean(e)
        #self.replay.add_priorities(idx,e)
        dL_dw = tape.gradient(L, self.VNN.trainable_variables)
        self.VNN_Adam.apply_gradients(zip(dL_dw, self.VNN.trainable_variables))

    def TD_secure(self):
        self.def_algorithm()
        St, At, rt, St_, st, d = self.replay.sample()
        A_ = self.ANN_t(St_)
        std_ = self.sNN_t(St_)
        Q_ = self.QNN_t([St_, A_, std_])
        Q = rt + (1-d)*self.gamma*Q_
        if self.type!="DDPG":
            Q = np.abs(Q)*np.tanh(Q)
        self.QNN_update(self.QNN, self.QNN_Adam, St, At, std_, Q)
        self.ANN_update(self.ANN, self.sNN, self.QNN, self.VNN, self.ANN_Adam, self.sNN_Adadelta, St)
        self.update_target()

    def update_target(self):
        self.tow_update(self.ANN_t, self.ANN, 0.001)
        self.tow_update(self.sNN_t, self.sNN, 0.001)
        self.tow_update(self.QNN_t, self.QNN, 0.001)

    def tow_update(self, target, online, tow):
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
        self.ANN.save('./models/actor_pred.h5')
        self.QNN.save('./models/critic_pred.h5')

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def gradual_start(self, t, start_t):
        if t<start_t:
            return t%(1+start_t//t)==0
        return True

    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        state_dim = len(self.env.reset())
        self.cnt, awr = 0, 0
        score_history = []
        print('ep: score, avg, | y | std | record size ')
        r_mean = 1.0

        for episode in range(self.n_episodes):
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            terminal_reward = False
            rewards = []
            for t in range(self.T):
                #self.env.render(mode="human")
                action, std = self.chose_action(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, self.state_dim)
                self.replay.buffer.append([state, action, reward, state_next, std, done])
                rewards.append(reward)
                self.cnt += 1

                if done or t>=(self.T-1):
                    r_max = np.max(np.abs(rewards[:-1]))
                    for t in range(self.n_steps+1):
                        if abs(reward)>10*r_max: terminal_reward = reward
                        if terminal_reward:
                            reward = terminal_reward/self.gamma_sum
                            self.stack.append([state, reward])
                        else:
                            state = np.array(state_next).reshape(1, self.state_dim)
                            action, std = self.chose_action(state)
                            state_next, reward, _, _ = self.env.step(action)  # step returns obs+1, reward, done
                            self.stack.append([state, reward])
                            state = state_next
                    break

                self.stack.append([state, reward])
                state = state_next

                if episode>1 and len(self.replay.buffer)>self.batch_size:
                    if t%(round(1/self.y))==0:
                        if self.gradual_start(self.cnt, self.explore_time): # starts training gradualy globally
                            self.TD_secure()

                if len(self.stack)>=self.batch_size and t%(int(self.batch_size/2)) == 0:
                    self.update_buffer()
                    if len(self.replay.record)>self.batch_size:
                        self.VNN_update()

            self.update_buffer()
            if len(self.replay.record)>self.batch_size:
                self.VNN_update()
            self.stack = []

            if episode>=10 and episode%10==0:
                self.save()

            score = sum(rewards)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print(self.type, '%d: %f, %f, | once in %d step | std %f | buffer %d| record/step %d' % (episode, score, avg_score, round(1/self.y), std, len(self.replay.buffer), self.cnt))

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
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
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

option = 4

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
    max_time_steps = 200
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
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 max_record_size = 1000000,
                 batch_size = 64, # batch size for training actor and critic networks
                 max_time_steps = max_time_steps,# no of time steps per epoch
                 n_steps = 50, # Q is calculated till n-steps, even after termination for correctness
                 gamma  = 0.99,
                 explore_time = 10000,
                 actor_learning_rate = actor_learning_rate,
                 critic_learning_rate = critic_learning_rate,
                 n_episodes = 1000000) # no of episodes to run

ddpg.train()
            #dist = tfd.Normal(loc=A, scale=std)
            #At = dist.sample(A.shape)
