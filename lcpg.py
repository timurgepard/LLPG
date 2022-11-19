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
from buffer import Record
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
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 max_tape_size = 1000000,
                 gamma = 0.99,
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 n_steps = 25,
                 discount_factor  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.max_tape_size = max_tape_size
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

        self.x = -3.0
        self.y = 1.0
        self.gamma = gamma
        self.type = "DDPG"

        observation_dim = len(env.reset())
        self.state_dim = state_dim = observation_dim

        self.n_steps = n_steps
        self.gamma_sum=sum([self.gamma**i for i in range(self.n_steps)])
        self.T = max_time_steps  ## Time limit for a episode
        self.stack_steps = max_time_steps
        self.stack = []
        self.record = Record(self.max_buffer_size, self.max_tape_size, self.batch_size)


        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)
        self.VNN_Adam = Adam(self.critic_learning_rate)
        self.Adadelta = Adadelta(self.critic_learning_rate)



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
                    Qt = 0.0
                    for k in range(t, t+self.n_steps):
                        Qt += self.gamma**k*self.stack[k][1] # here Q is calcualted
                    self.record.add_roll_outs(St,Qt)
        self.stack = self.stack[-self.n_steps:]

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def def_algorithm(self):
        self.y = 1.0-self.sigmoid(self.x)
        self.x += self.critic_learning_rate*0.01
        if self.x<=0:
            self.type == "DDPG"
        elif 0.0<self.x<=1.0:
            self.type == "TD3"
        elif 1.0<self.x<=2.0:
            self.type == "SAC"
        elif self.x>2.0:
            self.type == "GAE"

    def ANN_update(self, ANN, DNN, QNN, VNN, opt_a, opt_std, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            std = DNN(St)
            Q = QNN([St, A, std])
            if self.type=="SAC" or "GAE": #soft
                #At is a sample from normal dist
                At = tf.random.normal(A.shape, 0.0, std[0])
                #calculate log(Guassian dist)=log_prob, gauss const = log(1/sqrt(2pi))
                log_prob = self.gauss_const-tf.math.log(std)-((A-At)/std)**2
                if self.type=="SAC":
                    Q = Q*(1-0.01*log_prob) #1% of R => log_prob entropy
                elif self.type=="GAE":
                    V = VNN(St)
                    Q = log_prob*(Q-V) # log_prob now directs sign of gradient
            R = tf.math.abs(Q)*tf.math.tanh(Q)  #exponential linear x: atanh, to smooth gradient
            R = -tf.math.reduce_mean(R)
        dR_dW = tape.gradient(R, ANN.trainable_variables)
        opt_a.apply_gradients(zip(dR_dW, ANN.trainable_variables))
        if self.type=="SAC" or self.type=="GAE":
            dR_dw = tape.gradient(R, DNN.trainable_variables)
            opt_std.apply_gradients(zip(dR_dw, DNN.trainable_variables))

    def QNN_update(self,QNN,opt,St,At,st,Q):
        with tf.GradientTape() as tape:
            e = Q-QNN([St, At, st])
            e = e*tf.math.tanh(e)   #differetiable abs(x): xtanh
            L = tf.math.reduce_mean(e)
        dL_dw = tape.gradient(L, QNN.trainable_variables)
        opt.apply_gradients(zip(dL_dw, QNN.trainable_variables))

    def VNN_update(self):
        St, Vt, idx = self.record.restore()
        with tf.GradientTape() as tape:
            e = Vt-self.VNN(St)
            e = e*tf.math.tanh(e)   #differetiable abs(x): xtanh
            L = tf.math.reduce_mean(e)
        self.record.add_priorities(idx,e)
        dL_dw = tape.gradient(L, self.VNN.trainable_variables)
        self.VNN_Adam.apply_gradients(zip(dL_dw, self.VNN.trainable_variables))

    def TD_secure(self):
        self.def_algorithm()
        St, At, rt, St_, st, d = self.record.sample()
        A_ = self.ANN_t(St_)
        std_ = self.sNN_t(St_)
        Q_ = self.QNN_t([St_, A_, std_])
        Q = rt + (1-d)*self.gamma*Q_
        if self.type == "TD3" or self.type=="SAC" or self.type=="GAE":
            Q = np.abs(Q)*np.tanh(Q) #exponential linear x: atanh, to smooth prediction, TD3 alternative
        self.QNN_update(self.QNN, self.QNN_Adam, St, At, std_, Q)
        self.ANN_update(self.ANN, self.sNN, self.QNN, self.VNN, self.ANN_Adam, self.Adadelta, St)
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
            state = np.array(env.reset(), dtype='float32').reshape(1, state_dim)
            terminal_reward = False
            rewards = []
            for t in range(self.T):
                #self.env.render(mode="human")
                action, std = self.chose_action(state)
                state_next, reward, done, info = env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, self.state_dim)
                self.record.buffer.append([state, action, reward, state_next, std, done])
                rewards.append(reward)

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

                self.cnt += 1
                if episode>1 and len(self.record.buffer)>self.batch_size:
                    if t%(round(1/self.y))==0:
                        if self.gradual_start(self.cnt, self.explore_time): # starts training gradualy globally
                            if self.type=="DDPG":
                                self.TD_secure()
                            elif self.type=="TD3":
                                self.TD_secure()
                            elif self.type=="SAC":
                                self.TD_secure()
                            elif self.type=="GAE":
                                self.TD_secure()

                if len(self.stack)>=self.batch_size and t%(int(self.batch_size/2)) == 0:
                    self.update_buffer()
                    if len(self.record.tape)>self.batch_size:
                        self.VNN_update()

            self.update_buffer()
            if len(self.record.tape)>self.batch_size:
                self.VNN_update()
            self.stack = []

            if episode>=10 and episode%10==0:
                self.save()

            score = sum(rewards)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f, | %f | %f | record size %d' % (episode, score, avg_score, self.y, std, len(self.record.buffer)))

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
                #self.env.render()
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
    env = gym.make('Pendulum-v0').env
    max_time_steps = 200
    actor_learning_rate = 0.001
    critic_learning_rate = 0.01
elif option == 2:
    env = gym.make('LunarLanderContinuous-v2').env
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 3:
    env = gym.make('BipedalWalker-v3').env
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 4:
    env = gym.make('HumanoidPyBulletEnv-v0').env
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 5:
    env = gym.make('HalfCheetahPyBulletEnv-v0').env
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
elif option == 6:
    env = gym.make('MountainCarContinuous-v0').env
    max_time_steps = 200
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 max_tape_size = 100000,
                 gamma = 0.97,
                 batch_size = 64, # batch size for training actor and critic networks
                 max_time_steps = max_time_steps,# no of time steps per epoch
                 n_steps = 100, # Q is calculated till n-steps, even after termination for correctness
                 discount_factor  = 0.97,
                 explore_time = 5000,
                 actor_learning_rate = actor_learning_rate,
                 critic_learning_rate = critic_learning_rate,
                 n_episodes = 1000000) # no of episodes to run

ddpg.train()
            #dist = tfd.Normal(loc=A, scale=std)
            #At = dist.sample(A.shape)
