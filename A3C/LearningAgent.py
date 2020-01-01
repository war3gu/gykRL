from collections import deque

import gym
import numpy as np
import argparse

from build_network import *


def policy_loss(adventage=0., beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        #下面第一行是目标的相反数，目标越大越好，相反数越小越好
        #第二行是减去y_pred的熵，熵越大，则loss越小。策略熵越大，有助于探索，有助于策略的稳定
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(adventage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss


# -----

class LearningAgent(object):
    def __init__(self, action_space, batch_size=32, screen=(84, 84), swap_freq=200, beta=0.01):
        from keras.optimizers import RMSprop
        # -----
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen
        self.batch_size = batch_size

        _, _, self.train_net, adventage = build_network(self.observation_shape, action_space.n)

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(adventage, beta)])

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        # ----- 下面的predict需要2个input，因为定义model的时候就是2个input。后面的self.unroll没有起任何作用，换成ignore也行
        ignore = np.zeros_like(self.unroll)
        values, policy = self.train_net.predict([last_observations, ignore])
        # -----
        self.targets.fill(0.)
        adventage = rewards - values.flatten()
        self.targets[self.unroll, actions] = 1.  #设置action的y_true
        # -----
        loss = self.train_net.train_on_batch([last_observations, adventage], [rewards, self.targets]) #多个outputs，需要多个y_true
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.pol_loss.append(loss[2]) #loss[0]是所有loss的和
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val), end='')
        # -----
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False

