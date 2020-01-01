
from scipy.misc import imresize
from skimage.color import rgb2gray
from collections import deque

import numpy as np

from build_network import *

class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84), n_step=8, discount=0.99):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 3
        self.observation_shape = (self.input_depth * self.past_range,) + self.screen

        self.value_net, self.policy_net, self.load_net, _ = build_network(self.observation_shape, action_space.n)

        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)   #一共储存3帧
        self.last_observations = np.zeros_like(self.observations)
        # -----
        self.n_step_observations = deque(maxlen=n_step)
        self.n_step_actions = deque(maxlen=n_step)
        self.n_step_rewards = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= args.reward_scale
        # -----
        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(self.counter): #按照时间上从后向前的顺序计算Q值并上传给主线程
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    def choose_action(self):
        sss = self.observations[None, ...]
        policy = self.policy_net.predict(sss)[0]
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0) #将最旧的一帧调到最后
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation) #最后一帧被新内容覆盖

    def transform_screen(self, data):
        img = imresize(data, self.screen)
        gray = rgb2gray(img)
        return gray[None, ...]

