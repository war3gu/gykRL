
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam



class Actor:
    def __init__(self, action_space, observation_space):
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        self.learning_rate = 0.001
        self.epsilon = 0.1  # exploration rate
        self.model = self._build_model()
        return

    def _build_model(self):
        state_obs = Input(shape=(self.observation_size,), name='state_obs')
        hidden = Dense(24, activation='relu')(state_obs)
        hidden = Dense(24, activation='relu')(hidden)
        action_value_out = Dense(self.action_size, activation='linear')(hidden)

        model = Model(inputs=[state_obs], outputs=[action_value_out])
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def predict(self, obs):  #具体的value
        act_values = self.model.predict(obs)
        return act_values

    def act(self, obs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), None, None
        act_values = self.predict(obs)
        return np.argmax(act_values[0]), None, None  # returns action

    def train(self, obs, value):
        self.model.fit(obs, value, epochs=1, verbose=0)

