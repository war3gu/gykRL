
import numpy as np

from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam



class Critic:
    def __init__(self, action_space, observation_space):
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        self.gamma = 0.99

        self.model = self._build_model()
        return

    def _build_model(self):
        state_obs = Input(shape=(self.observation_size, ), name='state_obs')

        hidden = Dense(256, activation='relu')(state_obs)
        hidden = Dense(256, activation='relu')(hidden)

        state_value = Dense(1,kernel_initializer='random_uniform')(hidden)

        model = Model(inputs=[state_obs], outputs=[state_value])

        model.compile(optimizer=Adam(lr=1e-4),loss='mean_squared_error')

        return model

    def predict(self, obs):
        state_value = self.model.predict(obs)
        return state_value

    def train(self, obs, state_next_value):
        return self.model.fit([obs], [state_next_value])