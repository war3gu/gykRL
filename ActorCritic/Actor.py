

import numpy as np

from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam

def actor_critic_loss(state_value, state_pre_value, action_ytrue):

    state_adv_value = state_pre_value - state_value

    print(state_value.shape)
    print(state_pre_value.shape)
    print(action_ytrue.shape)

    def loss(oldPre, newPre):
        prob = K.sum(action_ytrue * newPre, axis=-1, keepdims=True)

        prob = -K.log(prob)

        prob = prob*state_adv_value

        prob = K.sum(prob)

        return prob

    return loss


class Actor:
    def __init__(self, action_space, observation_space):
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        #self.gamma = 0.99
        self.model = self._build_model()

        self.DUMMY_STATE_VALUE = np.zeros((1, 1))
        self.DUMMY_STATE_PRE_VALUE = np.zeros((1, 1))
        self.DUMMY_ACTION_YTRUE = np.zeros((1, self.action_size))

    def _build_model(self):
        state_obs = Input(shape=(self.observation_size,), name='state_obs')
        state_value = Input(shape=(1,), name='state_value')
        state_pre_value = Input(shape=(1,), name='state_pre_value')
        action_ytrue = Input(shape=(self.action_size,), name='action_ytrue')

        hidden = Dense(256, activation='relu')(state_obs)
        hidden = Dense(256, activation='relu')(hidden)

        action_out = Dense(self.action_size, activation='softmax', name='action_out')(hidden)

        model = Model(inputs=[state_obs, state_value, state_pre_value, action_ytrue], outputs=[action_out])

        model.compile(optimizer=Adam(lr=5e-4),loss=[actor_critic_loss(state_value=state_value, state_pre_value=state_pre_value, action_ytrue=action_ytrue)])

        return model

    def predict(self, obs):
        p = self.model.predict([obs, self.DUMMY_STATE_VALUE, self.DUMMY_STATE_PRE_VALUE, self.DUMMY_ACTION_YTRUE])
        action = np.random.choice(self.action_size, p=p[0])
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1

        return action, action_matrix

    def train(self, obs, value, value_pre, ytrue):
        old_prediction = np.zeros((obs.shape[0], self.action_size))
        return self.model.fit([obs, value, value_pre, ytrue], [old_prediction])