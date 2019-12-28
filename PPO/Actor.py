
import numpy as np

from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
from keras.optimizers import Adam


from NoisyDense import NoisyDense


def ppo_loss(state_value, state_next_value, action_previous_ytrue):

    state_adv_value = state_next_value - state_value

    print(state_value.shape)
    print(state_next_value.shape)
    print(action_previous_ytrue.shape)

    def loss(action_previous_prob, action_new_prob):
        prob_new = K.sum(action_previous_ytrue * action_new_prob, axis=-1, keepdims=True)

        prob_previous = K.sum(action_previous_ytrue * action_previous_prob, axis=-1, keepdims=True)

        prob = prob_new/(prob_previous + 1e-10)

        m1 = prob * state_adv_value
        m2 = K.clip(prob, min_value=0.8, max_value=1.2) * state_adv_value
        m3 = -K.sum(K.minimum(m1, m2))

        return m3

    return loss


class Actor:
    def __init__(self, action_space, observation_space):
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        self.model = self._build_model()

        self.DUMMY_STATE_VALUE = np.zeros((1, 1))
        self.DUMMY_STATE_NEXT_VALUE = np.zeros((1,1))
        self.DUMMY_ACTION_PREVIOUS_YTRUE = np.zeros((1, self.action_size))

        return

    def _build_model(self):
        state_obs = Input(shape=(self.observation_size,), name='state_obs')
        state_value = Input(shape=(1,), name='state_value')
        state_next_value = Input(shape=(1,), name='state_next_value')
        action_previous_ytrue = Input(shape=(self.action_size,), name='action_previous_ytrue')
        #action_previous_prob = Input(shape=(self.action_size,), name='action_previous_prob')

        hidden = Dense(256, activation='relu')(state_obs)
        hidden = Dense(256, activation='relu')(hidden)

        action_out = NoisyDense(self.action_size, activation='softmax', sigma_init=0.1, name='action_out')(hidden)

        model = Model(inputs=[state_obs, state_value, state_next_value, action_previous_ytrue], outputs=[action_out])

        model.compile(optimizer=Adam(lr=1e-4),loss=[ppo_loss(state_value=state_value,
                                                             state_next_value=state_next_value,
                                                             action_previous_ytrue=action_previous_ytrue)])

        return model

    def predict(self, obs):
        p = self.model.predict([obs, self.DUMMY_STATE_VALUE, self.DUMMY_STATE_NEXT_VALUE, self.DUMMY_ACTION_PREVIOUS_YTRUE])
        action = np.random.choice(self.action_size, p=p[0])
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1

        return action, action_matrix, p[0]

    def train(self, obs, state_value, state_next_value, action_previous_ytrue, action_previous_prob):
        return self.model.fit([obs, state_value, state_next_value, action_previous_ytrue], [action_previous_prob])