
import numpy as np


from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam



def policy_gradient_loss(state_value, action_ytrue):

    print(state_value.shape)
    print(action_ytrue.shape)

    def loss(oldPre, newPre):
        prob = K.sum(action_ytrue * newPre, axis=-1, keepdims=True)

        prob = -K.log(prob)

        prob = prob*state_value

        prob = K.sum(prob)

        return prob

    return loss



class Actor:
    def __init__(self, action_space, observation_space):
        self.action_size = action_space.n
        self.observation_size = observation_space.shape[0]
        self.model = self._build_model()
        #self.gamma = 0.99


        self.DUMMY_STATE_VALUE = np.zeros((1, 1))
        self.DUMMY_ACTION_YTRUE = np.zeros((1, self.action_size))

        print(self.action_size)
        print(self.observation_size)

    def _build_model(self):
        state_obs = Input(shape=(self.observation_size,), name='state_obs')
        state_value = Input(shape=(1,), name='state_value')
        action_ytrue = Input(shape=(self.action_size,), name='action_ytrue')

        hidden = Dense(256, activation='relu')(state_obs)
        hidden = Dense(256, activation='relu')(hidden)

        action_out = Dense(self.action_size, activation='softmax', name='output')(hidden)

        model = Model(inputs=[state_obs, state_value, action_ytrue], outputs=[action_out])


        model.compile(optimizer=Adam(lr=10e-4),loss=[policy_gradient_loss(state_value=state_value, action_ytrue=action_ytrue)])

        return model

    def predict(self, obs):     #obs应该在外面reshape
        obs1 = obs.reshape(1, self.observation_size)
        p = self.model.predict([obs1, self.DUMMY_STATE_VALUE, self.DUMMY_ACTION_YTRUE])
        action = np.random.choice(self.action_size, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1

        return action, action_matrix

    def train(self, obs, value, ytrue):
        old_prediction = np.zeros((obs.shape[0], self.action_size))
        self.model.fit([obs, value, ytrue], [old_prediction])

