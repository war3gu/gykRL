
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Add,Activation,Lambda
from keras.optimizers import Adam
import keras.backend as K


class Critic:
    def __init__(self, sess, action_space, observation_space):
        self.sess = sess
        self.action_size = action_space.shape[0]
        self.observation_size = observation_space.shape[0]
        self.lr = 0.002
        self.tau = 0.01
        self.mainModel, self.state, self.actions = self._build_model()
        self.targetModel, _, _ = self._build_model()
        self.action_grads = tf.gradients(self.mainModel.output, self.actions)
        # self.funcAction_grads = K.function([self.state, self.actions], [self.action_grads, self.mainModel.output])#尝试使用function



    def _build_model(self):
        input_obs = Input(shape=(self.observation_size,))
        input_actions = Input(shape=(self.action_size,))
        h = Dense(64)(input_obs)
        h = Activation('relu')(h)
        # h = BatchNormalization()(h)
        temp1 = Dense(64)(h)
        action_abs = Dense(64)(input_actions)
        # action_abs = Activation('relu')(action_abs)
        # action_abs = BatchNormalization()(action_abs)
        h = Add()([temp1, action_abs])
        # h = Dense(64)(h)
        h = Activation('relu')(h)
        # h = BatchNormalization()(h)
        pred = Dense(1, kernel_initializer='random_uniform')(h)
        model = Model(inputs=[input_obs, input_actions], outputs=pred)
        model.compile(optimizer='Adam', loss='mean_squared_error')
        return model, input_obs, input_actions

    def action_gradients(self, states, actions):
        # x1, x2 = self.funcAction_grads([states, actions])
        # return x1[0]
        r = self.sess.run(self.action_grads, feed_dict={self.state: states, self.actions: actions})

        return r[0]

    def predict(self, state, actions):
        x = np.ndarray((actions.shape[1], self.action_size))
        for j in range(actions.shape[1]):
            x[j] = np.concatenate([y[j] for y in actions])
        return self.mainModel.predict([state, x])

    def predict_target(self, state, actions):
        return self.targetModel.predict([state, actions])

    def update_target(self):
        wMain = self.mainModel.get_weights()
        wTarget = self.targetModel.get_weights()
        for i in range(len(wMain)):
            wTarget[i] = self.tau * wMain[i] + (1 - self.tau) * wTarget[i]
        self.targetModel.set_weights(wTarget)

    def train(self, state, actions, labels):
        self.mainModel.train_on_batch([state, actions], labels)