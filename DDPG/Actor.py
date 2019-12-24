
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Add,Activation,Lambda
from keras.optimizers import Adam
import keras.backend as K

class Actor:
    def __init__(self, sess, action_space, observation_space):
        self.sess = sess
        self.action_size = action_space.shape[0]
        self.observation_size = observation_space.shape[0]
        self.lr = 0.002
        self.tau = 0.01
        self.action_bound = action_space.high

        self.mainModel, self.mainModel_weights, self.mainModel_state = self._build_model()
        self.targetModel, self.targetModel_weights, _ = self._build_model()

        self.action_gradient = Input(shape=(self.action_size, ))
        #self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])  # actor算出来的Q对action的梯度
        self.params_grad = tf.gradients(self.mainModel.output, self.mainModel_weights, -self.action_gradient)  # 综合action对actor参数的梯度得到Q对actor参数的梯度
        #self.action_gradient = Input(shape=(self.action_size,))
        #self.params_grad = K.gradients(self.mainModel.output, self.mainModel_weights, -self.action_gradient)

        grads = zip(self.params_grad, self.mainModel_weights)  # Q对actor参数梯度，actor参数
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)  # 将actor参数向Q增大方向移动




    def _build_model(self):
        input_obs = Input(shape=(self.observation_size,))
        h = Dense(64)(input_obs)
        h = Activation('relu')(h)
        # h = BatchNormalization()(h)
        h = Dense(64)(h)
        h = Activation('relu')(h)
        # h = BatchNormalization()(h)
        h = Dense(self.action_size)(h)
        h = Activation('tanh')(h)
        pred = Lambda(lambda h: h * self.action_bound)(h)
        model = Model(inputs=input_obs, outputs=pred)
        model.compile(optimizer='Adam', loss='categorical_crossentropy')
        return model, model.trainable_weights, input_obs

    def act(self, state, noise):
        act = self.mainModel.predict(state) + noise
        return act

    def predict(self, state):
        return self.mainModel.predict(state)

    def predict_target(self, state):
        return self.targetModel.predict(state)


    def update_target(self):
        wMain = self.mainModel.get_weights()
        wTarget = self.targetModel.get_weights()
        for i in range(len(wMain)):
            wTarget[i] = self.tau * wMain[i] + (1 - self.tau) * wTarget[i]
        self.targetModel.set_weights(wTarget)

    def train(self, state, action_grad):
        self.sess.run(self.optimize, feed_dict={self.mainModel_state: state, self.action_gradient: action_grad})

