


def build_network(input_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense
    # -----
    state = Input(shape=input_shape)  #input_shape （3,84,84）, state (?,3,84,84)
    h = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first')(state)
    h = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    adventage = Input(shape=(1,))
    train_network = Model(inputs=[state, adventage], outputs=[value, policy])

    return value_network, policy_network, train_network, adventage