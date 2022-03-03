
from keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model


def emotion_recognition(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2))(X)
    X = Flatten()(X)
    X = Dense(200, activation='relu')(X)
    X = Dropout(0.6)(X)
    X = Dense(7, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X)

    return model