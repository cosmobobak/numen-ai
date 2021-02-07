import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import name
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, SimpleRNN, MaxPooling3D, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model

BATCH_SIZE = 64

class NetMaker:
    def __init__(self) -> None:
        inputLayer = Input(
            shape=(10, 8, 8), batch_size=BATCH_SIZE, name="Input")
        #################################################################
        ##################### CONVOLUTIONAL BLOCK #######################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="FrontConv")(inputLayer)
        a = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="FrontNorm")(x)
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv1")(x)
        b = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm1")(x)
        x = concatenate([a, b])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv2")(x)
        c = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm2")(x)
        x = concatenate([b, c])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv3")(x)
        d = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm3")(x)
        x = concatenate([c, d])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv4")(x)
        e = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm4")(x)
        x = concatenate([d, e])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv5")(x)
        f = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm5")(x)
        x = concatenate([e, f])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv6")(x)
        b = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm6")(x)
        x = concatenate([f, b])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv7")(x)
        c = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm7")(x)
        x = concatenate([b, c])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv8")(x)
        d = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm8")(x)
        x = concatenate([c, d])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv9")(x)
        e = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm9")(x)
        x = concatenate([d, e])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv10")(x)
        f = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm10")(x)
        x = concatenate([e, f])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv11")(x)
        b = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm11")(x)
        x = concatenate([f, b])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv12")(x)
        c = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm12")(x)
        x = concatenate([b, c])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv13")(x)
        d = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm13")(x)
        x = concatenate([c, d])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv14")(x)
        e = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm14")(x)
        x = concatenate([d, e])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv15")(x)
        f = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm15")(x)
        x = concatenate([e, f])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv16")(x)
        b = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm16")(x)
        x = concatenate([f, b])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv17")(x)
        c = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm17")(x)
        x = concatenate([b, c])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv18")(x)
        d = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm18")(x)
        x = concatenate([c, d])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv19")(x)
        e = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm19")(x)
        x = concatenate([d, e])
        #################################################################
        ##################### RESIDUAL BLOCK ############################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="ResConv20")(x)
        f = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="ResNorm20")(x)
        x = concatenate([e, f])
        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(x)
        x = Dense(256, activation="relu", name="Dense1")(x)
        outputLayer = Dense(1, name="eval", activation="tanh")(x)

        self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

        self.evalModel.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel


class NetMaker2:
    def __init__(self) -> None:
        inputLayer = Input(
            shape=(10, 8, 8), batch_size=BATCH_SIZE, name="Input")
        #################################################################
        ##################### CONVOLUTIONAL BLOCK #######################
        #################################################################
        x = Conv2D(64, kernel_size=(1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="FrontConv")(inputLayer)
        x = BatchNormalization(axis=-1, center=False,
                               scale=False, epsilon=1e-5, name="FrontNorm")(x)
        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(x)
        x = Dense(512, activation="relu", name="Dense1")(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dense(64, activation="relu", name="Dense3")(x)
        outputLayer = Dense(1, name="eval", activation="tanh")(x)

        self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

        self.evalModel.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel

if __name__ == "__main__":
    n = NetMaker()
