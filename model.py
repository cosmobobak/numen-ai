
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Input, concatenate, Reshape, MaxPooling2D, Add
import os

from tensorflow.python.keras.layers.core import Dropout


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from hyperparams import BATCH_SIZE
from tensorflow.keras.models import Model
from tensorflow import keras

def residual_block(x):
    _a = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
               data_format='channels_last', activation="relu", kernel_initializer='he_uniform')(x)
    a = Add()([_a, x])
    _b = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                data_format='channels_last', activation="relu", kernel_initializer='he_uniform')(a)
    b = Add()([a, _b])
    _c = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                data_format='channels_last', activation="relu", kernel_initializer='he_uniform')(b)
    return Add()([b, _c])

class ChessNet:
    def __init__(self, dims=(8, 8, 11), xbatch_size=BATCH_SIZE) -> None:
        print(xbatch_size)
        inputLayer = Input(shape=dims, batch_size=xbatch_size, name="input")
        x = inputLayer
        #################################################################
        ##################### CONVOLUTIONAL BLOCK #######################
        #################################################################
        a = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv1", activation="relu", kernel_initializer='he_uniform')(x)
        _b = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv2", activation="relu", kernel_initializer='he_uniform')(a)
        b = Add()([a, _b])
        _c = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv3", activation="relu", kernel_initializer='he_uniform')(b)
        c = Add()([b, _c])
        x = c
        #x = residual_block(x)
        #x = residual_block(x)
        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(512, activation="relu", name="Dense1")(x)
        x = Dropout(0.1)(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation="relu", name="Dense3")(x)
        outputLayer = Dense(1, name="eval", activation="tanh")(x)

        self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

        self.evalModel.compile(
            optimizer=keras.optimizers.SGD(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel


class SimpleChessNet:
    def __init__(self, dims=(8, 8, 11), xbatch_size=BATCH_SIZE) -> None:
        print(xbatch_size)
        inputLayer = Input(shape=dims, batch_size=xbatch_size, name="input")
        x = inputLayer
        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(x)
        x = Dense(512, activation="relu", name="Dense1")(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dense(128, activation="relu", name="Dense3")(x)
        
        outputLayer = Dense(1, name="eval", activation="tanh")(x)
        self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

        self.evalModel.compile(
            optimizer=keras.optimizers.SGD(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()],
        )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel

class MainCNN:
    def __init__(self, dims=(10, 8, 8), eval_model=True, out_dims=7, xbatch_size=BATCH_SIZE) -> None:
        print(xbatch_size)
        rawInputLayer = Input(shape=42, batch_size=xbatch_size, name="Input")
        inputLayer = Reshape(dims, input_shape=(42,))(rawInputLayer)
        #################################################################
        ##################### CONVOLUTIONAL BLOCK #######################
        #################################################################
        a = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv1", activation="relu", kernel_initializer='he_uniform')(inputLayer)
        b = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv2", activation="relu", kernel_initializer='he_uniform')(a)
        b = Add()([a, b])
        c = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv3", activation="relu", kernel_initializer='he_uniform')(b)
        c = Add()([b, c])
        x = MaxPooling2D((2, 2), padding="same", name="pool1")(c)
        a = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv4", activation="relu", kernel_initializer='he_uniform')(x)
        b = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv5", activation="relu", kernel_initializer='he_uniform')(a)
        b = Add()([a, b])
        c = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv6", activation="relu", kernel_initializer='he_uniform')(b)
        c = Add()([b, c])
        x = MaxPooling2D((2, 2), padding="same", name="pool2")(c)
        a = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv7", activation="relu", kernel_initializer='he_uniform')(x)
        b = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv8", activation="relu", kernel_initializer='he_uniform')(a)
        b = Add()([a, b])
        c = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding="same",
                   data_format='channels_last', name="Conv9", activation="relu", kernel_initializer='he_uniform')(b)
        c = Add()([b, c])
        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(c)
        x = Dense(512, activation="relu", name="Dense1")(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dense(128, activation="relu", name="Dense3")(x)
        if eval_model:
            outputLayer = Dense(3, name="eval", activation="softmax")(x)
            self.evalModel = Model(inputs=rawInputLayer, outputs=outputLayer)

            self.evalModel.compile(
                optimizer=keras.optimizers.SGD(),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()],
            )
        else:
            outputLayer = Dense(out_dims, name="move", activation="softmax")(x)
            self.evalModel = Model(inputs=rawInputLayer, outputs=outputLayer)

            self.evalModel.compile(
                optimizer=keras.optimizers.SGD(),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()],
            )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel

class MLPMaker:
    def __init__(self, dims=(10, 8, 8), eval_model=True, out_dims=7, xbatch_size=BATCH_SIZE) -> None:
        rawInputLayer = Input(shape=42, batch_size=xbatch_size, name="input")
        inputLayer = Reshape(dims, input_shape=(42,))(rawInputLayer)

        #################################################################
        ##################### FULLY CONNECTED OUT #######################
        #################################################################
        x = Flatten()(inputLayer)
        x = Dense(2048, activation="relu", name="Dense0")(x)
        x = Dense(1024, activation="relu", name="Dense1")(x)
        x = Dense(256, activation="relu", name="Dense2")(x)
        x = Dense(128, activation="relu", name="Dense3")(x)
        x = Dense(64, activation="relu", name="Dense4")(x)
        if eval_model:
            outputLayer = Dense(1, name="eval", activation="tanh")(x)
            self.evalModel = Model(inputs=inputLayer, outputs=outputLayer)

            self.evalModel.compile(
                optimizer=keras.optimizers.SGD(),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()],
            )
        else:
            outputLayer = Dense(out_dims, name="move", activation="softmax")(x)
            self.evalModel = Model(inputs=rawInputLayer, outputs=outputLayer)

            self.evalModel.compile(
                optimizer=keras.optimizers.SGD(),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()],
            )

        self.evalModel.summary()

    def __call__(self) -> Model:
        return self.evalModel

#  img_height = 256
#     img_width = 256
#     img_channels = 1

#     input_shape = (img_height, img_width, img_channels)
#     img_input = k.Input(shape=input_shape)
#     conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_input)
#     conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
