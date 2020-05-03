import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler


class AE(Model):

    def __init__(self,
                 n_input=None,
                 code_nodes=None,
                 summary=False,
                 _code_activation='relu',
                 _output_activation='sigmoid',
                 _inner_activation='tanh',
                 _red_factor=7.5,
                 *args,
                 **kwargs):

        # data = np.array(data)
        # data = __standardize__(data)
        self.input_nodes = n_input #data.shape[0]

        # either a number of code nodes or reduction factor can be specified
        self.code_nodes = __even__(self.input_nodes / _red_factor) if not code_nodes else code_nodes

        # number of layers -> how to define
        n_layers = 5

        a = np.log(self.code_nodes)
        b = np.log(self.input_nodes)

        encoder_layers = np.flip(np.logspace(a, b, n_layers, base=np.e))[1:-1]

        decoder_layers = np.logspace(a, b, n_layers, base=np.e)[1:-1]

        _LAYERS = []

        # GENERATE LAYERS
        self.input_layer = Input(shape=(self.input_nodes,))
        _LAYERS.append(self.input_layer)

        for n in encoder_layers:
            _LAYERS.append(
                Dense(__even__(n), activation = _inner_activation,)(_LAYERS[-1])
            )

        self.code_layer = Dense(self.code_nodes, activation = _code_activation)(_LAYERS[-1])
        _LAYERS.append(self.code_layer)

        for n in decoder_layers:
            _LAYERS.append(
                Dense(__even__(n), activation = _inner_activation,)(_LAYERS[-1])
            )

        output_layer = Dense(self.input_nodes, activation=_output_activation)(_LAYERS[-1])

        # SETUP MODEL
        super().__init__(self.input_layer, output_layer, *args, **kwargs)

        if summary:
            self.summary()

    # def compile_fit(self, ):
    #     self.

    def generate_encoder(self):
        encoder = Model(self.input_layer, self.code_layer)
        return encoder


def __standardize__(data):
    indexes = []
    for col in range(data.shape[1]):
        if min(data[:, col]) < 0 or max(data[:, col]) > 1:
            indexes.append(col)

    if len(indexes) > 0:
        to_scale = data[:, indexes]
        not_to_scale = np.delete(data, indexes, axis=1)
        scaled = MinMaxScaler().fit_transform(to_scale)

        minmaxdata = np.concatenate((scaled, not_to_scale), axis = 1)
        return minmaxdata
    else:
        return data


def __even__(f):
    return math.ceil(f / 2.) * 2

