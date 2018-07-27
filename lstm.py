import typing

from keras.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import CuDNNLSTM

import matplotlib.pyplot as plt
import numpy as np


class LossHistory(Callback):

    def __init__(self):
        super(Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.maes = []
        self.mapes = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.maes.append(logs.get('val_mean_squared_error'))
        self.mapes.append(logs.get('val_mean_absolute_percentage_error'))


def process_data_for_lstm(data: np.ndarray, num_timesteps: int, timesteps_ahead: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    The data needs to be adapted before being used to train the LSTM network.
    This function receives the data from the training and will produce two numpy arrays that follows this pattern:
    If for example we have this data:
    38
    12
    99
    41
    37
    33
    45
    88
    23

    It will be reorganized like that:
    41 37 33 45 88 23
    99 41 37 33 45 88
    12 99 41 37 33 45
    38 12 99 41 37 33

    Each has new row has the data from the column above and is of length num_timesteps.
    Each new row has the data more and more shifted to the right so that:
    first row has the data of [0:6]
    second row has the data of [1:7]
    etc...
    strongly influence by source code from https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter06/econs_stateful.py
    """
    x = np.zeros((data.shape[0] - num_timesteps - timesteps_ahead, num_timesteps, data.shape[1]))
    y = np.zeros((data.shape[0] - num_timesteps - timesteps_ahead, num_timesteps, 1))

    for i in range(len(data) - num_timesteps - timesteps_ahead):
        x[i] = data[i:i + num_timesteps]
        y[i] = data[i + timesteps_ahead:i + num_timesteps + timesteps_ahead, 0:1]

    return x, y


def divide_data_into_train_test(x: np.ndarray,
                                y: np.ndarray,
                                ratio: float,
                                batch_size: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    receives the training data as numpy arrays. Will divide the data as follow:
    training data: main data used for training. will be the ratio of x and y
    test data: used to validate while training the model: will be 1.0 - ratio
    """
    sp = int(ratio * len(x))

    xtrain, xtest, ytrain, ytest = x[0:sp], x[sp:], y[0:sp], y[sp:]
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    train_size = (xtrain.shape[0] // batch_size) * batch_size
    test_size = (xtest.shape[0] // batch_size) * batch_size

    xtrain, ytrain = xtrain[0:train_size], ytrain[0:train_size]
    xtest, ytest = xtest[0:test_size], ytest[0:test_size]

    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    return xtrain, xtest, ytrain, ytest


def create_model(units: int,
                 n_hidden: int,
                 num_timesteps: int,
                 batch_size: int,
                 optimizer: str,
                 loss: str) -> Sequential:
    """
    create and return the model
    """
    model = Sequential()
    model.add(CuDNNLSTM(n_hidden,
                        stateful=True,
                        input_shape=(num_timesteps, units),
                        batch_input_shape=(batch_size, num_timesteps, units),
                        return_sequences=True))
    model.add(Dropout(0.1))
    model.add(CuDNNLSTM(n_hidden, stateful=True, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_squared_error', 'mape'])
    print(model.summary())
    return model


def train_model(model: Sequential,
                num_epochs: int,
                batch_size: int,
                xtrain: np.ndarray,
                ytrain: np.ndarray,
                xtest: np.ndarray,
                ytest: np.ndarray,
                extraCallback=[]
                ):
    """
    train the model
    according to https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter06/econs_stateful.py
    we need to do a for loop and reset the state if we want to train a LSTM in a stateful way.
    """
    for i in range(num_epochs):
        print("Epoch {:d}/{:d}".format(i + 1, num_epochs))

        model.fit(xtrain,
                  ytrain,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(xtest, ytest),
                  shuffle=False,
                  callbacks=extraCallback)

        model.reset_states()


def load_trained_model() -> Sequential:
    """last trained model can be reloaded from disk by calling this function"""
    # load model from single file
    model = load_model('models/lstm.h5')
    return model


def try_prediction(data: np.ndarray, model: Sequential, batch_size: int):
    """
    receives data thgt wasn't used during training and the trained model
    do a prediction and returns the result.
    """
    prediction = data[0:batch_size]
    prediction = model.predict(prediction, batch_size=batch_size)
    return prediction[-1]


def show_prediction(prediction: np.ndarray, reality: np.ndarray, ticker='', file_name='plot.png'):
    """
    receives the predicted data and the expected reality
    plot them side by side.
    viewable in pycharm scientific mode
    :param prediction:
    :param reality:
    :return:
    """

    predicted_var = ((prediction[-1, 0] - prediction[1, 0]) / prediction[-1, 0]) * 100
    actual_var = ((reality[-1, 0] - reality[1, 0]) / reality[-1, 0]) * 100
    plt.close('all')
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(prediction[:, 0])
    ax.plot(reality[:, 0])
    plt.title(f'{ticker}')
    ax.legend(('prediction (' + str(round(predicted_var, 2)) + ')', 'reality (' + str(round(actual_var, 2)) + ')'))

    fig.savefig(file_name)
    plt.show()
    plt.close('all')
