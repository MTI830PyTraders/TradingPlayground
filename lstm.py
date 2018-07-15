import os
import typing

import download_data

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, CuDNNLSTM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Parameters of what data to select
BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'
TICKER = 'MSFT'

# These parameters will tweak the model
BATCH_SIZE = 30
LOSS = 'mae'
N_HIDDEN = 1000
NUM_EPOCHS = 20
NUM_TIMESTEPS = 90
OPTIMIZER = 'adam'
TIMESTEPS_AHEAD = 1
VERBOSE = 0

# percentage of the data that will be used to train the model.
# the test ration is the ratio below minus the backtest ratio.
# so for example: 1.0 - 0.75 - 0.1 = 0.15 for test data
TRAIN_TEST_RATIO = 0.8

# setting the random seed allow to make the experiment reproductible
SEED = 1337
np.random.seed(SEED)


def get_training_data(beginning_date: str, ending_date: str, ticker: str) -> typing.Tuple[np.ndarray, preprocessing.MinMaxScaler]:
    """
    Select the data for training the model.
    All the data will be normalized to a value between -1 and 1 to make it easier to train the model.
    The function returns the following values in a tuple:
    :return: data: a numpy array containing 3 columns: sentiment, stock value at closing time, free cash flow
    :return: *_scaler: the MinMaxScalers used to normalize the data. Used to transform the data back to normal
    """
    train_data = download_data.get_training_dataset(beginning_date, ending_date, ticker)

    sentiments_scaler = preprocessing.MinMaxScaler()
    close_scaler = preprocessing.MinMaxScaler()
    close_p_scaler = preprocessing.MinMaxScaler()
    fcf_scaler = preprocessing.MinMaxScaler()

    sentiments = train_data['Sentiment'].values.astype(float)
    sentiments = np.array(sentiments).reshape((len(sentiments), 1))
    sentiments = sentiments_scaler.fit_transform(sentiments)

    close = train_data['close'].values.astype(float)
    close = np.array(close).reshape((len(close), 1))

    close_p = (close[1:] - close[:-1]) / close[1:]
    close_p = close_p_scaler.fit_transform(close_p)

    close = close_scaler.fit_transform(close)

    fcf = train_data['fcf'].values.astype(float)
    fcf = np.array(fcf).reshape((len(fcf), 1))
    fcf = fcf_scaler.fit_transform(fcf)

    data = np.concatenate((close[1:], close_p, sentiments[1:], fcf[1:]), axis=1)
    return data, close_scaler


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
    x = np.zeros((data.shape[0], num_timesteps, data.shape[1]))
    y = np.zeros((data.shape[0], num_timesteps, data.shape[1]))

    for i in range(len(data) - num_timesteps - timesteps_ahead):
        x[i] = data[i:i + num_timesteps]
        y[i] = data[i + timesteps_ahead:i + num_timesteps + timesteps_ahead]

    return x, y


def divide_data_into_train_test(x: np.ndarray,
                                y: np.ndarray,
                                ratio: float,
                                batch_size: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    receives the training data as numpy arrays. Will divide the data as follow:
    training data: main data used for training. will be the ratio of x and y
    test data: used to validate while training the model: will be 1.0 - ratio - bt_ratio
    backtest data: used to do a prediction once training is finished. will be bt_ratio of the data
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
                 num_timesteps: int,
                 batch_size: int,
                 optimizer: str,
                 loss: str) -> Sequential:
    """
    create and return the model
    """
    model = Sequential()
    model.add(CuDNNLSTM(N_HIDDEN,
                        input_shape=(num_timesteps, units),
                        batch_input_shape=(batch_size, num_timesteps, units),
                        return_sequences=True))
    model.add(Dropout(0.1))
    model.add(CuDNNLSTM(N_HIDDEN, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(units))
    model.add(Activation('linear'))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def train_model(model: Sequential,
                num_epochs: int,
                batch_size: int,
                xtrain: np.ndarray,
                ytrain: np.ndarray,
                xtest: np.ndarray,
                ytest: np.ndarray):
    """
    train the model
    according to https://github.com/PacktPublishing/Deep-Learning-with-Keras/blob/master/Chapter06/econs_stateful.py
    we need to do a for loop and reset the state if we want to train a LSTM in a stateful way.
    """
    for i in range(1):
        print("Epoch {:d}/{:d}".format(i + 1, num_epochs))
        model.fit(xtrain,
                  ytrain,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(xtest, ytest),
                  shuffle=False)
        model.reset_states()


def load_trained_model() -> Sequential:
    """last trained model can be reloaded from disk by calling this function"""
    # load model from single file
    model = load_model('models/lstm.h5')
    return model


def try_prediction(data: np.ndarray, model: Sequential):
    """
    receives data thgt wasn't used during training and the trained model
    do a prediction and returns the result.
    """
    prediction = data[0:BATCH_SIZE]
    prediction = model.predict(prediction, batch_size=BATCH_SIZE)
    return prediction[-1]


def show_prediction(prediction: np.ndarray, reality: np.ndarray):
    """
    receives the predicted data and the expected reality
    plot them side by side.
    viewable in pycharm scientific mode
    :param prediction:
    :param reality:
    :return:
    """
    print('Predicted variation:', ((prediction[-1, 0] - prediction[1, 0]) / prediction[-1, 0]) * 100)
    print('Actual variation:', ((reality[-1, 0] - reality[1, 0]) / reality[-1, 0]) * 100)

    print('Expected sum value: ', sum(prediction[:, 0]))
    print('Real sum value: ', sum(reality[:, 0]))

    plt.plot(prediction[:, 0], label='prediction')
    plt.plot(reality[:, 0], label='reality')
    plt.legend(('prediction', 'reality'))
    plt.show()


def scale_back_to_normal(data: np.ndarray, scaler: preprocessing.MinMaxScaler) -> np.ndarray:
    """
    receives data that was normalized as well as the MinMaxScalers used to do so.
    put the data back to normal and returns the result.
    """
    if len(data.shape) == 3:
        col = data[:, 0]
        col = col.reshape(len(col), 1)
        col = scaler.inverse_transform(col)
        return np.concatenate((col, data[:, 1:]), axis=1)
    else:
        return scaler.inverse_transform(data)


if __name__ == '__main__':
    # actual code is done here. It is not wrapped in a function so that variables can be checked in a python console.
    data, scaler = get_training_data(BEGINNING_DATE, ENDING_DATE, TICKER)
    x, y = process_data_for_lstm(data, NUM_TIMESTEPS, TIMESTEPS_AHEAD)
    xtrain, xtest, ytrain, ytest = divide_data_into_train_test(x, y, TRAIN_TEST_RATIO, BATCH_SIZE)
    #model = load_trained_model()
    model = create_model(data.shape[1], NUM_TIMESTEPS, BATCH_SIZE, OPTIMIZER, LOSS)
    train_model(model, NUM_EPOCHS, BATCH_SIZE, xtrain, ytrain, xtest, ytest)
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/lstm.h5')

    prediction = try_prediction(xtest, model)
    prediction = scale_back_to_normal(prediction, scaler)
    test_data = scale_back_to_normal(ytest[0], scaler)
    show_prediction(prediction, test_data)
