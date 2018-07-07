import os
import typing

import download_data

from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers import LSTM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'
TICKER = 'TSLA'

NUM_EPOCHS = 25
BATCH_SIZE = 30
VERBOSE = 0
OPTIMIZER = 'adam'
N_HIDDEN = 20000
NUM_TIMESTEPS = 90
TRAIN_TEST_RATIO = 0.75
BACKTEST_RATIO = 0.1
LOSS = 'mean_squared_error'
SEED =1337
TIMESTEPS_AHEAD = 1
np.random.seed(SEED)


def get_training_data(beginning_date: str, ending_date: str, ticker: str) -> typing.Tuple[np.ndarray, preprocessing.MinMaxScaler, preprocessing.MinMaxScaler, preprocessing.MinMaxScaler]:
    train_data = download_data.get_training_dataset(beginning_date, ending_date, ticker)

    sentiments_scaler = preprocessing.MinMaxScaler()
    close_scaler = preprocessing.MinMaxScaler()
    fcf_scaler = preprocessing.MinMaxScaler()

    sentiments = train_data['Sentiment'].values.astype(float)
    sentiments = np.array(sentiments).reshape((len(sentiments), 1))
    close = train_data['close'].values.astype(float)
    close = np.array(close).reshape((len(close), 1))
    fcf = train_data['fcfps'].values.astype(float)
    fcf = np.array(fcf).reshape((len(fcf), 1))

    sentiments = sentiments_scaler.fit_transform(sentiments)
    close = close_scaler.fit_transform(close)
    fcf = fcf_scaler.fit_transform(fcf)
    data = np.concatenate((sentiments, close, fcf), axis=1)
    return data, sentiments_scaler, close_scaler, fcf_scaler


def process_data_for_lstm(data: pd.DataFrame, num_timesteps: int, timesteps_ahead: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((data.shape[0], num_timesteps, data.shape[1]))
    y = np.zeros((data.shape[0], num_timesteps, data.shape[1]))

    for i in range(len(data) - num_timesteps - timesteps_ahead):
        x[i] = data[i:i + num_timesteps]
        y[i] = data[i + timesteps_ahead:i + num_timesteps + timesteps_ahead]

    return x, y


def divide_data_into_train_test(x: np.ndarray,
                                y: np.ndarray,
                                ratio: float,
                                bt_ratio: float,
                                batch_size: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sp = int(ratio * len(x))
    bt = int(bt_ratio * len(x))

    xtrain, xtest, ytrain, ytest, xbtest, ybtest = x[0:sp], x[sp:sp+bt], y[0:sp], y[sp:sp+bt], x[:bt], y[:bt]
    print(xtrain.shape, xtest.shape, xbtest.shape, ytrain.shape, ytest.shape, ybtest.shape)

    train_size = (xtrain.shape[0] // batch_size) * batch_size
    test_size = (xtest.shape[0] // batch_size) * batch_size
    btest_size = (xbtest.shape[0] // batch_size) * batch_size

    xtrain, ytrain = xtrain[0:train_size], ytrain[0:train_size]
    xtest, ytest = xtest[0:test_size], ytest[0:test_size]
    xbtest, ybtest = xbtest[0:btest_size], ybtest[0:btest_size]

    print(xtrain.shape, xtest.shape, xbtest.shape, ytrain.shape, ytest.shape, ybtest.shape)

    return xtrain, xtest, ytrain, ytest, xbtest, ybtest


def create_model(units: int,
                 num_timesteps: int,
                 batch_size: int,
                 optimizer: str,
                 loss: str) -> Sequential:
    model = Sequential()
    model.add(LSTM(num_timesteps,
                   input_shape=(num_timesteps, units),
                   batch_input_shape=(batch_size, num_timesteps, units),
                   return_sequences=True))
    model.add(Dense(units))

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def train_model(model: Sequential,
                num_epochs: int,
                batch_size: int,
                xtrain: np.ndarray,
                ytrain: np.ndarray,
                xtest: np.ndarray,
                ytest: np.ndarray):
    for i in range(num_epochs):
        print("Epoch {:d}/{:d}".format(i + 1, num_epochs))
        model.fit(xtrain,
                  ytrain,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(xtest, ytest),
                  shuffle=False)
        model.reset_states()


def load_trained_model() -> Sequential:
    # load model from single file
    model = load_model('models/lstm.h5')
    return model


def try_prediction(data: np.ndarray, model: Sequential):
    prediction = data[0:BATCH_SIZE]
    #for i in range(0, NUM_TIMESTEPS):
    #    prediction = model.predict(prediction, batch_size=BATCH_SIZE)
    #    #prediction = prediction[-1]
    return prediction[-1]


def show_prediction(prediction: np.ndarray, reality: np.ndarray):
    plt.plot(prediction[:, 0])
    plt.plot(reality[:, 0])
    plt.show()
    plt.plot(prediction[:, 1])
    plt.plot(reality[:, 1])
    plt.show()
    plt.plot(prediction[:, 2])
    plt.plot(reality[:, 2])
    plt.show()


def scale_back_to_normal(data: np.ndarray,
                         sentiments_scaler: preprocessing.MinMaxScaler,
                         close_scaler: preprocessing.MinMaxScaler,
                         fcf_scaler: preprocessing.MinMaxScaler):
    sents = data[:, 0]
    sents = sents.reshape(len(sents), 1)
    sents = sentiments_scaler.inverse_transform(sents)
    close = data[:, 1]
    close = close.reshape(len(close), 1)
    close = close_scaler.inverse_transform(close)
    fcf = data[:, 2]
    fcf = fcf.reshape(len(fcf), 1)
    fcf = fcf_scaler.inverse_transform(fcf)
    return np.concatenate((sents, close, fcf), axis=1)


if __name__ == '__main__':
    data, sentiments_scaler, close_scaler, fcf_scaler = get_training_data(BEGINNING_DATE, ENDING_DATE, TICKER)
    x, y = process_data_for_lstm(data, NUM_TIMESTEPS, TIMESTEPS_AHEAD)
    xtrain, xtest, ytrain, ytest, xbtest, ybtest = divide_data_into_train_test(x, y, TRAIN_TEST_RATIO, BACKTEST_RATIO, BATCH_SIZE)
    #model = load_trained_model()
    model = create_model(data.shape[1], NUM_TIMESTEPS, BATCH_SIZE, OPTIMIZER, LOSS)
    train_model(model, NUM_EPOCHS, BATCH_SIZE, xtrain, ytrain, xtest, ytest)
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/lstm.h5')

    prediction = try_prediction(xbtest, model)
    prediction = scale_back_to_normal(prediction, sentiments_scaler, close_scaler, fcf_scaler)
    test_data = scale_back_to_normal(xbtest[BATCH_SIZE+NUM_TIMESTEPS], sentiments_scaler, close_scaler, fcf_scaler)
    show_prediction(prediction, test_data)
