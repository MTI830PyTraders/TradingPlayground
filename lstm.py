import typing

import download_data

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'
TICKER = 'TSLA'

NUM_EPOCHS = 12
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = 'adam'
N_HIDDEN = 112
NUM_TIMESTEPS = 180
TRAIN_TEST_RATIO = 0.75


def get_training_data(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    train_data = download_data.get_training_dataset(beginning_date, ending_date, ticker)

    min_max_scaler = preprocessing.MinMaxScaler()

    data = train_data[['Sentiment', 'close', 'fcf']].values.astype(float)
    data = min_max_scaler.fit_transform(data)
    return data


def process_data_for_lstm(data: pd.DataFrame, num_timesteps: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((data.shape[0], num_timesteps, data.shape[1]))
    Y = np.zeros((data.shape[0], num_timesteps, data.shape[1]))

    for i in range(len(data) - num_timesteps - 1):
        X[i] = data[i:i + num_timesteps]
        Y[i] = data[i + 1:i + num_timesteps + 1]

    return X, Y


def divide_data_into_train_test(X: np.ndarray, Y: np.ndarray, ratio: float, batch_size: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sp = int(ratio * len(X))
    Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    train_size = (Xtrain.shape[0] // batch_size) * batch_size
    test_size = (Xtest.shape[0] // batch_size) * batch_size
    Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
    Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

    return Xtrain, Xtest, Ytrain, Ytest


def create_model(units: int,
                 num_timesteps: int,
                 batch_size: int,
                 optimizer: str) -> Sequential:
    model = Sequential()
    model.add(LSTM(num_timesteps,
                   input_shape=(num_timesteps, units),
                   batch_input_shape=(batch_size, num_timesteps, units),
                   return_sequences=True))
    model.add(Dense(units))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

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
                  validation_data=(xtest, ytest))
        model.reset_states()


def main():
    data = get_training_data(BEGINNING_DATE, ENDING_DATE, TICKER)
    x, y = process_data_for_lstm(data, NUM_TIMESTEPS)
    xtrain, xtest, ytrain, ytest = divide_data_into_train_test(x, y, TRAIN_TEST_RATIO, BATCH_SIZE)
    model = create_model(data.shape[1], NUM_TIMESTEPS, BATCH_SIZE, OPTIMIZER)
    train_model(model, NUM_EPOCHS, BATCH_SIZE, xtrain, ytrain, xtest, ytest)


if __name__ == '__main__':
    main()
