import numpy as np
import pandas as pd
from sklearn import preprocessing
import typing


def generate_input_data(train_data: pd.DataFrame, input_str: str) -> typing.Tuple[np.ndarray, preprocessing.MinMaxScaler]:

    input_scaler = preprocessing.MinMaxScaler()
    input_data = train_data[input_str].values.astype(float)
    input_data = np.array(input_data).reshape((len(input_data), 1))
    input_data = input_scaler.fit_transform(input_data)
    return input_data, input_scaler


def prepare_training_data(train_data: pd.DataFrame, extra_input_list_str=[]) -> typing.Tuple[np.ndarray, preprocessing.MinMaxScaler]:
    """
    Receive the data for training the model.
    All the data will be normalized to a value between -1 and 1 to make it easier to train the model.
    The function returns the following values in a tuple:
    :return: data: a numpy array containing 3 columns: sentiment, stock value at closing time, free cash flow
    :return: *_scaler: the MinMaxScaler used to normalize the data. Used to transform the data back to normal
    """

    close = train_data['close'].values.astype(float)
    close = np.array(close).reshape((len(close), 1))
    close_scaler = preprocessing.MinMaxScaler()
    close = close_scaler.fit_transform(close)
    # close_p_scaler = preprocessing.MinMaxScaler()
    # close_p = (close[1:] - close[:-1]) / close[1:]
    # close_p = close_p_scaler.fit_transform(close_p)

    extra_input_list = [close[1:], ]

    if 'eps' in train_data.keys():
        peg_ratio_scaler = preprocessing.MinMaxScaler()
        peg_ratio = (train_data['close'] / train_data['eps'])
        peg_ratio = np.array(peg_ratio).reshape((len(peg_ratio), 1))
        peg_ratio = peg_ratio_scaler.fit_transform(peg_ratio)
        extra_input_list.append(peg_ratio[1:])

    if extra_input_list_str:
        for input in extra_input_list_str:
            extra_input, extra_scaler = generate_input_data(train_data, input)
            extra_input_list = extra_input_list + [extra_input[1:]]

    data_to_concat = tuple(extra_input_list)
    data = np.concatenate(data_to_concat, axis=1)

    return data, close_scaler


def scale_back_to_normal(data: np.ndarray, scaler: preprocessing.MinMaxScaler) -> np.ndarray:
    """
    receives data that was normalized as well as the MinMaxScaler used to do so.
    put the data back to normal and returns the result.
    """
    if len(data.shape) == 3:
        col = data[:, 0]
        col = col.reshape(len(col), 1)
        col = scaler.inverse_transform(col)
        return np.concatenate((col, data[:, 1:]), axis=1)
    else:
        return scaler.inverse_transform(data)
