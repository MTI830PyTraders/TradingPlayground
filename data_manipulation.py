import numpy as np
import pandas as pd

from sklearn import preprocessing
import typing


def prepare_training_data(train_data: pd.DataFrame) -> typing.Tuple[np.ndarray, preprocessing.MinMaxScaler]:
    """
    Receive the data for training the model.
    All the data will be normalized to a value between -1 and 1 to make it easier to train the model.
    The function returns the following values in a tuple:
    :return: data: a numpy array containing 3 columns: sentiment, stock value at closing time, free cash flow
    :return: *_scaler: the MinMaxScaler used to normalize the data. Used to transform the data back to normal
    """
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
