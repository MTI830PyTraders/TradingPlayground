import download_data
import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM

BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'
TICKER = 'TSLA'

train_data = download_data.get_training_dataset(BEGINNING_DATE, ENDING_DATE, TICKER)

NUM_EPOCHS = 12
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = 'adam'
N_HIDDEN = 112
NUM_TIMESTEPS = 180

min_max_scaler = preprocessing.MinMaxScaler()

data = train_data[['Sentiment', 'close', 'fcf']].values.astype(float)
data = min_max_scaler.fit_transform(data)

X = np.zeros((data.shape[0], NUM_TIMESTEPS, data.shape[1]))
Y = np.zeros((data.shape[0], NUM_TIMESTEPS, data.shape[1]))


for i in range(len(train_data) - NUM_TIMESTEPS - 1):
    X[i] = data[i:i + NUM_TIMESTEPS]
    Y[i] = data[i + 1:i + NUM_TIMESTEPS + 1]

sp = int(0.75 * len(train_data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(LSTM(NUM_TIMESTEPS,
               input_shape=(NUM_TIMESTEPS, data.shape[1]),
               batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, data.shape[1]),
               return_sequences=True))
model.add(Dense(data.shape[1]))

model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)

train_size = (Xtrain.shape[0] // BATCH_SIZE) * BATCH_SIZE
test_size = (Xtest.shape[0] // BATCH_SIZE) * BATCH_SIZE
Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
for i in range(NUM_EPOCHS):
    print("Epoch {:d}/{:d}".format(i + 1, NUM_EPOCHS))
    model.fit(Xtrain,
              Ytrain,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(Xtest, Ytest))
    model.reset_states()

