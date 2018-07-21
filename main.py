import data_manipulation
import lstm
import numpy as np
import os
import xarray as xr
import math
from keras.optimizers import Adam

# Parameters of what data to select
BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-06-31'
TICKER = 'MSFT'

# These parameters will tweak the model
BATCH_SIZE = 90
LOSS = 'mae'
N_HIDDEN = 1000
NUM_EPOCHS = 100
SAVE_EVERY = 10
NUM_TIMESTEPS = 180
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
TIMESTEPS_AHEAD = 90
VERBOSE = 0

# percentage of the data that will be used to train the model.
TRAIN_TEST_RATIO = 0.8

# setting the random seed allow to make the experiment reproductible
SEED = 1337
np.random.seed(SEED)

# actual code is done here. It is not wrapped in a function so that variables can be checked in a python console.

final_xr = xr.open_dataset("final_xr.nc", chunks=30)
ds = final_xr.sel(ticker=TICKER)
train_data = xr.Dataset({'Sentiment': ds.sentiment, 'close': ds['close'], 'fcf': ds.fcf}).to_dataframe().interpolate(limit_direction='both')

data, scaler = data_manipulation.prepare_training_data(train_data)

if not os.path.exists('plots'):
    os.mkdir('plots')

x, y = lstm.process_data_for_lstm(data, NUM_TIMESTEPS, TIMESTEPS_AHEAD)
xtrain, xtest, ytrain, ytest = lstm.divide_data_into_train_test(x, y, TRAIN_TEST_RATIO, BATCH_SIZE)
#model = load_trained_model()
model = lstm.create_model(data.shape[1], N_HIDDEN, NUM_TIMESTEPS, BATCH_SIZE, OPTIMIZER, LOSS)

for i in range(NUM_EPOCHS // SAVE_EVERY):
    lstm.train_model(model, SAVE_EVERY, BATCH_SIZE, xtrain, ytrain, xtest, ytest)
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/lstm.h5')

    prediction = lstm.try_prediction(xtest, model, BATCH_SIZE)
    prediction = lstm.scale_back_to_normal(prediction, scaler)
    test_data = lstm.scale_back_to_normal(ytest[BATCH_SIZE], scaler)
    lstm.show_prediction(prediction, test_data, f'plots/{TICKER}_{i}.png')

score, _ = model.evaluate(xtest, ytest, batch_size=BATCH_SIZE)
rmse = math.sqrt(score)
print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))
