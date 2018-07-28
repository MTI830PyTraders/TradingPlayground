import matplotlib
# matplotlib.use('Agg')
import data_manipulation
import lstm
import numpy as np
import os
import xarray as xr
import keras
import math
from keras.optimizers import Adam

import matplotlib.pyplot as plt
# plt.ioff()
from keras.callbacks import CSVLogger
import pdb

# Parameters of what data to select
BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-06-31'


# These parameters will tweak the model
BATCH_SIZE = 90
LOSS = 'mae'
N_HIDDEN = 1000
NUM_EPOCHS = 100
SAVE_EVERY = 10
NUM_TIMESTEPS = 180
OPTIMIZER = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# OPTIMIZER = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

TIMESTEPS_AHEAD = 90
VERBOSE = 0
CONFIG_ID = "1"

if not os.path.exists('RESULTS'):
    os.mkdir('RESULTS')
if not os.path.exists(f'RESULTS/{CONFIG_ID}'):
    os.mkdir(f'RESULTS/{CONFIG_ID}')
csv_logger = CSVLogger(f'RESULTS/{CONFIG_ID}/keras.log.csv', append=True, separator=';')
import logging
# fh = logging.FileHandler('spam.log')
# os.mknod("RESULTS" + str(CONFIG_ID) + "logfile.txt")
logging.basicConfig(filename=f'RESULTS/{CONFIG_ID}/logfile.txt', filemode='w')
stderrLogger = logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)


# percentage of the data that will be used to train the model.
TRAIN_TEST_RATIO = 0.8

# setting the random seed allow to make the experiment reproductible
SEED = 1337
np.random.seed(SEED)

# actual code is done here. It is not wrapped in a function so that variables can be checked in a python console.


base_input = ['close']
extra_input = ['sentiment', 'pe']
input_list = base_input + extra_input

final_xr = xr.open_dataset("final_xr.nc", chunks=30)[input_list]

tickers = final_xr.get('ticker').values
model = None

tickers = ['MCD', 'DIS', 'INTC', 'AAPL', 'MSFT', 'WDC']


totalEpoch = 1
history = lstm.LossHistory()

for ticker in tickers:
    ds = final_xr.sel(ticker=ticker)
    train_data = ds.to_dataframe().interpolate(limit_direction='both')
    data, scaler = data_manipulation.prepare_training_data(train_data, extra_input_list_str=extra_input)

    if not model:
        model = lstm.create_model(data.shape[1], N_HIDDEN, NUM_TIMESTEPS,
                                  BATCH_SIZE, OPTIMIZER, LOSS)

    x, y = lstm.process_data_for_lstm(data, NUM_TIMESTEPS, TIMESTEPS_AHEAD)
    xtrain, xtest, ytrain, ytest = lstm.divide_data_into_train_test(x, y, TRAIN_TEST_RATIO, BATCH_SIZE)
    for i in range(NUM_EPOCHS // SAVE_EVERY):
        print(f"epoch number: {totalEpoch}")
        lstm.train_model(model, SAVE_EVERY, BATCH_SIZE, xtrain, ytrain,
                         xtest, ytest, extraCallback=[csv_logger, history])
        if not os.path.exists('models'):
            os.mkdir('models')
        model.save('models/lstm.h5')

        prediction = lstm.try_prediction(xtest, model, BATCH_SIZE)
        prediction = data_manipulation.scale_back_to_normal(prediction, scaler)
        test_data = data_manipulation.scale_back_to_normal(ytest[BATCH_SIZE], scaler)
        if not os.path.exists(f'RESULTS/{CONFIG_ID}/predict'):
            os.mkdir(f'RESULTS/{CONFIG_ID}/predict')
        lstm.show_prediction(prediction, test_data, ticker,
                             file_name=f"RESULTS/{CONFIG_ID}/predict/predictEpoch{totalEpoch}.png")

        plt.close("all")
        history
        plt.plot(history.losses)
        plt.plot(history.val_losses)
        plt.title("model train vs validation loss - Ticker: " + ticker)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(f'RESULTS/{CONFIG_ID}/loss.png')
        plt.show()
        plt.close("all")

        plt.figure()
        plt.plot(history.mapes)
        plt.title("mape - ticker: " + ticker)
        plt.ylabel('mape')
        plt.xlabel('epoch')
        plt.legend(['mape'], loc='upper right')
        plt.savefig(f'RESULTS/{CONFIG_ID}/mape.png')
        plt.show()
        plt.close("all")
        totalEpoch = totalEpoch + 1
    # plt.show()
    # plt.show()


score, _, _ = model.evaluate(xtest, ytest, batch_size=BATCH_SIZE)
rmse = math.sqrt(score)
logging.info(r"\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))
