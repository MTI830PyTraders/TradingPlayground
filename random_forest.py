# https://github.com/jalajthanaki/stock_price_prediction/blob/master/Stock_Price_Prediction.ipynb

import data_manipulation
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt


from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split



# Parameters of what data to select
BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'

BEGINNING_DATE_TRAIN = BEGINNING_DATE
ENDING_DATE_TRAIN = '2017-03-31'
BEGINNING_DATE_TEST = '2017-04-01'
ENDING_DATE_TEST = ENDING_DATE

TICKER = 'MSFT'


final_xr = xr.open_dataset("final_xr.nc", chunks=30)
ds = final_xr.sel(ticker=TICKER)
train_data = xr.Dataset({'Sentiment': ds.sentiment, 'close': ds['close'], 'fcf': ds.fcf}).to_dataframe().interpolate(limit_direction='both')
data, scaler = data_manipulation.prepare_training_data(train_data)

train = df.loc[train_start_date : train_end_date]
test = df.loc[test_start_date:test_end_date]


rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)
