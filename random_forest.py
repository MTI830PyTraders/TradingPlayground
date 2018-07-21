# https://github.com/jalajthanaki/stock_price_prediction/blob/master/Stock_Price_Prediction.ipynb

import numpy as np
import pandas as pd
import xarray as xr

from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor


TICKER = 'MSFT'

# Parameters of what data to select
BEGINNING_DATE = '2013-03-31'
ENDING_DATE = '2018-03-31'

BEGINNING_DATE_TRAIN = BEGINNING_DATE
ENDING_DATE_TRAIN = '2017-03-31'
BEGINNING_DATE_TEST = '2017-04-01'
ENDING_DATE_TEST = ENDING_DATE


final_xr = xr.open_dataset("final_xr.nc", chunks=30)
ds = final_xr.sel(ticker=TICKER)
df = xr.Dataset({'Sentiment': ds.sentiment, 'close': ds['close'], 'fcf': ds.fcf}).to_dataframe().interpolate(limit_direction='both')


train = df.loc[BEGINNING_DATE_TRAIN:ENDING_DATE_TRAIN]
train_sentiments = train['Sentiment'].values.astype(float)
train_sentiments = np.array(train_sentiments).reshape((len(train_sentiments), 1))
train_fcf = train['fcf'].values.astype(float)
train_fcf = np.array(train_fcf).reshape((len(train_fcf), 1))
train_final = np.concatenate((train_sentiments, train_fcf), axis=1)


test = df.loc[BEGINNING_DATE_TEST:ENDING_DATE_TEST]
test_sentiments = test['Sentiment'].values.astype(float)
test_sentiments = np.array(test_sentiments).reshape((len(test_sentiments), 1))
test_fcf = test['fcf'].values.astype(float)
test_fcf = np.array(test_fcf).reshape((len(test_fcf), 1))
test_final = np.concatenate((test_sentiments, test_fcf), axis=1)


y_train = train['close'].values.astype(float)
y_test = test['close'].values.astype(float)


rf = RandomForestRegressor()
rf.fit(train_final, y_train)


prediction, bias, contributions = ti.predict(rf, test_final)
rf.score(test_final, y_test)
idx = pd.date_range(BEGINNING_DATE_TEST, ENDING_DATE_TEST)[:-1]
predictions_df = pd.DataFrame(data=prediction[0:], index=idx, columns=['close'])

ax = predictions_df.rename(columns={'close': "predicted_close"}).plot(title='Random Forest predicted prices')
ax.set_xlabel("Dates")
ax.set_ylabel("Adj Close Prices")

actuals_df = pd.DataFrame(data=y_test[0:], index=idx, columns=['close'])
fig = actuals_df.rename(columns={'close': "actual_close"}).plot(ax=ax).get_figure()
fig.show()

# fig.savefig("./graphs/random forest without smoothing.png")
