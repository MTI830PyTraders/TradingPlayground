# https://github.com/jalajthanaki/stock_price_prediction/blob/master/Stock_Price_Prediction.ipynb

import numpy as np
import pandas as pd
import xarray as xr

from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor


TICKER = 'AAPL'

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
train_close = train['close'].values.astype(float)
train_close = np.array(train_close).reshape((len(train_close), 1))

train_final = np.concatenate((train_sentiments, train_fcf, train_close), axis=1)


test = df.loc[BEGINNING_DATE_TEST:ENDING_DATE_TEST]
test_sentiments = test['Sentiment'].values.astype(float)
test_sentiments = np.array(test_sentiments).reshape((len(test_sentiments), 1))
test_fcf = test['fcf'].values.astype(float)
test_fcf = np.array(test_fcf).reshape((len(test_fcf), 1))
test_close = test['close'].values.astype(float)
test_close = np.array(test_close).reshape((len(test_close), 1))
test_final = np.concatenate((test_sentiments, test_fcf, test_close), axis=1)


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



from datetime import datetime, timedelta
temp_date = BEGINNING_DATE_TEST
average_last_5_days_test = 0
total_days = 10
for i in range(total_days):
    average_last_5_days_test += test.loc[temp_date, 'close']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Reducing one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
    temp_date = difference.strftime('%Y-%m-%d')
    #print temp_date
average_last_5_days_test = average_last_5_days_test / total_days
print(average_last_5_days_test)

temp_date = BEGINNING_DATE_TEST
average_upcoming_5_days_predicted = 0
for i in range(total_days):
    average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'close']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Adding one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
    temp_date = difference.strftime('%Y-%m-%d')
    #print temp_date
average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
print(average_upcoming_5_days_predicted)
#average train.loc['2013-12-31', 'prices'] - advpredictions_df.loc['2014-01-01', 'prices']
difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
print(difference_test_predicted_prices)


predictions_df['close'] = predictions_df['close'] + difference_test_predicted_prices
ax = predictions_df.rename(columns={"close": "predicted_close"}).plot(title='Random Forest predicted prices after aligning')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
actuals_df = pd.DataFrame(data=y_test[0:], index=idx, columns=['close'])
fig = actuals_df.rename(columns={'close': "actual_close"}).plot(ax=ax).get_figure()
fig.show()
# fig.savefig("./graphs/random forest with aligning.png")




# fig.savefig("./graphs/random forest without smoothing.png")
predictions_df['ewma'] = predictions_df["close"].ewm(span=60).mean()
predictions_df['actual_close'] = test['close']
predictions_df['actual_close_ewma'] = predictions_df["actual_close"].ewm(span=60).mean()
predictions_df.columns = ['predicted_close', 'average_predicted_close', 'actual_close', 'average_actual_close']
predictions_plot = predictions_df.plot(title='Random Forest predicted prices after aligning & smoothing')
predictions_plot.set_xlabel("Dates")
predictions_plot.set_ylabel("Stock Prices")
fig = predictions_plot.get_figure()
fig.show()

predictions_df_average = predictions_df[['average_predicted_close', 'actual_close']]
predictions_plot = predictions_df_average.plot(title='Random Forest after aligning & smoothing')
predictions_plot.set_xlabel("Dates")
predictions_plot.set_ylabel("Stock Prices")
fig = predictions_plot.get_figure()
fig.show()
