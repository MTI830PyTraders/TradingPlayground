import pandas as pd
# cheap fix from https://stackoverflow.com/a/50415484
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np
import quandl
from datetime import datetime
import os


def _get_cashflow_data(beginning_date: str, ending_date: str, ticker: str, api_key: str) -> pd.DataFrame:
    quandl.ApiConfig.api_key = api_key
    cashflow_df = quandl.get_table('SHARADAR/SF1',
                           ticker=ticker,
                           dimension='MRQ',
                           calendardate={'gte': beginning_date, 'lte': ending_date})
    check_empty_df(cashflow_df)
    cashflow_df = cashflow_df.sort_values(by='calendardate')
    cashflow_df = cashflow_df[['calendardate', 'fcf']].set_index('calendardate')
    idx = pd.date_range(beginning_date, ending_date)
    cashflow_df.index = pd.DatetimeIndex(cashflow_df.index)
    cashflow_df = cashflow_df.reindex(idx, fill_value=np.NaN)
    cashflow_df = cashflow_df.interpolate(limit_direction='both')
    return cashflow_df


def _get_stocks_data_from_wikip(beginning_date: str, ending_date: str, ticker: str, api_key: str) -> pd.DataFrame:
    quandl.ApiConfig.api_key = api_key
    stocks_df = quandl.get_table('WIKI/PRICES', date={'gte': beginning_date, 'lte': ending_date}, ticker=ticker)
    check_empty_df(stocks_df)
    stocks_df = stocks_df[['date', 'close']]
    stocks_df = stocks_df.sort_values(by='date')
    stocks_df = stocks_df.set_index('date')
    idx = pd.date_range(beginning_date, ending_date)
    stocks_df.index = pd.DatetimeIndex(stocks_df.index)
    stocks_df = stocks_df.reindex(idx, fill_value=np.NaN)
    stocks_df = stocks_df.interpolate(limit_direction='both')
    return stocks_df


def _get_stocks_data_from_morningstar(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    # doesn't work like this. Must reindex it properly.
    start = datetime.strptime(beginning_date, '%Y-%m-%d')
    end = datetime.strptime(ending_date, '%Y-%m-%d')
    stocks_df = web.DataReader(ticker, 'morningstar', start, end)
    check_empty_df(stocks_df)
    stocks_df = stocks_df[['Close']]
    #stocks_df = stocks_df.sort_values(by='Date')
    stocks_df = stocks_df.set_index('Date')
    idx = pd.date_range(beginning_date, ending_date)
    stocks_df.index = pd.DatetimeIndex(stocks_df.index)
    stocks_df = stocks_df.reindex(idx, fill_value=np.NaN)
    stocks_df = stocks_df.interpolate(limit_direction='both')

    return stocks_df


def _get_sentiments_data(beginning_date: str, ending_date: str, ticker: str, api_key: str) -> pd.DataFrame:
    quandl.ApiConfig.api_key = api_key
    sentiments_df = quandl.get(f'NS1/{ticker}_US', start_date=beginning_date, end_date=ending_date)
    check_empty_df(sentiments_df)
    sentiments_df = sentiments_df[['Sentiment']]
    return sentiments_df


_SHARADAR_API_KEY = open("sharadar.secret", "r").readline().strip()
_WIKI_PRICES_KEY = open("wikiprices.secret", "r").readline().strip()
_FINSENTS_KEY = open("finsents.secret", "r").readline().strip()


def get_cashflow_data(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    return _get_cashflow_data(beginning_date, ending_date, ticker, _SHARADAR_API_KEY)


def get_stocks_data(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    return _get_stocks_data_from_wikip(beginning_date, ending_date, ticker, _WIKI_PRICES_KEY)
    # return _get_stocks_data_from_morningstar(beginning_date, ending_date, ticker)


def get_sentiments_data(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    return _get_sentiments_data(beginning_date, ending_date, ticker, _FINSENTS_KEY)


def get_training_dataset(beginning_date: str, ending_date: str, ticker: str) -> pd.DataFrame:
    saved_file = f'cached_data/{ticker}_{beginning_date}_{ending_date}.csv'
    if os.path.exists(saved_file):
        return pd.read_csv(saved_file)

    cashflow_df = get_cashflow_data(beginning_date, ending_date, ticker)
    stocks_df = get_stocks_data(beginning_date, ending_date, ticker)
    sentiments_df = get_sentiments_data(beginning_date, ending_date, ticker)

    merge1 = pd.merge(cashflow_df, stocks_df, left_index=True, right_index=True)
    final_df: pd.DataFrame = pd.merge(merge1, sentiments_df, left_index=True, right_index=True)

    if not os.path.exists('cached_data'):
        os.mkdir('cached_data')
    final_df.to_csv(saved_file)

    return final_df


def check_empty_df(df: pd.DataFrame):
    if df.empty:
        raise ValueError('df is empty.')
