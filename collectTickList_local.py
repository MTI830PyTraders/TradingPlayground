import download_data
import csv
import pandas as pd
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from quandl.errors.quandl_error import NotFoundError
import xarray as xr


BEGINNING_DATE = '2017-03-31'
ENDING_DATE = '2018-03-31'

idx = pd.date_range(BEGINNING_DATE, ENDING_DATE)


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


sharadar_sf1_df = pd.read_csv('SF1.cleaned.csv',
                              parse_dates={'datetime': ['datekey']}, date_parser=dateparse, index_col=['datetime', 'ticker'])

sharadar_sep_df = pd.read_csv('SEP.cleaned.csv',
                              parse_dates={'datetime': ['date']}, index_col=['datetime', 'ticker'], date_parser=dateparse)


finsentsHeadersNames = ["ticker", "date", "sentiment", "sentimentHigh",  "sentimentLow", "newVolume", "newBuzz"]
finsents_df = pd.read_csv('NS1.cleaned.csv', names=finsentsHeadersNames,
                          parse_dates={'datetime': ['date']}, date_parser=dateparse,
                          index_col=['datetime', 'ticker'])


sharadar_sf1_df = sharadar_sf1_df.rename(columns={'lastupdated': 'lastupdated_sf1'})
sharadar_sep_df = sharadar_sep_df.rename(columns={'lastupdated': 'lastupdated_sep'})

finsents_xr = finsents_df.to_xarray()
sharadar_sf1_xr = sharadar_sf1_df.to_xarray()
sharadar_sep_xr = sharadar_sep_df.to_xarray()

sharadar_sf1_xr.to_netcdf('sf1.nc')
sharadar_sep_xr.to_netcdf('sep.nc')
finsents_xr.to_netcdf('ns1.nc')


finsents_xr = xr.open_dataset("ns1.nc", chunks=30)
sharadar_sep_xr = xr.open_dataset("sep.nc", chunks=30)
sharadar_sf1_xr = xr.open_dataset("sf1.nc", chunks=30)


sharadar_sf1_xr = sharadar_sf1_xr.stack(z=('datetime', 'ticker'))
# print(sharadar_sf1_xr)
finsents_xr = finsents_xr.stack(z=('datetime', 'ticker'))
sharadar_sep_xr = sharadar_sep_xr.stack(z=('datetime', 'ticker'))


sharadar_sf1_xr = sharadar_sf1_xr.reindex_like(sharadar_sep_xr)
print(sharadar_sf1_xr)
merge1 = xr.merge([sharadar_sf1_xr, sharadar_sep_xr])

final_xr = xr.merge([merge1, finsents_xr])
final_xr.to_netcdf('final.nc')
print(final_xr)
