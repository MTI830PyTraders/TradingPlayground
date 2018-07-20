import pandas as pd
import numpy as np
import xarray as xr


def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')


sharadar_sf1_df = pd.read_csv('SF1.cleaned.csv',
                              parse_dates={'datetime': ['datekey'], 'lastupdated_sf1': ['lastupdated']},
                              date_parser=dateparse, index_col=['datetime', 'ticker'],
                              dtype={'dimension': 'str', 'reportperiod': 'datetime64', 'calendardate': 'datetime64'}, engine='python')

sharadar_sep_df = pd.read_csv('SEP.cleaned.csv',
                              parse_dates={'datetime': ['date'], 'lastupdated_sep': ['lastupdated']},
                              index_col=['datetime', 'ticker'], date_parser=dateparse)

finsentsHeadersNames = ["ticker", "date", "sentiment", "sentimentHigh",  "sentimentLow", "newsVolume", "newsBuzz"]
finsents_df = pd.read_csv('NS1.cleaned.csv', names=finsentsHeadersNames,
                          parse_dates={'datetime': ['date']}, date_parser=dateparse,
                          index_col=['datetime', 'ticker'])

finsents_xr = finsents_df.to_xarray()
sharadar_sf1_xr = sharadar_sf1_df.to_xarray()
sharadar_sep_xr = sharadar_sep_df.to_xarray()

sharadar_sf1_xr.to_netcdf('sf1.nc')
sharadar_sep_xr.to_netcdf('sep.nc')
finsents_xr.to_netcdf('ns1.nc')


finsents_xr = xr.open_dataset("ns1.nc", chunks=30)
sharadar_sep_xr = xr.open_dataset("sep.nc", chunks=30)
sharadar_sf1_xr = xr.open_dataset("sf1.nc", chunks=30)


# sharadar_sf1_xr = sharadar_sf1_xr.reindex_like(sharadar_sep_xr)
print(sharadar_sf1_xr)
merge1 = xr.merge([sharadar_sf1_xr, sharadar_sep_xr])

final_xr = xr.merge([merge1, finsents_xr])
print(final_xr)

time = pd.date_range(final_xr.indexes.get('datetime').min(), final_xr.indexes.get('datetime').max(), freq='D')
ds = xr.Dataset({'datetime': time})
final_xr.reindex_like(ds)
final_xr = final_xr.interpolate_na(method='linear')
final_xr.to_netcdf('final_xr.nc')
