import download_data
import csv
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

BEGINNING_DATE = '2017-03-31'
ENDING_DATE = '2018-03-31'
_SHARADAR_API_KEY = open("sharadar.secret", "r").readline().strip()
_WIKI_PRICES_KEY = open("wikiprices.secret", "r").readline().strip()
_FINSENTS_KEY = open("finsents.secret", "r").readline().strip()
MAX_WORKER = 10


def collect_thread_list(ticker):
    trainning_dataSet = pd.DataFrame()
    try:
        trainning_dataSet = download_data.get_training_dataset(BEGINNING_DATE, ENDING_DATE, ticker)

    except ValueError:
        print("Maybe, ", ticker, " isn't available.")
    return trainning_dataSet


newDF = pd.DataFrame()

tickerList = []
futures_list = []
executor = ThreadPoolExecutor(MAX_WORKER=10)

with open('NS1-datasets-codes.csv', 'r') as csvfile:
    fieldnames = ['ticker', 'companyName']
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    for row in reader:
        ticker, contry_code = row['ticker'].split('/')[1].split('_')[0:2]
        print("proceed ticker named: ", ticker)
        if contry_code == "US":
            try:
                tickerList.append(ticker)
            except ValueError:
                print("Maybe, ", ticker, " isn't available.")

for f in executor.map(collect_thread_list, tickerList):
    print('result: {}'.format(f))
    newDF.append(f)

# Final dataframe after appendinf ticks
newDF.describe()
