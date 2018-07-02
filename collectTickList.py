import download_data
import csv
BEGINNING_DATE = '2017-03-31'
ENDING_DATE = '2018-03-31'
_SHARADAR_API_KEY = open("sharadar.secret", "r").readline().strip()
_WIKI_PRICES_KEY = open("wikiprices.secret", "r").readline().strip()
_FINSENTS_KEY = open("finsents.secret", "r").readline().strip()

with open('NS1-datasets-codes.csv', 'r') as csvfile:
    fieldnames = ['ticker', 'companyName']
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    for row in reader:
        ticker, contry_code = row['ticker'].split('/')[1].split('_')[0], row['ticker'].split('/')[1].split('_')[1]
        print("ticker")
        if contry_code == "US":
            try:
                print(download_data.get_training_dataset(BEGINNING_DATE, ENDING_DATE, ticker))
            except ValueError:
                print("Maybe, ", ticker, " isn't available.")
