# Python script to extract desired data from a given asset using yahooquery api

from yahooquery import Ticker

asset = 'AAPL'
ticker = Ticker(asset)

data = ticker.summary_detail

desired_keys = ['regularMarketPrice', 'regularMarketDayLow', 'regularMarketDayHigh', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'forwardPE']

for key, value in data[asset].items():
    if key in desired_keys:
        print(key, value)