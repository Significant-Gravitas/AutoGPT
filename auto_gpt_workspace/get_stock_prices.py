from typing import List, Optional
import yfinance as yf
import pandas as pd

def get_stock_prices(tickers: List[str], interval: str = '1d', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    '''
    Retrieves the stock prices for a list of tickers using the Yahoo Finance API.

    Parameters:
    tickers (list): A list of strings representing the ticker symbols of the companies whose stock prices should be retrieved.
    interval (str): The interval for which to retrieve the stock prices (e.g., '1d' for daily, '1wk' for weekly, etc.).
    start_date (Optional[str]): A string representing the start date for which to retrieve the stock prices in ISO format (YYYY-MM-DD).
    end_date (Optional[str]): A string representing the end date for which to retrieve the stock prices in ISO format (YYYY-MM-DD).

    Returns:
    A Pandas DataFrame containing the retrieved stock prices.
    '''
    data = yf.download(
        tickers=tickers,
        interval=interval,
        start=start_date,
        end=end_date,
        group_by='ticker'
    )

    # reshape data
    data.columns = data.columns.droplevel(0)
    data = data.reset_index()
    data = data.rename(columns={'Datetime': 'date'})
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")

    return data

def main():
    # Example usage:
    tickers = ['AAPL', 'MSFT', 'AMZN']
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    stock_prices = get_stock_prices(tickers, start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    main()
