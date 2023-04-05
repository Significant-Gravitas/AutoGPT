from yahooquery import Ticker
import json



def write_to_file(file_path, data):
    """
    Writes data to a file in JSON format.

    :param file_path: The path of the file to write to.
    :param data: The data to write to the file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    """
    Retrieves financial data for a specified stock symbol and saves it to JSON files.
    """
    # Define Ticker object
    symbol = 'AAPL'
    aapl = Ticker(symbol)
    
    # Get relevant financial metrics for AAPL
    financials = aapl.financial_data
    earnings = aapl.earnings
    key_metrics = aapl.key_stats
    recommendations = aapl.recommendations
    
    # Save data to files
    write_to_file(f'{symbol}_financials.json', financials)
    write_to_file(f'{symbol}_earnings.json', earnings)
    write_to_file(f'{symbol}_key_metrics.json', key_metrics)
    write_to_file(f'{symbol}_recommendations.json', recommendations)


if __name__ == '__main__':
    main()