"""
Finance information about companies
Using yahoo finance
"""
import yfinance as yf
import json

from .registry import ability

ability(
    name="get_ticker_info",
    description="Get information about a specific ticker symbol",
    parameters=[
        {
            "name": "ticker_symbol",
            "description": "ticker symbol",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)
async def get_ticker_info(
    agent,
    task_id: str,
    ticker_symbol: str
) -> str:
    
    # get ticker then financial information as dict
    # dict has structure where timestamp are keys
    stock = yf.Ticker(ticker_symbol)
    return json.dumps(stock.get_info())

@ability(
    name="get_financials_year",
    description="Get financial information of a company for a specific year",
    parameters=[
        {
            "name": "ticker_symbol",
            "description": "ticker symbol",
            "type": "string",
            "required": True
        },
        {
            "name": "year",
            "description": "year wanted for financials",
            "type": "integer",
            "required": True
        }
    ],
    output_type="str"
)
async def get_financials_year(
    agent,
    task_id: str,
    ticker_symbol: str,
    year: int
) -> str:
    
    # get ticker then financial information as dict
    # dict has structure where timestamp are keys
    stock = yf.Ticker(ticker_symbol)
    financials_dict = stock.financials.to_dict()

    year_financial_data = {}

    # get financial informatio and return dict
    for timestamp, financial_data in financial_data.items():
        key_year = int(str(timestamp).split("-")[0])

        if key_year == year:
            year_financial_data = financials_dict

    return json.dumps(year_financial_data)
