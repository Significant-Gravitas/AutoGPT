import concurrent.futures

from newsapi import NewsApiClient

from autogpt.commands.command import command
from autogpt.config import Config

CFG = Config()

categories = ["technology", "business", "entertainment", "health", "sports", "science"]
news_api_client = NewsApiClient(CFG.news_api_key)


def get_news_headlines_for_category(category: str):
    """
    Get top news headlines for category specified.

    Args:

        category (str) : The category specified. Must be one of technology, business, entertainment, health, sports or science.

    Returns:
        list(str): A list of top 5 news headlines for the specified category.


    """
    result = news_api_client.get_top_headlines(
        category=category, language="en", country="us", page=1
    )
    return [article["title"] for article in result["articles"][:3]]


@command(
    "aggregate_news_headlines", "Aggregate news headlines accross different sectors"
)
def aggregate_top_news_headlines():
    with concurrent.futures.ThreadPoolExecutor() as tp:
        futures = []
        for cat in categories:
            futures.append(tp.submit(get_news_headlines_for_category, category=cat))

        aggregated_headlines = []
        for fut in concurrent.futures.wait(futures)[0]:
            aggregated_headlines.append(fut.result())

        return aggregated_headlines
