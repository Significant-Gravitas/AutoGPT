import concurrent.futures
from typing import List

from newsapi import NewsApiClient

categories = ["technology", "business", "entertainment", "health", "sports", "science"]

class NewsSearch(object):

    def __init__(self, api_key):
        self.news_api_client = NewsApiClient(api_key)

    def news_headlines_search(self, category: str, query: str) -> List[str]:
        """
        Get top news headlines for category specified.
        Args:
            category (str) : The category specified. Must be one of technology, business, entertainment, health, sports or science.
        Returns:
            list(str): A list of top news headlines for the specified category.
        """
        result = self.news_api_client.get_top_headlines(
            category=category, language="en", country="us", page=1, q=query
        )
        return [article["title"] for article in result["articles"][:3]]


    def news_search(self, query: str) -> List[str]:
        """
        Aggregates top news headlines from the categories.
        Returns:
            list(str): A list of top news headlines aggregated from all categories.
        """
        with concurrent.futures.ThreadPoolExecutor() as tp:
            futures = []
            for cat in categories:
                futures.append(tp.submit(self.news_headlines_search, category=cat, query=query))

            aggregated_headlines = []
            for fut in concurrent.futures.wait(futures)[0]:
                aggregated_headlines.append(fut.result())

            return aggregated_headlines