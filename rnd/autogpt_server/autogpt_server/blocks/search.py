from typing import Any
from urllib.parse import quote

import requests

from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class GetRequest:
    @classmethod
    def get_request(cls, url: str, json=False) -> Any:
        response = requests.get(url)
        response.raise_for_status()
        return response.json() if json else response.text


class WikipediaSummaryBlock(Block, GetRequest):
    class Input(BlockSchema):
        topic: str

    class Output(BlockSchema):
        summary: str
        error: str

    def __init__(self):
        super().__init__(
            id="h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m",
            input_schema=WikipediaSummaryBlock.Input,
            output_schema=WikipediaSummaryBlock.Output,
            test_input={"topic": "Artificial Intelligence"},
            test_output=("summary", "summary content"),
            test_mock={"get_request": lambda url, json: {"extract": "summary content"}},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            topic = input_data.topic
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = self.get_request(url, json=True)
            yield "summary", response["extract"]

        except requests.exceptions.HTTPError as http_err:
            yield "error", f"HTTP error occurred: {http_err}"

        except requests.RequestException as e:
            yield "error", f"Request to Wikipedia failed: {e}"

        except KeyError as e:
            yield "error", f"Error parsing Wikipedia response: {e}"


class WebSearchBlock(Block, GetRequest):
    class Input(BlockSchema):
        query: str  # The search query

    class Output(BlockSchema):
        results: str  # The search results including content from top 5 URLs
        error: str  # Error message if the search fails

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-6f7g-8h9i-0j1k-l2m3n4o5p6q7",
            input_schema=WebSearchBlock.Input,
            output_schema=WebSearchBlock.Output,
            test_input={"query": "Artificial Intelligence"},
            test_output=("results", "search content"),
            test_mock={"get_request": lambda url, json: "search content"},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Encode the search query
            encoded_query = quote(input_data.query)

            # Prepend the Jina Search URL to the encoded query
            jina_search_url = f"https://s.jina.ai/{encoded_query}"

            # Make the request to Jina Search
            response = self.get_request(jina_search_url, json=False)

            # Output the search results
            yield "results", response

        except requests.exceptions.HTTPError as http_err:
            yield "error", f"HTTP error occurred: {http_err}"

        except requests.RequestException as e:
            yield "error", f"Request to Jina Search failed: {e}"


class WebScraperBlock(Block, GetRequest):
    class Input(BlockSchema):
        url: str  # The URL to scrape

    class Output(BlockSchema):
        content: str  # The scraped content from the URL
        error: str

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",  # Unique ID for the block
            input_schema=WebScraperBlock.Input,
            output_schema=WebScraperBlock.Output,
            test_input={"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
            test_output=("content", "scraped content"),
            test_mock={"get_request": lambda url, json: "scraped content"},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Prepend the Jina-ai Reader URL to the input URL
            jina_url = f"https://r.jina.ai/{input_data.url}"

            # Make the request to Jina-ai Reader
            response = self.get_request(jina_url, json=False)

            # Output the scraped content
            yield "content", response

        except requests.exceptions.HTTPError as http_err:
            yield "error", f"HTTP error occurred: {http_err}"

        except requests.RequestException as e:
            yield "error", f"Request to Jina-ai Reader failed: {e}"
