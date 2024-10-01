from typing import Any
from urllib.parse import quote

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class GetRequest:
    @classmethod
    def get_request(cls, url: str, json=False) -> Any:
        response = requests.get(url)
        response.raise_for_status()
        return response.json() if json else response.text


class GetWikipediaSummaryBlock(Block, GetRequest):
    class Input(BlockSchema):
        topic: str

    class Output(BlockSchema):
        summary: str
        error: str

    def __init__(self):
        super().__init__(
            id="f5b0f5d0-1862-4d61-94be-3ad0fa772760",
            description="This block fetches the summary of a given topic from Wikipedia.",
            categories={BlockCategory.SEARCH},
            input_schema=GetWikipediaSummaryBlock.Input,
            output_schema=GetWikipediaSummaryBlock.Output,
            test_input={"topic": "Artificial Intelligence"},
            test_output=("summary", "summary content"),
            test_mock={"get_request": lambda url, json: {"extract": "summary content"}},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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


class SearchTheWebBlock(Block, GetRequest):
    class Input(BlockSchema):
        query: str  # The search query

    class Output(BlockSchema):
        results: str  # The search results including content from top 5 URLs
        error: str  # Error message if the search fails

    def __init__(self):
        super().__init__(
            id="87840993-2053-44b7-8da4-187ad4ee518c",
            description="This block searches the internet for the given search query.",
            categories={BlockCategory.SEARCH},
            input_schema=SearchTheWebBlock.Input,
            output_schema=SearchTheWebBlock.Output,
            test_input={"query": "Artificial Intelligence"},
            test_output=("results", "search content"),
            test_mock={"get_request": lambda url, json: "search content"},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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


class ExtractWebsiteContentBlock(Block, GetRequest):
    class Input(BlockSchema):
        url: str  # The URL to scrape
        raw_content: bool = SchemaField(
            default=False,
            title="Raw Content",
            description="Whether to do a raw scrape of the content or use Jina-ai Reader to scrape the content",
            advanced=True,
        )

    class Output(BlockSchema):
        content: str  # The scraped content from the URL
        error: str

    def __init__(self):
        super().__init__(
            id="436c3984-57fd-4b85-8e9a-459b356883bd",
            description="This block scrapes the content from the given web URL.",
            categories={BlockCategory.SEARCH},
            input_schema=ExtractWebsiteContentBlock.Input,
            output_schema=ExtractWebsiteContentBlock.Output,
            test_input={"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
            test_output=("content", "scraped content"),
            test_mock={"get_request": lambda url, json: "scraped content"},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if input_data.raw_content:
            url = input_data.url
        else:
            url = f"https://r.jina.ai/{input_data.url}"

        try:
            content = self.get_request(url, json=False)
            yield "content", content
        except requests.exceptions.HTTPError as http_err:
            yield "error", f"HTTP error occurred: {http_err}"
        except requests.RequestException as e:
            yield "error", f"Request to URL failed: {e}"


class GetWeatherInformationBlock(Block, GetRequest):
    class Input(BlockSchema):
        location: str
        api_key: BlockSecret = SecretField(key="openweathermap_api_key")
        use_celsius: bool = True

    class Output(BlockSchema):
        temperature: str
        humidity: str
        condition: str
        error: str

    def __init__(self):
        super().__init__(
            id="f7a8b2c3-6d4e-5f8b-9e7f-6d4e5f8b9e7f",
            input_schema=GetWeatherInformationBlock.Input,
            output_schema=GetWeatherInformationBlock.Output,
            description="Retrieves weather information for a specified location using OpenWeatherMap API.",
            test_input={
                "location": "New York",
                "api_key": "YOUR_API_KEY",
                "use_celsius": True,
            },
            test_output=[
                ("temperature", "21.66"),
                ("humidity", "32"),
                ("condition", "overcast clouds"),
            ],
            test_mock={
                "get_request": lambda url, json: {
                    "main": {"temp": 21.66, "humidity": 32},
                    "weather": [{"description": "overcast clouds"}],
                }
            },
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            units = "metric" if input_data.use_celsius else "imperial"
            api_key = input_data.api_key.get_secret_value()
            location = input_data.location
            url = f"http://api.openweathermap.org/data/2.5/weather?q={quote(location)}&appid={api_key}&units={units}"
            weather_data = self.get_request(url, json=True)

            if "main" in weather_data and "weather" in weather_data:
                yield "temperature", str(weather_data["main"]["temp"])
                yield "humidity", str(weather_data["main"]["humidity"])
                yield "condition", weather_data["weather"][0]["description"]
            else:
                yield "error", f"Expected keys not found in response: {weather_data}"

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 403:
                yield "error", "Request to weather API failed: 403 Forbidden. Check your API key and permissions."
            else:
                yield "error", f"HTTP error occurred: {http_err}"
        except requests.RequestException as e:
            yield "error", f"Request to weather API failed: {e}"
        except KeyError as e:
            yield "error", f"Error processing weather data: {e}"
