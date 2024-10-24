from typing import Any, Dict, Literal, Optional
from urllib.parse import quote

import requests
from autogpt_libs.supabase_integration_credentials_store import APIKeyCredentials

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    BlockSecret,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    SecretField,
)


class GetRequest:
    @classmethod
    def get_request(
        cls, url: str, json=False, headers: Optional[Dict[str, str]] = None
    ) -> Any:
        headers = headers or {}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json() if json else response.text


class GetWikipediaSummaryBlock(Block, GetRequest):
    class Input(BlockSchema):
        topic: str = SchemaField(description="The topic to fetch the summary for")

    class Output(BlockSchema):
        summary: str = SchemaField(description="The summary of the given topic")
        error: str = SchemaField(
            description="Error message if the summary cannot be retrieved"
        )

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
        topic = input_data.topic
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        response = self.get_request(url, json=True)
        if "extract" not in response:
            raise RuntimeError(f"Unable to parse Wikipedia response: {response}")
        yield "summary", response["extract"]


class SearchTheWebBlock(Block, GetRequest):
    class Input(BlockSchema):
        query: str = SchemaField(description="The search query to search the web for")

    class Output(BlockSchema):
        results: str = SchemaField(
            description="The search results including content from top 5 URLs"
        )
        error: str = SchemaField(description="Error message if the search fails")

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
        # Encode the search query
        encoded_query = quote(input_data.query)

        # Prepend the Jina Search URL to the encoded query
        jina_search_url = f"https://s.jina.ai/{encoded_query}"

        # Make the request to Jina Search
        response = self.get_request(jina_search_url, json=False)

        # Output the search results
        yield "results", response


class ExtractWebsiteContentBlock(Block, GetRequest):
    class Input(BlockSchema):
        url: str = SchemaField(description="The URL to scrape the content from")
        raw_content: bool = SchemaField(
            default=False,
            title="Raw Content",
            description="Whether to do a raw scrape of the content or use Jina-ai Reader to scrape the content",
            advanced=True,
        )

    class Output(BlockSchema):
        content: str = SchemaField(description="The scraped content from the given URL")
        error: str = SchemaField(
            description="Error message if the content cannot be retrieved"
        )

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

        content = self.get_request(url, json=False)
        yield "content", content


class GetWeatherInformationBlock(Block, GetRequest):
    class Input(BlockSchema):
        location: str = SchemaField(
            description="Location to get weather information for"
        )
        api_key: BlockSecret = SecretField(key="openweathermap_api_key")
        use_celsius: bool = SchemaField(
            default=True,
            description="Whether to use Celsius or Fahrenheit for temperature",
        )

    class Output(BlockSchema):
        temperature: str = SchemaField(
            description="Temperature in the specified location"
        )
        humidity: str = SchemaField(description="Humidity in the specified location")
        condition: str = SchemaField(
            description="Weather condition in the specified location"
        )
        error: str = SchemaField(
            description="Error message if the weather information cannot be retrieved"
        )

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
            raise RuntimeError(f"Expected keys not found in response: {weather_data}")


class FactCheckerBlock(Block, GetRequest):
    class Input(BlockSchema):
        statement: str = SchemaField(
            description="The statement to check for factuality"
        )
        credentials: CredentialsMetaInput[Literal["jina"], Literal["api_key"]] = (
            CredentialsField(
                provider="jina",
                supported_credential_types={"api_key"},
                description="The Jina AI API key for getting around the API rate limit.",
            )
        )

    class Output(BlockSchema):
        factuality: float = SchemaField(
            description="The factuality score of the statement"
        )
        result: bool = SchemaField(description="The result of the factuality check")
        reason: str = SchemaField(description="The reason for the factuality result")
        error: str = SchemaField(description="Error message if the check fails")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
            description="This block checks the factuality of a given statement using Jina AI's Grounding API.",
            categories={BlockCategory.SEARCH},
            input_schema=FactCheckerBlock.Input,
            output_schema=FactCheckerBlock.Output,
            test_input={
                "statement": "Jina AI was founded in 2020 in Berlin.",
                "credentials": {
                    "id": "test-credentials-id",
                    "provider": "jina",
                    "type": "api_key",
                    "title": "Mock Jina API key",
                },
            },
            test_output=[
                ("factuality", 0.95),
                ("result", True),
                ("reason", "The statement is supported by multiple sources."),
            ],
            test_mock={
                "get_request": lambda url, json, headers: {
                    "data": {
                        "factuality": 0.95,
                        "result": True,
                        "reason": "The statement is supported by multiple sources.",
                    }
                }
            },
            test_credentials=APIKeyCredentials(
                id="test-credentials-id",
                provider="jina",
                api_key="mock-api-key",
                title="Mock Jina API key",
                expires_at=None,
            ),
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        encoded_statement = quote(input_data.statement)
        url = f"https://g.jina.ai/{encoded_statement}"

        headers = {"Accept": "application/json", "Authorization": credentials.bearer()}

        response = self.get_request(url, json=True, headers=headers)

        if "data" in response:
            data = response["data"]
            yield "factuality", data["factuality"]
            yield "result", data["result"]
            yield "reason", data["reason"]
        else:
            raise RuntimeError(f"Expected 'data' key not found in response: {response}")
