from typing import Literal
from urllib.parse import quote

from pydantic import SecretStr

from backend.blocks.helpers.http import GetRequest
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName


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


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="openweathermap",
    api_key=SecretStr("mock-openweathermap-api-key"),
    title="Mock OpenWeatherMap API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


class GetWeatherInformationBlock(Block, GetRequest):
    class Input(BlockSchema):
        location: str = SchemaField(
            description="Location to get weather information for"
        )
        credentials: CredentialsMetaInput[
            Literal[ProviderName.OPENWEATHERMAP], Literal["api_key"]
        ] = CredentialsField(
            description="The OpenWeatherMap integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
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
                "use_celsius": True,
                "credentials": TEST_CREDENTIALS_INPUT,
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
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        units = "metric" if input_data.use_celsius else "imperial"
        api_key = credentials.api_key
        location = input_data.location
        url = f"http://api.openweathermap.org/data/2.5/weather?q={quote(location)}&appid={api_key}&units={units}"
        weather_data = self.get_request(url, json=True)

        if "main" in weather_data and "weather" in weather_data:
            yield "temperature", str(weather_data["main"]["temp"])
            yield "humidity", str(weather_data["main"]["humidity"])
            yield "condition", weather_data["weather"][0]["description"]
        else:
            raise RuntimeError(f"Expected keys not found in response: {weather_data}")
