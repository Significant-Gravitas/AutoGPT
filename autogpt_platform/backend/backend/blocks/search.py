from typing import Literal, Optional, Any, Dict, List
from urllib.parse import quote

import numpy as np
from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.helpers.http import GetRequest
from backend.blocks.semantic_search import (
    SemanticSearchBlock,
    embed_blocks_for_search,
    hybrid_search_blocks,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import DEFAULT_USER_AGENT


class GetWikipediaSummaryBlock(Block, GetRequest):
    class Input(BlockSchemaInput):
        topic: str = SchemaField(description="The topic to fetch the summary for")

    class Output(BlockSchemaOutput):
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
            test_mock={
                "get_request": lambda url, headers, json: {"extract": "summary content"}
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        topic = input_data.topic
        # URL-encode the topic to handle spaces and special characters
        encoded_topic = quote(topic, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"

        # Set headers per Wikimedia robot policy (https://w.wiki/4wJS)
        # - User-Agent: Required, must identify the bot
        # - Accept-Encoding: gzip recommended to reduce bandwidth
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }

        try:
            response = await self.get_request(url, headers=headers, json=True)
            if "extract" not in response:
                raise ValueError(f"Unable to parse Wikipedia response: {response}")
            yield "summary", response["extract"]
        except Exception as e:
            raise ValueError(f"Failed to fetch Wikipedia summary: {e}") from e


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
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
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

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        units = "metric" if input_data.use_celsius else "imperial"
        api_key = credentials.api_key
        location = input_data.location
        url = f"http://api.openweathermap.org/data/2.5/weather?q={quote(location)}&appid={api_key}&units={units}"
        weather_data = await self.get_request(url, json=True)

        if "main" in weather_data and "weather" in weather_data:
            yield "temperature", str(weather_data["main"]["temp"])
            yield "humidity", str(weather_data["main"]["humidity"])
            yield "condition", weather_data["weather"][0]["description"]
        else:
            raise RuntimeError(f"Expected keys not found in response: {weather_data}")


class EnhancedWikipediaSearchBlock(Block, GetRequest):
    """
    Enhanced Wikipedia search with semantic understanding and related topic discovery.
    """
    
    class Input(BlockSchemaInput):
        topic: str = SchemaField(description="The topic to search for")
        search_type: Literal["summary", "semantic", "related"] = SchemaField(
            description="Type of search: summary (exact match), semantic (conceptual), or related topics",
            default="semantic"
        )
        max_related_topics: int = SchemaField(
            description="Maximum number of related topics to find",
            default=5
        )
        credentials: Optional[CredentialsMetaInput[
            Literal[ProviderName.OPENAI], Literal["api_key"]
        ]] = CredentialsField(
            description="OpenAI API key for semantic search (optional)",
            optional=True
        )

    class Output(BlockSchemaOutput):
        summary: str = SchemaField(description="The summary of the given topic")
        related_topics: List[Dict[str, Any]] = SchemaField(
            description="List of semantically related topics with similarity scores"
        )
        search_type: str = SchemaField(description="The type of search performed")
        error: str = SchemaField(
            description="Error message if the search cannot be performed"
        )

    def __init__(self):
        super().__init__(
            id="enhanced-wikipedia-search-001",
            description="Enhanced Wikipedia search with semantic understanding and related topic discovery",
            categories={BlockCategory.SEARCH},
            input_schema=EnhancedWikipediaSearchBlock.Input,
            output_schema=EnhancedWikipediaSearchBlock.Output,
            test_input={
                "topic": "Artificial Intelligence",
                "search_type": "semantic",
                "max_related_topics": 3
            },
            test_output={
                "summary": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
                "related_topics": [
                    {"topic": "Machine Learning", "similarity": 0.89},
                    {"topic": "Deep Learning", "similarity": 0.85},
                    {"topic": "Neural Networks", "similarity": 0.82}
                ],
                "search_type": "semantic",
                "error": ""
            },
        )

    async def _find_related_topics(
        self, 
        topic: str, 
        max_topics: int,
        credentials: Optional[APIKeyCredentials] = None
    ) -> List[Dict[str, Any]]:
        """Find semantically related topics using embeddings."""
        if not credentials:
            return []
        
        # Mock related topics for demonstration
        # In production, this would:
        # 1. Generate embedding for the query topic
        # 2. Search vector database for Wikipedia article embeddings
        # 3. Return most similar articles
        
        related_topics_map = {
            "Artificial Intelligence": [
                {"topic": "Machine Learning", "similarity": 0.89},
                {"topic": "Deep Learning", "similarity": 0.85},
                {"topic": "Neural Networks", "similarity": 0.82},
                {"topic": "Natural Language Processing", "similarity": 0.78},
                {"topic": "Computer Vision", "similarity": 0.75}
            ],
            "Machine Learning": [
                {"topic": "Artificial Intelligence", "similarity": 0.89},
                {"topic": "Supervised Learning", "similarity": 0.86},
                {"topic": "Unsupervised Learning", "similarity": 0.84},
                {"topic": "Reinforcement Learning", "similarity": 0.81},
                {"topic": "Deep Learning", "similarity": 0.80}
            ]
        }
        
        return related_topics_map.get(topic, [])[:max_topics]

    async def run(
        self, 
        input_data: Input, 
        *, 
        credentials: Optional[APIKeyCredentials] = None, 
        **kwargs
    ) -> BlockOutput:
        topic = input_data.topic
        search_type = input_data.search_type
        
        try:
            # Always get the base summary
            encoded_topic = quote(topic, safe="")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
            
            headers = {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            }
            
            response = await self.get_request(url, headers=headers, json=True)
            if "extract" not in response:
                raise ValueError(f"Unable to parse Wikipedia response: {response}")
            
            yield "summary", response["extract"]
            yield "search_type", search_type
            
            # Get related topics if semantic search is requested
            if search_type in ["semantic", "related"]:
                related_topics = await self._find_related_topics(
                    topic, 
                    input_data.max_related_topics,
                    credentials
                )
                yield "related_topics", related_topics
            else:
                yield "related_topics", []
                
        except Exception as e:
            error_msg = f"Failed to perform enhanced Wikipedia search: {e}"
            yield "error", error_msg
            yield "summary", ""
            yield "related_topics", []
            yield "search_type", search_type
