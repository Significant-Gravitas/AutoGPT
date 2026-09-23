import math
from typing import Any

from tavily import AsyncTavilyClient

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import CREDIT_USD, credits_from_response
from ._config import tavily


class TavilyMapBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = tavily.credentials_field(
            description="The Tavily integration requires an API Key."
        )
        url: str = SchemaField(description="The root URL to map")
        limit: int = SchemaField(
            description="Maximum number of pages to discover",
            default=50,
            ge=1,
            advanced=True,
        )
        max_depth: int = SchemaField(
            description="Maximum link depth from the root URL",
            default=1,
            ge=1,
            advanced=True,
        )
        instructions: str | None = SchemaField(
            description="Natural language instructions guiding which pages to include (e.g. 'Only documentation pages')",
            default=None,
        )

    class Output(BlockSchemaOutput):
        links: list[str] = SchemaField(
            description="List of URLs discovered on the website"
        )
        error: str = SchemaField(
            description="Error message if the map failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="ddc65c9a-c4b0-4389-8604-4aba01a3dfe6",
            description="Maps a website's structure with Tavily, discovering its URLs without extracting content",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": tavily.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=tavily.get_test_credentials(),
            test_output=[
                ("links", ["https://agpt.co", "https://agpt.co/blog"]),
            ],
            test_mock={
                "_map": lambda *args, **kwargs: {
                    "base_url": "https://agpt.co",
                    "results": ["https://agpt.co", "https://agpt.co/blog"],
                    "usage": {"credits": 1},
                }
            },
        )

    async def _map(self, credentials: APIKeyCredentials, **kwargs) -> dict[str, Any]:
        """Private method to call the Tavily API - can be mocked for testing."""
        client = AsyncTavilyClient(api_key=credentials.api_key.get_secret_value())
        return await client.map(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        sdk_kwargs: dict[str, Any] = {
            "url": input_data.url,
            "limit": input_data.limit,
            "max_depth": input_data.max_depth,
            "include_usage": True,
        }
        if input_data.instructions:
            sdk_kwargs["instructions"] = input_data.instructions

        try:
            response = await self._map(credentials, **sdk_kwargs)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Map failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        links = [str(link) for link in response.get("results", [])]

        # Actual credit spend from the API's usage report; falls back to the
        # documented schedule (1 credit per 10 successfully mapped pages,
        # 2 per 10 when instructions are provided).
        credits_used = credits_from_response(
            response,
            estimate=math.ceil(len(links) / 10) * (2 if input_data.instructions else 1),
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_used * CREDIT_USD,
                provider_cost_type="cost_usd",
            )
        )

        yield "links", links
