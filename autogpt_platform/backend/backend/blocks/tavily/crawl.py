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

from ._api import (
    CREDIT_USD,
    TavilyExtractDepth,
    TavilyFormat,
    TavilyPageContent,
    credits_from_response,
)
from ._config import tavily


class TavilyCrawlBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = tavily.credentials_field(
            description="The Tavily integration requires an API Key."
        )
        url: str = SchemaField(description="The root URL to start crawling from")
        limit: int = SchemaField(
            description="Maximum number of pages to crawl",
            default=10,
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
            description="Natural language instructions guiding which pages to crawl (e.g. 'Find all the API reference pages')",
            default=None,
        )
        extract_depth: TavilyExtractDepth = SchemaField(
            description="Depth of the extraction: basic or advanced (retrieves more data, including tables and embedded content)",
            default=TavilyExtractDepth.BASIC,
            advanced=True,
        )
        format: TavilyFormat = SchemaField(
            description="The format of the extracted content",
            default=TavilyFormat.MARKDOWN,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[TavilyPageContent] = SchemaField(
            description="List of crawled pages with their content"
        )
        result: TavilyPageContent = SchemaField(description="Single crawled page")
        error: str = SchemaField(
            description="Error message if the crawl failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="ad89e82c-cdab-4784-bf54-9666ab0d2952",
            description="Crawls a website with Tavily, following links from the root URL and extracting page content",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": tavily.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
                "limit": 2,
            },
            test_credentials=tavily.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 2),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("result", lambda x: x.url == "https://agpt.co/blog"),
            ],
            test_mock={
                "_crawl": lambda *args, **kwargs: {
                    "base_url": "https://agpt.co",
                    "results": [
                        {
                            "url": "https://agpt.co",
                            "raw_content": "AutoGPT is a platform for building AI agents.",
                        },
                        {
                            "url": "https://agpt.co/blog",
                            "raw_content": "AutoGPT blog.",
                        },
                    ],
                    "usage": {"credits": 2},
                }
            },
        )

    async def _crawl(self, credentials: APIKeyCredentials, **kwargs) -> dict[str, Any]:
        """Private method to call the Tavily API - can be mocked for testing."""
        client = AsyncTavilyClient(api_key=credentials.api_key.get_secret_value())
        return await client.crawl(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        sdk_kwargs: dict[str, Any] = {
            "url": input_data.url,
            "limit": input_data.limit,
            "max_depth": input_data.max_depth,
            "extract_depth": input_data.extract_depth.value,
            "format": input_data.format.value,
            "include_usage": True,
        }
        if input_data.instructions:
            sdk_kwargs["instructions"] = input_data.instructions

        try:
            response = await self._crawl(credentials, **sdk_kwargs)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Crawl failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        results = [
            TavilyPageContent(
                url=r.get("url") or "",
                raw_content=r.get("raw_content") or "",
            )
            for r in response.get("results", [])
        ]

        # Actual credit spend from the API's usage report; falls back to the
        # documented schedule (crawl combines mapping and extraction costs:
        # 1 credit per 10 pages mapped, 2 per 10 with instructions, plus
        # 1 credit per 5 pages extracted, 2 per 5 for advanced extraction).
        pages = len(results)
        map_credits = math.ceil(pages / 10) * (2 if input_data.instructions else 1)
        extract_credits = math.ceil(pages / 5) * (
            2 if input_data.extract_depth == TavilyExtractDepth.ADVANCED else 1
        )
        credits_used = credits_from_response(
            response, estimate=map_credits + extract_credits
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_used * CREDIT_USD,
                provider_cost_type="cost_usd",
            )
        )

        yield "results", results
        for result in results:
            yield "result", result
