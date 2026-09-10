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


class TavilyExtractBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = tavily.credentials_field(
            description="The Tavily integration requires an API Key."
        )
        urls: list[str] = SchemaField(
            description="The URLs to extract content from (up to 20 per request)",
            max_length=20,
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
            description="List of successfully extracted pages"
        )
        result: TavilyPageContent = SchemaField(description="Single extracted page")
        failed_urls: list[str] = SchemaField(
            description="URLs that could not be extracted"
        )
        error: str = SchemaField(
            description="Error message if the extraction failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d90fc44b-d462-42be-8327-09b75ad06721",
            description="Extracts page content from one or more URLs using Tavily, optimized for LLM consumption",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": tavily.get_test_credentials().model_dump(),
                "urls": ["https://agpt.co"],
            },
            test_credentials=tavily.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("failed_urls", []),
            ],
            test_mock={
                "_extract": lambda *args, **kwargs: {
                    "results": [
                        {
                            "url": "https://agpt.co",
                            "raw_content": "AutoGPT is a platform for building AI agents.",
                        }
                    ],
                    "failed_results": [],
                    "usage": {"credits": 1},
                }
            },
        )

    async def _extract(
        self, credentials: APIKeyCredentials, **kwargs
    ) -> dict[str, Any]:
        """Private method to call the Tavily API - can be mocked for testing."""
        client = AsyncTavilyClient(api_key=credentials.api_key.get_secret_value())
        return await client.extract(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            response = await self._extract(
                credentials,
                urls=input_data.urls,
                extract_depth=input_data.extract_depth.value,
                format=input_data.format.value,
                include_usage=True,
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Extract failed: {e}",
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
        # documented schedule (1 credit per 5 successfully extracted URLs,
        # 2 per 5 for advanced extraction; failed extractions are free).
        # The usage report is incremental: extract charges at each 5-URL
        # account boundary, so 0 credits on a given request is a valid
        # report, not a missing one.
        credits_per_batch = (
            2 if input_data.extract_depth == TavilyExtractDepth.ADVANCED else 1
        )
        credits_used = credits_from_response(
            response, estimate=math.ceil(len(results) / 5) * credits_per_batch
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

        yield "failed_urls", [
            f.get("url", "") for f in response.get("failed_results", [])
        ]
