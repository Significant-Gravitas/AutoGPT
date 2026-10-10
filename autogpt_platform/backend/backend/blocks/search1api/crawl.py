from typing import Any

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import CREDIT_USD, Search1APIClient
from ._config import search1api


class Search1APICrawlBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = search1api.credentials_field(
            description="The Search1API integration requires an API Key."
        )
        url: str = SchemaField(description="The URL of the page to crawl")

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the crawled page")
        title: str = SchemaField(description="Title of the crawled page")
        content: str = SchemaField(
            description="Page content as markdown (untrusted web content)"
        )

    def __init__(self):
        super().__init__(
            id="8b37052a-fc7e-4403-a2eb-8396abdb5c09",
            description="Crawls a single URL with Search1API and returns its main "
            "content as clean markdown, ready for LLM input",
            categories={BlockCategory.SEARCH},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": search1api.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=search1api.get_test_credentials(),
            test_output=[
                ("url", "https://agpt.co"),
                ("title", "AutoGPT"),
                ("content", lambda x: "AI agents" in x),
            ],
            test_mock={
                "_crawl": lambda *args, **kwargs: {
                    "crawlParameters": {"url": "https://agpt.co"},
                    "results": {
                        "title": "AutoGPT",
                        "link": "https://agpt.co",
                        "content": "AutoGPT is a platform for building AI agents.",
                    },
                }
            },
        )

    async def _crawl(self, credentials: APIKeyCredentials, url: str) -> Any:
        """POST /crawl and return the parsed JSON body (mockable)."""
        return await Search1APIClient(credentials).post("/crawl", {"url": url})

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            response = await self._crawl(credentials, input_data.url)
            page = response.get("results") if isinstance(response, dict) else None
            if not isinstance(page, dict):
                raise ValueError("malformed Search1API response: missing results")
            url = page.get("link") or input_data.url
            title = page.get("title") or ""
            content = page.get("content")
            if not all(isinstance(v, str) for v in (url, title, content)):
                raise ValueError("malformed Search1API response fields")
        except Exception as e:
            raise BlockExecutionError(
                message=f"Crawl failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        self.merge_stats(
            NodeExecutionStats(provider_cost=CREDIT_USD, provider_cost_type="cost_usd")
        )

        yield "url", url
        yield "title", title
        yield "content", content
