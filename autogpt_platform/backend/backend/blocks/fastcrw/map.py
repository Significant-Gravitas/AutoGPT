from typing import Any

from firecrawl import FirecrawlApp

from backend.blocks.fastcrw._api import get_fastcrw_api_url
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

from ._config import fastcrw


class FastCRWMapWebsiteBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = fastcrw.credentials_field()

        url: str = SchemaField(description="The website url to map")

    class Output(BlockSchemaOutput):
        links: list[str] = SchemaField(description="List of URLs found on the website")
        results: list[dict[str, Any]] = SchemaField(
            description="List of search results with url, title, and description"
        )
        error: str = SchemaField(
            description="Error message if the map failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c38252c5-77f8-4e9a-aa60-012e9daa5a41",
            description="fastCRW maps a website to extract all the links.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(
            api_key=credentials.api_key.get_secret_value(),
            api_url=get_fastcrw_api_url(),
        )

        # Sync call
        map_result = app.map(
            url=input_data.url,
        )
        # fastCRW mirrors Firecrawl's credit model: 1 credit (~$0.001) per map
        # request.
        self.merge_stats(
            NodeExecutionStats(provider_cost=0.001, provider_cost_type="cost_usd")
        )

        # Convert SearchResult objects to dicts
        results_data = [
            {
                "url": link.url,
                "title": link.title,
                "description": link.description,
            }
            for link in map_result.links
        ]

        yield "links", [link.url for link in map_result.links]
        yield "results", results_data
