from typing import Any

from firecrawl import FirecrawlApp
from firecrawl.v2.types import ScrapeOptions

from backend.blocks.fastcrw._api import ScrapeFormat, get_fastcrw_api_url
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
from ._format_utils import convert_to_format_options


class FastCRWSearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = fastcrw.credentials_field()
        query: str = SchemaField(description="The query to search for")
        limit: int = SchemaField(description="The number of pages to crawl", default=10)
        max_age: int = SchemaField(
            description="The maximum age of the page in milliseconds - default is 1 hour",
            default=3600000,
        )
        wait_for: int = SchemaField(
            description="Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load.",
            default=200,
        )
        formats: list[ScrapeFormat] = SchemaField(
            description="Returns the content of the search if specified", default=[]
        )

    class Output(BlockSchemaOutput):
        data: dict[str, Any] = SchemaField(description="The result of the search")
        site: dict[str, Any] = SchemaField(description="The site of the search")
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="0c721b99-9b95-4031-8fc0-5df3b8511308",
            description="fastCRW searches the web for the given query.",
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
        scrape_result = app.search(
            input_data.query,
            limit=input_data.limit,
            scrape_options=ScrapeOptions(
                formats=convert_to_format_options(input_data.formats) or None,
                max_age=input_data.max_age,
                wait_for=input_data.wait_for,
            ),
        )
        # fastCRW mirrors Firecrawl's credit model: billed per returned web
        # result (~1 credit each). The SearchResponse structure exposes `.web`
        # when scrape_options was requested; fall back to `limit` as an upper
        # bound estimate.
        web_results = getattr(scrape_result, "web", None) or []
        billed_units = max(len(web_results), 1)
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=billed_units * 0.001,
                provider_cost_type="cost_usd",
            )
        )
        yield "data", scrape_result
        if hasattr(scrape_result, "web") and scrape_result.web:
            for site in scrape_result.web:
                yield "site", site
