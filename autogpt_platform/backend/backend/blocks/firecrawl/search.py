from typing import Any

from firecrawl import FirecrawlApp
from firecrawl.v2.types import ScrapeOptions

from backend.blocks.firecrawl._api import ScrapeFormat
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import firecrawl
from ._format_utils import convert_to_format_options


class FirecrawlSearchBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
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

    class Output(BlockSchema):
        data: dict[str, Any] = SchemaField(description="The result of the search")
        site: dict[str, Any] = SchemaField(description="The site of the search")

    def __init__(self):
        super().__init__(
            id="f8d2f28d-b3a1-405b-804e-418c087d288b",
            description="Firecrawl searches the web for the given query.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(api_key=credentials.api_key.get_secret_value())

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
        yield "data", scrape_result
        if hasattr(scrape_result, "web") and scrape_result.web:
            for site in scrape_result.web:
                yield "site", site
