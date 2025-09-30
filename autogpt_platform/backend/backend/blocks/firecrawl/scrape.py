from typing import Any

from firecrawl import FirecrawlApp

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


class FirecrawlScrapeBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
        url: str = SchemaField(description="The URL to crawl")
        limit: int = SchemaField(description="The number of pages to crawl", default=10)
        only_main_content: bool = SchemaField(
            description="Only return the main content of the page excluding headers, navs, footers, etc.",
            default=True,
        )
        max_age: int = SchemaField(
            description="The maximum age of the page in milliseconds - default is 1 hour",
            default=3600000,
        )
        wait_for: int = SchemaField(
            description="Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load.",
            default=200,
        )
        formats: list[ScrapeFormat] = SchemaField(
            description="The format of the crawl", default=[ScrapeFormat.MARKDOWN]
        )

    class Output(BlockSchema):
        data: dict[str, Any] = SchemaField(description="The result of the crawl")
        markdown: str = SchemaField(description="The markdown of the crawl")
        html: str = SchemaField(description="The html of the crawl")
        raw_html: str = SchemaField(description="The raw html of the crawl")
        links: list[str] = SchemaField(description="The links of the crawl")
        screenshot: str = SchemaField(description="The screenshot of the crawl")
        screenshot_full_page: str = SchemaField(
            description="The screenshot full page of the crawl"
        )
        json_data: dict[str, Any] = SchemaField(
            description="The json data of the crawl"
        )
        change_tracking: dict[str, Any] = SchemaField(
            description="The change tracking of the crawl"
        )

    def __init__(self):
        super().__init__(
            id="ac444320-cf5e-4697-b586-2604c17a3e75",
            description="Firecrawl scrapes a website to extract comprehensive data while bypassing blockers.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(api_key=credentials.api_key.get_secret_value())

        scrape_result = app.scrape(
            input_data.url,
            formats=convert_to_format_options(input_data.formats),
            only_main_content=input_data.only_main_content,
            max_age=input_data.max_age,
            wait_for=input_data.wait_for,
        )
        yield "data", scrape_result

        for f in input_data.formats:
            if f == ScrapeFormat.MARKDOWN:
                yield "markdown", scrape_result.markdown
            elif f == ScrapeFormat.HTML:
                yield "html", scrape_result.html
            elif f == ScrapeFormat.RAW_HTML:
                yield "raw_html", scrape_result.raw_html
            elif f == ScrapeFormat.LINKS:
                yield "links", scrape_result.links
            elif f == ScrapeFormat.SCREENSHOT:
                yield "screenshot", scrape_result.screenshot
            elif f == ScrapeFormat.SCREENSHOT_FULL_PAGE:
                yield "screenshot_full_page", scrape_result.screenshot
            elif f == ScrapeFormat.CHANGE_TRACKING:
                yield "change_tracking", scrape_result.change_tracking
            elif f == ScrapeFormat.JSON:
                yield "json", scrape_result.json
