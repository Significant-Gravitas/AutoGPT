from types import SimpleNamespace
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


class FastCRWCrawlBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = fastcrw.credentials_field()
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
            default=0,
        )
        formats: list[ScrapeFormat] = SchemaField(
            description="The format of the crawl", default=[ScrapeFormat.MARKDOWN]
        )

    class Output(BlockSchemaOutput):
        data: list[dict[str, Any]] = SchemaField(description="The result of the crawl")
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
        error: str = SchemaField(
            description="Error message if the crawl failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="bed8f214-6204-4794-af9b-423c87494170",
            description="fastCRW crawls websites to extract comprehensive data while bypassing blockers.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": fastcrw.get_test_credentials().model_dump(),
                "url": "https://example.com",
                "limit": 1,
                "formats": [ScrapeFormat.MARKDOWN.value],
            },
            test_credentials=fastcrw.get_test_credentials(),
            test_output=[
                ("data", lambda x: isinstance(x, list) and len(x) == 1),
                ("markdown", "# Example"),
            ],
            test_mock={
                "_crawl": lambda *args, **kwargs: SimpleNamespace(
                    data=[
                        SimpleNamespace(
                            markdown="# Example",
                            html="<h1>Example</h1>",
                            raw_html="<html></html>",
                            links=["https://example.com"],
                            screenshot="",
                            change_tracking={},
                            json={},
                        )
                    ]
                )
            },
        )

    def _crawl(self, app: FirecrawlApp, input_data: Input) -> Any:
        """SDK call isolated so it can be mocked in block self-tests."""
        limit = max(0, input_data.limit)
        max_age = max(0, input_data.max_age)
        wait_for = max(0, input_data.wait_for)

        return app.crawl(
            input_data.url,
            limit=limit,
            scrape_options=ScrapeOptions(
                formats=convert_to_format_options(input_data.formats),
                only_main_content=input_data.only_main_content,
                max_age=max_age,
                wait_for=wait_for,
            ),
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(
            api_key=credentials.api_key.get_secret_value(),
            api_url=get_fastcrw_api_url(),
        )

        # Sync call
        crawl_result = self._crawl(app, input_data)
        # fastCRW mirrors Firecrawl's credit model: 1 credit (~$0.001) per
        # crawled page. crawl_result.data is the list of scraped pages returned.
        pages_data = crawl_result.data or []
        pages = len(pages_data)
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=pages * 0.001, provider_cost_type="cost_usd"
            )
        )
        yield "data", pages_data

        for data in pages_data:
            for f in input_data.formats:
                if f == ScrapeFormat.MARKDOWN:
                    yield "markdown", data.markdown
                elif f == ScrapeFormat.HTML:
                    yield "html", data.html
                elif f == ScrapeFormat.RAW_HTML:
                    yield "raw_html", data.raw_html
                elif f == ScrapeFormat.LINKS:
                    yield "links", data.links
                elif f == ScrapeFormat.SCREENSHOT:
                    yield "screenshot", data.screenshot
                elif f == ScrapeFormat.SCREENSHOT_FULL_PAGE:
                    yield "screenshot_full_page", data.screenshot
                elif f == ScrapeFormat.CHANGE_TRACKING:
                    yield "change_tracking", data.change_tracking
                elif f == ScrapeFormat.JSON:
                    yield "json_data", data.json
