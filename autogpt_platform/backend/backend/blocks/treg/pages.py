"""Treg blocks that read web pages and map websites."""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._api import TregError
from ._base import TregBlockBase, TregRouted, TregRoutedInput, TregRoutedOutput, rows
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details

# Where providers put a page's text, most useful first.
PAGE_TEXT_KEYS = ("markdown", "content", "text", "raw_content", "html")


def page_text(routed: TregRouted) -> str | None:
    """The first page's text, from whichever field its provider uses."""
    for page in rows(routed, "pages")[:1]:
        for key in PAGE_TEXT_KEYS:
            if isinstance(page.get(key), str) and page[key].strip():
                return page[key]
    return None


class TregExtractWebPageBlock(TregBlockBase):
    class Input(TregRoutedInput):
        url: str = SchemaField(
            description="The page to read, e.g. 'https://modal.com/pricing'.",
            min_length=1,
        )

    class Output(TregRoutedOutput):
        content: str = SchemaField(
            description="The page's main content as clean text or Markdown."
        )
        pages: list[dict[str, Any]] = SchemaField(
            description="The extracted page, in the serving provider's own format, "
            "with any metadata it returns."
        )

    def __init__(self):
        super().__init__(
            id="3e5b9d7f-2a1c-4b84-8e6d-0f4a7c2e9b31",
            description=(
                "Read a web page's main content as clean text, through Treg's "
                "cheapest working page extractor."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://modal.com/pricing",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("content", "# Pricing\nPay per second."),
                (
                    "pages",
                    [
                        {
                            "url": "https://modal.com/pricing",
                            "markdown": "# Pricing\nPay per second.",
                        }
                    ],
                ),
                *routed_details(served_by="crawl4ai.extract", cost_usd=0.0),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "pages": [
                            {
                                "url": "https://modal.com/pricing",
                                "markdown": "# Pricing\nPay per second.",
                            }
                        ],
                        "count": 1,
                    },
                    served_by="crawl4ai.extract",
                    cost_usd=0.0,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials, "web.extract", {"url": input_data.url}, input_data
        )
        pages = rows(routed, "pages")
        content = page_text(routed)
        if content is not None:
            yield "content", content
        if pages:
            yield "pages", pages
        for output in self.route_details(routed, bool(pages)):
            yield output


class TregExtractStructuredDataBlock(TregBlockBase):
    class Input(TregRoutedInput):
        url: str = SchemaField(description="The page to read.", min_length=1)
        instruction: str = SchemaField(
            default="",
            description="What to pull out, in plain words, e.g. 'every pricing plan "
            "with its monthly price'.",
        )
        json_schema: dict[str, Any] = SchemaField(
            default_factory=dict,
            description="A JSON schema the data must follow. Give this, an "
            "instruction, or both.",
        )

    class Output(TregRoutedOutput):
        data: Any = SchemaField(
            description="The extracted data, in the shape your schema or "
            "instruction asked for."
        )

    def __init__(self):
        super().__init__(
            id="b0c4f8e2-7d3a-4196-a5b8-9e1d6c3f0a47",
            description=(
                "Pull typed data out of a web page, described by a JSON schema or a "
                "plain-English instruction, through Treg."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://modal.com/pricing",
                "instruction": "Every plan with its monthly price",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("data", [{"plan": "Starter", "monthly_price": 0}]),
                *routed_details(served_by="firecrawl.extract", cost_usd=0.003),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"data": [{"plan": "Starter", "monthly_price": 0}]},
                    served_by="firecrawl.extract",
                    cost_usd=0.003,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.instruction and not input_data.json_schema:
            raise TregError(
                "Nothing to extract: give an instruction, a JSON schema, or both."
            )
        routed = await self.route(
            credentials,
            "web.extract.structured",
            {
                "url": input_data.url,
                "instruction": input_data.instruction,
                "schema": input_data.json_schema,
            },
            input_data,
        )
        data = routed.get("data")
        found = data is not None
        if found:
            yield "data", data
        for output in self.route_details(routed, found):
            yield output


class TregMapWebsiteBlock(TregBlockBase):
    class Input(TregRoutedInput):
        url: str = SchemaField(
            description="The site to map, e.g. 'https://modal.com'.", min_length=1
        )
        search: str = SchemaField(
            default="",
            description="Only URLs about this, e.g. 'pricing', where the provider "
            "supports it.",
        )
        limit: int = SchemaField(
            default=50, ge=1, le=1000, description="The most URLs to return."
        )

    class Output(TregRoutedOutput):
        urls: list[str] = SchemaField(description="URLs found on the site.")
        results: list[dict[str, Any]] = SchemaField(
            description="The same pages in the serving provider's own format, "
            "with titles where it has them."
        )
        count: int = SchemaField(description="How many URLs were found.")

    def __init__(self):
        super().__init__(
            id="5c7e1a9f-4d6b-4f38-92c0-8a3e5b1d7f69",
            description="List the pages on a website, through Treg.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://modal.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("urls", ["https://modal.com/pricing"]),
                ("results", [{"url": "https://modal.com/pricing", "title": "Pricing"}]),
                ("count", 1),
                *routed_details(served_by="firecrawl.map", cost_usd=0.001),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "results": [
                            {"url": "https://modal.com/pricing", "title": "Pricing"}
                        ],
                        "count": 1,
                    },
                    served_by="firecrawl.map",
                    cost_usd=0.001,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials,
            "web.map",
            {"url": input_data.url, "q": input_data.search, "limit": input_data.limit},
            input_data,
        )
        # Some providers list bare URL strings rather than objects.
        listed = routed.get("results")
        listed = listed if isinstance(listed, list) else []
        results = [r if isinstance(r, dict) else {"url": r} for r in listed]
        urls = [r["url"] for r in results if isinstance(r.get("url"), str)]
        yield "urls", urls
        yield "results", results
        yield "count", len(urls)
        for output in self.route_details(routed, bool(results)):
            yield output
