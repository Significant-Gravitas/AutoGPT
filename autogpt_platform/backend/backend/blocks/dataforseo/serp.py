"""
DataForSEO Google Search Results block: a live Google results page for a keyword.
"""

from typing import Any, Literal
from urllib.parse import urlparse

from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)

from ._api import DataForSeoClient, record_cost
from ._config import dataforseo

SERP_PATH = "/v3/serp/google/organic/live/advanced"
# Google's click-through redirect. DataForSEO normally resolves it to the real
# page, but when it can't, the breadcrumb is the only clue to the site.
GOOGLE_REDIRECT_PREFIX = "https://google.com/goto?"


class OrganicResult(BaseModel):
    position: int
    title: str
    url: str
    domain: str
    description: str
    url_resolved: bool


class DataForSeoGoogleSearchResultsBlock(Block):
    """Live Google results for one keyword, as a searcher in that place sees them."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        keyword: str = SchemaField(description="The search query to look up")
        location_code: int = SchemaField(
            description="Location code for targeting (e.g., 2840 for USA)",
            default=2840,
        )
        language_code: str = SchemaField(
            description="Language code (e.g., 'en' for English)", default="en"
        )
        device: Literal["desktop", "mobile"] = SchemaField(
            description="Which device's results to fetch", default="desktop"
        )
        depth: int = SchemaField(
            description="How many results to fetch (10 per page; billed per 10)",
            default=10,
            ge=10,
            le=100,
        )

    class Output(BlockSchemaOutput):
        organic_results: list[OrganicResult] = SchemaField(
            description=(
                "Unpaid results in order. position counts organic listings only "
                "(1 = first organic result). url_resolved is false when Google's "
                "redirect link could not be resolved; domain then comes from the "
                "result's breadcrumb"
            )
        )
        ai_overview: str = SchemaField(
            description="Google's AI Overview as markdown, or empty if none showed"
        )
        people_also_ask: list[str] = SchemaField(
            description="Questions in the 'People also ask' box"
        )
        related_searches: list[str] = SchemaField(
            description="Searches Google suggests at the bottom of the page"
        )
        result_types: list[str] = SchemaField(
            description="Every kind of result on the page, e.g. organic, video, local_pack"
        )
        check_url: str = SchemaField(
            description="Google URL that shows the same results in a browser"
        )

    def __init__(self):
        super().__init__(
            id="77531cc5-598a-4245-8657-b6a5340606de",
            description=(
                "Get live Google search results for a keyword from DataForSEO: "
                "organic rankings, AI Overview, People also ask and related searches"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "keyword": "candle business",
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "organic_results",
                    [
                        OrganicResult(
                            position=1,
                            title="Start a candle business",
                            url="https://www.shopify.com/blog/candle-business",
                            domain="www.shopify.com",
                            description="A guide",
                            url_resolved=True,
                        ),
                        OrganicResult(
                            position=2,
                            title="Candle guide",
                            url="https://google.com/goto?url=abc",
                            domain="villagecraftandcandle.com",
                            description="",
                            url_resolved=False,
                        ),
                    ],
                ),
                ("ai_overview", ""),
                ("people_also_ask", ["Is a candle business profitable?"]),
                ("related_searches", ["candle business names"]),
                ("result_types", ["organic", "people_also_ask"]),
                ("check_url", "https://www.google.com/search?q=candle+business"),
            ],
            test_mock={"_fetch": lambda *args, **kwargs: _TEST_SERP},
            effect=BlockEffect.READ,
        )

    async def _fetch(self, client: DataForSeoClient, input_data: Input) -> list:
        return await client.post_task(
            SERP_PATH,
            {
                "keyword": input_data.keyword,
                "location_code": input_data.location_code,
                "language_code": input_data.language_code,
                "device": input_data.device,
                "depth": input_data.depth,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        client = DataForSeoClient(credentials)
        results = await self._fetch(client, input_data)
        record_cost(self, client)
        page = results[0] if results else {}
        items = page.get("items") or []
        yield "organic_results", parse_organic(items)
        yield "ai_overview", next(
            (i.get("markdown") or "" for i in items if i["type"] == "ai_overview"), ""
        )
        yield "people_also_ask", [
            q.get("title") or ""
            for i in items
            if i["type"] == "people_also_ask"
            for q in i.get("items") or []
        ]
        yield "related_searches", [
            s
            for i in items
            if i["type"] == "related_searches"
            for s in i.get("items") or []
        ]
        yield "result_types", page.get("item_types") or []
        yield "check_url", page.get("check_url") or ""


def parse_organic(items: list[dict[str, Any]]) -> list[OrganicResult]:
    organic = [i for i in items if i.get("type") == "organic"]
    return [
        _organic_result(position, item)
        for position, item in enumerate(organic, start=1)
    ]


def _organic_result(position: int, item: dict[str, Any]) -> OrganicResult:
    url = item.get("url") or ""
    resolved = not url.startswith(GOOGLE_REDIRECT_PREFIX)
    domain = (item.get("domain") or "") if resolved else _breadcrumb_domain(item)
    return OrganicResult(
        position=position,
        title=item.get("title") or "",
        url=url,
        domain=domain,
        description=item.get("description") or "",
        url_resolved=resolved,
    )


def _breadcrumb_domain(item: dict[str, Any]) -> str:
    """The site from a breadcrumb like 'https://example.com › Blog', else ''."""
    first = (item.get("breadcrumb") or "").split(" ")[0]
    return (urlparse(first).hostname or "") if first.startswith("http") else ""


_TEST_SERP = [
    {
        "check_url": "https://www.google.com/search?q=candle+business",
        "item_types": ["organic", "people_also_ask"],
        "items": [
            {
                "type": "organic",
                "title": "Start a candle business",
                "url": "https://www.shopify.com/blog/candle-business",
                "domain": "www.shopify.com",
                "description": "A guide",
            },
            {
                "type": "people_also_ask",
                "items": [{"title": "Is a candle business profitable?"}],
            },
            {
                "type": "organic",
                "title": "Candle guide",
                "url": "https://google.com/goto?url=abc",
                "domain": "google.com",
                "breadcrumb": "https://villagecraftandcandle.com › Blog",
            },
            {"type": "related_searches", "items": ["candle business names"]},
        ],
    }
]
