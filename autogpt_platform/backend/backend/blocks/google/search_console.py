import asyncio
from typing import Any

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._search_console_api import (
    WEBMASTERS_READONLY_SCOPE,
    build_query_body,
    build_search_console_service,
    list_sites,
    search_console_error,
)
from ._search_console_inputs import (
    SITE_URL_DESCRIPTION,
    resolve_dates,
    resolve_site_url,
)
from ._search_console_models import (
    Dimension,
    SearchConsoleFilter,
    SearchConsoleRow,
    SearchConsoleSite,
    SearchType,
    to_row,
    to_site,
)
from ._search_console_testdata import TEST_ANALYTICS_RESPONSE, TEST_SITE_ENTRIES

CATEGORIES = {BlockCategory.MARKETING, BlockCategory.DATA}


class GoogleSearchConsoleListSitesBlock(Block):
    """List the Search Console properties the connected account can see."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [WEBMASTERS_READONLY_SCOPE]
        )

    class Output(BlockSchemaOutput):
        sites: list[SearchConsoleSite] = SchemaField(
            description="Every property the account can see"
        )
        site: SearchConsoleSite = SchemaField(description="Each property")

    def __init__(self):
        sites = [to_site(entry) for entry in TEST_SITE_ENTRIES]
        super().__init__(
            id="e412ccaa-577a-4163-84b6-9bec67f9c542",
            description=(
                "List the Google Search Console properties (sites) the connected "
                "Google account can see, with its permission level for each. Pass "
                "a property's site_url to the other Search Console blocks."
            ),
            categories=CATEGORIES,
            input_schema=GoogleSearchConsoleListSitesBlock.Input,
            output_schema=GoogleSearchConsoleListSitesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("sites", sites), *(("site", site) for site in sites)],
            test_mock={"_list_sites": lambda *args, **kwargs: TEST_SITE_ENTRIES},
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_search_console_service(credentials)
        try:
            entries = await asyncio.to_thread(self._list_sites, service)
        except HttpError as e:
            raise search_console_error(e, self.name, self.id) from e

        sites = [to_site(entry) for entry in entries]
        yield "sites", sites
        for site in sites:
            yield "site", site

    @staticmethod
    def _list_sites(service) -> list[dict[str, Any]]:
        return list_sites(service)


class GoogleSearchConsoleGetPerformanceBlock(Block):
    """Search performance from Search Console's Search Analytics API."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [WEBMASTERS_READONLY_SCOPE]
        )
        site_url: str = SchemaField(
            description=SITE_URL_DESCRIPTION, placeholder="sc-domain:example.com"
        )
        start_date: str = SchemaField(
            description=(
                "First day to count: YYYY-MM-DD, today, yesterday or NdaysAgo, in "
                "Pacific Time like Search Console"
            ),
            default="30daysAgo",
            advanced=False,
        )
        end_date: str = SchemaField(
            description=(
                "Last day to count, in the same forms. Search Console's final "
                "numbers run 2-3 days behind, so the default is 3daysAgo."
            ),
            default="3daysAgo",
            advanced=False,
        )
        dimensions: list[Dimension] = SchemaField(
            description=(
                "What to break the numbers down by, in this order. Leave empty "
                "for one row of totals."
            ),
            default=["query"],
            advanced=False,
        )
        search_type: SearchType = SchemaField(
            description=(
                "Which results to count: web (the main results), image, video, "
                "news (the News tab), discover, or googleNews (news.google.com and "
                "the Google News app). Discover and Google News have no query "
                "dimension and no position."
            ),
            default="web",
            advanced=False,
        )
        filters: list[SearchConsoleFilter] = SchemaField(
            description=(
                "Only count rows that meet every one of these conditions, such as "
                "query contains 'shoes'"
            ),
            default=[],
            advanced=False,
        )
        row_limit: int = SchemaField(
            description="Most rows to return (Google allows up to 25,000)",
            default=100,
            ge=1,
            le=25000,
            advanced=False,
        )
        start_row: int = SchemaField(
            description="Rows to skip, to page through more rows than row_limit",
            default=0,
            ge=0,
        )
        include_fresh_data: bool = SchemaField(
            description=(
                "Also count the last few days, whose numbers aren't final yet and "
                "can still change. Set end_date to today or yesterday to see them."
            ),
            default=False,
        )

    class Output(BlockSchemaOutput):
        rows: list[SearchConsoleRow] = SchemaField(
            description="The rows, most clicks first (oldest first when grouped by date)"
        )
        row: SearchConsoleRow = SchemaField(description="Each row")
        site_url: str = SchemaField(description="The property the numbers are for")

    def __init__(self):
        rows = [
            to_row(row, ["query", "page"]) for row in TEST_ANALYTICS_RESPONSE["rows"]
        ]
        super().__init__(
            id="01ed30bc-1d35-427b-98ab-c8b2a7106937",
            description=(
                "Get a site's search performance from Google Search Console: "
                "clicks, impressions, CTR and average position, by query, page, "
                "country, device or date. This is the Search Analytics data behind "
                "Search Console's Performance report, for SEO questions such as a "
                "website's top search queries (keywords), its top pages and how "
                "they rank. It can filter rows and also report on image, video, "
                "news, Discover and Google News results."
            ),
            categories=CATEGORIES,
            input_schema=GoogleSearchConsoleGetPerformanceBlock.Input,
            output_schema=GoogleSearchConsoleGetPerformanceBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "site_url": "example.com",
                "dimensions": ["query", "page"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("rows", rows),
                *(("row", row) for row in rows),
                ("site_url", "sc-domain:example.com"),
            ],
            test_mock={
                "_list_sites": lambda *args, **kwargs: TEST_SITE_ENTRIES,
                "_query": lambda *args, **kwargs: TEST_ANALYTICS_RESPONSE,
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        start_date, end_date = resolve_dates(
            input_data.start_date, input_data.end_date, self.name, self.id
        )
        dimensions: list[str] = list(dict.fromkeys(input_data.dimensions))
        service = build_search_console_service(credentials)
        site_url = await resolve_site_url(
            input_data.site_url, lambda: self._list_sites(service), self.name, self.id
        )
        body = build_query_body(
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            search_type=input_data.search_type,
            filters=input_data.filters,
            row_limit=input_data.row_limit,
            start_row=input_data.start_row,
            include_fresh_data=input_data.include_fresh_data,
        )
        try:
            response = await asyncio.to_thread(self._query, service, site_url, body)
        except HttpError as e:
            raise search_console_error(e, self.name, self.id, site_url=site_url) from e

        rows = [to_row(row, dimensions) for row in response.get("rows", [])]
        yield "rows", rows
        for row in rows:
            yield "row", row
        yield "site_url", site_url

    @staticmethod
    def _list_sites(service) -> list[dict[str, Any]]:
        return list_sites(service)

    @staticmethod
    def _query(service, site_url: str, body: dict[str, Any]) -> dict[str, Any]:
        return service.searchanalytics().query(siteUrl=site_url, body=body).execute()
