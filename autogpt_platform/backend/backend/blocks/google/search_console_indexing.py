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
    build_search_console_service,
    list_sites,
    search_console_error,
)
from ._search_console_inputs import (
    SITE_URL_DESCRIPTION,
    require_page_url,
    resolve_site_url,
)
from ._search_console_models import SearchConsoleSitemap, inspection_outputs, to_sitemap
from ._search_console_testdata import (
    TEST_INSPECTED_URL,
    TEST_INSPECTION_RESULT,
    TEST_SITEMAPS_RESPONSE,
)

CATEGORIES = {BlockCategory.MARKETING, BlockCategory.DATA}


class GoogleSearchConsoleInspectURLBlock(Block):
    """Google's index status for one URL, from the URL Inspection API."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [WEBMASTERS_READONLY_SCOPE]
        )
        inspection_url: str = SchemaField(
            description="The full URL of the page to inspect, inside the property",
            placeholder="https://www.example.com/pricing",
        )
        site_url: str = SchemaField(
            description=(
                f"{SITE_URL_DESCRIPTION} For a bare domain, only properties that "
                "contain the URL count."
            ),
            placeholder="sc-domain:example.com",
        )
        language_code: str = SchemaField(
            description=(
                "Language for Google's issue messages, as a BCP-47 code such as "
                "en-US or de-CH"
            ),
            default="en-US",
        )

    class Output(BlockSchemaOutput):
        verdict: str = SchemaField(
            description=(
                "Google's verdict on the page: PASS (indexed), NEUTRAL (excluded, "
                "for example by noindex or as a duplicate) or FAIL (an error)"
            )
        )
        coverage_state: str = SchemaField(
            description=(
                "Why the page is or isn't indexed, in Search Console's words, such "
                "as 'Submitted and indexed' or 'Crawled - currently not indexed'"
            )
        )
        indexing_state: str = SchemaField(
            description=(
                "INDEXING_ALLOWED, or BLOCKED_BY_META_TAG or BLOCKED_BY_HTTP_HEADER "
                "when a noindex rule blocks it"
            )
        )
        robots_txt_state: str = SchemaField(
            description="ALLOWED or DISALLOWED by the site's robots.txt"
        )
        page_fetch_state: str = SchemaField(
            description=(
                "Whether Google could fetch the page: SUCCESSFUL, or a problem such "
                "as SOFT_404, NOT_FOUND, SERVER_ERROR or REDIRECT_ERROR"
            )
        )
        last_crawl_time: str = SchemaField(
            description="When Google last crawled the page (RFC 3339, UTC), if ever"
        )
        crawled_as: str = SchemaField(
            description="The crawler Google used: MOBILE or DESKTOP"
        )
        google_canonical: str = SchemaField(
            description="The URL Google picked as canonical, once the page is indexed"
        )
        user_canonical: str = SchemaField(
            description="The canonical URL the page declares, if it declares one"
        )
        sitemaps: list[str] = SchemaField(
            description="Sitemaps that Google knows list the URL (not always all of them)"
        )
        referring_urls: list[str] = SchemaField(
            description="Pages that Google knows link to the URL"
        )
        rich_results_verdict: str = SchemaField(
            description="PASS or FAIL for the page's rich results, if it has any"
        )
        rich_result_types: list[str] = SchemaField(
            description="The kinds of rich result found, such as Breadcrumbs or FAQ"
        )
        inspection_result_link: str = SchemaField(
            description="Link to the URL's report in Search Console"
        )
        inspection_result: dict[str, Any] = SchemaField(
            description="Google's full inspection result, including AMP and rich result issues"
        )
        site_url: str = SchemaField(description="The property the URL was inspected in")

    def __init__(self):
        super().__init__(
            id="41fd16c2-72b7-4c69-b1c7-16e66d49fc0b",
            description=(
                "Inspect a URL with Google Search Console: whether Google has "
                "indexed it and if not why, when it was last crawled, its canonical "
                "and any rich results. It reports the version in Google's index and "
                "can't test the live page."
            ),
            categories=CATEGORIES,
            input_schema=GoogleSearchConsoleInspectURLBlock.Input,
            output_schema=GoogleSearchConsoleInspectURLBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inspection_url": TEST_INSPECTED_URL,
                "site_url": "sc-domain:example.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *inspection_outputs(TEST_INSPECTION_RESULT),
                ("inspection_result", TEST_INSPECTION_RESULT),
                ("site_url", "sc-domain:example.com"),
            ],
            test_mock={
                "_inspect": lambda *args, **kwargs: {
                    "inspectionResult": TEST_INSPECTION_RESULT
                }
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        url = require_page_url(input_data.inspection_url, self.name, self.id)
        service = build_search_console_service(credentials)
        site_url = await resolve_site_url(
            input_data.site_url,
            lambda: self._list_sites(service),
            self.name,
            self.id,
            inspection_url=url,
        )
        language_code = input_data.language_code.strip() or "en-US"
        try:
            response = await asyncio.to_thread(
                self._inspect, service, url, site_url, language_code
            )
        except HttpError as e:
            raise search_console_error(
                e, self.name, self.id, site_url=site_url, inspection_url=url
            ) from e

        result = response.get("inspectionResult") or {}
        for name, value in inspection_outputs(result):
            yield name, value
        yield "inspection_result", result
        yield "site_url", site_url

    @staticmethod
    def _list_sites(service) -> list[dict[str, Any]]:
        return list_sites(service)

    @staticmethod
    def _inspect(
        service, inspection_url: str, site_url: str, language_code: str
    ) -> dict[str, Any]:
        body = {
            "inspectionUrl": inspection_url,
            "siteUrl": site_url,
            "languageCode": language_code,
        }
        return service.urlInspection().index().inspect(body=body).execute()


class GoogleSearchConsoleListSitemapsBlock(Block):
    """The sitemaps Search Console knows for a property."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [WEBMASTERS_READONLY_SCOPE]
        )
        site_url: str = SchemaField(
            description=SITE_URL_DESCRIPTION, placeholder="sc-domain:example.com"
        )
        sitemap_index: str = SchemaField(
            description=(
                "The full URL of a sitemap index, to list the sitemaps inside it "
                "instead"
            ),
            default="",
        )

    class Output(BlockSchemaOutput):
        sitemaps: list[SearchConsoleSitemap] = SchemaField(
            description="The sitemaps, with what Google made of each"
        )
        sitemap: SearchConsoleSitemap = SchemaField(description="Each sitemap")
        site_url: str = SchemaField(description="The property the sitemaps belong to")

    def __init__(self):
        sitemaps = [to_sitemap(item) for item in TEST_SITEMAPS_RESPONSE["sitemap"]]
        super().__init__(
            id="bc0030bc-5e07-4c2d-93a5-e99f05544cd6",
            description=(
                "List a site's sitemaps in Google Search Console: when Google last "
                "read each one, its errors and warnings, and how many URLs it "
                "lists. It can also list the sitemaps inside a sitemap index."
            ),
            categories=CATEGORIES,
            input_schema=GoogleSearchConsoleListSitemapsBlock.Input,
            output_schema=GoogleSearchConsoleListSitemapsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "site_url": "https://www.example.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("sitemaps", sitemaps),
                *(("sitemap", sitemap) for sitemap in sitemaps),
                ("site_url", "https://www.example.com/"),
            ],
            test_mock={
                "_list_sitemaps": lambda *args, **kwargs: TEST_SITEMAPS_RESPONSE
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_search_console_service(credentials)
        site_url = await resolve_site_url(
            input_data.site_url, lambda: self._list_sites(service), self.name, self.id
        )
        try:
            response = await asyncio.to_thread(
                self._list_sitemaps, service, site_url, input_data.sitemap_index.strip()
            )
        except HttpError as e:
            raise search_console_error(e, self.name, self.id, site_url=site_url) from e

        sitemaps = [to_sitemap(item) for item in response.get("sitemap", [])]
        yield "sitemaps", sitemaps
        for sitemap in sitemaps:
            yield "sitemap", sitemap
        yield "site_url", site_url

    @staticmethod
    def _list_sites(service) -> list[dict[str, Any]]:
        return list_sites(service)

    @staticmethod
    def _list_sitemaps(service, site_url: str, sitemap_index: str) -> dict[str, Any]:
        params = {"siteUrl": site_url}
        if sitemap_index:
            params["sitemapIndex"] = sitemap_index
        return service.sitemaps().list(**params).execute()
