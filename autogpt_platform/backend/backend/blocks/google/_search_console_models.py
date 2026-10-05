"""Models for the Google Search Console blocks, and the parsers that fill them."""

from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

Dimension = Literal["query", "page", "country", "device", "date", "searchAppearance"]
SearchType = Literal["web", "image", "video", "news", "discover", "googleNews"]

_ROW_FIELDS = {
    "query": "query",
    "page": "page",
    "country": "country",
    "device": "device",
    "date": "date",
    "searchAppearance": "search_appearance",
}
_INDEX_STATUS_OUTPUTS = {
    "verdict": "verdict",
    "coverageState": "coverage_state",
    "indexingState": "indexing_state",
    "robotsTxtState": "robots_txt_state",
    "pageFetchState": "page_fetch_state",
    "lastCrawlTime": "last_crawl_time",
    "crawledAs": "crawled_as",
    "googleCanonical": "google_canonical",
    "userCanonical": "user_canonical",
}


class SearchConsoleSite(BaseModel):
    """A Search Console property the connected Google account can see."""

    site_url: str = Field(
        description=(
            "The property as Search Console names it: sc-domain:example.com for a "
            "domain property, https://www.example.com/ for a URL-prefix property"
        )
    )
    permission_level: str = Field(
        description=(
            "The account's access: siteOwner, siteFullUser, siteRestrictedUser, or "
            "siteUnverifiedUser (listed, but it can't read the data)"
        )
    )


class SearchConsoleFilter(BaseModel):
    """A condition that every counted row must meet."""

    dimension: Literal["query", "page", "country", "device", "searchAppearance"] = (
        Field(
            description=(
                "What to test. country takes 3-letter codes such as usa or gbr; "
                "device takes DESKTOP, MOBILE or TABLET."
            )
        )
    )
    operator: Literal[
        "equals",
        "notEquals",
        "contains",
        "notContains",
        "includingRegex",
        "excludingRegex",
    ] = Field(
        description=(
            "How to compare. contains ignores case; equals is case-sensitive for "
            "query and page; the regex operators take RE2 patterns."
        )
    )
    expression: str = Field(
        min_length=1,
        max_length=4096,
        description="The value or pattern to compare with",
    )


class SearchConsoleRow(BaseModel):
    """One row of Search Console performance data."""

    query: Optional[str] = Field(
        default=None, description="The search query, when grouped by query"
    )
    page: Optional[str] = Field(
        default=None, description="The page URL, when grouped by page"
    )
    country: Optional[str] = Field(
        default=None,
        description="The country as a 3-letter code such as usa, when grouped by country",
    )
    device: Optional[str] = Field(
        default=None, description="DESKTOP, MOBILE or TABLET, when grouped by device"
    )
    date: Optional[str] = Field(
        default=None,
        description="The day as YYYY-MM-DD in Pacific Time, when grouped by date",
    )
    search_appearance: Optional[str] = Field(
        default=None,
        description=(
            "The search feature the result appeared as, such as a rich result, "
            "when grouped by searchAppearance"
        ),
    )
    clicks: int = Field(description="Clicks from Google to the site")
    impressions: int = Field(description="Times a link to the site was shown")
    ctr: float = Field(description="Clicks divided by impressions, from 0 to 1")
    position: Optional[float] = Field(
        default=None,
        description=(
            "Average position in the results, 1 being the top. Empty for Discover "
            "and Google News, which don't report it."
        ),
    )


class SearchConsoleSitemapContent(BaseModel):
    """How many items of one kind a sitemap lists."""

    type: str = Field(
        description="The kind of item: web, image, video, news, mobile, androidApp or iosApp"
    )
    submitted: int = Field(description="How many of them the sitemap lists")


class SearchConsoleSitemap(BaseModel):
    """A sitemap in Search Console, and what Google made of it."""

    path: str = Field(description="The sitemap's URL")
    type: str = Field(
        default="",
        description="sitemap, rssFeed, atomFeed, urlList, patternSitemap or notSitemap",
    )
    last_submitted: Optional[str] = Field(
        default=None, description="When it was last submitted (RFC 3339)"
    )
    last_downloaded: Optional[str] = Field(
        default=None, description="When Google last downloaded it (RFC 3339)"
    )
    is_pending: bool = Field(
        default=False, description="Whether Google has yet to process it"
    )
    is_sitemaps_index: bool = Field(
        default=False,
        description="Whether it's a sitemap index, a list of other sitemaps",
    )
    warnings: int = Field(
        default=0, description="Warnings Google found, mostly about URLs in it"
    )
    errors: int = Field(
        default=0,
        description="Errors in the sitemap itself, which stop Google reading it",
    )
    contents: list[SearchConsoleSitemapContent] = Field(
        default_factory=list,
        description="How many pages, images, videos and so on it lists, by kind",
    )


def to_site(entry: dict[str, Any]) -> SearchConsoleSite:
    return SearchConsoleSite(
        site_url=entry.get("siteUrl", ""),
        permission_level=camel_enum(entry.get("permissionLevel", "")),
    )


def to_row(row: dict[str, Any], dimensions: list[str]) -> SearchConsoleRow:
    """Map a Search Analytics row, whose keys follow the requested dimensions."""
    keys: dict[str, Any] = dict(
        zip((_ROW_FIELDS[name] for name in dimensions), row.get("keys") or [])
    )
    return SearchConsoleRow(
        **keys,
        clicks=round(row.get("clicks") or 0),
        impressions=round(row.get("impressions") or 0),
        ctr=row.get("ctr") or 0.0,
        position=row.get("position") or None,
    )


def to_sitemap(item: dict[str, Any]) -> SearchConsoleSitemap:
    """Map a sitemap resource, leaving out Google's deprecated indexed counts."""
    return SearchConsoleSitemap(
        path=item.get("path", ""),
        type=camel_enum(item.get("type", "")),
        last_submitted=item.get("lastSubmitted"),
        last_downloaded=item.get("lastDownloaded"),
        is_pending=bool(item.get("isPending")),
        is_sitemaps_index=bool(item.get("isSitemapsIndex")),
        warnings=int(item.get("warnings") or 0),
        errors=int(item.get("errors") or 0),
        contents=[
            SearchConsoleSitemapContent(
                type=camel_enum(content.get("type", "")),
                submitted=int(content.get("submitted") or 0),
            )
            for content in item.get("contents") or []
        ],
    )


def inspection_outputs(result: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """The inspection result's most useful fields, as the block's outputs.

    Google leaves out what doesn't apply, such as the canonical of a page it
    hasn't indexed or the rich results of a page without any, and so does this.
    The lists always come out, empty if need be. The retired mobile usability
    result is skipped.
    """
    status = result.get("indexStatusResult") or {}
    rich = result.get("richResultsResult") or {}
    for field, output in _INDEX_STATUS_OUTPUTS.items():
        if status.get(field):
            yield output, status[field]
    yield "sitemaps", status.get("sitemap") or []
    yield "referring_urls", status.get("referringUrls") or []
    if rich.get("verdict"):
        yield "rich_results_verdict", rich["verdict"]
    types = (item.get("richResultType") for item in rich.get("detectedItems") or [])
    yield "rich_result_types", list(dict.fromkeys(name for name in types if name))
    if link := result.get("inspectionResultLink"):
        yield "inspection_result_link", link


def camel_enum(value: str) -> str:
    """Google's documented camelCase for an enum value: SITE_OWNER -> siteOwner.

    The discovery document lists these enums in upper case while the API
    reference documents camelCase, so either form reads the same.
    """
    if not value.isupper():
        return value
    first, *rest = value.lower().split("_")
    return first + "".join(word.capitalize() for word in rest)
