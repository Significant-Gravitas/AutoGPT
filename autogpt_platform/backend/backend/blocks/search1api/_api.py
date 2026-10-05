from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

SEARCH1API_API_URL = "https://api.search1api.com"

# USD cost of one Search1API credit on the pay-as-you-go plan ($1 = 1,000).
CREDIT_USD = 0.001


class Search1APISearchService(Enum):
    """Engines and sources accepted by POST /search as search_service."""

    GOOGLE = "google"
    BING = "bing"
    BING_CN = "bingcn"
    DUCKDUCKGO = "duckduckgo"
    YAHOO = "yahoo"
    YANDEX = "yandex"
    BAIDU = "baidu"
    QUARK = "quark"
    SO_360 = "360"
    YOUTUBE = "youtube"
    X = "x"
    REDDIT = "reddit"
    GITHUB = "github"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    WECHAT = "wechat"
    BILIBILI = "bilibili"
    IMDB = "imdb"


class Search1APINewsService(Enum):
    """News sources accepted by POST /news as search_service."""

    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    YAHOO = "yahoo"
    HACKERNEWS = "hackernews"
    REUTERS = "reuters"


class Search1APITimeRange(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class Search1APIResult(BaseModel):
    """Schema for a single Search1API search or news result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(description="Snippet shown for the result", default="")
    content: str | None = SchemaField(
        description="Full page content, present when the result was crawled "
        "(crawl_results > 0)",
        default=None,
    )
    published_date: str | None = SchemaField(
        description="Publication date reported by the source, when available",
        default=None,
    )


class Search1APIQueryResults(BaseModel):
    """Results produced for one query of a batch search."""

    query: str = SchemaField(description="The query this result group belongs to")
    results: list[Search1APIResult] = SchemaField(
        description="Results for this query", default_factory=list
    )
    error: str = SchemaField(
        description="Error message if this query failed", default=""
    )


class Search1APIClient:
    """Thin async REST client for api.search1api.com.

    Errors come back as a non-2xx status with an ``{error, message}`` JSON
    body; the message is surfaced so users see e.g. an invalid key or an
    out-of-range parameter instead of a bare status code.
    """

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[SEARCH1API_API_URL],
            raise_for_status=False,
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}"
            },
        )

    async def post(self, path: str, payload: Any) -> Any:
        response = await self.requests.post(f"{SEARCH1API_API_URL}{path}", json=payload)
        if not response.ok:
            raise ValueError(
                f"Search1API {path} request failed with HTTP {response.status}: "
                f"{_error_message(response)}"
            )
        return response.json()


def _error_message(response) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text()[:200] or "no response body"
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:200]
    return str(body)[:200]


def build_search_payload(
    query: str,
    *,
    search_service: Enum | None,
    max_results: int,
    crawl_results: int,
    include_sites: list[str],
    exclude_sites: list[str],
    language: str | None,
    time_range: Search1APITimeRange | None,
) -> dict[str, Any]:
    """Request body shared by /search and /news; unset options are omitted."""
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "crawl_results": crawl_results,
    }
    if search_service:
        payload["search_service"] = search_service.value
    if include_sites:
        payload["include_sites"] = include_sites
    if exclude_sites:
        payload["exclude_sites"] = exclude_sites
    if language:
        payload["language"] = language
    if time_range:
        payload["time_range"] = time_range.value
    return payload


def check_crawl_results(max_results: int, crawl_results: int) -> None:
    """The API only crawls pages it returned, so it rejects (HTTP 422) a
    crawl_results above max_results; catch that before a request is sent."""
    if crawl_results > max_results:
        raise ValueError("crawl_results cannot be greater than max_results")


def results_from_response(response: Any) -> list[Search1APIResult]:
    """Parse the ``results`` list of a /search or /news response."""
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ValueError("malformed Search1API response: missing results list")
    if not all(isinstance(r, dict) for r in response["results"]):
        raise ValueError("malformed Search1API response: non-object result entry")
    return [
        Search1APIResult(
            title=r.get("title") or "",
            url=r.get("link") or "",
            snippet=r.get("snippet") or "",
            content=r.get("content") or None,
            published_date=r.get("published_date") or None,
        )
        for r in response["results"]
    ]


def search_credits(results: list[Search1APIResult], crawl_results: int) -> int:
    """Credits charged for one /search or /news request.

    A request costs 1 credit, plus 1 for each page successfully crawled when
    crawl_results > 0. The single-request endpoints do not report the charge,
    so it is derived from the results that came back with page content.
    """
    crawled = sum(1 for r in results if r.content) if crawl_results > 0 else 0
    return 1 + crawled


def format_context(results: list[Search1APIResult]) -> str:
    """Render results as markdown for LLM input."""
    return "\n\n".join(
        f"[{r.title}]({r.url})\n{r.content or r.snippet}" for r in results
    )
