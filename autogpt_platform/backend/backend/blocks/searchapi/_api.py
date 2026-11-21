from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField
from backend.util.request import Response

SEARCHAPI_API_URL = "https://www.searchapi.io"

# Upper bound of the USD cost of one search: the Developer plan's $4 per
# 1,000 searches. Larger plans cost less per search and failed requests are
# not billed, but the response carries no price, so every successful call is
# recorded at this fixed rate on purpose, the same flat per-call estimate the
# Serply and Firecrawl blocks use.
SEARCH_COST_USD = 0.004

# Google returns at most 10 organic results per page; SearchApi honours
# ``num`` up to that.
MAX_RESULTS = 10


class SearchApiTimeRange(Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


# SearchApi ``time_period`` values for each time range.
TIME_RANGE_PARAM = {
    SearchApiTimeRange.HOUR: "last_hour",
    SearchApiTimeRange.DAY: "last_day",
    SearchApiTimeRange.WEEK: "last_week",
    SearchApiTimeRange.MONTH: "last_month",
    SearchApiTimeRange.YEAR: "last_year",
}


class SearchApiSearchResult(BaseModel):
    """Schema for a single SearchApi organic result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    description: str = SchemaField(description="Snippet shown for the result page")
    position: int = SchemaField(description="Rank of the result on the page (1-based)")


class SearchApiClient:
    """Thin async REST client for www.searchapi.io (GET /api/v1/search)."""

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[SEARCHAPI_API_URL],
            # Read the body of a 4xx ourselves: SearchApi explains the failure
            # in it ({"error": "Invalid API key."}).
            raise_for_status=False,
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
                "User-Agent": "AutoGPT",
            },
        )

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.get(
            f"{SEARCHAPI_API_URL}/api/v1/search", params=params
        )
        if not response.ok:
            raise ValueError(
                f"SearchApi search request failed with HTTP {response.status}: "
                f"{error_from_response(response)}"
            )
        return response.json()


def error_from_response(response: Response) -> str:
    """The API's own error message when the body carries one, else the HTTP
    reason phrase."""
    try:
        body = response.json()
    except ValueError:
        return response.reason or ""
    return body.get("error") or response.reason or ""


def results_from_response(response: dict[str, Any]) -> list[SearchApiSearchResult]:
    """Parse organic rows, dropping any whose link is not an http(s) URL.

    A page with no results is a 200 with an ``error`` string and no
    ``organic_results``; that is an empty list, not a failure. Links that
    Google wraps in a ``google.com/goto?url=`` redirect are passed through.
    """
    results: list[SearchApiSearchResult] = []
    for r in response.get("organic_results") or []:
        url = r.get("link") or ""
        if not url.startswith(("http://", "https://")):
            continue
        results.append(
            SearchApiSearchResult(
                title=r.get("title") or "",
                url=url,
                description=r.get("snippet") or "",
                position=r.get("position") or len(results) + 1,
            )
        )
    return results
