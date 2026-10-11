from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

SERPLY_API_URL = "https://api.serply.io"

# Upper bound of the USD cost of one uncached search (Micro pack, $2.00 per
# 1,000 credits). Cached responses are free, but /v1/search returns no
# cache-hit indicator in its body or headers, so every call is recorded at
# this fixed rate on purpose, the same flat per-call estimate firecrawl's
# scrape and map blocks use.
SEARCH_COST_USD = 0.002

# The API returns at most 10 organic results per page.
MAX_RESULTS = 10


class SerplyTimeRange(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


# Google tbs values for each time range.
TIME_RANGE_TBS = {
    SerplyTimeRange.DAY: "qdr:d",
    SerplyTimeRange.WEEK: "qdr:w",
    SerplyTimeRange.MONTH: "qdr:m",
    SerplyTimeRange.YEAR: "qdr:y",
}


class SerplySearchResult(BaseModel):
    """Schema for a single Serply search result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    description: str = SchemaField(description="Snippet shown for the result page")
    position: int = SchemaField(description="Rank of the result on the page (1-based)")


class SerplyClient:
    """Thin async REST client for api.serply.io (GET /v1/search)."""

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[SERPLY_API_URL],
            extra_headers={
                "X-Api-Key": credentials.api_key.get_secret_value(),
                "User-Agent": "AutoGPT",
            },
        )

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.get(f"{SERPLY_API_URL}/v1/search", params=params)
        if not response.ok:
            raise ValueError(
                f"Serply search request failed with HTTP {response.status}"
            )
        return response.json()


def results_from_response(response: dict[str, Any]) -> list[SerplySearchResult]:
    """Parse organic rows, dropping any whose link is not an http(s) URL."""
    results: list[SerplySearchResult] = []
    for r in response.get("results") or []:
        url = r.get("link") or ""
        if not url.startswith(("http://", "https://")):
            continue
        results.append(
            SerplySearchResult(
                title=r.get("title") or "",
                url=url,
                description=r.get("description") or "",
                position=r.get("position") or len(results) + 1,
            )
        )
    return results
