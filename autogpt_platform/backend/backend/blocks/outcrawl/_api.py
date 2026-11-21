from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

OUTCRAWL_API_URL = "https://api.outcrawl.ai"

# Pay-as-you-go price of one Outcrawl credit ($49 per 10,000 credits).
CREDIT_USD = 0.0049


class OutcrawlFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    LINKS = "links"


class OutcrawlSearchResult(BaseModel):
    """Schema for a single Outcrawl search hit."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(description="Snippet of the result page", default="")
    rank: int = SchemaField(description="Position of the hit in the results")
    markdown: str | None = SchemaField(
        description="Page content as markdown, when scrape_results is on",
        default=None,
    )


class OutcrawlClient:
    """Thin async REST client for api.outcrawl.ai (/v1/scrape, /v1/search)."""

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[OUTCRAWL_API_URL],
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}"
            },
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.post(f"{OUTCRAWL_API_URL}{path}", json=payload)
        if not response.ok:
            raise ValueError(
                f"Outcrawl {path} request failed with HTTP {response.status}"
            )
        return response.json()

    async def scrape(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/scrape", payload)

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/search", payload)


def credits_from_response(response: dict[str, Any], estimate: float) -> float:
    """Credits consumed per the response's usage report (a numeric string),
    falling back to the documented rate when absent."""
    credits = (response.get("usage") or {}).get("credits")
    if credits is None:
        return estimate
    try:
        return float(credits)
    except (TypeError, ValueError):
        return estimate


def result_from_dict(hit: dict[str, Any], rank: int) -> OutcrawlSearchResult:
    return OutcrawlSearchResult(
        title=hit.get("title") or "",
        url=hit.get("url") or "",
        snippet=hit.get("snippet") or "",
        rank=hit.get("rank") or rank,
        markdown=hit.get("markdown"),
    )
