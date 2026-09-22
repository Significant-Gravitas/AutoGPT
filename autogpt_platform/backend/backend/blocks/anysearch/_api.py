from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

ANYSEARCH_API_URL = "https://api.anysearch.com"


class AnySearchDomain(Enum):
    """Vertical domains accepted by the AnySearch domain search parameter."""

    ACADEMIC = "academic"
    AGRICULTURE = "agriculture"
    BUSINESS = "business"
    CODE = "code"
    ENERGY = "energy"
    ENVIRONMENT = "environment"
    FILM = "film"
    FINANCE = "finance"
    GAMING = "gaming"
    GENERAL = "general"
    HEALTH = "health"
    IP = "ip"
    LEGAL = "legal"
    RESOURCE = "resource"
    SECURITY = "security"
    SOCIAL_MEDIA = "social_media"
    TRAVEL = "travel"


class AnySearchResult(BaseModel):
    """Schema for a single AnySearch search result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(
        description="Query-relevant snippet of the result page", default=""
    )
    content: str | None = SchemaField(
        description="Page content returned for the result, when available",
        default=None,
    )


class AnySearchQueryResults(BaseModel):
    """Results produced for one query of a parallel AnySearch batch."""

    query: str = SchemaField(description="The query this result group belongs to")
    results: list[AnySearchResult] = SchemaField(
        description="Results for this query", default_factory=list
    )
    error: str = SchemaField(
        description="Error message if this query failed", default=""
    )


class AnySearchClient:
    """Thin async REST client for api.anysearch.com (/v1/search, /v1/extract).

    The service answers a {code, message, data} envelope; a non-zero code is
    an API-level error even on HTTP 200. No public per-call pricing is
    published, so callers record no provider_cost.
    """

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[ANYSEARCH_API_URL],
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}"
            },
        )

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.post(
            f"{ANYSEARCH_API_URL}/v1/search", json=payload
        )
        if not response.ok:
            raise ValueError(
                f"AnySearch search request failed with HTTP {response.status}"
            )
        return response.json()

    async def extract(self, url: str) -> dict[str, Any]:
        response = await self.requests.post(
            f"{ANYSEARCH_API_URL}/v1/extract", json={"url": url}
        )
        if not response.ok:
            raise ValueError(
                f"AnySearch extract request failed with HTTP {response.status}"
            )
        return response.json()


def unwrap_envelope(response: dict[str, Any]) -> dict[str, Any]:
    """Return the envelope's data payload; raise on API-level errors."""
    code = response.get("code")
    data = response.get("data")
    if isinstance(code, bool) or code != 0 or not isinstance(data, dict):
        message = response.get("message") or "unknown error"
        raise ValueError(f"AnySearch API error: {message}")
    return data


def result_from_dict(r: dict[str, Any]) -> AnySearchResult:
    return AnySearchResult(
        title=r.get("title") or "",
        url=r.get("url") or "",
        snippet=r.get("snippet") or "",
        content=r.get("content"),
    )
