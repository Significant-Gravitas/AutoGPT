from enum import Enum
from typing import Any
from urllib.parse import urlencode

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

KEENABLE_API_URL = "https://api.keenable.ai"
# The keyless public endpoints require the calling app to identify itself.
KEENABLE_APP_TITLE = "AutoGPT"
# Marker Keenable appends to fetched content cut at max_chars.
KEENABLE_TRUNCATION_MARKER = "_This content has been truncated"


class KeenableAuth(str, Enum):
    """How a Keenable block authenticates.

    API_KEY uses a stored Keenable credential; ANONYMOUS calls the keyless
    public endpoints (rate-limited per IP) and makes the credentials field
    optional - the same discriminator mechanism the AnySearch blocks use.
    """

    API_KEY = "api_key"
    ANONYMOUS = "anonymous"

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        json_schema = handler(schema)
        json_schema["enumNames"] = ["API key", "Anonymous (rate-limited)"]
        return json_schema


class KeenableResult(BaseModel):
    """Schema for a single Keenable search result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(
        description="Excerpt of the result page relevant to the query", default=""
    )
    published_at: str | None = SchemaField(
        description="Publication time of the page (ISO 8601), when known",
        default=None,
    )


class KeenableClient:
    """Thin async REST client for api.keenable.ai (/v1/search, /v1/fetch).

    Without credentials (or with an empty key) it calls the ``/public``
    variants and sends ``X-Keenable-Title``; with a key it calls the keyed
    endpoints with ``X-API-Key``.
    """

    def __init__(self, credentials: APIKeyCredentials | None):
        api_key = credentials.api_key.get_secret_value().strip() if credentials else ""
        self.keyed = bool(api_key)
        self.requests = Requests(
            trusted_origins=[KEENABLE_API_URL],
            extra_headers=(
                {"X-API-Key": api_key}
                if self.keyed
                else {"X-Keenable-Title": KEENABLE_APP_TITLE}
            ),
            sensitive_headers=("X-API-Key",),
        )

    def _url(self, path: str) -> str:
        return f"{KEENABLE_API_URL}{path}" + ("" if self.keyed else "/public")

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.post(self._url("/v1/search"), json=payload)
        if not response.ok:
            raise ValueError(
                f"Keenable search request failed with HTTP {response.status}"
            )
        return response.json()

    async def fetch(self, url: str, max_chars: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"url": url, "live": "true"}
        if max_chars:
            params["max_chars"] = max_chars
        response = await self.requests.get(
            f"{self._url('/v1/fetch')}?{urlencode(params)}"
        )
        if not response.ok:
            raise ValueError(
                f"Keenable fetch request failed with HTTP {response.status}"
            )
        return response.json()


def results_from_response(response: dict[str, Any]) -> list[KeenableResult]:
    """Map a /v1/search response to results, skipping rows without a URL."""
    rows = response.get("results")
    if not isinstance(rows, list):
        raise TypeError("Keenable search response has no results list")
    results = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("url"):
            continue
        results.append(
            KeenableResult(
                title=r.get("title") or "",
                url=r["url"],
                snippet=r.get("snippet") or r.get("description") or "",
                published_at=r.get("published_at"),
            )
        )
    return results
