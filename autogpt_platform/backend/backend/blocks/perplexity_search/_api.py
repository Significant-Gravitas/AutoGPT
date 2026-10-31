"""Thin async client for the Perplexity Search API (POST /search)."""

import asyncio
import re
from enum import Enum
from typing import Any

from aiohttp import ClientError, ClientTimeout

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

PERPLEXITY_API_URL = "https://api.perplexity.ai"

# Identifies this integration to Perplexity; the platform's default
# User-Agent (AutoGPT-Platform/...) is sent alongside it.
INTEGRATION_HEADER = {"X-Pplx-Integration": "autogpt"}

# Control characters aiohttp rejects in header values. Checked before any
# request so no transport error can describe the Authorization header.
_FORBIDDEN_HEADER_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")

MAX_DOMAIN_FILTERS = 20

# Published Search API prices per successful request, by search_type.
WEB_SEARCH_COST_USD = 0.005
FAST_SEARCH_COST_USD = 0.001

# Limit each attempt and the total time across retries/backoff to 60 seconds.
REQUEST_TIMEOUT = ClientTimeout(total=60)
REQUEST_DEADLINE_SECONDS = 60


class PerplexitySearchType(Enum):
    WEB = "web"
    FAST = "fast"


class PerplexityRecencyFilter(Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class PerplexitySearchResult(BaseModel):
    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(
        description="Query-relevant snippet extracted from the page", default=""
    )
    date: str | None = SchemaField(
        description="Publication date of the page, when known", default=None
    )
    last_updated: str | None = SchemaField(
        description="Date the page was last updated, when known", default=None
    )


def check_domain_filter(domains: list[str]) -> None:
    """Reject blank entries and mixed include/exclude modes."""
    if any(not domain.lstrip("-").strip() for domain in domains):
        raise ValueError("search_domain_filter entries cannot be blank")
    excluded = sum(1 for domain in domains if domain.startswith("-"))
    if 0 < excluded < len(domains):
        raise ValueError(
            "search_domain_filter cannot mix included and excluded domains: "
            "list only domains to include, or prefix every entry with '-' to "
            "exclude them"
        )


def search_cost_usd(search_type: PerplexitySearchType) -> float:
    if search_type == PerplexitySearchType.FAST:
        return FAST_SEARCH_COST_USD
    return WEB_SEARCH_COST_USD


class PerplexitySearchResponse(BaseModel):
    results: list[PerplexitySearchResult]


async def search(
    credentials: APIKeyCredentials, payload: dict[str, Any]
) -> PerplexitySearchResponse:
    """Return parsed search results. Raise ValueError for invalid keys, request
    failures, non-2xx responses, or invalid response data. Error text excludes
    credentials, transport text, and response bodies."""
    api_key = credentials.api_key.get_secret_value()
    if not api_key or _FORBIDDEN_HEADER_CHARS.search(api_key):
        raise ValueError(
            "The Perplexity API key is empty or contains characters that cannot "
            "be sent in an HTTP header. Check the credential for line breaks or "
            "other stray characters."
        )
    requests = Requests(
        trusted_origins=[PERPLEXITY_API_URL],
        raise_for_status=False,
        # Unbounded by default; cap it so a lasting 429/5xx reaches the
        # error message below instead of retrying until the block times out.
        retry_max_attempts=3,
        extra_headers={
            "Authorization": f"Bearer {api_key}",
            **INTEGRATION_HEADER,
        },
    )
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_SECONDS):
            response = await requests.post(
                f"{PERPLEXITY_API_URL}/search",
                # Requests re-sends extra_headers on every hop; never follow
                # a redirect carrying the bearer token to another host.
                allow_redirects=False,
                timeout=REQUEST_TIMEOUT,
                json=payload,
            )
    except (ClientError, TimeoutError, ValueError) as e:
        # Only the exception type is reported: aiohttp and the Requests
        # wrapper may describe the request (including headers) in their text.
        raise ValueError(
            f"Could not reach Perplexity ({type(e).__name__}). "
            "Check the network connection and try again."
        ) from None
    if not response.ok:
        raise ValueError(_http_error_message(response.status))
    try:
        return PerplexitySearchResponse.model_validate(response.json())
    except ValueError:
        # Covers JSON decoding and pydantic ValidationError, whose text quotes
        # the offending input; the body is not forwarded.
        raise ValueError(
            "Perplexity Search returned an invalid response. Retry the request; "
            "if it persists, contact Perplexity support."
        ) from None


def _http_error_message(status: int) -> str:
    if status in (401, 403):
        return (
            f"Perplexity rejected the API key (HTTP {status}). Check the "
            "credential in your integrations settings."
        )
    if status in (400, 422):
        return (
            f"Perplexity rejected the request parameters (HTTP {status}). Check "
            "the query, max_results, and filters."
        )
    if status == 429:
        return "Perplexity rate limited this request (HTTP 429). Try again shortly."
    if 500 <= status <= 599:
        return (
            f"Perplexity Search is unavailable right now (HTTP {status}). Try "
            "again later."
        )
    return (
        f"Perplexity Search request failed with HTTP {status}. Contact "
        "Perplexity support with this status code."
    )
