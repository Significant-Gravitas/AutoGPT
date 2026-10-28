"""Thin async client for the Perplexity Search API (POST /search)."""

import asyncio
import re
from enum import Enum
from typing import Any

from aiohttp import ClientError, ClientTimeout
from pydantic import Field

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

PERPLEXITY_API_URL = "https://api.perplexity.ai"

# Identifies this integration to Perplexity; the platform's default
# User-Agent (AutoGPT-Platform/...) is sent alongside it.
INTEGRATION_HEADER = {"X-Pplx-Integration": "autogpt"}

# Characters an HTTP header value may not contain. A key with one of these
# (usually a line break from a copy-paste) is rejected here with a fixed
# message, so no transport error that mentions the header is ever raised.
_FORBIDDEN_HEADER_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")

# search_domain_filter takes at most 20 domains, all include or all exclude.
MAX_DOMAIN_FILTERS = 20

# Published Search API prices per successful request, by search_type.
WEB_SEARCH_COST_USD = 0.005
FAST_SEARCH_COST_USD = 0.001

# A search answers in well under a minute; fail before aiohttp's 300 s default
# so a hung call doesn't hold an executor worker. ClientTimeout bounds one
# attempt; the deadline bounds all retries and backoff together.
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
    """Schema for a single Perplexity Search API result."""

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
    """Enforce the Search API's rules for search_domain_filter.

    Raises ValueError (surfaced as an input validation error) when the list
    has more than 20 entries, has a blank entry, or mixes include entries
    with '-' prefixed exclude entries in one request.
    """
    if len(domains) > MAX_DOMAIN_FILTERS:
        raise ValueError(
            f"search_domain_filter accepts at most {MAX_DOMAIN_FILTERS} domains"
        )
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
    """The parts of a Search API response body the block uses."""

    id: str | None = None
    results: list[PerplexitySearchResult] = Field(default_factory=list)


async def search(
    credentials: APIKeyCredentials, payload: dict[str, Any]
) -> PerplexitySearchResponse:
    """Send one Search API request and return the parsed response.

    Every failure raises ValueError, which the block turns into a
    BlockExecutionError on its error output. Error messages are fixed strings
    chosen by HTTP status; they never include the API key, transport
    exception text, or anything from the response body (a 422 body echoes
    request input, and a misbehaving upstream could echo headers).
    """
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
            "Perplexity Search returned a response the block could not parse."
        ) from None


def _http_error_message(status: int) -> str:
    """Fixed, status-based message for a non-2xx Search API response."""
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
    return f"Perplexity Search request failed with HTTP {status}."
