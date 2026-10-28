"""Thin async client for the Perplexity Search API (POST /search)."""

import asyncio
import re
from enum import Enum
from typing import Any

from aiohttp import ClientError, ClientTimeout
from pydantic import Field, ValidationError

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField
from backend.util.request import Response

PERPLEXITY_API_URL = "https://api.perplexity.ai"

# Identifies this integration to Perplexity; the platform's default
# User-Agent (AutoGPT-Platform/...) is sent alongside it.
INTEGRATION_HEADER = {"X-Pplx-Integration": "autogpt"}

# Characters an HTTP header value may not contain. A key with one of these
# (usually a line break from a copy-paste) is rejected here with a fixed
# message, so no transport error that mentions the header is ever raised.
_FORBIDDEN_HEADER_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")

# Longest provider error message echoed to the user.
MAX_ERROR_MESSAGE_CHARS = 500

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
    plus, for HTTP errors, the ``error.message`` field Perplexity returns.
    They never include the API key, transport exception text, or a raw
    response body.
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
        raise ValueError(
            f"Perplexity Search request failed with HTTP {response.status}: "
            f"{_error_message(response)}"
        )
    try:
        return PerplexitySearchResponse.model_validate(response.json())
    except ValueError:
        # Covers JSON decoding and pydantic ValidationError, whose text quotes
        # the offending input; the body is not forwarded.
        raise ValueError(
            "Perplexity Search returned a response the block could not parse."
        ) from None


class _PerplexityError(BaseModel):
    message: str | None = None


class _PerplexityDetail(BaseModel):
    msg: str | None = None


class _PerplexityErrorBody(BaseModel):
    # 401 answers {"error": {"message": ..., "type": ..., "code": 401}};
    # 422 answers a FastAPI {"detail": [...]} list (or a plain string).
    error: _PerplexityError | None = None
    detail: str | list[_PerplexityDetail] | None = None

    def message(self) -> str | None:
        if self.error and self.error.message:
            return self.error.message
        match self.detail:
            case str() as text:
                return text
            case list() as items:
                return "; ".join(item.msg or "validation error" for item in items)
        return None


def _error_message(response: Response) -> str:
    """Pull the human-readable message out of a Perplexity error body.

    Falls back to a fixed string; the raw body is never echoed.
    """
    try:
        body = _PerplexityErrorBody.model_validate_json(response.content)
    except ValidationError:
        body = _PerplexityErrorBody()
    message = body.message()
    if not message:
        return "no error details were provided"
    return message[:MAX_ERROR_MESSAGE_CHARS]
