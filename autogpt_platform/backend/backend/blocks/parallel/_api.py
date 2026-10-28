"""Thin async client for the Parallel Decisions API (POST /v1/decisions)."""

import asyncio
from typing import Any

from aiohttp import ClientError, ClientTimeout
from pydantic import ValidationError

from backend.sdk import APIKeyCredentials, BaseModel, Requests
from backend.util.request import Response

PARALLEL_API_URL = "https://api.parallel.ai"

# Parallel's published price: $1 per 1,000 Decisions requests, regardless of
# how many questions a request carries.
DECISION_REQUEST_COST_USD = 0.001

# Parallel quotes 200 ms-1 s per request; fail well before aiohttp's 300 s
# default so a hung call doesn't hold an executor worker. ClientTimeout
# bounds one attempt; the deadline bounds all retries and backoff together.
REQUEST_TIMEOUT = ClientTimeout(total=60)
REQUEST_DEADLINE_SECONDS = 60


async def make_decision(
    credentials: APIKeyCredentials,
    state: Any,
    questions: dict[str, dict[str, Any]],
    include_excerpts: bool,
) -> dict[str, Any]:
    """Send one Decisions request and return the parsed response body.

    Refusals are part of a successful (200) response and are returned as-is.
    Every failure raises ValueError, which the executor puts on the block's
    error output as a handled failure.
    """
    requests = Requests(
        trusted_origins=[PARALLEL_API_URL],
        raise_for_status=False,
        # Unbounded by default; cap it so a lasting 429/5xx reaches the
        # error message below instead of retrying until the block times out.
        retry_max_attempts=3,
        extra_headers={"x-api-key": credentials.api_key.get_secret_value()},
    )
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_SECONDS):
            response = await _post(requests, state, questions, include_excerpts)
    except (ClientError, TimeoutError) as e:
        # Network failures and timeouts become handled block errors (the
        # error output) rather than unexpected crashes.
        raise ValueError(f"Could not reach Parallel: {type(e).__name__} {e}") from e
    if not response.ok:
        raise ValueError(
            f"Parallel Decisions request failed with HTTP {response.status}: "
            f"{_error_message(response)}"
        )
    return response.json()


async def _post(
    requests: Requests,
    state: Any,
    questions: dict[str, dict[str, Any]],
    include_excerpts: bool,
) -> Response:
    return await requests.post(
        f"{PARALLEL_API_URL}/v1/decisions",
        # Requests re-sends extra_headers on every hop and only strips
        # Authorization/Cookie, so a redirect would leak x-api-key.
        allow_redirects=False,
        timeout=REQUEST_TIMEOUT,
        json={
            "state": state,
            "questions": questions,
            "include_excerpts": include_excerpts,
        },
    )


class _ParallelError(BaseModel):
    message: str


class _ParallelErrorBody(BaseModel):
    # Documented: {"type": "error", "error": {"message": ...}}. The live API
    # also answers auth failures as {"code": 16, "message": ...}.
    error: _ParallelError | None = None
    message: str | None = None


def _error_message(response: Response) -> str:
    """Pull the human-readable message out of Parallel's error body."""
    try:
        body = _ParallelErrorBody.model_validate_json(response.content)
    except ValidationError:
        body = _ParallelErrorBody()
    message = body.error.message if body.error else body.message
    # Anything else (e.g. a FastAPI 422 "detail" list) is passed through raw.
    return message or response.text()[:500]
