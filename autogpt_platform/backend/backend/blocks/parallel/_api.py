"""Thin async client for the Parallel Decisions API (POST /v1/decisions)."""

from typing import Any

from pydantic import ValidationError

from backend.sdk import APIKeyCredentials, BaseModel, Requests
from backend.util.request import Response

PARALLEL_API_URL = "https://api.parallel.ai"

# Parallel's published price: $1 per 1,000 Decisions requests, regardless of
# how many questions a request carries.
DECISION_REQUEST_COST_USD = 0.001


async def make_decision(
    credentials: APIKeyCredentials,
    state: Any,
    questions: dict[str, dict[str, Any]],
    include_excerpts: bool,
) -> dict[str, Any]:
    """Send one Decisions request and return the parsed response body.

    Refusals are part of a successful (200) response and are returned as-is;
    only HTTP-level failures raise.
    """
    requests = Requests(
        trusted_origins=[PARALLEL_API_URL],
        raise_for_status=False,
        # Unbounded by default; cap it so a lasting 429/5xx reaches the
        # error message below instead of retrying until the block times out.
        retry_max_attempts=3,
        extra_headers={"x-api-key": credentials.api_key.get_secret_value()},
    )
    response = await requests.post(
        f"{PARALLEL_API_URL}/v1/decisions",
        # Requests re-sends extra_headers on every hop and only strips
        # Authorization/Cookie, so a redirect would leak x-api-key.
        allow_redirects=False,
        json={
            "state": state,
            "questions": questions,
            "include_excerpts": include_excerpts,
        },
    )
    if not response.ok:
        raise ValueError(
            f"Parallel Decisions request failed with HTTP {response.status}: "
            f"{_error_message(response)}"
        )
    return response.json()


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
