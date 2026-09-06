"""Transport helpers for the RMFG client: error envelopes and polling."""

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

from pydantic import TypeAdapter, ValidationError

from backend.sdk import BaseModel
from backend.util.request import Response

POLL_INITIAL_SECONDS = 2.0
POLL_MAX_SECONDS = 10.0

_JSON_OBJECT = TypeAdapter(dict[str, Any])

T = TypeVar("T")


class APIError(BaseModel):
    """The ``error`` object of an RMFG error response."""

    type: str = ""
    code: str = ""
    message: str = ""
    param: Optional[str] = None
    request_id: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: APIError


class RMFGError(ValueError):
    """An error envelope returned by the API, rendered for a person."""

    def __init__(self, status: int, code: str, message: str, request_id: str = ""):
        self.status = status
        self.code = code
        self.request_id = request_id
        suffix = f" (request {request_id})" if request_id else ""
        super().__init__(f"RMFG {code or status}: {message}{suffix}")


def parse_body(response: Response) -> dict[str, Any]:
    """Return the JSON object body, or raise a readable error for a failure."""
    text = response.text()
    try:
        payload = _JSON_OBJECT.validate_python(response.json()) if text else {}
    except (ValueError, ValidationError) as exc:
        if response.ok:
            raise RMFGError(
                response.status, "invalid_response", f"not JSON: {text[:200]}"
            ) from exc
        raise RMFGError(response.status, "http_error", text[:200] or "no body")
    if response.ok:
        return payload
    raise _error_from(response.status, payload)


def _error_from(status: int, payload: dict[str, Any]) -> RMFGError:
    try:
        error = ErrorEnvelope.model_validate(payload).error
    except ValidationError:
        return RMFGError(status, "http_error", f"HTTP {status}")
    message = error.message or f"HTTP {status}"
    if error.param:
        message = f"{message} (field: {error.param})"
    if status in (401, 403):
        message = f"{message}. Check the RMFG API key and its scopes."
    return RMFGError(status, error.code or error.type, message, error.request_id or "")


async def poll(
    fetch: Callable[[], Awaitable[T]],
    *,
    initial: T,
    is_pending: Callable[[T], bool],
    timeout_seconds: float,
    what: str,
) -> T:
    """Re-fetch a resource with growing delays until it is no longer pending."""
    resource = initial
    deadline = time.monotonic() + timeout_seconds
    delay = POLL_INITIAL_SECONDS
    while is_pending(resource):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s waiting for {what}; "
                "fetch it again later with its ID."
            )
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, POLL_MAX_SECONDS)
        resource = await fetch()
    return resource
