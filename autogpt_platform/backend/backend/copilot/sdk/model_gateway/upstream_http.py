"""HTTPX requests with no redirects, bounded errors and safe transport failures."""

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager

import httpx

from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    timed_out,
    unreachable,
)
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

_ERROR_BODY_LIMIT = 64 * 1024
_CONNECT_TIMEOUT_SECONDS = 15.0


def build_http_client(
    timeout_seconds: float, *, codex: bool = False
) -> httpx.AsyncClient:
    """A client that follows no redirect and reads no ambient credentials.

    ``trust_env=False`` keeps proxy variables and ``.netrc`` away from a
    request that carries the route's key, and an unfollowed redirect means
    the key never reaches a host the operator did not configure.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
        follow_redirects=False,
        # Only the fixed public Codex endpoint trusts operator-controlled egress proxy configuration.
        trust_env=codex,
    )


@contextmanager
def retain_usage(latest: Callable[[], GatewayUsage | None]) -> Iterator[None]:
    try:
        yield
    except GatewayError as error:
        if (usage := latest()) is not None:
            error.usage = usage
        raise


@asynccontextmanager
async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
) -> AsyncIterator[httpx.Response]:
    """POST ``payload`` and hand back the response with its body unread."""
    request = client.build_request(
        "POST",
        url,
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "content-type": "application/json"},
    )
    try:
        response = await client.send(request, stream=True, follow_redirects=False)
    except httpx.TimeoutException:
        raise timed_out() from None
    except httpx.HTTPError:
        raise unreachable() from None
    try:
        yield response
    finally:
        await response.aclose()


async def response_lines(response: httpx.Response) -> AsyncIterator[str]:
    """The body's lines; a dropped or stalled stream becomes a gateway error."""
    try:
        async for line in response.aiter_lines():
            yield line
    except httpx.TimeoutException:
        raise timed_out() from None
    except httpx.HTTPError:
        raise unreachable() from None


async def read_body(response: httpx.Response, *, limit: int | None = None) -> bytes:
    """The body, or its first ``limit`` bytes when one is given."""
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if limit is not None and size >= limit:
                break
    except httpx.TimeoutException:
        raise timed_out() from None
    except httpx.HTTPError:
        raise unreachable() from None
    body = b"".join(chunks)
    return body[:limit] if limit is not None else body


async def read_error_body(response: httpx.Response) -> bytes:
    return await read_body(response, limit=_ERROR_BODY_LIMIT)
