"""Sending a Claude Code CLI's Messages API call on to Anthropic with a user's
key, for the relays that hold the key so the CLI never does."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import ssl
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager

import aiohttp
from yarl import URL

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"

RELAYED_PATHS = ("/v1/messages", "/v1/messages/count_tokens")
# The Messages API takes requests of up to 32 MB.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
_CONNECT_TIMEOUT_SECONDS = 30
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# The CLI's credential is the relay's token; the client sets the other two.
_NOT_SENT_UPSTREAM = _HOP_BY_HOP | {
    "host",
    "content-length",
    "x-api-key",
    "authorization",
}
_NOT_SENT_BACK = _HOP_BY_HOP | {"content-length"}
# Headers aiohttp would add itself: the request goes upstream as the CLI made it.
_NOT_ADDED = ("Accept", "Accept-Encoding", "Content-Type", "User-Agent")


def sent_credentials(headers: Mapping[str, str]) -> list[str]:
    """What the CLI sent as its credential, in either header it uses."""
    authorization = headers.get("Authorization", "")
    bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
    return [sent for sent in (headers.get("x-api-key", ""), bearer) if sent]


def holds_token(headers: Mapping[str, str], token: str) -> bool:
    """Whether *headers* carry *token* as the CLI's credential."""
    expected = token.encode()
    return any(
        secrets.compare_digest(sent.encode("utf-8", "surrogateescape"), expected)
        for sent in sent_credentials(headers)
    )


@asynccontextmanager
async def upstream_session() -> AsyncIterator[aiohttp.ClientSession]:
    """A client for the Messages API: proxy settings as the CLI had them, bytes
    and encoding passed through as sent, and no deadline but the CLI's own on
    a long reply."""
    tls = await asyncio.to_thread(_upstream_tls)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=tls),
        trust_env=True,
        auto_decompress=False,
        skip_auto_headers=_NOT_ADDED,
        timeout=aiohttp.ClientTimeout(
            total=None, sock_connect=_CONNECT_TIMEOUT_SECONDS
        ),
    ) as session:
        yield session


@asynccontextmanager
async def forward(
    session: aiohttp.ClientSession,
    *,
    upstream_url: str,
    path: str,
    query: str,
    headers: Iterable[tuple[str, str]],
    body: bytes,
    api_key: str,
    as_bearer_too: bool = False,
) -> AsyncIterator[aiohttp.ClientResponse]:
    """Send one call to *path* (a relayed route, never the path as sent) at
    *upstream_url* with *api_key* in place of the CLI's credential. A redirect
    comes back unfollowed, so the key reaches only *upstream_url*."""
    sent = [
        (name, value)
        for name, value in headers
        if name.lower() not in _NOT_SENT_UPSTREAM
    ]
    sent.append(("x-api-key", api_key))
    if as_bearer_too:
        sent.append(("Authorization", f"Bearer {api_key}"))
    url = URL(
        f"{upstream_url.rstrip('/')}{path}{'?' + query if query else ''}", encoded=True
    )
    async with session.post(
        url, data=body, headers=sent, allow_redirects=False
    ) as upstream:
        yield upstream


def headers_back(upstream: aiohttp.ClientResponse) -> list[tuple[str, str]]:
    """The upstream reply's headers that go back to the CLI."""
    return [
        (name, value)
        for name, value in upstream.headers.items()
        if name.lower() not in _NOT_SENT_BACK
    ]


def error_body(error_type: str, message: str) -> dict[str, object]:
    """An error in the Messages API's own shape, which the CLI reports as such."""
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _upstream_tls() -> ssl.SSLContext:
    """The CAs the backend trusts, plus any the deployment told the CLI to
    trust (``NODE_EXTRA_CA_CERTS``), since the relay makes the CLI's calls."""
    context = ssl.create_default_context()
    if extra_cas := os.environ.get("NODE_EXTRA_CA_CERTS"):
        try:
            context.load_verify_locations(cafile=extra_cas)
        except (OSError, ssl.SSLError) as error:
            logger.warning(f"Ignoring NODE_EXTRA_CA_CERTS={extra_cas}: {error}")
    return context
