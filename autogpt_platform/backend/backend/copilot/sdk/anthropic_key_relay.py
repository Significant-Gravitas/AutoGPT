"""A loopback relay that puts a user's own key on one CLI launch's model calls,
so the Claude Code CLI never holds that key.

The launch gets this relay as its ``ANTHROPIC_BASE_URL`` and a one-off token
only the relay accepts; the relay swaps the token for the key on the Messages
API, and serves nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import ssl
from functools import partial

import aiohttp
from aiohttp import web
from yarl import URL

from backend.copilot.sdk.env import _LOOPBACK_NO_PROXY_HOSTS, _no_proxy_value

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"

_RELAYED_PATHS = ("/v1/messages", "/v1/messages/count_tokens")
# The Messages API takes requests of up to 32 MB.
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
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


class AnthropicKeyRelay:
    """Relays one CLI launch's Messages API calls to *upstream_url* with
    *api_key*, from a loopback port only that launch holds the token for.

    *as_bearer_too* is for an upstream that takes the key as ``x-api-key`` or
    as a bearer token (OpenRouter): the launch sends its token both ways, and
    the key goes upstream both ways."""

    def __init__(
        self, upstream_url: str, api_key: str, *, as_bearer_too: bool = False
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._api_key = api_key
        self._as_bearer_too = as_bearer_too
        self._token = secrets.token_urlsafe(32)
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> dict[str, str]:
        """Start relaying; returns the environment that points a CLI launch at
        the relay, reached directly even where a proxy is configured."""
        application = web.Application(client_max_size=_MAX_REQUEST_BYTES)
        for path in _RELAYED_PATHS:
            application.router.add_post(path, partial(self._relay, path=path))
        # A call the CLI gives up on is given up on upstream too.
        runner = web.AppRunner(
            application, access_log=None, shutdown_timeout=2, handler_cancellation=True
        )
        self._runner = runner
        try:
            tls = await asyncio.to_thread(_upstream_tls)
            # Proxy settings as the CLI had them; bytes and encoding passed
            # through as sent; no deadline but the CLI's own on a long reply.
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=tls),
                trust_env=True,
                auto_decompress=False,
                skip_auto_headers=_NOT_ADDED,
                timeout=aiohttp.ClientTimeout(
                    total=None, sock_connect=_CONNECT_TIMEOUT_SECONDS
                ),
            )
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            port = runner.addresses[0][1]
        except BaseException:
            await self.close()
            raise
        no_proxy = _no_proxy_value(*_LOOPBACK_NO_PROXY_HOSTS)
        env = {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_API_KEY": self._token,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        if self._as_bearer_too:
            env["ANTHROPIC_AUTH_TOKEN"] = self._token
        return env

    async def close(self) -> None:
        """Stop relaying. Never raises, so the caller's cleanup after it runs."""
        runner, self._runner = self._runner, None
        session, self._session = self._session, None
        for closing in (runner and runner.cleanup(), session and session.close()):
            if closing is None:
                continue
            try:
                await closing
            except Exception as error:
                logger.warning(f"Anthropic key relay did not close cleanly: {error}")

    async def _relay(self, request: web.Request, *, path: str) -> web.StreamResponse:
        session = self._session
        if session is None or not self._holds_token(request):
            return _error(401, "authentication_error", "invalid x-api-key")
        headers = [
            (name, value)
            for name, value in request.headers.items()
            if name.lower() not in _NOT_SENT_UPSTREAM
        ]
        headers.append(("x-api-key", self._api_key))
        if self._as_bearer_too:
            headers.append(("Authorization", f"Bearer {self._api_key}"))
        body = await request.read()
        # The path the route was registered for, never the path as sent.
        query = request.rel_url.raw_query_string
        url = URL(
            f"{self._upstream_url}{path}{'?' + query if query else ''}", encoded=True
        )
        response: web.StreamResponse | None = None
        try:
            # A redirect goes back to the CLI, so the key reaches only the
            # configured endpoint.
            async with session.post(
                url, data=body, headers=headers, allow_redirects=False
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=[
                        (name, value)
                        for name, value in upstream.headers.items()
                        if name.lower() not in _NOT_SENT_BACK
                    ],
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, TimeoutError) as error:
            if response is not None and response.prepared:
                # Cut the reply short, for the CLI to see that it was.
                raise
            host = URL(self._upstream_url).host
            logger.warning(
                f"Anthropic key relay could not reach {host}: {type(error).__name__}"
            )
            return _error(
                502, "api_error", f"Could not reach the Anthropic API: {error}"
            )

    def _holds_token(self, request: web.Request) -> bool:
        """Whether *request* carries the launch's token, in either header the
        CLI sends a credential in."""
        authorization = request.headers.get("Authorization", "")
        bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
        token = self._token.encode()
        return any(
            secrets.compare_digest(sent.encode("utf-8", "surrogateescape"), token)
            for sent in (request.headers.get("x-api-key", ""), bearer)
        )


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


def _error(status: int, error_type: str, message: str) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status=status,
    )
