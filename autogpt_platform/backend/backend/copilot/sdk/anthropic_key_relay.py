"""A loopback relay that puts a user's own key on one CLI launch's model calls,
so the Claude Code CLI never holds that key.

The launch gets this relay as its ``ANTHROPIC_BASE_URL`` and a one-off token
only the relay accepts; the relay swaps the token for the key on the Messages
API, and serves nothing else.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import AsyncExitStack
from functools import partial

import aiohttp
from aiohttp import web
from yarl import URL

from backend.copilot.sdk.env import _LOOPBACK_NO_PROXY_HOSTS, _no_proxy_value
from backend.util.anthropic_forwarding import ANTHROPIC_API_URL as ANTHROPIC_API_URL
from backend.util.anthropic_forwarding import (
    MAX_REQUEST_BYTES,
    RELAYED_PATHS,
    error_body,
    forward,
    headers_back,
    holds_token,
    upstream_session,
)

logger = logging.getLogger(__name__)


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
        self._resources = AsyncExitStack()

    async def start(self) -> dict[str, str]:
        """Start relaying; returns the environment that points a CLI launch at
        the relay, reached directly even where a proxy is configured."""
        application = web.Application(client_max_size=MAX_REQUEST_BYTES)
        for path in RELAYED_PATHS:
            application.router.add_post(path, partial(self._relay, path=path))
        # A call the CLI gives up on is given up on upstream too.
        runner = web.AppRunner(
            application, access_log=None, shutdown_timeout=2, handler_cancellation=True
        )
        self._runner = runner
        try:
            self._session = await self._resources.enter_async_context(
                upstream_session()
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
        self._session = None
        resources, self._resources = self._resources, AsyncExitStack()
        for closing in (runner and runner.cleanup(), resources.aclose()):
            if closing is None:
                continue
            try:
                await closing
            except Exception as error:
                logger.warning(f"Anthropic key relay did not close cleanly: {error}")

    async def _relay(self, request: web.Request, *, path: str) -> web.StreamResponse:
        session = self._session
        if session is None or not holds_token(request.headers, self._token):
            return web.json_response(
                error_body("authentication_error", "invalid x-api-key"), status=401
            )
        body = await request.read()
        response: web.StreamResponse | None = None
        try:
            async with forward(
                session,
                upstream_url=self._upstream_url,
                path=path,
                query=request.rel_url.raw_query_string,
                headers=request.headers.items(),
                body=body,
                api_key=self._api_key,
                as_bearer_too=self._as_bearer_too,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=headers_back(upstream),
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
            return web.json_response(
                error_body("api_error", f"Could not reach the Anthropic API: {error}"),
                status=502,
            )
