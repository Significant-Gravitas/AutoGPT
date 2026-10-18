"""Authenticated loopback front; meters successful and failed requests separately."""

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

from aiohttp import web

from backend.copilot.provider_failure import ProviderFailure
from backend.copilot.sdk.model_gateway.adapters.anthropic_passthrough import (
    AnthropicPassthrough,
)
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.adapters.openai_chat import OpenAIChat
from backend.copilot.sdk.model_gateway.errors import GatewayError, redact
from backend.copilot.sdk.model_gateway.relay import (
    SSE_HEADERS,
    end_quietly,
    error_response,
    json_body,
    relay,
    write_quietly,
)
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.sse import encode_event
from backend.copilot.sdk.model_gateway.stream import (
    Event,
    StreamItem,
    Upstream,
    collect_message,
)
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    UsageMeter,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

# Anthropic's own request-size limit.
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def build_upstream(route: GatewayRoute) -> Upstream:
    if route.adapter == "anthropic_passthrough":
        return AnthropicPassthrough(route)
    if route.adapter == "openai_chat":
        return OpenAIChat(route)
    return CodexResponses(route)


class ModelGateway:
    def __init__(
        self, route: GatewayRoute, *, upstream: Upstream | None = None
    ) -> None:
        self._route = route
        self._upstream = upstream
        self._auth_token = secrets.token_urlsafe(32)
        self._runner: web.AppRunner | None = None
        self._base_url: str | None = None
        self._closed = False
        self.usage = GatewayUsage()
        self.failed_usage = GatewayUsage()
        self.requests = 0
        self.peak_prompt_tokens = 0
        self.last_failure: ProviderFailure | None = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("The model gateway is not running")
        return self._base_url

    @property
    def auth_token(self) -> str:
        return self._auth_token

    @property
    def route(self) -> GatewayRoute:
        return self._route

    async def __aenter__(self) -> "ModelGateway":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._runner is not None:
            return
        if self._closed:
            raise RuntimeError("A closed model gateway cannot be restarted")
        try:
            if self._upstream is None:
                self._upstream = build_upstream(self._route)
            self._runner = web.AppRunner(
                self._app(), access_log=None, shutdown_timeout=2
            )
            await self._runner.setup()
            await web.TCPSite(self._runner, "127.0.0.1", 0).start()
            host, port = self._runner.addresses[0][:2]
            self._base_url = f"http://{host}:{port}"
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runner, self._runner = self._runner, None
        self._base_url = None
        try:
            if runner is not None:
                await asyncio.wait_for(runner.cleanup(), timeout=5)
        finally:
            if self._upstream is not None:
                await self._upstream.aclose()

    def _app(self) -> web.Application:
        @web.middleware
        async def require_token(
            request: web.Request, handler: _Handler
        ) -> web.StreamResponse:
            if not self._authorized(request):
                return error_response(
                    GatewayError(401, "authentication_error", "Unauthorized")
                )
            return await handler(request)

        app = web.Application(
            client_max_size=_MAX_REQUEST_BYTES, middlewares=[require_token]
        )
        app.router.add_post("/v1/messages", self._messages)
        app.router.add_post("/v1/messages/count_tokens", self._count_tokens)
        app.router.add_get("/v1/models", self._models)
        app.router.add_route("HEAD", "/api/hello", self._hello)
        app.router.add_get("/healthz", self._health)
        return app

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        body = await json_body(request)
        if isinstance(body, web.Response):
            return body
        upstream = self._upstream
        if upstream is None or self._closed:
            return self._failed(
                GatewayError(503, "api_error", "The model gateway is shutting down")
            )
        started = time.monotonic()
        meter = UsageMeter()
        streamed = body.get("stream") is True
        if streamed:
            response = await self._stream(request, upstream.stream(body), meter)
        else:
            response = await self._complete(upstream, body, meter)
        self._account(meter, response.status, streamed, started)
        return response

    async def _stream(
        self,
        request: web.Request,
        items: AsyncGenerator[StreamItem, None],
        meter: UsageMeter,
    ) -> web.StreamResponse:
        try:
            first = await anext(items)
        except StopAsyncIteration:
            return self._failed(
                GatewayError(502, "api_error", "The model sent nothing"), meter
            )
        except Exception as exc:
            await items.aclose()
            return self._failed(self._as_gateway_error(exc), meter)
        response = web.StreamResponse(status=200, headers=SSE_HEADERS)
        await response.prepare(request)
        try:
            await relay(response, first, items, meter)
        except ConnectionError:
            meter.fail()
            # The CLI hung up (an interrupt, or its own retry); nothing to say.
            return response
        except Exception as exc:
            error = self._as_gateway_error(exc)
            meter.fail(error.usage)
            self._note(error)
            await write_quietly(response, encode_event(error.body()))
        finally:
            await items.aclose()
        await end_quietly(response)
        return response

    async def _complete(
        self, upstream: Upstream, body: Event, meter: UsageMeter
    ) -> web.Response:
        try:
            if self._route.adapter == "anthropic_passthrough":
                message = await upstream.complete(body)
                meter.observe_message(message)
            else:
                message = await collect_message(upstream.stream(body), meter)
        except Exception as exc:
            return self._failed(self._as_gateway_error(exc), meter)
        return web.json_response(message)

    def _as_gateway_error(self, exc: Exception) -> GatewayError:
        """A ``GatewayError`` as it is; anything else is our own fault.

        Only the exception's type is logged: a message from a validation
        error can quote the request it choked on.
        """
        if isinstance(exc, GatewayError):
            return exc
        logger.error(
            f"model gateway: adapter={self._route.adapter} failed "
            f"unexpectedly: {type(exc).__name__}"
        )
        return GatewayError(500, "api_error", "The model gateway failed")

    async def _count_tokens(self, request: web.Request) -> web.Response:
        body = await json_body(request)
        if isinstance(body, web.Response):
            return body
        return web.json_response({"input_tokens": estimate_tokens(body)})

    async def _models(self, request: web.Request) -> web.Response:
        model = self._route.model
        listing = {
            "type": "model",
            "id": model,
            "display_name": model,
            "created_at": "2026-01-01T00:00:00Z",
        }
        return web.json_response(
            {"data": [listing], "has_more": False, "first_id": model, "last_id": model}
        )

    async def _hello(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def _authorized(self, request: web.Request) -> bool:
        expected = self._auth_token.encode()
        authorization = request.headers.get("Authorization", "")
        bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        bearer_ok = secrets.compare_digest(bearer.encode("utf-8", "replace"), expected)
        key_ok = secrets.compare_digest(api_key.encode("utf-8", "replace"), expected)
        return bearer_ok or key_ok

    def _failed(
        self, error: GatewayError, meter: UsageMeter | None = None
    ) -> web.Response:
        if meter is not None:
            meter.fail(error.usage)
        self._note(error)
        return error_response(error)

    def _note(self, error: GatewayError) -> None:
        if error.failure is not None:
            self.last_failure = error.failure.model_copy(
                update={
                    "message": redact(
                        error.failure.message, self._route.secret_values()
                    )
                    or "The model server request failed"
                }
            )

    def _account(
        self, meter: UsageMeter, status: int, streamed: bool, started: float
    ) -> None:
        usage = meter.total
        self.requests += 1
        if meter.failed:
            self.failed_usage = self.failed_usage.plus(usage)
        else:
            self.usage = self.usage.plus(usage)
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens)
        logger.info(
            f"model gateway: adapter={self._route.adapter} model={self._route.model} "
            f"status={status} stream={streamed} failed={meter.failed} "
            f"ms={int((time.monotonic() - started) * 1000)} "
            f"input={usage.input_tokens} cache_read={usage.cache_read_tokens} "
            f"cache_write={usage.cache_write_tokens} output={usage.output_tokens} "
            f"reasoning={usage.reasoning_tokens}"
        )
