"""For servers that serve Anthropic's ``/v1/messages`` themselves.

llama.cpp, Ollama 0.14.3+, vLLM, LM Studio 0.4.1+ and LiteLLM all do, and
the real CLI completes tool rounds, compaction and resume against them
directly. The request and the stream go through unchanged, with four
exceptions:

- the upstream's own key is added here, so it never enters the CLI's env;
- the CLI's billing line is dropped from ``system`` (see ``normalize``);
- ``model`` is the route's, whatever the CLI asked for;
- an over-long-prompt error is rewritten into the only wording the CLI
  compacts on, and every other error is restated in Anthropic's envelope.
"""

import json
from collections.abc import AsyncGenerator

import httpx

from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    error_from_payload,
    parse_upstream_error,
    status_for_type,
)
from backend.copilot.sdk.model_gateway.normalize import strip_billing_header
from backend.copilot.sdk.model_gateway.overflow import upstream_failure
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.sse import read_frames
from backend.copilot.sdk.model_gateway.stream import Event, StreamItem
from backend.copilot.sdk.model_gateway.upstream_http import (
    build_http_client,
    post_json,
    read_body,
    read_error_body,
    response_lines,
)
from backend.copilot.sdk.model_gateway.usage import estimate_tokens

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicPassthrough:
    def __init__(
        self, route: GatewayRoute, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._route = route
        self._url = f"{route.upstream_base}/v1/messages"
        self._owns_client = client is None
        self._client = client or build_http_client(route.timeout_seconds)

    async def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        payload = self._payload(body, stream=True)
        async with post_json(
            self._client, self._url, payload, self._headers(stream=True)
        ) as response:
            if not response.is_success:
                raise self._failure(
                    response.status_code,
                    await read_error_body(response),
                    response.headers,
                    payload,
                )
            async for frame in read_frames(response_lines(response)):
                event = frame.payload()
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "error":
                    raise self._stream_failure(event, payload)
                yield StreamItem(event=event, raw=frame.encoded())

    async def complete(self, body: Event) -> Event:
        payload = self._payload(body, stream=False)
        async with post_json(
            self._client, self._url, payload, self._headers(stream=False)
        ) as response:
            raw = await read_body(response)
            if not response.is_success:
                raise self._failure(
                    response.status_code, raw, response.headers, payload
                )
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            raise GatewayError(
                502, "api_error", "The model server's reply was not JSON"
            )
        return message

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _payload(self, body: Event, *, stream: bool) -> Event:
        payload: Event = {**body, "model": self._route.model, "stream": stream}
        if "system" in body:
            payload["system"] = strip_billing_header(body["system"])
        return payload

    def _headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "accept": "text/event-stream" if stream else "application/json",
        }
        key = self._route.api_key()
        if key:
            # Servers differ on which one they read; both carry the same key
            # to the same operator-configured host.
            headers["x-api-key"] = key
            headers["authorization"] = f"Bearer {key}"
        return headers

    def _failure(
        self,
        status: int,
        raw: bytes,
        headers: httpx.Headers,
        payload: Event,
    ) -> GatewayError:
        return upstream_failure(
            parse_upstream_error(raw),
            status=status,
            estimate=estimate_tokens(payload),
            window=self._route.context_window,
            secrets=self._route.secret_values(),
            headers=dict(headers),
        )

    def _stream_failure(self, event: Event, payload: Event) -> GatewayError:
        error = error_from_payload(event)
        if error is None:
            return GatewayError(502, "api_error", "The model server's stream failed")
        return upstream_failure(
            error,
            status=status_for_type(error.error_type),
            estimate=estimate_tokens(payload),
            window=self._route.context_window,
            secrets=self._route.secret_values(),
        )
