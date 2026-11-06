"""Chat Completions translation for operator-configured upstreams.

Tool results precede user text; inline images follow tool outputs.
Reasoning deltas are displayed, but history reasoning is dropped because
Chat Completions has no standard field for it. Effort comes from the route."""

from collections.abc import AsyncGenerator

import httpx

from backend.copilot.sdk.model_gateway.adapters.openai_chat_stream import (
    ChatStreamMapper,
)
from backend.copilot.sdk.model_gateway.adapters.openai_request import chat_payload
from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    UpstreamError,
    error_from_payload,
    parse_upstream_error,
)
from backend.copilot.sdk.model_gateway.normalize import normalize_request
from backend.copilot.sdk.model_gateway.overflow import upstream_failure
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.sse import read_frames
from backend.copilot.sdk.model_gateway.stream import (
    AnthropicStreamBuilder,
    Event,
    StreamItem,
    Upstream,
    closing_items,
)
from backend.copilot.sdk.model_gateway.upstream_http import (
    build_http_client,
    post_json,
    read_error_body,
    response_lines,
    retain_usage,
)
from backend.copilot.sdk.model_gateway.usage import estimate_tokens


class OpenAIChat(Upstream):
    def __init__(
        self, route: GatewayRoute, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._route = route
        self._url = f"{route.upstream_base}/chat/completions"
        self._owns_client = client is None
        self._client = client or build_http_client(route.timeout_seconds)

    async def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        request = normalize_request(body)
        payload, names = chat_payload(request, self._route)
        estimate = estimate_tokens(body)
        mapper = ChatStreamMapper(
            AnthropicStreamBuilder(model=self._route.model, input_estimate=estimate),
            names,
        )
        with retain_usage(lambda: mapper.usage):
            async with post_json(
                self._client, self._url, payload, self._headers()
            ) as response:
                if not response.is_success:
                    error = parse_upstream_error(await read_error_body(response))
                    raise self._failure(
                        error, response.status_code, estimate, response.headers
                    )
                async for frame in read_frames(response_lines(response)):
                    if frame.data.strip() == "[DONE]":
                        mapper.saw_done = True
                        break
                    chunk = frame.payload()
                    if not isinstance(chunk, dict):
                        continue
                    if "error" in chunk:
                        raise self._chunk_failure(chunk, estimate)
                    for event in mapper.feed(chunk):
                        yield StreamItem(event=event)
            if not mapper.complete:
                raise GatewayError(
                    502, "api_error", "The model server ended the stream early"
                )
            for item in closing_items(mapper.finish(), mapper.usage):
                yield item

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "text/event-stream"}
        key = self._route.api_key()
        if key:
            headers["authorization"] = f"Bearer {key}"
        return headers

    def _failure(
        self,
        error: UpstreamError,
        status: int,
        estimate: int,
        headers: httpx.Headers | None = None,
    ) -> GatewayError:
        return upstream_failure(
            error,
            status=status,
            estimate=estimate,
            window=self._route.context_window,
            secrets=self._route.secret_values(),
            headers=dict(headers) if headers is not None else None,
        )

    def _chunk_failure(self, chunk: Event, estimate: int) -> GatewayError:
        """An error a server sent as a stream chunk after a 200."""
        error = error_from_payload(chunk)
        if error is None:
            return GatewayError(502, "api_error", "The model server's stream failed")
        return self._failure(error, 500, estimate)
