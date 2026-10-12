"""For the ChatGPT Codex backend, on the connected account's own lease.

Every call sends the whole conversation as Responses ``input`` items with
``store: false``, so ChatGPT holds no state between calls and neither does
the gateway: system text becomes ``instructions``, user and assistant turns
become messages with their real roles, tool rounds become ``function_call``
and ``function_call_output`` items, images become ``input_image`` parts, and
the model's earlier reasoning comes back from the thinking blocks' sealed
signatures (``include: ["reasoning.encrypted_content"]``).

The endpoint, headers, account id and client come from
``backend.integrations.codex.http_client.build_client``; the model and
effort come from the route the platform resolved. Retries are left to the
CLI, so its own policy and the ``retry-after`` the gateway passes on decide
when to try again.
"""

import math
import re
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from backend.copilot.provider_failure import (
    ProviderFailure,
    ProviderFailureKind,
    classify,
)
from backend.copilot.sdk.model_gateway.adapters.codex_input import responses_payload
from backend.copilot.sdk.model_gateway.adapters.codex_signature import reasoning_tag
from backend.copilot.sdk.model_gateway.adapters.codex_stream import CodexStreamMapper
from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    UpstreamError,
    error_for_status,
    error_from_payload,
    redact,
    retry_headers,
    timed_out,
    unreachable,
)
from backend.copilot.sdk.model_gateway.normalize import normalize_request
from backend.copilot.sdk.model_gateway.overflow import detect_overflow, prompt_too_long
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.sse import read_frames
from backend.copilot.sdk.model_gateway.stream import (
    AnthropicStreamBuilder,
    Event,
    StreamItem,
    closing_items,
    collect_message,
)
from backend.copilot.sdk.model_gateway.usage import estimate_tokens
from backend.integrations.codex.http_client import (
    account_id_for,
    build_client,
    parse_rate_limits,
)
from backend.integrations.codex.models import CodexRateLimitsSnapshot
from backend.integrations.codex.transport import codex_credentials

_USAGE_LIMITS = frozenset({"usage_limit_reached", "usage_not_included"})
_TRY_AGAIN = re.compile(r"(?i)try again in (\d+(?:\.\d+)?)\s*(ms|s)\b")
_STREAM_FAILURE_STATUS = {
    "rate_limit_exceeded": 429,
    "usage_limit_reached": 429,
    "usage_not_included": 429,
    "invalid_prompt": 400,
    "server_error": 500,
}


class CodexResponses:
    def __init__(
        self, route: GatewayRoute, *, client: AsyncOpenAI | None = None
    ) -> None:
        credential = route.credential
        if credential.kind != "codex_lease":
            raise ValueError("The Codex adapter needs the account's credential lease")
        credentials = codex_credentials(credential.lease)
        self._route = route
        self._credential_id = credentials.id
        self._tag = reasoning_tag(
            account_id_for(credentials) or credentials.id, route.model
        )
        self._owns_client = client is None
        self._client = client or build_client(credentials).with_options(max_retries=0)
        self.rate_limits: CodexRateLimitsSnapshot | None = None

    async def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        request = normalize_request(body)
        payload, names = responses_payload(request, self._route, self._tag)
        estimate = estimate_tokens(body)
        builder = AnthropicStreamBuilder(
            model=self._route.model, input_estimate=estimate
        )
        mapper = CodexStreamMapper(builder, names, self._tag)
        # The body is built as plain JSON; the SDK only forwards it.
        arguments: dict[str, Any] = {**payload}
        try:
            async with self._client.responses.with_streaming_response.create(
                **arguments
            ) as response:
                self.rate_limits = parse_rate_limits(response.headers)
                async for frame in read_frames(response.iter_lines()):
                    event = frame.payload()
                    if not isinstance(event, dict):
                        continue
                    events = mapper.feed(event)
                    if mapper.failure is not None:
                        raise self._reported(mapper.failure, estimate)
                    for out in events:
                        yield StreamItem(event=out)
        except APIStatusError as exc:
            raise self._refused(exc, estimate) from None
        except (APITimeoutError, httpx.TimeoutException):
            raise timed_out() from None
        except (APIConnectionError, httpx.HTTPError):
            raise unreachable() from None
        if not mapper.completed:
            raise GatewayError(502, "api_error", "ChatGPT ended the response early")
        for item in closing_items(mapper.finish(), mapper.usage):
            yield item

    async def complete(self, body: Event) -> Event:
        return await collect_message(self.stream(body))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    def _refused(self, exc: APIStatusError, estimate: int) -> GatewayError:
        """An HTTP error from ChatGPT, before anything streamed."""
        # A refusal carries the quota headers too, and a spent quota is when
        # the platform most needs to show them.
        self.rate_limits = parse_rate_limits(exc.response.headers)
        error = error_from_payload(exc.body) or UpstreamError(
            message=str(exc.body or "")
        )
        message = redact(error.message, self._route.secret_values())
        failure = classify(
            exc,
            auth_provider="codex",
            credential_id=self._credential_id,
            message=message or None,
        )
        return self._restated(
            error, exc.status_code, estimate, dict(exc.response.headers), failure
        )

    def _reported(self, error: UpstreamError, estimate: int) -> GatewayError:
        """A failure ChatGPT reported inside the stream."""
        code = error.code or error.error_type or ""
        failure = None
        if code in _USAGE_LIMITS:
            failure = ProviderFailure(
                kind=ProviderFailureKind.USAGE_LIMIT,
                message=redact(error.message, self._route.secret_values())
                or "Your ChatGPT usage limit has been reached.",
                auth_provider="codex",
                credential_id=self._credential_id,
            )
        status = _STREAM_FAILURE_STATUS.get(code, 500)
        return self._restated(error, status, estimate, {}, failure)

    def _restated(
        self,
        error: UpstreamError,
        status: int,
        estimate: int,
        headers: Mapping[str, str],
        failure: ProviderFailure | None,
    ) -> GatewayError:
        overflow = detect_overflow(error)
        if overflow is not None:
            return prompt_too_long(
                overflow, estimate=estimate, window=self._route.context_window
            )
        message = redact(error.message, self._route.secret_values())
        if status == 429 or (error.code or error.error_type) in _USAGE_LIMITS:
            return _rate_limited(error, message, headers, failure)
        restated = error_for_status(status, message, headers=headers)
        restated.failure = failure
        return restated


def _rate_limited(
    error: UpstreamError,
    message: str,
    headers: Mapping[str, str],
    failure: ProviderFailure | None,
) -> GatewayError:
    """429 with the wait ChatGPT reported, and no retries against a quota."""
    hints = retry_headers(headers)
    if "retry-after" not in hints and (seconds := _reset_seconds(error)) is not None:
        hints["retry-after"] = str(seconds)
    if (error.code or error.error_type) in _USAGE_LIMITS:
        hints["x-should-retry"] = "false"
    return GatewayError(
        429,
        "rate_limit_error",
        message or "ChatGPT's rate limit was reached",
        headers=hints,
        failure=failure,
    )


def _reset_seconds(error: UpstreamError) -> int | None:
    """The wait ChatGPT reported: a field on a quota error, or the
    "try again in 1.9s" its rate-limit messages carry."""
    seconds = error.fields.get("resets_in_seconds")
    if isinstance(seconds, int) and not isinstance(seconds, bool) and seconds >= 0:
        return seconds
    resets_at = error.fields.get("resets_at")
    if isinstance(resets_at, int) and not isinstance(resets_at, bool):
        return max(0, resets_at - int(time.time()))
    if match := _TRY_AGAIN.search(error.message):
        scale = 0.001 if match.group(2).lower() == "ms" else 1.0
        return math.ceil(float(match.group(1)) * scale)
    return None
