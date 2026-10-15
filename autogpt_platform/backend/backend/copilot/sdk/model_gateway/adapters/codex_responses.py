"""Stateless Responses over HTTPX, with current lease authentication per request."""

from collections.abc import AsyncGenerator, Mapping
from uuid import uuid4

import httpx

from backend.copilot.provider_failure import (
    ProviderFailure,
    ProviderFailureKind,
    reset_time_from_headers,
)
from backend.copilot.sdk.model_gateway.adapters.codex_input import responses_payload
from backend.copilot.sdk.model_gateway.adapters.codex_quota import rate_limited
from backend.copilot.sdk.model_gateway.adapters.codex_signature import reasoning_tag
from backend.copilot.sdk.model_gateway.adapters.codex_stream import CodexStreamMapper
from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    UpstreamError,
    parse_upstream_error,
    redact,
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
from backend.integrations.codex.http_client import (
    API_BASE,
    account_id_for,
    inference_headers,
    parse_rate_limits,
)
from backend.integrations.codex.models import CodexRateLimitsSnapshot
from backend.integrations.codex.transport import codex_credentials

_USAGE_LIMITS = frozenset({"usage_limit_reached", "usage_not_included"})
_STREAM_FAILURE_STATUS = {
    "rate_limit_exceeded": 429,
    "usage_limit_reached": 429,
    "usage_not_included": 429,
    "invalid_prompt": 400,
    "server_error": 500,
}
_FAILURE_KINDS = {
    401: ProviderFailureKind.AUTH_EXPIRED,
    403: ProviderFailureKind.POLICY_DENIED,
    404: ProviderFailureKind.MODEL_UNAVAILABLE,
    429: ProviderFailureKind.USAGE_LIMIT,
}


class CodexResponses(Upstream):
    def __init__(
        self, route: GatewayRoute, *, client: httpx.AsyncClient | None = None
    ) -> None:
        if route.credential.kind != "codex_lease":
            raise ValueError("The Codex adapter needs the account's credential lease")
        self._session_id = str(uuid4())
        self._route = route
        self._owns_client = client is None
        self._client = client or build_http_client(route.timeout_seconds, codex=True)
        self._url = f"{str(client.base_url).rstrip('/') if client is not None and str(client.base_url) else API_BASE}/responses"
        self.rate_limits: CodexRateLimitsSnapshot | None = None

    async def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        credential = self._route.credential
        if credential.kind != "codex_lease":
            raise ValueError("The Codex adapter needs the account's credential lease")
        credentials = codex_credentials(credential.lease)
        secrets = (credentials.access_token.get_secret_value(),)
        tag = reasoning_tag(
            account_id_for(credentials) or credentials.id, self._route.model
        )
        payload, names = responses_payload(normalize_request(body), self._route, tag)
        estimate = estimate_tokens(body)
        mapper = CodexStreamMapper(
            AnthropicStreamBuilder(model=self._route.model, input_estimate=estimate),
            names,
            tag,
        )
        headers = {
            **inference_headers(credentials),
            "session_id": self._session_id,
            "authorization": f"Bearer {secrets[0]}",
            "accept": "text/event-stream",
        }
        with retain_usage(lambda: mapper.usage):
            async with post_json(self._client, self._url, payload, headers) as response:
                self.rate_limits = parse_rate_limits(response.headers)
                async for item in self._read_response(
                    response, mapper, estimate, secrets, credentials.id
                ):
                    yield item
            if not mapper.completed:
                raise GatewayError(502, "api_error", "ChatGPT ended the response early")
            for item in closing_items(mapper.finish(), mapper.usage):
                yield item

    async def _read_response(
        self,
        response: httpx.Response,
        mapper: CodexStreamMapper,
        estimate: int,
        secrets: tuple[str, ...],
        credential_id: str,
    ) -> AsyncGenerator[StreamItem, None]:
        if not response.is_success:
            error = parse_upstream_error(await read_error_body(response))
            raise self._restated(
                error,
                response.status_code,
                estimate,
                response.headers,
                secrets,
                credential_id,
            )
        async for frame in read_frames(response_lines(response)):
            event = frame.payload()
            if not isinstance(event, dict):
                continue
            events = mapper.feed(event)
            if mapper.failure is not None:
                code = mapper.failure.code or mapper.failure.error_type or ""
                raise self._restated(
                    mapper.failure,
                    _STREAM_FAILURE_STATUS.get(code, 500),
                    estimate,
                    {},
                    secrets,
                    credential_id,
                )
            for out in events:
                yield StreamItem(event=out)
            if mapper.completed:
                return

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _restated(
        self,
        error: UpstreamError,
        status: int,
        estimate: int,
        headers: Mapping[str, str],
        secrets: tuple[str, ...],
        credential_id: str,
    ) -> GatewayError:
        restated = upstream_failure(
            error,
            status=status,
            estimate=estimate,
            window=self._route.context_window,
            secrets=secrets,
            headers=headers,
        )
        if restated.status == 429 or (error.code or error.error_type) in _USAGE_LIMITS:
            restated = rate_limited(error, restated.message, headers)
        kind = _FAILURE_KINDS.get(restated.status)
        if kind is None and restated.status >= 500:
            kind = ProviderFailureKind.TRANSIENT
        if kind is not None:
            restated.failure = ProviderFailure(
                kind=kind,
                message=redact(error.message, secrets) or "The ChatGPT request failed",
                resets_at=reset_time_from_headers(headers),
                auth_provider="codex",
                credential_id=credential_id,
            )
        return restated
