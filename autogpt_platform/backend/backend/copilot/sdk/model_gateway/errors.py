"""Errors the gateway answers with, in Anthropic's envelope.

The CLI reads an error's HTTP status to decide whether to retry and its
``error.type``/``error.message`` to tell the user what happened, so every
upstream failure is restated in Anthropic's shape before it leaves here. An
upstream's message can echo what it was sent, credentials included, so it is
redacted on the way through.
"""

import json
import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from backend.copilot.provider_failure import ProviderFailure
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

REDACTED = "[REDACTED]"
_MAX_MESSAGE_CHARS = 1_000
_ANTHROPIC_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "overloaded_error",
    }
)
_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}
_STATUS_BY_TYPE = {kind: status for status, kind in _TYPE_BY_STATUS.items()}
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;\"']+")
_KEY_LIKE = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{8,}")
_JWT_LIKE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"authorization|x-api-key|password|secret)\b(\s*[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_RETRY_AFTER = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")


class GatewayError(Exception):
    """A failure the front answers with ``status`` and Anthropic's envelope."""

    def __init__(
        self,
        status: int,
        error_type: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
        failure: ProviderFailure | None = None,
        usage: GatewayUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message
        self.headers = dict(headers or {})
        self.failure = failure
        self.usage = usage

    def body(self) -> dict[str, object]:
        return {
            "type": "error",
            "error": {"type": self.error_type, "message": self.message},
        }


class UpstreamError(BaseModel):
    """What an upstream said in an error body, whatever its dialect."""

    model_config = ConfigDict(frozen=True)

    message: str = ""
    error_type: str | None = None
    code: str | None = None
    fields: dict[str, object] = Field(default_factory=dict)


def error_for_status(
    status: int,
    message: str,
    *,
    error_type: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> GatewayError:
    """Restate an upstream failure, keeping its status and Anthropic type.

    A status that is not an error (a redirect the client never follows) is
    the upstream misbehaving, so it reaches the CLI as a 502.
    """
    text = message or f"The model server answered HTTP {status}"
    if not 400 <= status <= 599:
        status = 502
    kind = error_type if error_type in _ANTHROPIC_ERROR_TYPES else None
    kind = kind or _TYPE_BY_STATUS.get(status, "api_error")
    return GatewayError(status, kind, text, headers=retry_headers(headers or {}))


def status_for_type(error_type: str | None) -> int:
    """The HTTP status Anthropic pairs with an error type (500 if unknown)."""
    return _STATUS_BY_TYPE.get(error_type or "", 500)


def unreachable() -> GatewayError:
    return GatewayError(502, "api_error", "The model server could not be reached")


def timed_out() -> GatewayError:
    return GatewayError(504, "api_error", "The model server did not answer in time")


def parse_upstream_error(raw: bytes | str) -> UpstreamError:
    """Read an error body in Anthropic, OpenAI, llama.cpp or plain-text form."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    try:
        payload = json.loads(text)
    except ValueError:
        return UpstreamError(message=text.strip())
    return error_from_payload(payload) or UpstreamError(message=text.strip())


def error_from_payload(payload: object) -> UpstreamError | None:
    """The error object inside a decoded body or stream event, if any."""
    if not isinstance(payload, dict):
        return None
    inner = payload.get("error", payload)
    if isinstance(inner, str):
        return UpstreamError(message=inner)
    if not isinstance(inner, dict):
        return None
    message = inner.get("message") or inner.get("detail") or ""
    return UpstreamError(
        message=message if isinstance(message, str) else json.dumps(message),
        error_type=_text(inner.get("type")),
        code=_text(inner.get("code")),
        fields={str(key): value for key, value in inner.items()},
    )


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Strip credentials and anything shaped like one, and bound the length."""
    for secret in sorted((s for s in secrets if len(s) >= 4), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    text = _BEARER.sub(f"Bearer {REDACTED}", text)
    text = _ASSIGNED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    text = _KEY_LIKE.sub(REDACTED, text)
    text = _JWT_LIKE.sub(REDACTED, text)
    text = " ".join(text.split())
    if len(text) > _MAX_MESSAGE_CHARS:
        return text[: _MAX_MESSAGE_CHARS - 3] + "..."
    return text


def retry_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The upstream's retry hints the CLI understands, and nothing else."""
    lowered = {key.lower(): value for key, value in headers.items()}
    kept: dict[str, str] = {}
    for name in ("retry-after", "retry-after-ms"):
        value = lowered.get(name)
        if value is not None and _RETRY_AFTER.match(value):
            kept[name] = value.strip()
    return kept


def _text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None
