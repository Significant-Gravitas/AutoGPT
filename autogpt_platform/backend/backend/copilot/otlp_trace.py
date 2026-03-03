"""Lightweight OTLP JSON trace exporter for CoPilot LLM calls.

Sends spans to a remote OTLP-compatible endpoint (e.g. Product Intelligence)
in the ExportTraceServiceRequest JSON format.  Payload construction and the
HTTP POST run in background asyncio tasks so streaming latency is unaffected.

Configuration (via backend.util.settings.Secrets):
    OTLP_TRACING_HOST  – base URL of the trace ingestion service
                         (e.g. "https://traces.example.com")
    OTLP_TRACING_TOKEN – optional Bearer token for authentication
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_settings = Settings()

# Resolve the endpoint once at import time.
_TRACING_HOST = (_settings.secrets.otlp_tracing_host or "").rstrip("/")
_TRACING_TOKEN = _settings.secrets.otlp_tracing_token.get_secret_value()
_TRACING_ENABLED = bool(_TRACING_HOST)

# Shared async client — created lazily on first use.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if _TRACING_TOKEN:
            headers["Authorization"] = f"Bearer {_TRACING_TOKEN}"
        _client = httpx.AsyncClient(headers=headers, timeout=10.0)
    return _client


def _nano(ts: float) -> str:
    """Convert a ``time.time()`` float to nanosecond string for OTLP."""
    return str(int(ts * 1_000_000_000))


def _kv(key: str, value: Any) -> dict | None:
    """Build an OTLP KeyValue entry, returning None for missing values."""
    if value is None:
        return None
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value}}
    if isinstance(value, bool):
        return {"key": key, "value": {"stringValue": str(value).lower()}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    # Fallback: serialise as string
    return {"key": key, "value": {"stringValue": str(value)}}


def _build_completion_text(
    assistant_content: str | None,
    tool_calls: list[dict[str, Any]] | None,
) -> str | None:
    """Build completion text that includes tool calls in the format
    the Product Intelligence system can parse: ``tool_name{json_args}``.
    """
    parts: list[str] = []
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
            if name:
                parts.append(f"{name}{args}")
    if assistant_content:
        parts.append(assistant_content)
    return "\n".join(parts) if parts else None


def _model_provider_slug(model: str) -> str:
    text = (model or "").strip().lower()
    if not text:
        return "unknown"
    return text.split("/", 1)[0]


def _model_provider_name(slug: str) -> str:
    known = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "meta": "Meta",
        "mistral": "Mistral",
        "deepseek": "DeepSeek",
        "x-ai": "xAI",
        "xai": "xAI",
        "qwen": "Qwen",
        "nvidia": "NVIDIA",
        "cohere": "Cohere",
    }
    return known.get(slug, slug)


@dataclass
class TraceContext:
    """Accumulates trace data during LLM streaming for OTLP emission.

    Used by both SDK and non-SDK CoPilot paths to collect usage metrics,
    tool calls, and timing information in a consistent structure.
    """

    model: str = ""
    user_id: str | None = None
    session_id: str | None = None
    start_time: float = 0.0

    # Accumulated during streaming
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None

    def emit(
        self,
        *,
        finish_reason: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build and emit the trace as a fire-and-forget background task."""
        fr = finish_reason or ("tool_calls" if self.tool_calls else "stop")
        emit_trace(
            model=self.model,
            messages=messages or [],
            assistant_content="".join(self.text_parts) or None,
            finish_reason=fr,
            prompt_tokens=(self.usage.get("prompt") or self.usage.get("input_tokens")),
            completion_tokens=(
                self.usage.get("completion") or self.usage.get("output_tokens")
            ),
            total_tokens=self.usage.get("total"),
            total_cost_usd=self.cost_usd,
            cache_creation_input_tokens=self.usage.get("cache_creation_input_tokens"),
            cache_read_input_tokens=(
                self.usage.get("cached") or self.usage.get("cache_read_input_tokens")
            ),
            reasoning_tokens=self.usage.get("reasoning"),
            user_id=self.user_id,
            session_id=self.session_id,
            tool_calls=self.tool_calls or None,
            start_time=self.start_time,
            end_time=time.time(),
        )


def _build_otlp_payload(
    *,
    trace_id: str,
    model: str,
    messages: list[dict[str, Any]],
    assistant_content: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    total_cost_usd: float | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict:
    """Build an ``ExportTraceServiceRequest`` JSON payload."""
    provider_slug = _model_provider_slug(model)
    provider_name = _model_provider_name(provider_slug)

    prompt_payload: str | None = None
    if messages:
        prompt_payload = json.dumps({"messages": messages}, default=str)

    completion_payload: str | None = None
    completion_text = _build_completion_text(assistant_content, tool_calls)
    if completion_text is not None:
        completion_obj: dict[str, Any] = {
            "completion": completion_text,
            "reasoning": None,
            "rawRequest": {
                "model": model,
                "stream": True,
                "stream_options": {"include_usage": True},
                "tool_choice": "auto",
                "user": user_id,
                "posthogDistinctId": user_id,
                "session_id": session_id,
            },
        }
        completion_payload = json.dumps(completion_obj, default=str)

    attrs: list[dict] = []
    for kv in [
        _kv("trace.name", "OpenRouter Request"),
        _kv("span.type", "generation"),
        _kv("span.level", "DEFAULT"),
        _kv("gen_ai.operation.name", "chat"),
        _kv("gen_ai.system", provider_slug),
        _kv("gen_ai.provider.name", provider_slug),
        _kv("gen_ai.request.model", model),
        _kv("gen_ai.response.model", model),
        _kv("gen_ai.response.finish_reason", finish_reason),
        _kv("gen_ai.response.finish_reasons", json.dumps([finish_reason])),
        _kv("gen_ai.usage.input_tokens", prompt_tokens),
        _kv("gen_ai.usage.output_tokens", completion_tokens),
        _kv("gen_ai.usage.total_tokens", total_tokens),
        _kv("gen_ai.usage.input_tokens.cached", cache_read_input_tokens),
        _kv(
            "gen_ai.usage.input_tokens.cache_creation",
            cache_creation_input_tokens,
        ),
        _kv("gen_ai.usage.output_tokens.reasoning", reasoning_tokens),
        _kv("user.id", user_id),
        _kv("session.id", session_id),
        _kv("trace.metadata.openrouter.source", "openrouter"),
        _kv("trace.metadata.openrouter.user_id", user_id),
        _kv("gen_ai.usage.total_cost", total_cost_usd),
        _kv("trace.metadata.openrouter.provider_name", provider_name),
        _kv("trace.metadata.openrouter.provider_slug", provider_slug),
        _kv("trace.metadata.openrouter.finish_reason", finish_reason),
    ]:
        if kv is not None:
            attrs.append(kv)

    if prompt_payload is not None:
        attrs.append({"key": "trace.input", "value": {"stringValue": prompt_payload}})
        attrs.append({"key": "span.input", "value": {"stringValue": prompt_payload}})
        attrs.append({"key": "gen_ai.prompt", "value": {"stringValue": prompt_payload}})

    if completion_payload is not None:
        attrs.append(
            {
                "key": "trace.output",
                "value": {"stringValue": completion_payload},
            }
        )
        attrs.append(
            {
                "key": "span.output",
                "value": {"stringValue": completion_payload},
            }
        )
        attrs.append(
            {
                "key": "gen_ai.completion",
                "value": {"stringValue": completion_payload},
            }
        )

    span = {
        "traceId": trace_id,
        "startTimeUnixNano": _nano(start_time or time.time()),
        "endTimeUnixNano": _nano(end_time or time.time()),
        "attributes": attrs,
    }

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "openrouter"},
                        },
                        {
                            "key": "openrouter.trace.id",
                            "value": {
                                "stringValue": (
                                    f"gen-{int(end_time or time.time())}"
                                    f"-{trace_id[:20]}"
                                )
                            },
                        },
                    ]
                },
                "scopeSpans": [{"spans": [span]}],
            }
        ]
    }


async def _send_trace(payload: dict) -> None:
    """POST the OTLP payload to the configured tracing host."""
    url = f"{_TRACING_HOST}/v1/traces"
    try:
        client = _get_client()
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logger.debug(
                "[OTLP] Trace POST returned %d: %s",
                resp.status_code,
                resp.text[:200],
            )
        else:
            logger.debug("[OTLP] Trace sent successfully (%d)", resp.status_code)
    except Exception as e:
        logger.warning("[OTLP] Failed to send trace: %s", e)


# Background task set with backpressure cap.
_bg_tasks: set[asyncio.Task[Any]] = set()
_MAX_BG_TASKS = 64


async def _build_and_send_trace(
    *,
    model: str,
    messages: list[dict[str, Any]],
    assistant_content: str | None,
    finish_reason: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    total_cost_usd: float | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    reasoning_tokens: int | None,
    user_id: str | None,
    session_id: str | None,
    tool_calls: list[dict[str, Any]] | None,
    start_time: float | None,
    end_time: float | None,
) -> None:
    """Build the OTLP payload and send it — runs entirely in a background task."""
    trace_id = uuid.uuid4().hex
    payload = _build_otlp_payload(
        trace_id=trace_id,
        model=model,
        messages=messages,
        assistant_content=assistant_content,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        reasoning_tokens=reasoning_tokens,
        user_id=user_id,
        session_id=session_id,
        tool_calls=tool_calls,
        start_time=start_time,
        end_time=end_time,
    )
    await _send_trace(payload)


def emit_trace(
    *,
    model: str,
    messages: list[dict[str, Any]],
    assistant_content: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    total_cost_usd: float | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    """Fire-and-forget: build and send an OTLP trace span.

    Safe to call from async context — both payload serialization and the
    HTTP POST run in a background task so they never block the event loop.
    """
    if not _TRACING_ENABLED:
        return

    if len(_bg_tasks) >= _MAX_BG_TASKS:
        logger.warning(
            "[OTLP] Backpressure: dropping trace (%d tasks queued)",
            len(_bg_tasks),
        )
        return

    task = asyncio.create_task(
        _build_and_send_trace(
            model=model,
            messages=messages,
            assistant_content=assistant_content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            reasoning_tokens=reasoning_tokens,
            user_id=user_id,
            session_id=session_id,
            tool_calls=tool_calls,
            start_time=start_time,
            end_time=end_time,
        )
    )
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
