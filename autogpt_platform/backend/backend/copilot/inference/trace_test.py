"""``trace`` gives a background call the Langfuse trace chat gives a turn."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from . import trace as trace_mod
from .context import (
    InferenceContext,
    InferenceError,
    InferenceJob,
    InferenceScope,
    InferenceUsage,
    RouteDecision,
)
from .trace import trace

_CTX = InferenceContext(
    scope=InferenceScope(user_id="u1", expert_id="e1"),
    job=InferenceJob(
        kind="dream",
        phase="recombine",
        correlation_id="pass-1",
        latency_class="deferred",
        tier="advanced",
    ),
    route=RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider="open_router",
        model="anthropic/claude-opus-5-5",
        payer="platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider="open_router",
        reason="open_router chat transport, advanced tier (fast_advanced_model)",
    ),
)

_USAGE = InferenceUsage(
    model="anthropic/claude-opus-5-5",
    input_tokens=120,
    output_tokens=30,
    cache_read_tokens=7,
    cache_creation_tokens=3,
    cost_usd=0.0123,
    cost_source="provider",
    payer="platform_allowance",
)


class _FakeSpan:
    """The current OTEL span: inside someone else's trace or not."""

    def __init__(self, *, inside_a_trace: bool) -> None:
        self.attributes: dict[str, object] = {}
        self._context = SimpleNamespace(is_valid=inside_a_trace)

    def get_span_context(self):
        return self._context

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _FakeLangfuse:
    def __init__(self) -> None:
        self.spans: list[dict] = []
        self.exits: list[type[BaseException] | None] = []

    @contextmanager
    def start_as_current_span(self, *, name, metadata):
        self.spans.append({"name": name, "metadata": metadata})
        try:
            yield SimpleNamespace()
        except BaseException as exc:
            self.exits.append(type(exc))
            raise
        else:
            self.exits.append(None)

    def get_current_trace_id(self) -> str:
        return "trace-1"


@pytest.fixture
def langfuse(monkeypatch):
    """Langfuse configured, with a fake client, tracer and propagation."""
    client = _FakeLangfuse()
    propagated: list[dict] = []

    @contextmanager
    def fake_propagate(**kwargs):
        propagated.append(kwargs)
        yield

    span = _FakeSpan(inside_a_trace=False)
    monkeypatch.setattr(trace_mod, "_langfuse_configured", lambda: True)
    monkeypatch.setattr(trace_mod, "get_client", lambda: client)
    monkeypatch.setattr(trace_mod, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(
        trace_mod, "otel_trace", SimpleNamespace(get_current_span=lambda: span)
    )
    return SimpleNamespace(client=client, propagated=propagated, span=span)


@pytest.mark.asyncio
async def test_without_langfuse_it_does_nothing(monkeypatch):
    monkeypatch.setattr(trace_mod, "_langfuse_configured", lambda: False)
    get_client = MagicMock()
    with patch.object(trace_mod, "get_client", get_client):
        async with trace(_CTX) as call:
            call.usage = _USAGE
    get_client.assert_not_called()
    assert call.ctx is _CTX


@pytest.mark.asyncio
async def test_without_langfuse_errors_still_propagate(monkeypatch):
    monkeypatch.setattr(trace_mod, "_langfuse_configured", lambda: False)
    with pytest.raises(InferenceError):
        async with trace(_CTX):
            raise InferenceError("bad json")


@pytest.mark.asyncio
async def test_a_background_call_starts_a_trace_like_a_chat_turn(langfuse):
    async with trace(_CTX) as call:
        call.usage = _USAGE

    (span,) = langfuse.client.spans
    assert span["name"] == "dream:recombine"
    (attributes,) = langfuse.propagated
    assert attributes["user_id"] == "u1"
    assert attributes["session_id"] == "pass-1"
    assert attributes["trace_name"] == "dream:recombine"
    assert attributes["tags"] == ["dream", "execution_path:sync_baseline", "expert:e1"]
    assert attributes["metadata"]["model"] == "anthropic/claude-opus-5-5"
    assert attributes["metadata"]["payer"] == "platform_allowance"
    # Langfuse drops non-string propagated metadata, so none is sent.
    assert all(isinstance(v, str) for v in attributes["metadata"].values())
    assert call.ctx.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_usage_lands_on_the_span_as_chat_writes_it(langfuse):
    async with trace(_CTX) as call:
        call.usage = _USAGE

    assert langfuse.span.attributes == {
        "gen_ai.usage.prompt_tokens": 120,
        "gen_ai.usage.completion_tokens": 30,
        "gen_ai.usage.cache_read_tokens": 7,
        "gen_ai.usage.cache_creation_tokens": 3,
        "gen_ai.usage.cost_usd": 0.0123,
    }
    assert langfuse.client.exits == [None]


@pytest.mark.asyncio
async def test_a_billed_failure_still_reports_its_usage(langfuse):
    with pytest.raises(InferenceError):
        async with trace(_CTX):
            raise InferenceError("did not parse", _USAGE)

    assert langfuse.span.attributes["gen_ai.usage.prompt_tokens"] == 120
    # The span closes with the error, so the trace shows the call failed.
    assert langfuse.client.exits == [InferenceError]


@pytest.mark.asyncio
async def test_inside_a_chat_turn_it_is_a_child_span_only(langfuse):
    """A consult runs inside the asking turn's trace; renaming that trace or
    re-tagging it would clobber the chat's own attributes."""
    langfuse.span._context = SimpleNamespace(is_valid=True)

    async with trace(_CTX):
        pass

    assert [s["name"] for s in langfuse.client.spans] == ["dream:recombine"]
    assert langfuse.propagated == []


@pytest.mark.asyncio
async def test_a_span_that_will_not_open_never_fails_the_call(langfuse, monkeypatch):
    def broken_client():
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(trace_mod, "get_client", broken_client)

    async with trace(_CTX) as call:
        call.usage = _USAGE

    assert call.ctx.trace_id is None
