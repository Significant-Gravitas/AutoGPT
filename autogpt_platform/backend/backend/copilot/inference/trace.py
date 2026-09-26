"""A Langfuse trace around one background LLM call, the way chat traces a turn.

Chat opens a parent span per turn, propagates the user, session, trace name,
tags and metadata onto it (``sdk/service.py``, ``baseline/service.py``) and
writes the turn's ``gen_ai.usage.*`` onto the span as it closes. ``trace(ctx)``
does the same for a background call: a span named ``kind:phase``, the scope's
user, the job's correlation id as the session, tags for the kind, the
execution path and the expert, and the usage the caller hands back on
``TracedCall.usage`` (or the billed usage an ``InferenceError`` carries).

A call made inside a span someone else opened (a consult inside the chat turn
that asked for it) becomes a child span and leaves that trace's own
attributes alone. Without Langfuse keys ``trace`` does nothing, and a tracing
failure never fails the call.
"""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from types import TracebackType

from langfuse import get_client, propagate_attributes
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from backend.util.settings import Settings

from .context import InferenceContext, InferenceError, InferenceUsage

logger = logging.getLogger(__name__)

settings = Settings()

_ExcInfo = tuple[type[BaseException] | None, BaseException | None, TracebackType | None]
_NO_EXCEPTION: _ExcInfo = (None, None, None)


class TracedCall(BaseModel):
    """One traced call: its context (with the trace id once one is open) and
    the usage to write on the span when it closes."""

    ctx: InferenceContext
    usage: InferenceUsage | None = None


@asynccontextmanager
async def trace(ctx: InferenceContext) -> AsyncIterator[TracedCall]:
    """Trace one call; set ``usage`` on what this yields once the call returns."""
    spans = _open_spans(ctx)
    if spans is None:
        yield TracedCall(ctx=ctx)
        return
    call = TracedCall(ctx=ctx.model_copy(update={"trace_id": _current_trace_id()}))
    exc_info = _NO_EXCEPTION
    try:
        yield call
    except InferenceError as exc:
        call.usage = call.usage or exc.usage
        exc_info = sys.exc_info()
        raise
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _write_usage(call.usage)
        _close_spans(spans, exc_info)


def _open_spans(ctx: InferenceContext) -> ExitStack | None:
    """The call's span, plus the trace's attributes when the call starts the
    trace; ``None`` when Langfuse is off or the span would not open."""
    if not _langfuse_configured():
        return None
    spans = ExitStack()
    try:
        starts_trace = not otel_trace.get_current_span().get_span_context().is_valid
        spans.enter_context(
            get_client().start_as_current_span(
                name=ctx.job.label, metadata=_metadata(ctx)
            )
        )
        if starts_trace:
            spans.enter_context(
                propagate_attributes(
                    user_id=ctx.scope.user_id,
                    session_id=ctx.job.correlation_id,
                    trace_name=ctx.job.label,
                    tags=_tags(ctx),
                    metadata=_metadata(ctx),
                )
            )
    except Exception:
        logger.debug("Langfuse span for %s did not open", ctx.job.label, exc_info=True)
        _close_spans(spans, _NO_EXCEPTION)
        return None
    return spans


def _close_spans(spans: ExitStack, exc_info: _ExcInfo) -> None:
    try:
        spans.__exit__(*exc_info)
    except Exception:
        logger.warning("Langfuse span teardown failed", exc_info=True)


def _write_usage(usage: InferenceUsage | None) -> None:
    """The ``gen_ai.usage.*`` attributes chat writes on its turn span."""
    if usage is None:
        return
    try:
        span = otel_trace.get_current_span()
        if not span.is_recording():
            return
        span.set_attribute("gen_ai.usage.prompt_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.completion_tokens", usage.output_tokens)
        span.set_attribute("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens)
        span.set_attribute(
            "gen_ai.usage.cache_creation_tokens", usage.cache_creation_tokens
        )
        if usage.cost_usd is not None:
            span.set_attribute("gen_ai.usage.cost_usd", usage.cost_usd)
    except Exception:
        logger.debug("Failed to set OTEL usage attributes", exc_info=True)


def _current_trace_id() -> str | None:
    try:
        return get_client().get_current_trace_id()
    except Exception:
        logger.debug("Failed to read the Langfuse trace id", exc_info=True)
        return None


def _tags(ctx: InferenceContext) -> list[str]:
    tags = [ctx.job.kind, f"execution_path:{ctx.route.execution_path}"]
    if ctx.scope.expert_id:
        tags.append(f"expert:{ctx.scope.expert_id}")
    return tags


def _metadata(ctx: InferenceContext) -> dict[str, str]:
    """String values only: Langfuse drops any other kind from propagated
    metadata."""
    metadata = {
        "job_kind": ctx.job.kind,
        "phase": ctx.job.phase,
        "correlation_id": ctx.job.correlation_id,
        "latency_class": ctx.job.latency_class,
        "tier": ctx.job.tier,
        "model": ctx.route.model,
        "provider": ctx.route.provider,
        "payer": ctx.route.payer,
        "execution_path": ctx.route.execution_path,
        "route_reason": ctx.route.reason,
        "expert_id": ctx.scope.expert_id,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _langfuse_configured() -> bool:
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )
