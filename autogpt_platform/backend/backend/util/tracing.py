"""One convention for Sentry transactions and the timings recorded on them.

HTTP requests get a transaction from the SDK's integrations. Work that arrives
any other way (a copilot turn off a queue, a graph node run by the executor)
starts its own here, so each unit of work is a bounded transaction with a
stable, low-cardinality name that can be compared release over release:

* ``start_transaction`` runs a block as one transaction.
* ``set_timing`` / ``set_data`` put numbers on it as span data, which Sentry's
  span explorer aggregates as a typed attribute, e.g.
  ``p95(tags[copilot.turn.ttft_ms,number])``.
* ``suppress_child_spans`` keeps I/O that repeats for as long as a unit of work
  lasts (stream polls, lock keepalives) from filling the SDK's span limit.
* ``trace_propagation_headers`` carries the active trace across a handover that
  is not an HTTP request, such as a queue message.

Sentry's server-side data scrubbing drops span data whose name contains
``token``, ``auth``, ``secret`` or ``password``, so names must avoid them:
``ttft_ms``, not ``first_token_ms``.

Nothing here raises into the caller. When Sentry is off or a call fails, the
work runs untraced.
"""

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

import sentry_sdk
from sentry_sdk.tracing import NoOpSpan, Span, Transaction, TransactionSource

logger = logging.getLogger(__name__)

SENTRY_TRACE_HEADER = "sentry-trace"
BAGGAGE_HEADER = "baggage"

# The probe and scrape endpoints our services serve, matched by exact path: a
# route that merely ends in one of these names (a store agent slugged
# "health") is real traffic.
_UNTRACED_PATHS = frozenset(
    {
        # WebSocket server: Kubernetes probe.
        "/",
        # Every FastAPI service: Prometheus scrape.
        "/metrics",
        # REST API and WebSocket server: Kubernetes probe, load balancer check.
        "/health",
        # Every RPC service (AppService): Kubernetes probe.
        "/health_check",
        "/health_check_async",
        # Chat service health check.
        "/api/chat/health",
        # Codex compatibility gateway.
        "/healthz",
    }
)


def traces_sampler(sampling_context: Mapping[str, Any]) -> float:
    """Trace everything except probes, metric scrapes and WebSocket connections.

    A WebSocket transaction lasts as long as its connection, so it is neither
    bounded nor comparable; probes and scrapes are frequent and say nothing.
    Any other transaction keeps its parent's decision when it has one.
    """
    try:
        if _is_untraced(sampling_context):
            return 0.0
        parent_sampled = sampling_context.get("parent_sampled")
        if parent_sampled is not None:
            return 1.0 if parent_sampled else 0.0
    except Exception:
        logger.debug("traces_sampler could not classify a transaction", exc_info=True)
    return 1.0


def _is_untraced(sampling_context: Mapping[str, Any]) -> bool:
    transaction = sampling_context.get("transaction_context") or {}
    asgi_scope = sampling_context.get("asgi_scope") or {}
    if (
        transaction.get("op") == "websocket.server"
        or asgi_scope.get("type") == "websocket"
    ):
        return True
    return _request_path(sampling_context) in _UNTRACED_PATHS


def _request_path(sampling_context: Mapping[str, Any]) -> str | None:
    """The request's path, without scheme, host or query string.

    It comes from the ASGI scope or the aiohttp request when the integration
    passes one, or else from the transaction name, which at sampling time is
    still the raw URL.
    """
    path = (sampling_context.get("asgi_scope") or {}).get("path")
    if isinstance(path, str):
        return path
    path = getattr(sampling_context.get("aiohttp_request"), "path", None)
    if isinstance(path, str):
        return path
    transaction = sampling_context.get("transaction_context") or {}
    name = transaction.get("name")
    if transaction.get("op") != "http.server" or not isinstance(name, str):
        return None
    return urlsplit(name).path


@contextmanager
def start_transaction(
    op: str,
    name: str,
    *,
    trace_headers: Mapping[str, str] | None = None,
    trace_id: str | None = None,
    tags: Mapping[str, str] | None = None,
    data: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Run the block as its own transaction and yield it.

    ``op`` and ``name`` must be low-cardinality; ids belong in ``data``. The
    trace is, in order: the one ``trace_headers`` carries (work handed over
    from elsewhere), the transaction already running in this scope (the block
    becomes its child), ``trace_id`` (units of work that share a trace by key),
    or a new one.

    The block runs in a forked Sentry scope, so neither the transaction nor
    anything the block sets on the scope reaches concurrent work that shares
    the caller's. An exception leaving the block marks the transaction failed,
    or ``cancelled`` for a cancellation, and propagates unchanged; a block that
    returns without setting a status leaves it ``ok``.
    """
    if not _tracing_active():
        yield NoOpSpan()
        return
    with sentry_sdk.isolation_scope():
        span = _begin_transaction(op, name, trace_headers, trace_id, tags, data)
        if span is None:
            yield NoOpSpan()
            return
        try:
            yield span
        except BaseException as exc:
            set_status(_status_for(exc), span=span)
            raise
        else:
            if span.status is None:
                set_status("ok", span=span)
        finally:
            _finish(span)


def _begin_transaction(
    op: str,
    name: str,
    trace_headers: Mapping[str, str] | None,
    trace_id: str | None,
    tags: Mapping[str, str] | None,
    data: Mapping[str, Any] | None,
) -> Span | None:
    """The started and entered transaction, or None if it could not start."""
    try:
        transaction = sentry_sdk.continue_trace(
            _incoming_trace(trace_headers, trace_id),
            op=op,
            name=name,
            source=TransactionSource.TASK,
        )
        for key, value in (tags or {}).items():
            transaction.set_tag(key, value)
        for key, value in (data or {}).items():
            transaction.set_data(key, value)
        span = sentry_sdk.start_transaction(transaction)
        span.__enter__()
        return span
    except Exception:
        logger.warning("Could not start Sentry transaction %s", name, exc_info=True)
        return None


def _incoming_trace(
    trace_headers: Mapping[str, str] | None, trace_id: str | None
) -> dict[str, str]:
    if trace_headers and trace_headers.get(SENTRY_TRACE_HEADER):
        return {
            key: value
            for key, value in trace_headers.items()
            if key in (SENTRY_TRACE_HEADER, BAGGAGE_HEADER)
        }
    if active := trace_propagation_headers():
        return active
    if trace_id:
        return {SENTRY_TRACE_HEADER: trace_id}
    return {}


def _finish(span: Span) -> None:
    try:
        span.__exit__(None, None, None)
    except Exception:
        logger.warning("Could not finish Sentry transaction", exc_info=True)


def _status_for(exc: BaseException) -> str:
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return "cancelled"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "aborted"
    return "internal_error"


def set_status(status: str, *, span: Span | None = None) -> None:
    """Set the status of ``span`` or else the current transaction, from
    Sentry's vocabulary (``ok``, ``cancelled``, ``internal_error``,
    ``invalid_argument``, ...)."""
    try:
        target = span if span is not None else current_transaction()
        if target is not None:
            target.set_status(status)
    except Exception:
        logger.debug("Could not set Sentry span status", exc_info=True)


def current_transaction() -> Transaction | None:
    """The transaction running in the current scope, if any."""
    try:
        return sentry_sdk.get_current_scope().transaction
    except Exception:
        return None


def set_timing(name: str, milliseconds: float, *, span: Span | None = None) -> None:
    """Record a duration in milliseconds, named ``<area>.<what>_ms``.

    It lands on ``span`` or else the current transaction, as numeric span data
    that the span explorer can aggregate. Negative durations, which only clock
    skew between hosts produces, are recorded as zero.
    """
    try:
        value = round(max(float(milliseconds), 0.0), 3)
    except (TypeError, ValueError):
        logger.debug("Not a duration for %s: %r", name, milliseconds)
        return
    set_data(name, value, span=span)


def set_data(name: str, value: Any, *, span: Span | None = None) -> None:
    """Attach a value to ``span`` or else the current transaction."""
    try:
        target = span if span is not None else current_transaction()
        if target is not None:
            target.set_data(name, value)
    except Exception:
        logger.debug("Could not set Sentry span data %s", name, exc_info=True)


@contextmanager
def suppress_child_spans() -> Iterator[None]:
    """Run the block without recording its spans on the active transaction.

    For I/O that repeats for as long as a unit of work lasts. The block's spans
    are dropped, not its errors: anything it captures is still reported, on the
    same trace.
    """
    if not _tracing_active() or not _span_active():
        yield
        return
    try:
        scope_manager = sentry_sdk.new_scope()
        scope = scope_manager.__enter__()
        scope.span = None
    except Exception:
        logger.debug("Could not suppress Sentry child spans", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        scope_manager.__exit__(None, None, None)


def trace_propagation_headers() -> dict[str, str]:
    """The active transaction's ``sentry-trace`` and ``baggage`` headers.

    Empty outside a transaction, so a handover from untraced code starts a new
    trace instead of joining whichever trace its thread last saw.
    """
    try:
        if not _tracing_active():
            return {}
        span = sentry_sdk.get_current_scope().span
        if span is None or span.containing_transaction is None:
            return {}
        return dict(span.iter_headers())
    except Exception:
        logger.debug("Could not read Sentry trace headers", exc_info=True)
        return {}


def trace_id_for(key: str) -> str:
    """A trace id derived from ``key``, for units of work that share a trace.

    A UUID maps to its own 32 hex digits, so such a trace can be opened
    straight from the id it was keyed by.
    """
    try:
        return uuid.UUID(key).hex
    except (AttributeError, TypeError, ValueError):
        return hashlib.sha256(str(key).encode()).hexdigest()[:32]


def _tracing_active() -> bool:
    try:
        return sentry_sdk.get_client().is_active()
    except Exception:
        return False


def _span_active() -> bool:
    """Whether a span started now would get a parent, as ``Scope.start_span``
    decides it."""
    try:
        return (
            sentry_sdk.get_current_scope().span is not None
            or sentry_sdk.get_isolation_scope().span is not None
        )
    except Exception:
        return True
