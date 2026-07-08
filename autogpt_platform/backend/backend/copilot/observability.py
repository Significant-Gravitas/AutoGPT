"""Best-effort Langfuse span helpers for the copilot baseline path.

Tracing must never break a turn: every Langfuse interaction is wrapped so a
misconfigured or unreachable Langfuse degrades to a no-op instead of an error
in the streaming path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import get_client

logger = logging.getLogger(__name__)


@contextmanager
def langfuse_span(name: str, *, input: Any = None) -> Iterator[Any]:
    """Open a named Langfuse span around a block; yields the span or ``None``.

    The span nests under whatever trace/span is active in the current OTEL
    context (the baseline turn's root span), giving per-phase / per-tool
    timing, input and output in the trace view.
    """
    try:
        cm = get_client().start_as_current_span(name=name, input=input)
        span = cm.__enter__()
    except Exception:
        logger.debug("[copilot] Langfuse span %r setup failed", name)
        yield None
        return
    try:
        yield span
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("[copilot] Langfuse span %r teardown failed", name)


def update_span(span: Any, **kwargs: Any) -> None:
    """Best-effort ``span.update`` — ignores tracing failures."""
    if span is None:
        return
    try:
        span.update(**kwargs)
    except Exception:
        logger.debug("[copilot] Langfuse span update failed")
