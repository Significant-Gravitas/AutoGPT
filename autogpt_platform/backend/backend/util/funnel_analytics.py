"""Server-side funnel events for the experts loop (SECRT-2526 / SECRT-2552).

PostHog is the store for these; the Sentry breadcrumb puts the same event on
the timeline of any error raised after it. Nothing here needs Prisma, so the
Prisma-less briefing scheduler and executor call it directly rather than
through the DB manager.
"""

import logging
from typing import Any

import sentry_sdk

from backend.util.posthog_client import get_posthog_client

logger = logging.getLogger(__name__)


def emit_funnel_event(
    user_id: str, event: str, data: dict[str, Any], data_index: str | None = None
) -> None:
    """Record a funnel event without blocking or raising into the caller.

    The breadcrumb is added in the caller's own scope, which is the only place
    it attaches to an error from the action being measured — one added on a
    background task or another thread lands in that scope and reaches nothing.
    PostHog's client batches on its own thread, so the capture does not wait on
    the network either.

    ``data_index`` is an idempotency key for a redelivery or requeue that can
    emit the same event twice.
    """
    try:
        sentry_sdk.add_breadcrumb(
            category="funnel", message=event, data=dict(data), level="info"
        )
        client = get_posthog_client()
        if client is None:
            return
        properties: dict[str, Any] = {**data}
        if data_index is not None:
            properties["$insert_id"] = data_index
        client.capture(event=event, distinct_id=user_id, properties=properties)
    except Exception:
        logger.exception(f"Failed to emit funnel event {event} for user {user_id}")
