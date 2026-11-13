"""Server-side funnel events for the experts loop (SECRT-2526 / SECRT-2552).

PostHog is the store for these; the Sentry breadcrumb puts the same event on
the timeline of any error raised after it. Nothing here needs Prisma, so the
Prisma-less briefing scheduler and executor call it directly rather than
through the DB manager.
"""

import logging
from typing import Any

import sentry_sdk

from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)


def emit_funnel_event(
    user_id: str,
    event: PostHogEvent,
    data: dict[str, Any],
    data_index: str | None = None,
) -> None:
    """Record a funnel event without blocking or raising into the caller.

    The breadcrumb is added in the caller's own scope, which is the only place
    it attaches to an error from the action being measured — one added on a
    background task or another thread lands in that scope and reaches nothing.
    PostHog's client batches on its own thread, so the capture does not wait on
    the network either.

    ``data_index`` is an idempotency key for a redelivery or requeue that can
    emit the same event twice; it is carried both as ``$insert_id`` and as a
    deterministic event uuid.
    """
    event_name = event.value
    # Separate boundaries: a breadcrumb failure must not cost the capture.
    try:
        sentry_sdk.add_breadcrumb(
            category="funnel", message=event_name, data=dict(data), level="info"
        )
    except Exception:
        logger.exception(f"Failed to breadcrumb funnel event {event_name}")

    posthog_client.capture(user_id, event, data, dedup_key=data_index)
