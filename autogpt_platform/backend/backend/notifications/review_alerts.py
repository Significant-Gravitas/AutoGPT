"""Keeping the "waiting on your review" alert in step with reality.

This is one of the conditions the notification surface this replaces could not
report at all: an agent holding outputs that nothing will send until a human
approves or dismisses them. Because it is a *state* rather than an event, the
alert is re-derived from the live count every time the queue changes — so
approving the last item resolves the condition instead of leaving a stale alert
behind.
"""

import logging

from prisma.enums import AlertCause, ReviewStatus
from prisma.models import PendingHumanReview

from backend.data import alerts as alerts_db
from backend.data.graph import get_graph_metadata
from backend.notifications.alert_causes import AwaitingReviewCause
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[ReviewAlerts]")


async def sync_awaiting_review(user_id: str, graph_id: str) -> None:
    """Raise, update or clear the review-queue alert for one agent.

    Never raises: a notification must not fail the review flow that triggered
    it.
    """
    try:
        waiting = await PendingHumanReview.prisma().find_many(
            where={
                "userId": user_id,
                "graphId": graph_id,
                "status": ReviewStatus.WAITING,
            },
            order={"createdAt": "asc"},
        )
        cause_key = f"awaiting_review:{graph_id}"
        if not waiting:
            await alerts_db.resolve_condition(user_id, cause_key)
            return

        oldest = waiting[0].createdAt
        metadata = await get_graph_metadata(graph_id=graph_id)
        cause = AwaitingReviewCause(
            cta_path=f"/library/agents/{graph_id}/reviews",
            agent=metadata.name if metadata else f"Agent {graph_id[:8]}",
            count=len(waiting),
            since_label=f"{oldest.day} {oldest.strftime('%b')}, {oldest.strftime('%H:%M')}",
        )
        await alerts_db.raise_condition(
            user_id=user_id,
            cause=AlertCause.AWAITING_REVIEW,
            cause_key=cause_key,
            data=cause.model_dump(mode="json"),
        )
    except Exception:
        logger.warning(
            f"Could not sync the awaiting-review alert for user {user_id} agent "
            f"{graph_id}",
            exc_info=True,
        )
