"""Keeping the "waiting on your review" alert in step with reality.

This is one of the conditions the notification surface this replaces could not
report at all: an agent holding outputs that nothing will send until a human
approves or dismisses them. Because it is a *state* rather than an event, the
alert is re-derived from the live count every time the queue changes — so
approving the last item resolves the condition instead of leaving a stale alert
behind.
"""

import logging
from urllib.parse import quote

from prisma.enums import AlertCause, ReviewStatus
from prisma.models import PendingHumanReview
from prisma.types import PendingHumanReviewWhereInput

from backend.copilot.constants import AUTOPILOT_NAME
from backend.data import alerts as alerts_db
from backend.data.graph import get_graph_metadata
from backend.notifications.alert_causes import AwaitingReviewCause
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[ReviewAlerts]")


async def sync_awaiting_review(
    user_id: str, graph_id: str | None = None, *, session_id: str | None = None
) -> None:
    """Raise, update or clear the review-queue alert for one agent or one chat.

    Never raises: a notification must not fail the review flow that triggered
    it.
    """
    scope = f"chat {session_id}" if session_id else f"agent {graph_id}"
    try:
        if session_id:
            where: PendingHumanReviewWhereInput = {"sessionId": session_id}
            cause_key = f"awaiting_review:chat:{session_id}"
        elif graph_id:
            where = {"graphId": graph_id}
            cause_key = f"awaiting_review:{graph_id}"
        else:
            return
        waiting = await PendingHumanReview.prisma().find_many(
            where={**where, "userId": user_id, "status": ReviewStatus.WAITING},
            order={"createdAt": "asc"},
        )
        if not waiting:
            await alerts_db.resolve_alert_condition(user_id, cause_key)
            return

        oldest = waiting[0].createdAt
        if session_id:
            agent = AUTOPILOT_NAME
            cta_path = f"/copilot?sessionId={quote(session_id)}"
        else:
            assert graph_id
            metadata = await get_graph_metadata(graph_id=graph_id)
            agent = metadata.name if metadata else f"Agent {graph_id[:8]}"
            cta_path = f"/library/agents/{graph_id}/reviews"
        cause = AwaitingReviewCause(
            cta_path=cta_path,
            agent=agent,
            count=len(waiting),
            since_label=f"{oldest.day} {oldest.strftime('%b')}, {oldest.strftime('%H:%M')}",
        )
        await alerts_db.raise_alert_condition(
            user_id=user_id,
            cause=AlertCause.AWAITING_REVIEW,
            cause_key=cause_key,
            data=cause.model_dump(mode="json"),
        )
    except Exception:
        logger.warning(
            f"Could not sync the awaiting-review alert for user {user_id} {scope}",
            exc_info=True,
        )
