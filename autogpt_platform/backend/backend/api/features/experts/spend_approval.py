"""Spend-approval gate for expert work (SECRT-2599).

The weekly budget (``scheduling.enforce_expert_run_budget``) is the hard stop
that pauses an expert's schedules. This is the softer line below it: once her
credit spend in the window reaches the approval threshold, new work is parked
as a ``PendingHumanReview`` row — the row Home's "Needs You" list, the run
page and the chat card already render and decide — and proceeds once the
user approves. One approval unlocks the rest of the window.
"""

import logging
import uuid

import prisma.models
from prisma.enums import ResourceVisibility, ReviewStatus
from pydantic import BaseModel

from backend.copilot import db as chat_db
from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
)
from backend.data import human_review
from backend.data.execution import ExecutionStatus, update_graph_execution_stats
from backend.data.expert_spend import SpendWindow, get_spend, window_start
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Present in every spend-approval review id, on both the graph-execution and
# the chat shape, so one ``contains`` lookup finds an expert's decisions.
SPEND_REVIEW_MARKER = "expert-spend:"
_POST_NAMESPACE = uuid.UUID("2b7d1c4e-5a6f-4e8b-9c0d-1e2f3a4b5c6d")


class SpendApprovalParkFailed(Exception):
    """The execution could not be held, so it must not be allowed to run."""


class SpendApprovalNeeded(BaseModel):
    expert_id: str
    expert_name: str
    spent: int
    threshold: int
    window: SpendWindow

    @property
    def period(self) -> str:
        return "this week" if self.window == "week" else "today"

    @property
    def headline(self) -> str:
        return (
            f"{self.expert_name} has used {self.spent} of {self.threshold} credits "
            f"{self.period} and needs your approval to keep spending"
        )


def approval_threshold() -> int | None:
    """Credits per window before work waits for approval; None = disabled."""
    value = settings.config.expert_spend_approval_threshold_default
    return value if value > 0 else None


def approval_window() -> SpendWindow:
    return settings.config.expert_spend_approval_window


def is_spend_review(node_exec_id: str) -> bool:
    return SPEND_REVIEW_MARKER in node_exec_id


def spend_review_id(expert_id: str, graph_exec_id: str) -> str:
    return f"{SPEND_REVIEW_MARKER}{expert_id}:{graph_exec_id}"


async def spend_approval_required(
    user_id: str, expert_id: str
) -> SpendApprovalNeeded | None:
    """What the user must approve before this expert spends more, or None
    when she may proceed: check disabled, spend below the threshold, or an
    approval already given in the current window.

    Like the weekly gate, the spend read is a snapshot: concurrent starts can
    each pass before any of them meters, bounded by their own cost."""
    threshold = approval_threshold()
    if threshold is None:
        return None
    if not await is_feature_enabled(Flag.EXPERT_SPEND_APPROVAL, user_id):
        return None
    window = approval_window()
    spent = await get_spend(expert_id, window)
    if spent < threshold:
        return None
    if await _approved_in_window(user_id, expert_id, window):
        return None
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    return SpendApprovalNeeded(
        expert_id=expert_id,
        expert_name=expert.name if expert else "Your expert",
        spent=spent,
        threshold=threshold,
        window=window,
    )


async def _approved_in_window(
    user_id: str, expert_id: str, window: SpendWindow
) -> bool:
    row = await prisma.models.PendingHumanReview.prisma().find_first(
        where={
            "userId": user_id,
            "nodeExecId": {"contains": f"{SPEND_REVIEW_MARKER}{expert_id}:"},
            "status": ReviewStatus.APPROVED,
            "reviewedAt": {"gte": window_start(window)},
        }
    )
    return row is not None


async def park_execution_for_spend_approval(
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    graph_version: int,
    needed: SpendApprovalNeeded,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> None:
    """Hold a just-created, unpublished execution: one review row keyed on
    it, status REVIEW so the run page shows the approval, and one message in
    the expert's thread per window."""
    review_id = spend_review_id(needed.expert_id, graph_exec_id)
    await human_review.get_or_create_human_review(
        user_id=user_id,
        node_exec_id=review_id,
        graph_exec_id=graph_exec_id,
        graph_id=graph_id,
        graph_version=graph_version,
        input_data=_review_payload(needed),
        message=needed.headline,
        editable=False,
        organization_id=organization_id,
        team_id=team_id,
    )
    if (
        await update_graph_execution_stats(
            graph_exec_id=graph_exec_id, status=ExecutionStatus.REVIEW
        )
        is None
    ):
        # The resume gate reads the durable status, so a waiting review over an
        # execution left in another status is worse than no gate: the run
        # requeues unapproved while the card still asks. Undo and fail instead.
        await human_review.delete_review_by_node_exec_id(review_id, user_id)
        raise SpendApprovalParkFailed(
            f"Execution #{graph_exec_id} could not be set to REVIEW"
        )
    await _post_thread_message(user_id, needed)


async def parked_spend_decision(
    user_id: str, expert_id: str, graph_exec_id: str
) -> ReviewStatus | None:
    """The user's decision on a parked execution; None when it was never parked."""
    review_id = spend_review_id(expert_id, graph_exec_id)
    reviews = await human_review.get_reviews_by_node_exec_ids([review_id], user_id)
    review = reviews.get(review_id)
    return review.status if review else None


async def open_chat_spend_review(
    user_id: str,
    session_id: str,
    needed: SpendApprovalNeeded,
    block_name: str,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> str:
    """Park a paid ``run_block`` on the session's review rails and return the
    review id. An open row for the same expert is reused so a model retry
    does not stack cards."""
    synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
    node_id = f"{COPILOT_NODE_PREFIX}{SPEND_REVIEW_MARKER}{needed.expert_id}"
    for review in await human_review.get_pending_reviews_for_execution(
        synthetic_graph_id, user_id
    ):
        if review.node_id == node_id:
            return review.node_exec_id
    review_id = f"{node_id}{COPILOT_NODE_EXEC_ID_SEPARATOR}{uuid.uuid4().hex[:8]}"
    await human_review.get_or_create_human_review(
        user_id=user_id,
        node_exec_id=review_id,
        graph_exec_id=synthetic_graph_id,
        graph_id=synthetic_graph_id,
        graph_version=1,
        input_data=_review_payload(needed, block=block_name),
        message=needed.headline,
        editable=False,
        organization_id=organization_id,
        team_id=team_id,
    )
    return review_id


def _review_payload(needed: SpendApprovalNeeded, block: str | None = None) -> dict:
    payload: dict = {
        "expert": needed.expert_name,
        "spent_credits": needed.spent,
        "threshold_credits": needed.threshold,
        "window": needed.window,
    }
    if block:
        payload["block"] = block
    return payload


async def _post_thread_message(user_id: str, needed: SpendApprovalNeeded) -> None:
    """Once per expert per window, deduplicated by message id. Never raises —
    a failed post must not affect the park."""
    bucket = window_start(needed.window).date().isoformat()
    message_id = str(
        uuid.uuid5(_POST_NAMESPACE, f"spend-approval:{needed.expert_id}:{bucket}")
    )
    content = (
        f"I've used {needed.spent} of my {needed.threshold} credits "
        f"{needed.period}, so I'm holding new work until you approve more "
        "spending — it's waiting under Needs You on your Home page."
    )
    try:
        await chat_db.append_expert_run_message(
            user_id=user_id,
            expert_id=needed.expert_id,
            content=content,
            message_id=message_id,
        )
    except Exception as e:
        logger.warning(
            f"Failed to post spend-approval message for expert "
            f"#{needed.expert_id}: {type(e).__name__}: {e}"
        )
