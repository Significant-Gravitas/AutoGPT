"""In-app feedback prompts: one row per showing, and the limits on showing them.

A claim decides whether a prompt may appear and records that it did as one
atomic step, so two tabs or devices cannot both be told to ask. Each answer
then patches the row it claimed. Every query is scoped to the caller.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

from prisma import Prisma
from prisma.enums import AgentExecutionStatus, SubscriptionTier
from prisma.models import AgentGraphExecution, ChatSession, FeedbackPrompt, User
from prisma.types import FeedbackPromptUpdateManyMutationInput

from backend.data.db import transaction
from backend.util.json import sanitize_string

FeedbackPromptKind = Literal[
    "trial_cancel",
    "upgrade_checkout",
    "work_check",
    "otto_mistake",
    "top_up",
    "nps",
]

TrialCancelReason = Literal[
    "too_expensive", "no_results_yet", "missing_something", "only_trying", "other"
]

# A work check and a mistake flag offer the same list, so their answers can be
# read together.
_WORK_REASONS = frozenset(
    {
        "wrong_facts",
        "missed_ask",
        "made_up",
        "too_slow",
        "too_much_help",
        "not_usable",
        "other",
    }
)

# The closed list of answers each prompt offers. The values stand in for the
# labels in the frontend's ``FeedbackPrompts/helpers.ts``.
REASONS_BY_KIND: dict[FeedbackPromptKind, frozenset[str]] = {
    "trial_cancel": frozenset(get_args(TrialCancelReason)),
    "upgrade_checkout": frozenset(
        {"price", "unclear_value", "just_looking", "card_failed", "other"}
    ),
    "work_check": _WORK_REASONS,
    "otto_mistake": _WORK_REASONS,
    "top_up": frozenset(
        {"lower_price", "smaller_amounts", "credits_unclear", "just_trying", "other"}
    ),
    "nps": frozenset(),
}

# The prompts that ask out of the blue rather than answer something the person
# just did. They share one wait, so someone is asked one of them a week at most.
SHARED_KINDS: tuple[FeedbackPromptKind, ...] = ("work_check", "top_up", "nps")
SHARED_WAIT = timedelta(days=7)
# Two of them closed with "Not now" in a row, and none of them asks for a while.
QUIET_AFTER_TWO_NOT_NOWS = timedelta(days=30)
TOP_UP_WAIT = timedelta(days=30)
# The day this shipped. An account whose day 14 fell before it is not asked.
NPS_ROLLOUT_AT = datetime(2026, 11, 9, tzinfo=UTC)
NPS_AFTER = timedelta(days=14)
# NPS asks only someone paying who got something done in this window.
NPS_ACTIVE_WITHIN = timedelta(days=7)
_UNPAID_TIERS = (None, SubscriptionTier.NO_TIER, SubscriptionTier.TRIAL)
SCORES_BY_KIND: dict[FeedbackPromptKind, frozenset[int]] = {
    "nps": frozenset(range(11)),
    "work_check": frozenset({0, 1, 2}),
    "otto_mistake": frozenset({0}),
}


async def claim_feedback_prompt(
    *,
    user_id: str,
    kind: FeedbackPromptKind,
    session_id: str | None = None,
    message_id: str | None = None,
    now: datetime | None = None,
) -> str | None:
    """Record a showing of *kind* if its limit allows one, and return its id.

    Returns ``None`` when the prompt must not appear. The lock is per user and
    kind and lasts until the transaction ends, so a concurrent claim waits and
    then sees this one's row.
    """
    now = now or datetime.now(UTC)
    # The shared kinds take one lock, so two of them cannot both pass the wait.
    lock_group = "shared" if kind in SHARED_KINDS else kind
    async with transaction() as tx:
        # execute_raw, not query_raw: pg_advisory_xact_lock returns void,
        # which Prisma cannot deserialize as a result column.
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"{user_id}:{lock_group}",
        )
        if not await _is_allowed(tx, user_id, kind, session_id, now):
            return None
        row = await FeedbackPrompt.prisma(tx).create(
            data={
                "userId": user_id,
                "kind": kind,
                "sessionId": session_id,
                "messageId": message_id,
                "shownAt": now,
            }
        )
    return row.id


async def update_feedback_prompt(
    *,
    user_id: str,
    prompt_id: str,
    reason: str | None,
    note: str | None,
    score: int | None,
    dismissed: bool,
    now: datetime | None = None,
) -> FeedbackPrompt | None:
    """Save the fields given and leave the rest as they are.

    Returns the row as it now stands, or ``None`` when it is not *user_id*'s.
    """
    now = now or datetime.now(UTC)
    data = _changes(reason, note, score, dismissed, now)
    if data:
        updated = await FeedbackPrompt.prisma().update_many(
            where={"id": prompt_id, "userId": user_id}, data=data
        )
        if not updated:
            return None
    return await get_feedback_prompt(user_id=user_id, prompt_id=prompt_id)


async def get_feedback_prompt(*, user_id: str, prompt_id: str) -> FeedbackPrompt | None:
    return await FeedbackPrompt.prisma().find_first(
        where={"id": prompt_id, "userId": user_id}
    )


async def count_feedback_prompts(*, user_id: str, kind: FeedbackPromptKind) -> int:
    return await FeedbackPrompt.prisma().count(where={"userId": user_id, "kind": kind})


async def _is_allowed(
    tx: Prisma,
    user_id: str,
    kind: FeedbackPromptKind,
    session_id: str | None,
    now: datetime,
) -> bool:
    """Whether *kind* may be shown now. Every limit lives here."""
    if kind in SHARED_KINDS and not await _shared_wait_is_over(tx, user_id, now):
        return False
    if kind == "nps":
        return await _nps_is_due(tx, user_id, now)
    if kind == "top_up":
        recent = await FeedbackPrompt.prisma(tx).find_first(
            where={
                "userId": user_id,
                "kind": kind,
                "shownAt": {"gt": now - TOP_UP_WAIT},
            }
        )
        return recent is None
    if kind == "otto_mistake":
        # The limit is per chat, so without a chat it cannot be kept.
        if session_id is None:
            return False
        earlier = await FeedbackPrompt.prisma(tx).find_first(
            where={"userId": user_id, "kind": kind, "sessionId": session_id}
        )
        return earlier is None
    return True


async def _shared_wait_is_over(tx: Prisma, user_id: str, now: datetime) -> bool:
    """A week since any shared prompt showed, or 30 days after two "Not now"s."""
    latest = await FeedbackPrompt.prisma(tx).find_many(
        where={"userId": user_id, "kind": {"in": list(SHARED_KINDS)}},
        order={"shownAt": "desc"},
        take=2,
    )
    if latest and latest[0].shownAt > now - SHARED_WAIT:
        return False
    waved_off = len(latest) == 2 and all(_closed_unanswered(row) for row in latest)
    return not (waved_off and latest[0].shownAt > now - QUIET_AFTER_TWO_NOT_NOWS)


def _closed_unanswered(row: FeedbackPrompt) -> bool:
    return row.dismissedAt is not None and row.answeredAt is None


async def _nps_is_due(tx: Prisma, user_id: str, now: datetime) -> bool:
    """Once ever, from day 14, for accounts reaching day 14 since the rollout,
    and only while they pay and got something done in the last week."""
    earlier = await FeedbackPrompt.prisma(tx).find_first(
        where={"userId": user_id, "kind": "nps"}
    )
    if earlier is not None:
        return False
    user = await User.prisma(tx).find_unique(where={"id": user_id})
    if user is None or user.subscriptionTier in _UNPAID_TIERS:
        return False
    day_14 = user.createdAt + NPS_AFTER
    if not NPS_ROLLOUT_AT <= day_14 <= now:
        return False
    return await _got_something_done(tx, user_id, now - NPS_ACTIVE_WITHIN)


async def _got_something_done(tx: Prisma, user_id: str, since: datetime) -> bool:
    """A workflow run that finished, or a chat Otto replied in, since *since*."""
    run = await AgentGraphExecution.prisma(tx).find_first(
        where={
            "userId": user_id,
            "isDeleted": False,
            "executionStatus": AgentExecutionStatus.COMPLETED,
            "updatedAt": {"gt": since},
        }
    )
    if run is not None:
        return True
    chat = await ChatSession.prisma(tx).find_first(
        where={
            "userId": user_id,
            "updatedAt": {"gt": since},
            "totalCompletionTokens": {"gt": 0},
        }
    )
    return chat is not None


def _changes(
    reason: str | None,
    note: str | None,
    score: int | None,
    dismissed: bool,
    now: datetime,
) -> FeedbackPromptUpdateManyMutationInput:
    data: FeedbackPromptUpdateManyMutationInput = {}
    if reason is not None:
        data["reason"] = reason
    if note is not None:
        data["note"] = sanitize_string(note)
    if score is not None:
        data["score"] = score
    if data:
        data["answeredAt"] = now
    if dismissed:
        data["dismissedAt"] = now
    return data
