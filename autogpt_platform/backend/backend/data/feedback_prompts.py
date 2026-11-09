"""In-app feedback prompts: one row per showing, and the limits on showing them.

A claim decides whether a prompt may appear and records that it did as one
atomic step, so two tabs or devices cannot both be told to ask. Each answer
then patches the row it claimed. Every query is scoped to the caller.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from prisma import Prisma
from prisma.models import FeedbackPrompt, User
from prisma.types import FeedbackPromptUpdateManyMutationInput

from backend.data.db import transaction
from backend.util.json import sanitize_string

FeedbackPromptKind = Literal[
    "trial_cancel",
    "trial_checkout",
    "upgrade_checkout",
    "work_check",
    "otto_mistake",
    "top_up",
    "brain_dump_skip",
    "nps",
]

# The closed list of answers each prompt offers. The values stand in for the
# labels in the frontend's ``FeedbackPrompts/helpers.ts``.
REASONS_BY_KIND: dict[FeedbackPromptKind, frozenset[str]] = {
    "trial_cancel": frozenset(
        {"too_expensive", "no_results_yet", "missing_something", "only_trying", "other"}
    ),
    "trial_checkout": frozenset(
        {"unclear_value", "no_card_yet", "looking_around", "card_failed", "other"}
    ),
    "upgrade_checkout": frozenset({"price", "unclear_value", "card_failed", "other"}),
    "work_check": frozenset(
        {"wrong_facts", "missed_ask", "too_slow", "too_much_help", "other"}
    ),
    "otto_mistake": frozenset(
        {"wrong_facts", "ignored_instructions", "made_up", "too_slow", "other"}
    ),
    "top_up": frozenset(
        {"lower_price", "smaller_amounts", "credits_unclear", "just_trying", "other"}
    ),
    "brain_dump_skip": frozenset(
        {"no_mic", "privacy", "unsure_what_to_say", "prefer_typing"}
    ),
    "nps": frozenset(),
}

# The day this shipped. An account whose day 14 fell before it is not asked.
NPS_ROLLOUT_AT = datetime(2026, 11, 9, tzinfo=UTC)
NPS_AFTER = timedelta(days=14)
WEEKLY_COOLDOWN = timedelta(days=7)
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
    async with transaction() as tx:
        # execute_raw, not query_raw: pg_advisory_xact_lock returns void,
        # which Prisma cannot deserialize as a result column.
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"{user_id}:{kind}",
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
    offer_taken: bool | None,
    now: datetime | None = None,
) -> FeedbackPrompt | None:
    """Save the fields given and leave the rest as they are.

    Returns the row as it now stands, or ``None`` when it is not *user_id*'s.
    """
    now = now or datetime.now(UTC)
    data = _changes(reason, note, score, dismissed, offer_taken, now)
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
    if kind == "nps":
        return await _nps_is_due(tx, user_id, now)
    if kind in ("work_check", "top_up"):
        recent = await FeedbackPrompt.prisma(tx).find_first(
            where={
                "userId": user_id,
                "kind": kind,
                "shownAt": {"gt": now - WEEKLY_COOLDOWN},
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


async def _nps_is_due(tx: Prisma, user_id: str, now: datetime) -> bool:
    """Once ever, from day 14, for accounts reaching day 14 since the rollout."""
    earlier = await FeedbackPrompt.prisma(tx).find_first(
        where={"userId": user_id, "kind": "nps"}
    )
    if earlier is not None:
        return False
    user = await User.prisma(tx).find_unique(where={"id": user_id})
    if user is None:
        return False
    day_14 = user.createdAt + NPS_AFTER
    return NPS_ROLLOUT_AT <= day_14 <= now


def _changes(
    reason: str | None,
    note: str | None,
    score: int | None,
    dismissed: bool,
    offer_taken: bool | None,
    now: datetime,
) -> FeedbackPromptUpdateManyMutationInput:
    data: FeedbackPromptUpdateManyMutationInput = {}
    if reason is not None:
        data["reason"] = reason
    if note is not None:
        data["note"] = sanitize_string(note)
    if score is not None:
        data["score"] = score
    if offer_taken is not None:
        data["offerTaken"] = offer_taken
    if data:
        data["answeredAt"] = now
    if dismissed:
        data["dismissedAt"] = now
    return data
