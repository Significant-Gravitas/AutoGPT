"""Chat-scoped rules: the user's own word on a subject for the rest of the chat.

A rejection sets ``ask``; a card answered with a rule sets ``allow`` or
``judge``. A subject holds one rule at a time and the latest answer replaces it.
"""

import logging
from typing import Literal, Mapping, get_args

from prisma.enums import ReviewStatus

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ChatRule = Literal["allow", "judge", "ask"]
_RULES: dict[str, ChatRule] = {rule: rule for rule in get_args(ChatRule)}

# Outlives any chat anyone is still in; dead chats expire out of Redis.
_TTL_SECONDS = 90 * 24 * 60 * 60
_ASK_KEY = "copilot:gate:ask:"

DECLINED = "You declined this action earlier in this chat."
# An outage asks rather than runs, but must not claim the user said no.
UNREADABLE = (
    "Your earlier decisions in this chat could not be checked, so this action "
    "needs your approval."
)


async def set_ask(session_id: str, rule_key: str) -> None:
    """Only the user's answer on a card replaces an ask, so re-proposing the
    call with a space added to its arguments cannot buy a fresh verdict."""
    await set_rule(session_id, rule_key, "ask")


async def set_rule(session_id: str, rule_key: str, rule: ChatRule) -> None:
    try:
        redis = await get_redis_async()
        await redis.setex(_key(session_id, rule_key), _TTL_SECONDS, rule)
    except Exception:
        logger.warning(
            f"Gate could not persist a {rule} rule for session {session_id}",
            exc_info=True,
        )


async def set_answer_rules(
    session_id: str,
    answered: Mapping[str, PendingHumanReviewModel],
    rules: Mapping[str, ChatRule | None],
    subject_keys: Mapping[str, str],
) -> None:
    """The rule each approved card asked for, on the subject it named.

    ``subject_keys`` come from the held calls the gate stored, so a click can
    only ever rule on a subject the server put on a card.
    """
    for review_id, rule in rules.items():
        row = answered.get(review_id)
        key = subject_keys.get(review_id)
        if rule and key and row is not None and row.status == ReviewStatus.APPROVED:
            await set_rule(session_id, key, rule)


async def rule_for(
    session_id: str, rule_key: str
) -> ChatRule | Literal["unreadable"] | None:
    """``unreadable`` asks like ``ask`` but must not claim the user said no."""
    try:
        redis = await get_redis_async()
        raw = await redis.get(_key(session_id, rule_key))
    except Exception:
        logger.warning(
            f"Gate could not read chat rules for session {session_id}; "
            "assuming the subject asks",
            exc_info=True,
        )
        return "unreadable"
    if raw is None:
        return None
    value = raw.decode() if isinstance(raw, bytes) else str(raw)
    # Rows written before rules had decisions hold "1", which meant ask.
    return _RULES.get(value, "ask")


def _key(session_id: str, rule_key: str) -> str:
    # A flag per (session, subject): the cluster client's set operations are
    # not typed as awaitable.
    return f"{_ASK_KEY}{session_id}:{rule_key}"
