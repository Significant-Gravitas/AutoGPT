"""Chat rules: the user's own word on a subject from now on.

A rejection sets ``ask``; a card answered with a rule sets ``allow`` or
``judge``. A rule holds in its chat, or wider when the user picks a scope: every
chat with that Expert (or with Otto), or every Expert on their team. The
narrowest rule on record decides, and the latest answer replaces a rule.
"""

import logging
from datetime import UTC, date, datetime
from typing import Literal, Mapping, get_args

from prisma.enums import ReviewStatus
from pydantic import BaseModel, ConfigDict

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.model import ChatSessionInfo, get_chat_session_metadata
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ChatRule = Literal["allow", "judge", "ask"]
_RULES: dict[str, ChatRule] = {rule: rule for rule in get_args(ChatRule)}
Scope = Literal["chat", "expert", "team"]
_WIDER: tuple[Literal["expert", "team"], ...] = ("expert", "team")

# Outlives any chat anyone is still in; dead chats expire out of Redis.
_TTL_SECONDS = 90 * 24 * 60 * 60
_ASK_KEY = "copilot:gate:ask:"
# Otto's own chats have no expert id; their rules share one key.
_OTTO = "otto"

DECLINED = "You declined this action earlier in this chat."
# An outage asks rather than runs, but must not claim the user said no.
UNREADABLE = (
    "Your earlier decisions in this chat could not be checked, so this action "
    "needs your approval."
)


class RuleHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: ChatRule | Literal["unreadable"]
    scope: Scope = "chat"
    since: date | None = None
    otto: bool = False

    @property
    def reason(self) -> str:
        if self.rule == "unreadable":
            return UNREADABLE
        if self.scope == "chat" or self.since is None:
            return DECLINED
        when = f"{self.since.day} {self.since:%b}"
        if self.scope == "team":
            return f"You declined this for every Expert on your team on {when}."
        who = AUTOPILOT_NAME if self.otto else "this Expert"
        return f"You declined this in every chat with {who} on {when}."


async def set_ask(
    session_id: str, rule_key: str, user_id: str, expert_id: str | None
) -> None:
    """Only the user's answer on a card replaces an ask, so re-proposing the
    call with a space added to its arguments cannot buy a fresh verdict.

    A rejection also revokes every wider rule that would have run the call.
    """
    await set_rule(session_id, rule_key, "ask")
    for scope in _WIDER:
        key = _scoped_key(scope, user_id, expert_id, rule_key)
        try:
            redis = await get_redis_async()
            held = await redis.get(key) is not None
        except Exception:
            logger.warning(
                f"Gate could not check {scope} rules for user {user_id}",
                exc_info=True,
            )
            continue
        if held:
            await _set_scoped(key, "ask")


async def set_rule(session_id: str, rule_key: str, rule: ChatRule) -> None:
    try:
        redis = await get_redis_async()
        await redis.setex(_key(session_id, rule_key), _TTL_SECONDS, rule)
    except Exception:
        logger.warning(
            f"Gate could not persist a {rule} rule for session {session_id}",
            exc_info=True,
        )


async def set_scoped_rule(
    scope: Literal["expert", "team"],
    user_id: str,
    expert_id: str | None,
    rule_key: str,
    rule: ChatRule,
) -> None:
    await _set_scoped(_scoped_key(scope, user_id, expert_id, rule_key), rule)


async def set_answer_rules(
    session_id: str,
    user_id: str,
    answered: Mapping[str, PendingHumanReviewModel],
    rules: Mapping[str, ChatRule | None],
    subject_keys: Mapping[str, str],
    scopes: Mapping[str, Scope] | None = None,
) -> None:
    """The rule each approved card asked for, on the subject it named, at the
    scope the user picked.

    ``subject_keys`` come from the held calls the gate stored, so a click can
    only ever rule on a subject the server put on a card.
    """
    scopes = scopes or {}
    chat: ChatSessionInfo | None = None
    for review_id, rule in rules.items():
        row = answered.get(review_id)
        key = subject_keys.get(review_id)
        if not (rule and key and row is not None):
            continue
        if row.status != ReviewStatus.APPROVED:
            continue
        await set_rule(session_id, key, rule)
        scope = scopes.get(review_id, "chat")
        if scope == "team":
            await set_scoped_rule("team", user_id, None, key, rule)
        elif scope == "expert":
            chat = chat or await get_chat_session_metadata(session_id, user_id)
            # An unknown chat must not rule for Otto's chats instead.
            if chat is not None:
                await set_scoped_rule("expert", user_id, chat.expert_id, key, rule)


async def rule_for(
    session_id: str, rule_key: str, user_id: str, expert_id: str | None
) -> RuleHit | None:
    """The narrowest rule on record: this chat's, then the Expert's, then the
    team's. ``unreadable`` asks like ``ask`` but must not claim the user said no."""
    try:
        redis = await get_redis_async()
        # The keys sit in different cluster slots, so a plain MGET is refused.
        chat, *wider = await redis.mget_nonatomic(
            [
                _key(session_id, rule_key),
                _scoped_key("expert", user_id, expert_id, rule_key),
                _scoped_key("team", user_id, expert_id, rule_key),
            ]
        )
        if chat is not None:
            # Rows written before rules had decisions hold "1", which meant ask.
            return RuleHit(rule=_RULES.get(_text(chat), "ask"))
        for scope, raw in zip(_WIDER, wider):
            if raw is not None:
                rule, _, since = _text(raw).partition(" ")
                return RuleHit(
                    rule=_RULES.get(rule, "ask"),
                    scope=scope,
                    since=date.fromisoformat(since),
                    otto=expert_id is None,
                )
    except Exception:
        logger.warning(
            f"Gate could not read chat rules for session {session_id}; "
            "assuming the subject asks",
            exc_info=True,
        )
        return RuleHit(rule="unreadable")
    return None


async def _set_scoped(key: str, rule: ChatRule) -> None:
    today = datetime.now(UTC).date().isoformat()
    try:
        redis = await get_redis_async()
        await redis.setex(key, _TTL_SECONDS, f"{rule} {today}")
    except Exception:
        logger.warning(f"Gate could not persist a scoped {rule} rule", exc_info=True)


def _key(session_id: str, rule_key: str) -> str:
    # A flag per (session, subject): the cluster client's set operations are
    # not typed as awaitable.
    return f"{_ASK_KEY}{session_id}:{rule_key}"


def _scoped_key(
    scope: Literal["expert", "team"],
    user_id: str,
    expert_id: str | None,
    rule_key: str,
) -> str:
    if scope == "team":
        return f"{_ASK_KEY}team:{user_id}:{rule_key}"
    return f"{_ASK_KEY}expert:{user_id}:{expert_id or _OTTO}:{rule_key}"


def _text(raw: bytes | str) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)
