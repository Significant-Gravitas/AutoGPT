"""Chat rules: the user's own word on a subject from now on.

A rejection sets ``ask``; a card answered with a rule sets ``allow`` or
``judge``. A rule holds in its chat, or wider when the user picks a scope: every
chat with that Expert (or with Otto), or every Expert on their team. The
narrowest rule on record decides, and the latest answer replaces a rule.

An allow or judge rule also has a lifetime (``Lifetime``): ``once`` is spent by
the first call it covers, ``turn`` holds for the task that first uses it,
``chat`` for as long as the chat, ``ttl`` for a number of hours and ``always``
until the user answers otherwise. The short lifetimes only make sense in one
chat, so they are kept to it whatever scope was asked for.

A chat a tool started on another chat's behalf (``delegated_by_session_id``)
can only be stricter than the chats above it: their declines hold in it, their
allows do not.
"""

import json
import logging
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Mapping, Sequence, get_args

from prisma.enums import ReviewStatus
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.model import ChatSessionInfo, get_chat_session_metadata
from backend.data.redis_client import AsyncRedisClient, get_redis_async

logger = logging.getLogger(__name__)

ChatRule = Literal["allow", "judge", "ask"]
_RULES: dict[str, ChatRule] = {rule: rule for rule in get_args(ChatRule)}
Scope = Literal["chat", "expert", "team"]
Lifetime = Literal["once", "turn", "chat", "ttl", "always"]
# What "Always allow" on a card saves when the user picks no variant.
DEFAULT_SCOPE: Scope = "chat"
DEFAULT_LIFETIME: Lifetime = "always"
# Lifetimes bound to one chat or one task: a wider scope would outlive them.
_CHAT_BOUND: frozenset[str] = frozenset({"once", "turn", "chat"})
MAX_TTL_HOURS = 24 * 365

# Outlives any chat anyone is still in; dead chats expire out of Redis.
_TTL_SECONDS = 90 * 24 * 60 * 60
# Long enough for the answered card's own turn, which a single use or a
# single task is meant for; a forgotten one does not linger for months.
_SHORT_TTL_SECONDS = 24 * 60 * 60
# How far up a chain of delegated chats a decline is looked for; matches
# ``tree.MAX_DEPTH``, past which nothing can be spawned.
_MAX_ANCESTORS = 3
_ASK_KEY = "copilot:gate:ask:"
# Otto's own chats have no expert id; their rules share one key.
_OTTO = "otto"

DECLINED = "You declined this action earlier in this chat."
# An outage asks rather than runs, but must not claim the user said no.
UNREADABLE = (
    "Your earlier decisions in this chat could not be checked, so this action "
    "needs your approval."
)


INHERITED = "You declined this in the chat that started this task."


class AllowGrant(BaseModel):
    """Where and for how long an allow (or judge) rule from a card holds."""

    model_config = ConfigDict(frozen=True)

    scope: Scope = DEFAULT_SCOPE
    lifetime: Lifetime = DEFAULT_LIFETIME
    # Only for ``ttl``.
    ttl_hours: int | None = Field(default=None, ge=1, le=MAX_TTL_HOURS)

    def normalized(self) -> "AllowGrant":
        """A lifetime bound to one chat holds in that chat only, and a ``ttl``
        without hours is a day."""
        scope = "chat" if self.lifetime in _CHAT_BOUND else self.scope
        hours = (self.ttl_hours or 24) if self.lifetime == "ttl" else None
        return AllowGrant(scope=scope, lifetime=self.lifetime, ttl_hours=hours)


class _Stored(BaseModel):
    """A rule as kept in Redis when it carries a lifetime."""

    rule: ChatRule
    lifetime: Lifetime = DEFAULT_LIFETIME
    since: date | None = None
    expires_at: datetime | None = None
    # The task a ``turn`` rule was first used in.
    turn: str | None = None


class RuleHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: ChatRule | Literal["unreadable"]
    scope: Scope = "chat"
    since: date | None = None
    otto: bool = False
    lifetime: Lifetime | None = None
    # A decline from a chat above this one in its delegation chain.
    inherited: bool = False

    @property
    def reason(self) -> str:
        if self.rule == "unreadable":
            return UNREADABLE
        if self.inherited:
            return INHERITED
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
    for scope in ("expert", "team"):
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
    grants: Mapping[str, AllowGrant] | None = None,
) -> None:
    """The rule each approved card asked for, on the subject it named, at the
    scope the user picked.

    ``subject_keys`` come from the held calls the gate stored, so a click can
    only ever rule on a subject the server put on a card. A card answered with
    a ``grants`` entry ("Always allow" and its variants) saves its rule with
    that scope and lifetime; without one, the rule lasts as long as the chat.
    """
    scopes = scopes or {}
    grants = grants or {}
    chat: ChatSessionInfo | None = None
    for review_id, rule in rules.items():
        row = answered.get(review_id)
        key = subject_keys.get(review_id)
        if not (rule and key and row is not None):
            continue
        if row.status != ReviewStatus.APPROVED:
            continue
        if (grant := grants.get(review_id)) is not None:
            grant = grant.normalized()
            expert_id: str | None = None
            if grant.scope == "expert":
                chat = chat or await get_chat_session_metadata(session_id, user_id)
                # An unknown chat must not rule for Otto's chats instead.
                if chat is None:
                    grant = grant.model_copy(update={"scope": "chat"})
                else:
                    expert_id = chat.expert_id
            await set_granted_rule(session_id, user_id, expert_id, key, rule, grant)
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


async def set_granted_rule(
    session_id: str,
    user_id: str,
    expert_id: str | None,
    rule_key: str,
    rule: ChatRule,
    grant: AllowGrant,
) -> None:
    """Save a rule with a lifetime at the grant's scope.

    Written on the same key as every other rule at that scope, so a later
    decline replaces it there and ``set_ask`` revokes it from a narrower chat.
    """
    grant = grant.normalized()
    now = datetime.now(UTC)
    stored = _Stored(rule=rule, lifetime=grant.lifetime, since=now.date())
    ttl: int | None
    if grant.lifetime == "ttl":
        hours = grant.ttl_hours or 24
        stored.expires_at = now + timedelta(hours=hours)
        ttl = hours * 60 * 60
    elif grant.lifetime in ("once", "turn"):
        ttl = _SHORT_TTL_SECONDS
    elif grant.lifetime == "always" and grant.scope != "chat":
        # Until the user answers otherwise; nothing about it ages out.
        ttl = None
    else:
        ttl = _TTL_SECONDS
    key = (
        _key(session_id, rule_key)
        if grant.scope == "chat"
        else _scoped_key(grant.scope, user_id, expert_id, rule_key)
    )
    try:
        redis = await get_redis_async()
        value = stored.model_dump_json()
        if ttl is None:
            await redis.set(key, value)
        else:
            await redis.setex(key, ttl, value)
    except Exception:
        logger.warning(
            f"Gate could not persist a {grant.lifetime} {rule} rule "
            f"for session {session_id}",
            exc_info=True,
        )


async def rule_for(
    session_id: str,
    rule_key: str,
    user_id: str,
    expert_id: str | None,
    *,
    ancestors: Sequence[str] = (),
    turn_id: str | None = None,
) -> RuleHit | None:
    """The narrowest rule on record: this chat's, then the Expert's, then the
    team's. ``unreadable`` asks like ``ask`` but must not claim the user said no.

    A decline in any of ``ancestors`` (the chats that started this one) comes
    first: a delegated chat may only be stricter than the chat above it.
    A ``once`` rule is spent by this lookup and a ``turn`` rule binds to
    ``turn_id``; one already spent or bound to another task is passed over, as
    if the next scope out were the narrowest on record.
    """
    try:
        redis = await get_redis_async()
        for ancestor in ancestors:
            raw = await redis.get(_key(ancestor, rule_key))
            stored = _parse(raw, legacy_scoped=False) if raw is not None else None
            if stored is not None and stored.rule == "ask":
                return RuleHit(rule="ask", inherited=True)
        lookups: list[tuple[Scope, str]] = [
            ("chat", _key(session_id, rule_key)),
            ("expert", _scoped_key("expert", user_id, expert_id, rule_key)),
            ("team", _scoped_key("team", user_id, expert_id, rule_key)),
        ]
        for scope, key in lookups:
            raw = await redis.get(key)
            if raw is None:
                continue
            stored = _parse(raw, legacy_scoped=scope != "chat")
            if not await _live(redis, key, stored, turn_id):
                continue
            if scope == "chat":
                return RuleHit(rule=stored.rule, lifetime=_lifetime_of(raw, stored))
            return RuleHit(
                rule=stored.rule,
                scope=scope,
                since=stored.since or datetime.now(UTC).date(),
                otto=expert_id is None,
                lifetime=_lifetime_of(raw, stored),
            )
    except Exception:
        logger.warning(
            f"Gate could not read chat rules for session {session_id}; "
            "assuming the subject asks",
            exc_info=True,
        )
        return RuleHit(rule="unreadable")
    return None


async def ancestors_of(
    session_id: str, parent_id: str | None, user_id: str
) -> list[str]:
    """The chats that started this one, nearest first, up to the spawn depth.

    Stops at a chat that cannot be read or is not this user's: the chain is
    only ever used to add declines, so a short chain can only ask less often
    than a full one would, never run what the user refused in this chat.
    """
    chain: list[str] = []
    seen = {session_id}
    current = parent_id
    while current and current not in seen and len(chain) < _MAX_ANCESTORS:
        chain.append(current)
        seen.add(current)
        try:
            parent = await get_chat_session_metadata(current, user_id)
        except Exception:
            logger.warning(
                f"Gate could not read parent chat {current} of {session_id}",
                exc_info=True,
            )
            break
        current = parent.metadata.delegated_by_session_id if parent else None
    return chain


async def _live(
    redis: AsyncRedisClient, key: str, stored: _Stored, turn_id: str | None
) -> bool:
    """Whether a stored rule still holds for this call, spending it if it is
    single-use. A declined subject never expires here: only an answer does."""
    if stored.rule == "ask":
        return True
    if stored.expires_at is not None and stored.expires_at <= datetime.now(UTC):
        await redis.delete(key)
        return False
    if stored.lifetime == "once" or (stored.lifetime == "turn" and not turn_id):
        # The delete IS the mutex: two parallel calls cannot both spend it.
        return bool(await redis.delete(key))
    if stored.lifetime == "turn":
        if stored.turn is None:
            stored.turn = turn_id
            await redis.setex(key, _SHORT_TTL_SECONDS, stored.model_dump_json())
            return True
        if stored.turn != turn_id:
            await redis.delete(key)
            return False
    return True


def _parse(raw: bytes | str, *, legacy_scoped: bool) -> _Stored:
    """Read any rule ever written: a lifetime rule's JSON, a scoped rule's
    ``"<rule> <day>"``, or a chat rule's bare word (``"1"`` before decisions
    existed, which meant ask). Anything unreadable asks."""
    text = _text(raw)
    if text.startswith("{"):
        try:
            return _Stored.model_validate(json.loads(text))
        except (ValueError, ValidationError):
            return _Stored(rule="ask")
    if legacy_scoped:
        rule, _, since = text.partition(" ")
        try:
            day = date.fromisoformat(since) if since else None
        except ValueError:
            day = None
        return _Stored(rule=_RULES.get(rule, "ask"), since=day)
    return _Stored(rule=_RULES.get(text, "ask"))


def _lifetime_of(raw: bytes | str, stored: _Stored) -> Lifetime | None:
    # Rules written before lifetimes existed last as long as their chat.
    return stored.lifetime if _text(raw).startswith("{") else None


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
