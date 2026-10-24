"""A proposal card is answered once.

The card stops asking once the chat holds its decision, but a tab opened
before that, or two tabs at once, can still send a second one. The chat
routes ask here before they record or save a decision: an id the chat already
answered is refused, and so is one another request claimed a moment ago.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from backend.copilot.gate.card_approval import (
    CARD_DECISION_KIND,
    parse_card_decisions,
    withdraw_card_approvals,
)
from backend.copilot.model import ChatMessage, get_chat_session
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ClaimResult = Literal["answered", "pending", "claimed", "held"]

_KEY = "copilot:gate:card_decided:"
# Far past a proposal's 15 minutes; the chat itself is the lasting record.
_TTL_SECONDS = 7 * 24 * 60 * 60
# A user-role row the server wrote, not the user: a delegated or subtask
# prompt, a held call's rows. A row queued behind a running turn carries the
# queue's own metadata and is still the user's.
_SERVER_WRITTEN_KEYS = frozenset(
    {"from_session_id", "held_call", "held_calls_answered"}
)
# A claim is "owner|pending|<deadline ms>" while that request is still saving
# or queueing the answer, and "owner|accepted" once it has: only then is the
# card answered. A pending claim can be given back if the request fails, and
# lapses at its deadline if the request died before it could (a killed
# worker): saving or queueing takes seconds, so a claim this old is no one's.
_PENDING_LEASE_MS = 60_000
# All or nothing: a request that names an id someone else holds takes none.
# 0: someone else's accepted answer, 3: someone else's still pending,
# 1: claimed now, 2: already this owner's.
_CLAIM = """
local now = redis.call('TIME')
now = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local pending = ARGV[1] .. '|pending|'
local accepted = ARGV[1] .. '|accepted'
local take = {}
for i = 4, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  local mine = value == accepted
    or (value and string.sub(value, 1, #pending) == pending)
  if not value then
    table.insert(take, ARGV[i])
  elseif not mine then
    if string.sub(value, -9) == '|accepted' then return 0 end
    local deadline = tonumber(string.match(value, '|pending|(%d+)$'))
    if deadline and deadline > now then return 3 end
    table.insert(take, ARGV[i])
  end
end
if #take == 0 then return 2 end
local claim = pending .. (now + tonumber(ARGV[3]))
for _, id in ipairs(take) do redis.call('HSET', KEYS[1], id, claim) end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""
# Only the request that holds a pending claim marks it accepted.
_ACCEPT = """
local pending = ARGV[1] .. '|pending|'
for i = 2, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  if value and string.sub(value, 1, #pending) == pending then
    redis.call('HSET', KEYS[1], ARGV[i], ARGV[1] .. '|accepted')
  end
end
return 1
"""
# Only the request that claimed an id gives it back, and only while pending.
_RELEASE = """
local pending = ARGV[1] .. '|pending|'
for i = 2, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  if value and string.sub(value, 1, #pending) == pending then
    redis.call('HDEL', KEYS[1], ARGV[i])
  end
end
return 1
"""


async def claim_answer(
    user_id: str, session_id: str, confirmation_ids: list[str], owner: str
) -> ClaimResult:
    """Claim these cards for this request, unless something else answered one.

    ``owner`` is the request's scoped client message id. "claimed": this
    request took the claim; it marks it accepted with ``accept_claim`` once
    the answer is saved or queued, or gives it back with ``release_claim``.
    "held": it was already this owner's (a retransmit of the answering
    request) or the store is down, so there is nothing to give back.
    "answered": the chat, or another request's accepted answer, came first.
    "pending": another request holds the card but has not saved yet; it may
    still fail, so this one is neither answered nor taken, and should retry.
    A pending claim whose request died lapses after ``_PENDING_LEASE_MS``.
    """
    try:
        session = await get_chat_session(session_id, user_id)
    except Exception:
        logger.warning(
            f"Could not read session {session_id} for card answers", exc_info=True
        )
        session = None
    if session and decided_ids(session.messages, owner) & set(confirmation_ids):
        return "answered"
    claimed = await _claim(session_id, confirmation_ids, owner)
    return "answered" if claimed == "taken" else claimed


async def accept_claim(
    session_id: str, confirmation_ids: list[str], owner: str
) -> None:
    """Mark this request's claim answered once its answer is saved or queued."""
    try:
        redis = await get_redis_async()
        await redis.eval(_ACCEPT, 1, f"{_KEY}{session_id}", owner, *confirmation_ids)
    except Exception:
        logger.warning(
            f"Could not accept card answers for session {session_id}", exc_info=True
        )


async def forget_unsaved_answers(
    user_id: str, session_id: str, messages: Iterable[str]
) -> None:
    """Forget the answers in queued messages dropped before they were saved.

    A stopped session drops its queued follow-ups; a card decision among them
    never reaches the chat, so the card is the user's to answer again. An id
    the chat already holds an answer for is left alone.
    """
    ids = {d.confirmation_id for m in messages for d in parse_card_decisions(m)}
    if not ids:
        return
    try:
        session = await get_chat_session(session_id, user_id)
    except Exception:
        logger.warning(
            f"Could not read session {session_id} for card answers", exc_info=True
        )
        return
    if session:
        ids -= decided_ids(session.messages, None)
    if not ids:
        return
    await withdraw_card_approvals(user_id, session_id, ids)
    try:
        redis = await get_redis_async()
        await redis.hdel(f"{_KEY}{session_id}", *ids)
    except Exception:
        logger.warning(
            f"Could not forget card answers for session {session_id}", exc_info=True
        )


def decided_ids(
    messages: Iterable[ChatMessage], skip_message_id: str | None
) -> set[str]:
    """The confirmation ids the user's rows in this chat already answered."""
    decided: set[str] = set()
    for message in messages:
        if message.role != "user":
            continue
        if skip_message_id is not None and message.id == skip_message_id:
            continue
        metadata = message.metadata or {}
        if is_server_written(metadata):
            continue
        if metadata.get("kind") == CARD_DECISION_KIND:
            decided |= {
                d["confirmation_id"]
                for d in metadata.get("decisions", [])
                if isinstance(d, dict) and isinstance(d.get("confirmation_id"), str)
            }
            continue
        # Typed, queued, or saved before rows were tagged: the line is the record.
        decided |= {
            d.confirmation_id for d in parse_card_decisions(message.content or "")
        }
    return decided


def is_server_written(metadata: Mapping[str, Any]) -> bool:
    """A user-role row the server wrote for something else says nothing about
    the user's answers, whatever its text."""
    kind = metadata.get("kind")
    return (
        any(key in metadata for key in _SERVER_WRITTEN_KEYS)
        or metadata.get("hidden") is True
        or (kind is not None and kind != CARD_DECISION_KIND)
    )


async def release_claim(
    session_id: str, confirmation_ids: list[str], owner: str
) -> None:
    """Give back a claim when the request that took it saved nothing, so a
    retry of the same answer under a new message id is not refused."""
    try:
        redis = await get_redis_async()
        await redis.eval(_RELEASE, 1, f"{_KEY}{session_id}", owner, *confirmation_ids)
    except Exception:
        logger.warning(
            f"Could not release card answers for session {session_id}", exc_info=True
        )


async def _claim(
    session_id: str, confirmation_ids: list[str], owner: str
) -> Literal["claimed", "held", "taken", "pending"]:
    try:
        redis = await get_redis_async()
        result = int(
            await redis.eval(
                _CLAIM,
                1,
                f"{_KEY}{session_id}",
                owner,
                _TTL_SECONDS,
                _PENDING_LEASE_MS,
                *confirmation_ids,
            )
        )
    except Exception:
        logger.warning(
            f"Could not claim card answers for session {session_id}", exc_info=True
        )
        # Fail open, with nothing taken and so nothing to give back.
        return "held"
    if result == 0:
        return "taken"
    if result == 3:
        return "pending"
    return "claimed" if result == 1 else "held"
