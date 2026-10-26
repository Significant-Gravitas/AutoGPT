"""A proposal card is answered once.

The card stops asking once the chat holds its decision, but a tab opened
before that, or two tabs at once, can still send a second one. The chat
routes ask here before they record or save a decision: an id the chat already
answered is refused, and so is one another request claimed a moment ago.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.copilot.gate.answers_in_flight import answers_on_their_way
from backend.copilot.gate.card_approval import (
    CARD_DECISION_KIND,
    parse_card_decisions,
    withdraw_card_approvals_recorded_by,
)
from backend.copilot.model import ChatMessage
from backend.copilot.pending_messages import PendingMessage
from backend.data.db_accessors import chat_db
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ClaimResult = Literal[
    "answered", "pending", "in_progress", "claimed", "replayed", "held"
]


class CardClaim(BaseModel):
    result: ClaimResult
    # The cards that were answered already, when ``result`` is "answered".
    answered: frozenset[str] = frozenset()
    # The claim each answered card held when it was read ("owner|accepted"),
    # so clearing a lost one can never clear a claim made since.
    seen: dict[str, str] = Field(default_factory=dict)
    # The cards this claim took, when ``result`` is "claimed".
    claimed: frozenset[str] = frozenset()


_KEY = "copilot:gate:card_decided:"
# Far past a proposal's 15 minutes; the chat itself is the lasting record.
_TTL_SECONDS = 7 * 24 * 60 * 60
# A user-role row the server wrote, not the user: a delegated or subtask
# prompt, a held call's rows. A row queued behind a running turn carries the
# queue's own metadata and is still the user's.
_SERVER_WRITTEN_KEYS = frozenset(
    {"from_session_id", "held_call", "held_calls_answered"}
)
# A claim is "owner|pending|attempt|<deadline ms>" while a request is still
# saving or queueing the answer, and "owner|accepted" once it has: only then
# is the card answered. The owner is the send's client id, which a retransmit
# shares; the attempt is the one request. A pending claim is given back if its
# request fails, and lapses at its deadline if the request died before it
# could (a killed worker): saving or queueing takes seconds.
_PENDING_LEASE_MS = 60_000
# All or nothing: a request that names an id someone else holds takes none.
# {0, id, claim, id, claim...}: answered already, by someone else, and the
# claim each held; {3}: someone else's still
# pending; {4}: another attempt of this send is still pending; {1, ids...}:
# claimed now, those ids; {2}: this send's answer was already accepted.
_CLAIM = """
local now = redis.call('TIME')
now = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local accepted = ARGV[1] .. '|accepted'
local sent = ARGV[1] .. '|pending|'
local mine = sent .. ARGV[2] .. '|'
local take = {}
local answered = {}
for i = 5, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  if not value or string.sub(value, 1, #mine) == mine then
    table.insert(take, ARGV[i])
  elseif value ~= accepted then
    if string.sub(value, -9) == '|accepted' then
      table.insert(answered, ARGV[i])
      table.insert(answered, value)
    else
      local deadline = tonumber(string.match(value, '|(%d+)$'))
      if deadline and deadline > now then
        if string.sub(value, 1, #sent) == sent then return {4} end
        return {3}
      end
      table.insert(take, ARGV[i])
    end
  end
end
if #answered > 0 then
  table.insert(answered, 1, 0)
  return answered
end
if #take == 0 then return {2} end
local claim = mine .. (now + tonumber(ARGV[4]))
for _, id in ipairs(take) do redis.call('HSET', KEYS[1], id, claim) end
redis.call('EXPIRE', KEYS[1], ARGV[3])
table.insert(take, 1, 1)
return take
"""
# Only the attempt that holds a pending claim marks it accepted.
_ACCEPT = """
local mine = ARGV[1] .. '|pending|' .. ARGV[2] .. '|'
for i = 3, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  if value and string.sub(value, 1, #mine) == mine then
    redis.call('HSET', KEYS[1], ARGV[i], ARGV[1] .. '|accepted')
  end
end
return 1
"""
# A lost answer's claim goes only while it is still the claim that was
# read: one cleared and claimed again since is someone else's answer.
_FORGET_SEEN = """
local gone = {}
for i = 1, #ARGV, 2 do
  if redis.call('HGET', KEYS[1], ARGV[i]) == ARGV[i + 1] then
    redis.call('HDEL', KEYS[1], ARGV[i])
    table.insert(gone, ARGV[i])
  end
end
return gone
"""
# A dropped message's sender takes back its own claim only: accepted, or
# still pending under any of its attempts.
_FORGET_OWNED = """
for i = 1, #ARGV, 2 do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  local sent = ARGV[i + 1] .. '|pending|'
  if value == ARGV[i + 1] .. '|accepted'
    or (value and string.sub(value, 1, #sent) == sent) then
    redis.call('HDEL', KEYS[1], ARGV[i])
  end
end
return 1
"""
# Only the attempt that claimed an id gives it back, and only while pending.
_RELEASE = """
local mine = ARGV[1] .. '|pending|' .. ARGV[2] .. '|'
for i = 3, #ARGV do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  if value and string.sub(value, 1, #mine) == mine then
    redis.call('HDEL', KEYS[1], ARGV[i])
  end
end
return 1
"""


async def claim_answer(
    user_id: str,
    session_id: str,
    confirmation_ids: list[str],
    owner: str,
    attempt: str,
) -> CardClaim:
    """Claim these cards for this request, unless something else answered one.

    ``owner`` is the send's scoped client message id, shared by its
    retransmits; ``attempt`` is this request. "claimed": this request took
    the ``claimed`` cards; it marks them accepted with ``accept_claim`` once
    the answer is saved or queued, or gives them back with ``release_claim``.
    "replayed": this very send was already saved or queued (the chat holds
    its row, or its claim was accepted): the request is a replay, to be
    deduplicated, and records nothing, since the approval it recorded may
    already be used up. "held": the store is down; the request goes on as if
    it claimed, with nothing to give back. "answered": the chat, or
    another send's accepted answer, came first, for the ``answered`` ids.
    "pending": another send holds a card but has not saved yet; it may still
    fail, so this one is neither answered nor taken, and should retry.
    "in_progress": the same, for an earlier attempt of this very send. A
    pending claim whose request died lapses after ``_PENDING_LEASE_MS``, and
    an accepted answer that never reached the chat stops holding the card
    once it is no longer on its way there either.
    """
    rows = await _saved_rows(user_id, session_id, confirmation_ids)
    if rows is not None:
        answered = decided_ids(rows, owner) & set(confirmation_ids)
        if answered:
            return CardClaim(result="answered", answered=frozenset(answered))
        # Answered by no row but this send's own: it was saved already.
        if decided_ids(rows, None) & set(confirmation_ids):
            return CardClaim(result="replayed")
    claim = await _claim(session_id, confirmation_ids, owner, attempt)
    # With the chat unread, an accepted answer cannot be told lost.
    if rows is None or claim.result != "answered":
        return claim
    lost = await _lost_answers(user_id, session_id, claim.answered)
    if not lost:
        return claim
    await _forget_lost_answers(
        user_id, session_id, {cid: claim.seen[cid] for cid in lost}
    )
    return await _claim(session_id, confirmation_ids, owner, attempt)


async def _lost_answers(
    user_id: str, session_id: str, confirmation_ids: frozenset[str]
) -> set[str]:
    """The accepted answers among these that are neither in the chat nor on
    their way to it. The chat is read last: a turn may have saved one since
    the claim was read, and then it is not lost."""
    gone = set(confirmation_ids) - await answers_on_their_way(
        session_id, confirmation_ids
    )
    if not gone:
        return set()
    return await unsaved_ids(user_id, session_id, gone) or set()


async def _forget_lost_answers(
    user_id: str, session_id: str, seen: dict[str, str]
) -> None:
    """Clear the claims of lost answers, each only if it is still the claim
    that was read, then their approvals, each only if that same send
    recorded it. Another request may have found the same lost answer,
    cleared it, and answered the card while this one was checking: its
    claim and its approval are not this one's to take."""
    try:
        redis = await get_redis_async()
        pairs = [part for cid, value in seen.items() for part in (cid, value)]
        gone = await redis.eval(_FORGET_SEEN, 1, f"{_KEY}{session_id}", *pairs)
    except Exception:
        logger.warning(
            f"Could not forget lost card answers for session {session_id}",
            exc_info=True,
        )
        return
    if not gone:
        return
    logger.info(f"Forgot {len(gone)} lost card answer(s) in {session_id}")
    await withdraw_card_approvals_recorded_by(
        user_id,
        session_id,
        {str(cid): seen[str(cid)].removesuffix("|accepted") for cid in gone},
    )


async def accept_claim(
    session_id: str, confirmation_ids: list[str], owner: str, attempt: str
) -> None:
    """Mark this request's claim answered once its answer is saved or queued."""
    try:
        redis = await get_redis_async()
        await redis.eval(
            _ACCEPT, 1, f"{_KEY}{session_id}", owner, attempt, *confirmation_ids
        )
    except Exception:
        logger.warning(
            f"Could not accept card answers for session {session_id}", exc_info=True
        )


async def forget_unsaved_answers(
    user_id: str, session_id: str, messages: Iterable[PendingMessage]
) -> None:
    """Forget the answers in queued messages dropped before they were saved.

    A stopped session drops its queued follow-ups; a card decision among them
    never reaches the chat, so the card is the user's to answer again. Only
    what each message's sender still holds goes: its claim, accepted or still
    being queued, and the approval it recorded. Once the message left the
    queue another tab may have found its answer lost and answered the card,
    and that answer is not this Stop's to take. A message queued without its
    sender is left to that recovery.
    """
    owned = [
        (d.confirmation_id, m.card_owner)
        for m in messages
        if m.card_owner
        for d in parse_card_decisions(m.content)
    ]
    if not owned:
        return
    try:
        redis = await get_redis_async()
        pairs = [part for cid, owner in owned for part in (cid, owner)]
        await redis.eval(_FORGET_OWNED, 1, f"{_KEY}{session_id}", *pairs)
    except Exception:
        logger.warning(
            f"Could not forget card answers for session {session_id}", exc_info=True
        )
    await withdraw_card_approvals_recorded_by(user_id, session_id, dict(owned))


async def unsaved_ids(
    user_id: str, session_id: str, confirmation_ids: Iterable[str]
) -> set[str] | None:
    """The ids among these the chat holds no answer for, or None when the
    chat could not be read. A request that failed may have saved its row
    first (a turn that failed to dispatch), and then it answered the card."""
    ids = set(confirmation_ids)
    rows = await _saved_rows(user_id, session_id, ids)
    return None if rows is None else ids - decided_ids(rows, None)


async def _saved_rows(
    user_id: str, session_id: str, confirmation_ids: Iterable[str]
) -> list[ChatMessage] | None:
    """The user's saved rows in this chat that name these cards, every one of
    them: a loaded session holds only its newest rows, and in a long chat an
    answer is older than that. None when the chat cannot be read."""
    try:
        return await chat_db().get_user_messages_mentioning(
            session_id, user_id, sorted(set(confirmation_ids))
        )
    except Exception:
        logger.warning(
            f"Could not read session {session_id} for card answers", exc_info=True
        )
        return None


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
    session_id: str, confirmation_ids: list[str], owner: str, attempt: str
) -> None:
    """Give back a claim when the request that took it saved nothing, so a
    retry of the same answer under a new message id is not refused."""
    try:
        redis = await get_redis_async()
        await redis.eval(
            _RELEASE, 1, f"{_KEY}{session_id}", owner, attempt, *confirmation_ids
        )
    except Exception:
        logger.warning(
            f"Could not release card answers for session {session_id}", exc_info=True
        )


async def _claim(
    session_id: str, confirmation_ids: list[str], owner: str, attempt: str
) -> CardClaim:
    try:
        redis = await get_redis_async()
        reply = await redis.eval(
            _CLAIM,
            1,
            f"{_KEY}{session_id}",
            owner,
            attempt,
            _TTL_SECONDS,
            _PENDING_LEASE_MS,
            *confirmation_ids,
        )
    except Exception:
        logger.warning(
            f"Could not claim card answers for session {session_id}", exc_info=True
        )
        # Fail open, with nothing taken and so nothing to give back.
        return CardClaim(result="held", claimed=frozenset(confirmation_ids))
    code, *ids = reply
    result = _CLAIM_RESULTS[int(code)]
    if result == "answered":
        seen = dict(zip(map(str, ids[::2]), map(str, ids[1::2])))
        return CardClaim(result=result, answered=frozenset(seen), seen=seen)
    return CardClaim(result=result, claimed=frozenset(str(cid) for cid in ids))


_CLAIM_RESULTS: dict[int, ClaimResult] = {
    0: "answered",
    1: "claimed",
    2: "replayed",
    3: "pending",
    4: "in_progress",
}
