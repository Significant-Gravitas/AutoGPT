"""A proposal card is answered once.

The card stops asking once the chat holds its decision, but a tab opened
before that, or two tabs at once, can still send a second one. The chat
routes ask here before they record or save a decision: an id the chat already
answered is refused, and so is one another request claimed a moment ago.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from backend.copilot.gate.card_approval import CARD_DECISION_KIND, parse_card_decisions
from backend.copilot.model import ChatMessage, get_chat_session
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

ClaimResult = Literal["answered", "claimed", "held"]

_KEY = "copilot:gate:card_decided:"
# Far past a proposal's 15 minutes; the chat itself is the lasting record.
_TTL_SECONDS = 7 * 24 * 60 * 60
# A user-role row the server wrote, not the user: a delegated or subtask
# prompt, a held call's rows. A row queued behind a running turn carries the
# queue's own metadata and is still the user's.
_SERVER_WRITTEN_KEYS = frozenset(
    {"from_session_id", "held_call", "held_calls_answered"}
)
# All or nothing: a request that names an id someone else claimed takes none.
# 0: taken by someone else, 1: claimed now, 2: already this owner's.
_CLAIM = """
local held = true
for i = 3, #ARGV do
  local owner = redis.call('HGET', KEYS[1], ARGV[i])
  if owner and owner ~= ARGV[1] then return 0 end
  if not owner then held = false end
end
if held then return 2 end
for i = 3, #ARGV do redis.call('HSET', KEYS[1], ARGV[i], ARGV[1]) end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""
# Only the request that claimed an id gives it back.
_RELEASE = """
for i = 2, #ARGV do
  if redis.call('HGET', KEYS[1], ARGV[i]) == ARGV[1] then
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
    request took the claim, and gives it back with ``release_claim`` if it
    saves nothing. "held": it was already this owner's (a retransmit of the
    answering request) or the store is down, so there is nothing to give back.
    "answered": the chat, or another request, answered first.
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
) -> Literal["claimed", "held", "taken"]:
    try:
        redis = await get_redis_async()
        result = int(
            await redis.eval(
                _CLAIM, 1, f"{_KEY}{session_id}", owner, _TTL_SECONDS, *confirmation_ids
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
    return "claimed" if result == 1 else "held"
