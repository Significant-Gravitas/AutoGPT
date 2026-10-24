"""A proposal card is answered once.

The card stops asking once the chat holds its decision, but a tab opened
before that, or two tabs at once, can still send a second one. The chat route
asks here before it records or saves a decision: an id the chat already
answered is refused, and so is one another request claimed a moment ago.
"""

import logging
from collections.abc import Iterable

from backend.copilot.gate.card_approval import CARD_DECISION_KIND, parse_card_decisions
from backend.copilot.model import ChatMessage, get_chat_session
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY = "copilot:gate:card_decided:"
# Far past a proposal's 15 minutes; the chat itself is the lasting record.
_TTL_SECONDS = 7 * 24 * 60 * 60
# All or nothing: a request that names an id someone else claimed takes none.
_CLAIM = """
for i = 3, #ARGV do
  local owner = redis.call('HGET', KEYS[1], ARGV[i])
  if owner and owner ~= ARGV[1] then return 0 end
end
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


async def is_already_answered(
    user_id: str, session_id: str, confirmation_ids: list[str], owner: str
) -> bool:
    """Whether any of these cards was answered by anything but this request.

    ``owner`` is the request's scoped client message id, so a retransmit of
    the request that answered is not a second answer. Fails open: an outage
    must not stop the user answering.
    """
    try:
        session = await get_chat_session(session_id, user_id)
    except Exception:
        logger.warning(
            f"Could not read session {session_id} for card answers", exc_info=True
        )
        session = None
    if session and decided_ids(session.messages, owner) & set(confirmation_ids):
        return True
    return not await _claim(session_id, confirmation_ids, owner)


def decided_ids(
    messages: Iterable[ChatMessage], skip_message_id: str | None
) -> set[str]:
    """The confirmation ids the user's rows in this chat already answered."""
    decided: set[str] = set()
    for message in messages:
        if message.role != "user" or message.id == skip_message_id:
            continue
        metadata = message.metadata
        if metadata:
            # A row the server wrote for something else, such as a subtask's
            # prompt, says nothing about the user's answers.
            if metadata.get("kind") == CARD_DECISION_KIND:
                decided |= {
                    d["confirmation_id"]
                    for d in metadata.get("decisions", [])
                    if isinstance(d, dict) and isinstance(d.get("confirmation_id"), str)
                }
            continue
        # Typed, or saved before rows were tagged: the line itself is the record.
        decided |= {
            d.confirmation_id for d in parse_card_decisions(message.content or "")
        }
    return decided


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


async def _claim(session_id: str, confirmation_ids: list[str], owner: str) -> bool:
    try:
        redis = await get_redis_async()
        claimed = await redis.eval(
            _CLAIM, 1, f"{_KEY}{session_id}", owner, _TTL_SECONDS, *confirmation_ids
        )
        return int(claimed) == 1
    except Exception:
        logger.warning(
            f"Could not claim card answers for session {session_id}", exc_info=True
        )
        return True
