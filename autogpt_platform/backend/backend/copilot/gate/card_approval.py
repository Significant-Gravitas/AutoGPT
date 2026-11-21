"""A proposal card's answer is the user's one answer to the change it shows.

A hire, raise, update or Soul-edit card sends the decision as a chat line naming
the proposal's one-time id (``decisionLine`` in ``ToolChain/ExpertCards.tsx``).
The chat route records that line from the user's own request, and the confirm
tools apply a change only on its recorded Approve, in every mode: a model can
put text into a chat, but never through that request.
"""

import logging
import re
from typing import Literal

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY = "copilot:gate:card_approval:"
# A proposal lives 15 minutes, and a later answer to it outlives it.
_TTL_SECONDS = 15 * 60
_DECISION = re.compile(
    r"^(?P<verdict>Approved|Not approved): .+ \(confirmation_id: (?P<id>[^\s()]+)\)\.$"
)

CardAnswer = Literal["approved", "declined"]


async def record_card_decisions(user_id: str, session_id: str, message: str) -> None:
    """Called by the chat route for a message the user sent, and by nothing else."""
    decisions = [
        (match["verdict"] == "Approved", match["id"])
        for line in message.splitlines()
        if (match := _DECISION.match(line.strip()))
    ]
    if not decisions:
        return
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            for approved, confirmation_id in decisions:
                if approved:
                    pipe.hset(_key(session_id), _field(user_id, confirmation_id), "1")
                else:
                    # Final: it also withdraws an approval no confirm has used yet.
                    pipe.hset(
                        _key(session_id), _declined(user_id, confirmation_id), "1"
                    )
                    pipe.hdel(_key(session_id), _field(user_id, confirmation_id))
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        # Unrecorded, the confirm refuses until the user answers again.
        logger.warning(
            f"Could not record card decisions for session {session_id}", exc_info=True
        )


async def take_card_answer(
    user_id: str, session_id: str, confirmation_id: str, *declined_with: str
) -> CardAnswer | None:
    """This user's answer to exactly this id's card in this chat, or None.

    Taking an Approve uses it up; a Decline stays, so it binds every later
    confirm. A decline of any id in ``declined_with`` (a hire's other choice)
    declines this one too.
    """
    redis = await get_redis_async()
    key = _key(session_id)
    ids = (confirmation_id, *declined_with)
    declines = await redis.hmget(key, [_declined(user_id, cid) for cid in ids])
    if any(decline is not None for decline in declines):
        return "declined"
    # HDEL's count is the answer, so two confirms of one approval cannot both pass.
    if await redis.hdel(key, _field(user_id, confirmation_id)) == 1:
        return "approved"
    return None


def _key(session_id: str) -> str:
    return f"{_KEY}{session_id}"


def _field(user_id: str, confirmation_id: str) -> str:
    return f"{user_id}:{confirmation_id}"


def _declined(user_id: str, confirmation_id: str) -> str:
    return f"declined:{user_id}:{confirmation_id}"
