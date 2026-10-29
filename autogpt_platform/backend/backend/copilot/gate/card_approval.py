"""A proposal card's Approve is the user's answer to the confirm it asks about.

A hire, raise, update or Soul-edit card sends the decision as a chat line naming
the proposal's one-time id (``decisionLine`` in ``ToolChain/ExpertCards.tsx``).
The chat route records that line from the user's own request, and the gate reads
the record: a model can put text into a chat, but never through that request.
"""

import logging
import re
from typing import Any

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_CONFIRMS = frozenset({"confirm_expert_change", "confirm_expert_soul_update"})
_KEY = "copilot:gate:card_approval:"
# A proposal lives 15 minutes, so an approval of it never needs longer.
_TTL_SECONDS = 15 * 60
_DECISION = re.compile(
    r"^(?P<verdict>Approved|Not approved): .+ \(confirmation_id: (?P<id>[^\s()]+)\)\.$"
)


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
                    pipe.hdel(_key(session_id), _field(user_id, confirmation_id))
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        # Unrecorded, the confirm just goes to the mode's verdict.
        logger.warning(
            f"Could not record card decisions for session {session_id}", exc_info=True
        )


async def approved_on_card(
    tool_name: str, args: dict[str, Any], user_id: str, session_id: str
) -> bool:
    """True once, when this user approved exactly this id on its card in this chat."""
    confirmation_id = args.get("confirmation_id")
    if tool_name not in _CONFIRMS or not isinstance(confirmation_id, str):
        return False
    try:
        redis = await get_redis_async()
        # HDEL's count is the answer, so two confirms of one approval cannot both pass.
        field = _field(user_id, confirmation_id)
        return await redis.hdel(_key(session_id), field) == 1
    except Exception:
        logger.warning(
            f"Could not read card approvals for session {session_id}", exc_info=True
        )
        return False


def _key(session_id: str) -> str:
    return f"{_KEY}{session_id}"


def _field(user_id: str, confirmation_id: str) -> str:
    return f"{user_id}:{confirmation_id}"
