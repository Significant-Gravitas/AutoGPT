"""Where a card answer is before the chat holds it.

A decision sent behind a running turn is accepted once it is queued, and
saved only when a turn takes it in. Until then it waits in the session's
pending buffer, then in the persist queue once a turn has drained it, or
inside a turn that is still running. An accepted answer in none of these,
and not in the chat, was lost: its buffer expired, or a turn dropped it.
"""

import logging
from collections.abc import Iterable

from backend.copilot.gate.card_approval import parse_card_decisions
from backend.copilot.pending_message_helpers import is_turn_in_flight
from backend.copilot.pending_messages import (
    peek_pending_for_persist,
    peek_pending_messages,
)

logger = logging.getLogger(__name__)


async def answers_on_their_way(
    session_id: str, confirmation_ids: Iterable[str]
) -> set[str]:
    """The ids among these whose answer is still on its way to the chat.

    The queues are read before the turn, so a message moving from one to
    the other is seen in one of them. All the ids when this cannot be told:
    an answer still on its way must never be taken for lost.
    """
    ids = set(confirmation_ids)
    try:
        queued = [
            *await peek_pending_messages(session_id),
            *await peek_pending_for_persist(session_id),
        ]
        if await is_turn_in_flight(session_id):
            return ids
    except Exception:
        logger.warning(
            f"Could not read the queue of session {session_id} for card answers",
            exc_info=True,
        )
        return ids
    return ids & {
        d.confirmation_id for m in queued for d in parse_card_decisions(m.content)
    }
