"""A proposal card's Approve is the user's answer to the confirm it asks about.

A hire, raise, update or Soul-edit card sends the decision as a chat line naming
the proposal's one-time id (``decisionLine`` in ``copilot/cardDecisions.ts``).
The chat route records that line from the user's own request, and the gate reads
the record: a model can put text into a chat, but never through that request.
The route also tags a message made only of such lines, so the chat shows it as
a note of what was decided instead of the line itself.
"""

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

CARD_DECISION_KIND = "card_decision"

_CONFIRMS = frozenset({"confirm_expert_change", "confirm_expert_soul_update"})
_KEY = "copilot:gate:card_approval:"
# A proposal lives 15 minutes, so an approval of it never needs longer.
_TTL_SECONDS = 15 * 60
# Pinned with decisionLine by the shared cardDecisionLines.json fixture.
_DECISION = re.compile(
    r"^(?P<verdict>Approved|Not approved): (?P<subject>.+) "
    r"\(confirmation_id: (?P<id>[^\s()]+)\)\.$"
)
_APPROVED_SUBJECT = re.compile(r"^(?P<verb>hire|create|update) (?P<name>.+)$")
_DECLINED_SUBJECT = re.compile(
    r"^do not (?P<verb>hire|create|update) (?P<name>.+), discard that proposal$"
)


class CardDecision(BaseModel):
    verdict: Literal["approved", "declined"]
    confirmation_id: str
    # Absent when the line names its proposal in some other wording.
    verb: Literal["hire", "create", "update"] | None = None
    name: str | None = None


async def record_card_decisions(user_id: str, session_id: str, message: str) -> None:
    """Called by the chat route for a message the user sent, and by nothing else."""
    decisions = parse_card_decisions(message)
    if not decisions:
        return
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            for decision in decisions:
                field = _field(user_id, decision.confirmation_id)
                if decision.verdict == "approved":
                    pipe.hset(_key(session_id), field, "1")
                else:
                    pipe.hdel(_key(session_id), field)
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        # Unrecorded, the confirm just goes to the mode's verdict.
        logger.warning(
            f"Could not record card decisions for session {session_id}", exc_info=True
        )


def parse_card_decisions(message: str) -> list[CardDecision]:
    """Every decision line in a message, in order; other lines are skipped."""
    return [
        decision for line in message.splitlines() if (decision := _parse_line(line))
    ]


def card_decision_metadata(message: str) -> dict[str, Any] | None:
    """Metadata for a user message made only of card decision lines.

    The text stays as it is, since the model reads it; the metadata lets the
    chat show what was decided rather than the line with its id. A message
    with anything else in it is the user's own words and keeps no metadata.
    """
    lines = [line for line in message.splitlines() if line.strip()]
    decisions = [decision for line in lines if (decision := _parse_line(line))]
    if not decisions or len(decisions) != len(lines):
        return None
    return {
        "kind": CARD_DECISION_KIND,
        "decisions": [decision.model_dump() for decision in decisions],
    }


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


def _parse_line(line: str) -> CardDecision | None:
    match = _DECISION.match(line.strip())
    if not match:
        return None
    approved = match["verdict"] == "Approved"
    subject = (_APPROVED_SUBJECT if approved else _DECLINED_SUBJECT).match(
        match["subject"]
    )
    return CardDecision.model_validate(
        {
            "verdict": "approved" if approved else "declined",
            "confirmation_id": match["id"],
            "verb": subject["verb"] if subject else None,
            "name": subject["name"] if subject else None,
        }
    )
