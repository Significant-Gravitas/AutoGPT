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
from collections.abc import Collection, Mapping
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
# Inside a line, neither the chat nor the gate reads past a \r, \u2028 or
# \u2029 (JavaScript's "." stops at them; Python's would not).
_DECISION = re.compile(
    r"^(?P<verdict>Approved|Not approved): (?P<subject>[^\r\u2028\u2029]+) "
    r"\(confirmation_id: (?P<id>[^\s()]+)\)\.$"
)
# Trimmed from both ends of a line, by the chat's trimLine too: control
# characters, the whitespace both languages' \s share, and a BOM. Plain
# str.strip() and String.trim() each miss some. A set, not a regex, so a long
# run inside a line costs one pass.
_EDGE_CHARS = (
    "".join(chr(c) for c in (*range(0x20), *range(0x7F, 0xA0)))
    + " \u00a0\u1680"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000\ufeff"
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


async def record_card_decisions(
    user_id: str, session_id: str, message: str, *, by: str
) -> None:
    """Called by the chat route for a message the user sent, and by nothing else.

    ``by`` is the request that recorded the approval ("owner|attempt"), so
    an approval can be withdrawn as that request's and no one else's."""
    decisions = parse_card_decisions(message)
    if not decisions:
        return
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            for decision in decisions:
                field = _field(user_id, decision.confirmation_id)
                if decision.verdict == "approved":
                    pipe.hset(_key(session_id), field, by)
                else:
                    pipe.hdel(_key(session_id), field)
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        # Unrecorded, the confirm just goes to the mode's verdict.
        logger.warning(
            f"Could not record card decisions for session {session_id}", exc_info=True
        )


# Only an approval still recorded by that request goes: the token is the
# request ("owner|attempt"), or a send ("owner"), matching any of its attempts.
_WITHDRAW_RECORDED_BY = """
for i = 1, #ARGV, 2 do
  local value = redis.call('HGET', KEYS[1], ARGV[i])
  local by = ARGV[i + 1]
  if value and (value == by or string.sub(value, 1, #by + 1) == by .. '|') then
    redis.call('HDEL', KEYS[1], ARGV[i])
  end
end
return 1
"""


async def withdraw_card_approvals_recorded_by(
    user_id: str, session_id: str, senders: Mapping[str, str]
) -> None:
    """Withdraw approvals by confirmation id, each only while the request
    named for it recorded it: ``senders`` maps an id to "owner|attempt" for
    one request, or to "owner" for any attempt of that send. A request that
    answered since keeps its approval."""
    pairs = [
        part
        for cid, sender in senders.items()
        for part in (_field(user_id, cid), sender)
    ]
    if not pairs:
        return
    try:
        redis = await get_redis_async()
        await redis.eval(_WITHDRAW_RECORDED_BY, 1, _key(session_id), *pairs)
    except Exception:
        logger.warning(
            f"Could not withdraw card approvals for session {session_id}",
            exc_info=True,
        )


def parse_card_decisions(message: str) -> list[CardDecision]:
    """Every decision line in a message, in order; other lines are skipped."""
    return [decision for line in _lines(message) if (decision := _parse_line(line))]


def strip_card_decisions(message: str, confirmation_ids: Collection[str]) -> str:
    """The message without its decision lines on these cards: what is left
    to send once those cards turn out to be answered already."""
    kept = [
        line
        for line in _lines(message)
        if not (
            (decision := _parse_line(line))
            and decision.confirmation_id in confirmation_ids
        )
    ]
    return "\n".join(kept).strip()


def card_decision_metadata(message: str) -> dict[str, Any] | None:
    """Metadata for a user message made only of card decision lines.

    The text stays as it is, since the model reads it; the metadata lets the
    chat show what was decided rather than the line with its id. A message
    with anything else in it, or a line that names no hire, create or update,
    is the user's own words and keeps no metadata.
    """
    lines = [line for line in _lines(message) if _trim(line)]
    decisions = [decision for line in lines if (decision := _parse_line(line))]
    if not decisions or len(decisions) != len(lines):
        return None
    # The card always names the change; a line in other words is shown as
    # typed, so nothing the model reads is hidden behind a vaguer note.
    if any(decision.verb is None for decision in decisions):
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


def _lines(message: str) -> list[str]:
    # Newlines alone end a line, as the chat splits it; splitlines() would
    # also break on \v, \f, \x85 and the like, and read a line apart.
    return message.split("\n")


def _trim(line: str) -> str:
    return line.strip(_EDGE_CHARS)


def _parse_line(line: str) -> CardDecision | None:
    match = _DECISION.match(_trim(line))
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
