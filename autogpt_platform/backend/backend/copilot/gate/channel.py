"""A held call's card as a linked chat channel shows it, and what a click answers.

The channel says what the web card says — headline, reason, a held read's
passage — and offers the web card's choices on the same row, with every rule
held to this chat: a click in a shared channel must not change how the
owner's other chats ask.
"""

import logging
from typing import Any, Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.constants import AUTOPILOT_NAME
from backend.data.db_accessors import review_db

from . import chat_rules, held
from .review import payload_headline

logger = logging.getLogger(__name__)

Choice = Literal["approve", "approve_chat", "judge", "reject"]
Outcome = Literal["answered", "answered_elsewhere", "expired", "failed"]

# What each choice answers: approved, and the rule it sets in this chat.
_ANSWERS: dict[Choice, tuple[bool, chat_rules.ChatRule | None]] = {
    "approve": (True, None),
    "approve_chat": (True, "allow"),
    "judge": (True, "judge"),
    "reject": (False, None),
}
_RECEIPTS: dict[Choice, str] = {
    "approve": "Approved",
    "approve_chat": "Approved for this chat",
    "judge": f"Approved, and {AUTOPILOT_NAME} judges it in this chat",
    "reject": "Rejected",
}
_PASSAGE_CHARS = 500


class CardOption(BaseModel):
    choice: Choice
    label: str
    # What replaces the buttons once this option answered the row.
    receipt: str


class CardView(BaseModel):
    text: str
    options: list[CardOption]


def card_for(row: PendingHumanReviewModel) -> CardView:
    payload = row.payload if isinstance(row.payload, dict) else {}
    headline = payload_headline(payload)
    lines = [f"⏸️ **{headline}**"]
    reason = _reason_line(payload)
    if reason:
        lines.append(reason)
    passage = str(payload.get("passage") or "")[:_PASSAGE_CHARS]
    if _is_held_read(payload) and passage:
        lines.append("\n".join(f"> {line}" for line in passage.splitlines()))
    return CardView(
        text="\n".join(lines),
        options=[
            CardOption(
                choice=choice,
                label=label,
                receipt=f"{'✅' if _ANSWERS[choice][0] else '✖️'} {headline} · {said}",
            )
            for choice, label, said in _offered(payload)
        ],
    )


async def answer(
    user_id: str, session_id: str, review_id: str, choice: Choice
) -> Outcome:
    """Answer the row as the web card's approve endpoint does, waking the turn
    that runs it."""
    approved, rule = _ANSWERS[choice]
    # Up to the commit a failure leaves the card answerable; after it, the
    # answer stands and only its rule can be lost.
    try:
        rows = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
        row = rows.get(review_id)
        if row is None or row.session_id != session_id:
            return "expired"
        if row.status != ReviewStatus.WAITING:
            return "answered_elsewhere"
        # Read before the answer lands, as the endpoint does: a turn claims held calls.
        keys = await held.subject_keys(session_id, [review_id]) if rule else {}
        status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        answered = await review_db().process_all_reviews_for_execution(
            user_id=user_id, review_decisions={review_id: (status, None, None)}
        )
    except ValueError:
        # Answered the other way a moment ago, on the web or by another click.
        return "answered_elsewhere"
    except Exception:
        logger.warning(f"Channel answer to {review_id} failed", exc_info=True)
        return "failed"
    if keys:
        try:
            await chat_rules.set_answer_rules(
                session_id,
                user_id,
                answered,
                {review_id: rule},
                keys,
                {review_id: "chat"},
            )
        except Exception:
            logger.warning(f"Rule from {review_id} not saved", exc_info=True)
    await held.wake(user_id, session_id, answered.values())
    return "answered"


def _offered(payload: dict[str, Any]) -> list[tuple[Choice, str, str]]:
    """The web card's choices on this row: (choice, label, receipt)."""
    if _is_held_read(payload):
        reader = str(payload.get("reader") or AUTOPILOT_NAME)
        return [
            ("approve", f"Release to {reader}", "Released"),
            ("reject", "Keep it out", "Kept out"),
        ]
    rules = payload.get("chat_rules_allowed") or []
    offered: list[tuple[Choice, str]] = [("approve", "Approve")]
    if "allow" in rules:
        offered.append(("approve_chat", "Approve for this chat"))
    if "judge" in rules:
        offered.append(("judge", f"Let {AUTOPILOT_NAME} judge in this chat"))
    offered.append(("reject", "Reject"))
    return [(choice, label, _RECEIPTS[choice]) for choice, label in offered]


def _reason_line(payload: dict[str, Any]) -> str | None:
    """The web card's reason line, in its words."""
    reason = str(payload.get("reason") or "")
    kind = payload.get("reason_kind")
    reader = str(payload.get("reader") or AUTOPILOT_NAME)
    if _is_held_read(payload) and payload.get("judged") is False:
        return (
            f"{AUTOPILOT_NAME} could not check this, so he asks. "
            f"{reader} hasn't seen it."
        )
    if _is_held_read(payload):
        return (
            f"It contains instructions aimed at {reader}, so it was held back. "
            f"{reader} hasn't seen it."
        )
    if kind == "mode":
        # A linked chat is always in Auto, whose one reason the web says once per queue.
        return (
            f"{AUTOPILOT_NAME} asks before anything that reaches outside the platform."
        )
    if not reason:
        return None
    if kind == "supervisor":
        return f"Not sure this is safe: {reason}"
    if kind in ("subject", "rule"):
        return reason
    if kind == "spend":
        # The web card draws these numbers as a block; here they are the sentence.
        return f"{reason[:1].upper()}{reason[1:]}."
    return None


def _is_held_read(payload: dict[str, Any]) -> bool:
    return payload.get("reason_kind") == "content"
