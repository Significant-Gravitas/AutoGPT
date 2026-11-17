"""Spending an answered approval before its held call runs, at most once.

The fold reads the card, then spends it just before the call runs, with the
gate on or off, so one approval buys one run. The gate then lets that one
call through on the card the fold spent (:func:`take_spent`) instead of
asking the user again.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel

from . import review as review_store

SPENT = (
    "Nothing ran: this approval was already used by an identical call, or "
    "withdrawn. Do not retry it; tell the user."
)
EXPIRED = (
    "Nothing ran: the approval expired an hour after it was given. "
    "Propose the call again if it is still needed."
)
UNCONFIRMED = (
    "Nothing ran: the approval could not be confirmed before the action ran. "
    "Tell the user, and propose the call again if it is still needed."
)

# The card the fold spent for the call it is running, by review id.
_spent: ContextVar[dict[str, PendingHumanReviewModel] | None] = ContextVar(
    "_held_spent", default=None
)


class UnconfirmedSpend(Exception):
    """A spend failed before its call ran: the card may be spent or may still
    stand, but nothing ran on it."""


def take_spent(review_id: str) -> PendingHumanReviewModel | None:
    """The card the fold spent for this call, if the fold is running it.

    ``review_id`` is the id the gate derives from the call's arguments
    (``review.review_id_for``), so it matches only when the tool hands the
    gate the arguments the call was held with. Taken once: a second identical
    call in the same run is gated as any other.
    """
    spent = _spent.get()
    return spent.pop(review_id, None) if spent else None


def expired(row: PendingHumanReviewModel) -> bool:
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    return datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL


async def spend(
    user_id: str, row: PendingHumanReviewModel
) -> tuple[Literal["expired", "closed"], str] | None:
    """Spend the approval before its call runs: None once spent, else what the
    chat is told. Expiry is checked again here, as the gate's own lookup
    would, since the fold may have waited since it read the card. A spend
    that fails raises :class:`UnconfirmedSpend`, and nothing runs."""
    if expired(row):
        await review_store.discard(row, user_id)
        return "expired", EXPIRED
    try:
        spent = await review_store.consume(row, user_id)
    except Exception as failed:
        raise UnconfirmedSpend(row.node_exec_id) from failed
    return None if spent else ("closed", SPENT)


@contextmanager
def running(review_id: str, row: PendingHumanReviewModel) -> Iterator[None]:
    """Run the call on the approval the fold just spent: the gate's seam, the
    tool's or the MCP handler's, lets this one call through, once. Cleared
    when the run ends, so no task the call left behind can use it."""
    passes = {review_id: row}
    token = _spent.set(passes)
    try:
        yield
    finally:
        passes.clear()
        _spent.reset(token)
