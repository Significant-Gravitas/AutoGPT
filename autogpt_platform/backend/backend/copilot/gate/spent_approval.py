"""Spending an answered approval just before its held call runs, at most once.

The fold reads the card and hands it to the call as a :class:`FoldPass`.
When the call reaches its gate (the tool's, or the MCP handler's seam), the
gate spends that card, with the gate on or off, and lets the call through
instead of asking the user again. One approval buys one run, and a call
refused before its gate never spends it. The pass records how far the spend
got, which is what the fold then reports.
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
    "Nothing ran: the approval could not be confirmed, so the action was not "
    "started. Tell the user, and propose the call again if it is still needed."
)

# How far a fold's pass got: never reached a gate, spent there, missed (the
# card was taken or changed), lapsed, a spend that failed, or unmatched (the
# call reached its gate under another review id: its arguments changed).
Use = Literal["unused", "spent", "missed", "expired", "unconfirmed", "unmatched"]

# The fold's pass for the call it is running, by review id.
_passes: ContextVar[dict[str, "FoldPass"] | None] = ContextVar(
    "_held_passes", default=None
)


class FoldPass:
    """The card the fold read, for the one call it is running, and how far
    the spend of it got at the call's gate."""

    def __init__(self, card: PendingHumanReviewModel) -> None:
        self.card = card
        self.use: Use = "unused"


class UnconfirmedSpend(Exception):
    """A spend failed before its call ran: the card may be spent or may still
    stand, but nothing ran on it."""


class SpentThenFailed(Exception):
    """The approval was spent, then the call or its delivery failed: the call
    may have run."""


def passing() -> bool:
    """Whether a fold is running a held call here, so the gate looks for its pass."""
    return bool(_passes.get())


def take_pass(review_id: str) -> FoldPass | None:
    """The fold's pass for this call, if the fold is running it.

    ``review_id`` is the id the gate derives from the call's arguments
    (``review.review_id_for``), so it matches only when the tool hands the
    gate the arguments the call was held with. Taken once: a second identical
    call in the same run is gated as any other.
    """
    passes = _passes.get()
    return passes.pop(review_id, None) if passes else None


def unmatched() -> None:
    """The call the fold is running reached its gate under another review id
    than its pass: its arguments changed on the way. The gate rules on it as
    on any call; the fold then drops the card it read, which must not stand
    for a later identical call once this one may have run.

    A tool that gates a sub-action before its own call (another review id)
    marks the pass here too. If the call then reaches its own gate, the pass
    is taken there and spent, which overrides the mark. If it never does, the
    card is dropped, which fails closed: the approval goes unused rather than
    standing for an identical call that may then run a second time.
    """
    for fold_pass in (_passes.get() or {}).values():
        if fold_pass.use == "unused":
            fold_pass.use = "unmatched"


async def use(fold_pass: FoldPass, user_id: str) -> str | None:
    """Spend the fold's card at the gate, just before its call runs: None once
    spent, else why the call can't run. Raises :class:`UnconfirmedSpend`
    when the spend fails, and the pass says so."""
    fold_pass.use = "unconfirmed"
    refusal = await spend(user_id, fold_pass.card)
    if refusal is None:
        fold_pass.use = "spent"
        return None
    fold_pass.use = "expired" if refusal[0] == "expired" else "missed"
    return refusal[1]


def expired(row: PendingHumanReviewModel) -> bool:
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    return datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL


async def spend(
    user_id: str, row: PendingHumanReviewModel
) -> tuple[Literal["expired", "closed"], str] | None:
    """Spend the card as read: None once spent, else what the chat is told.
    Expiry is checked again here, as the gate's own lookup would, since the
    fold may have waited since it read the card. A spend that fails raises
    :class:`UnconfirmedSpend`, and nothing runs."""
    if expired(row):
        await review_store.discard(row, user_id)
        return "expired", EXPIRED
    try:
        spent = await review_store.consume(row, user_id)
    except Exception as failed:
        raise UnconfirmedSpend(row.node_exec_id) from failed
    return None if spent else ("closed", SPENT)


@contextmanager
def running(review_id: str, fold_pass: FoldPass) -> Iterator[None]:
    """Run the call with the fold's pass for its gate to spend, once. Cleared
    when the run ends, so no task the call left behind can use it."""
    # One dict, deliberately shared by reference with any task the call
    # starts: whichever reaches the gate first takes the pass.
    passes = {review_id: fold_pass}
    token = _passes.set(passes)
    try:
        yield
    finally:
        passes.clear()
        _passes.reset(token)
