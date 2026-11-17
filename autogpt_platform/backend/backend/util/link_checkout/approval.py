"""In-chat purchase approvals, shared by the REST API and the copilot executor.

When Link lets AutoGPT collect the approval itself (see ``policy``), the request
tool opens a record here holding the purchase exactly as the chat shows it.
The customer's Approve or Decline click reaches the REST API, which records the
decision and its evidence; no copilot tool can write a decision. The complete
tool then reads the decision before it asks Link for a spend request that is
already approved.

A decision is written once (``SET NX``): the first click wins, a second click
of the same kind is idempotent, and nothing can overwrite it.
"""

import time
from typing import Literal

from pydantic import BaseModel, Field

from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor
from backend.util.link_checkout.models import CHECKOUT_ID, ApprovalDetails

# Outlives the checkout's ten-minute deadline, so a late click is refused with
# "expired" rather than "not found".
_TTL_SECONDS = 20 * 60


class PendingApproval(BaseModel):
    checkout_id: str = Field(pattern=CHECKOUT_ID)
    user_id: str
    session_id: str
    merchant_name: str
    # Where the card will be used, and the agent's account of what is bought:
    # the card shows both, from this record rather than the tool's output.
    merchant_url: str
    context: str
    amount: int
    currency: str
    test_mode: bool
    expires_at: float


class Decision(BaseModel):
    approved: bool
    decided_at: int
    user_agent: str | None = None


ApprovalState = Literal["awaiting", "approved", "declined", "expired"]


class ApprovalView(BaseModel):
    pending: PendingApproval
    state: ApprovalState
    decision: Decision | None = None


class ApprovalConflict(Exception):
    """The purchase was already decided the other way, or has expired."""


async def open_approval(pending: PendingApproval) -> None:
    redis = await get_redis_async()
    stored = await redis.set(
        _record_key(pending.checkout_id),
        JSONCryptor().encrypt(pending.model_dump(mode="json")),
        nx=True,
        ex=_TTL_SECONDS,
    )
    if not stored:
        raise RuntimeError("This checkout already has an approval record")


async def read_approval(checkout_id: str) -> ApprovalView | None:
    redis = await get_redis_async()
    record = await redis.get(_record_key(checkout_id))
    if record is None:
        return None
    pending = PendingApproval.model_validate(JSONCryptor().decrypt(_text(record)))
    raw_decision = await redis.get(_decision_key(checkout_id))
    decision = (
        Decision.model_validate(JSONCryptor().decrypt(_text(raw_decision)))
        if raw_decision is not None
        else None
    )
    return ApprovalView(
        pending=pending, state=_state(pending, decision), decision=decision
    )


async def decide(
    checkout_id: str,
    user_id: str,
    session_id: str,
    *,
    approve: bool,
    user_agent: str | None,
) -> ApprovalView | None:
    """Record the customer's decision. None when no such purchase is theirs."""
    view = await read_approval(checkout_id)
    if (
        view is None
        or view.pending.user_id != user_id
        or view.pending.session_id != session_id
    ):
        return None
    if view.decision is None:
        if view.state == "expired":
            raise ApprovalConflict("This purchase request has expired")
        decision = Decision(
            approved=approve,
            decided_at=int(time.time()),
            user_agent=(user_agent or None) and user_agent[:512],
        )
        redis = await get_redis_async()
        await redis.set(
            _decision_key(checkout_id),
            JSONCryptor().encrypt(decision.model_dump(mode="json")),
            nx=True,
            ex=_TTL_SECONDS,
        )
        view = await read_approval(checkout_id)
        if view is None or view.decision is None:
            raise RuntimeError("The approval could not be recorded")
    if view.decision.approved != approve:
        raise ApprovalConflict(
            "This purchase was already "
            + ("approved" if view.decision.approved else "declined")
        )
    return view


def approval_details(view: ApprovalView) -> ApprovalDetails:
    """Link's ``approval_details`` for an approved purchase."""
    if view.decision is None or not view.decision.approved:
        raise ValueError("The customer has not approved this purchase")
    return ApprovalDetails(
        approved_at=view.decision.decided_at,
        external_user_id=view.pending.user_id,
        external_session_id=view.pending.session_id,
        agent_log_id=view.pending.checkout_id,
        user_agent=view.decision.user_agent,
    )


def _state(pending: PendingApproval, decision: Decision | None) -> ApprovalState:
    if decision is not None:
        return "approved" if decision.approved else "declined"
    return "expired" if pending.expires_at <= time.time() else "awaiting"


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


# The hash tag keeps both keys of one checkout on the same cluster slot.
def _record_key(checkout_id: str) -> str:
    return f"copilot:link_approval:{{{checkout_id}}}:record"


def _decision_key(checkout_id: str) -> str:
    return f"copilot:link_approval:{{{checkout_id}}}:decision"
