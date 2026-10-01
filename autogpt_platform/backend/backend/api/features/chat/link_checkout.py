"""Approve or decline a Link purchase from its card in the chat.

Served under ``/api/chat``. The purchase on the card is the one the checkout
tool recorded (``backend.util.link_checkout.approval``); these routes only
record the signed-in customer's decision on it. No copilot tool can reach
them, so an approval exists only when the customer clicked. The next
``browser_complete_link_payment`` call reads it and asks Link for a spend
request that is already approved.
"""

from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Body, HTTPException, Path, Request, Security, status
from pydantic import BaseModel, Field

from backend.util.link_checkout.approval import (
    ApprovalConflict,
    ApprovalState,
    ApprovalView,
    decide,
    read_approval,
)
from backend.util.link_checkout.models import CHECKOUT_ID

router = APIRouter(tags=["chat"])

CheckoutId = Annotated[str, Path(pattern=CHECKOUT_ID)]
_NOT_FOUND = "Purchase not found"


class LinkPurchaseApproval(BaseModel):
    """A purchase waiting for, or given, the customer's decision in the chat.

    ``merchant_url`` is the page the card will be used on and ``context`` the
    agent's description of the purchase. With an in-chat approval Link shows
    the customer nothing, so this is everything they approve.
    """

    checkout_id: str
    state: ApprovalState
    merchant_name: str
    merchant_url: str
    context: str
    amount: int
    currency: str
    test_mode: bool
    expires_at: float
    # Bumped each time the agent raised the total; a decision names the
    # revision it is for. A raise also carries the total it replaced and why.
    revision: int = 0
    previous_amount: int | None = None
    reason: str = ""


class LinkPurchaseDecision(BaseModel):
    """The revision of the purchase the customer decided on, as their card
    showed it. Omitted by older clients, which only ever saw the first."""

    revision: int = Field(default=0, ge=0)


@router.get(
    "/sessions/{session_id}/link-checkouts/{checkout_id}",
    summary="Get a Link purchase approval",
    dependencies=[Security(auth.requires_user)],
    responses={404: {"description": "No such purchase in the caller's chat"}},
)
async def get_link_purchase_approval(
    session_id: str,
    checkout_id: CheckoutId,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> LinkPurchaseApproval:
    view = await read_approval(checkout_id)
    if (
        view is None
        or view.pending.user_id != user_id
        or view.pending.session_id != session_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _approval(view)


@router.post(
    "/sessions/{session_id}/link-checkouts/{checkout_id}/approve",
    summary="Approve a Link purchase",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "No such purchase in the caller's chat"},
        409: {"description": "Already declined, expired, or since raised"},
    },
)
async def approve_link_purchase(
    session_id: str,
    checkout_id: CheckoutId,
    request: Request,
    user_id: Annotated[str, Security(auth.get_user_id)],
    decision: Annotated[LinkPurchaseDecision | None, Body()] = None,
) -> LinkPurchaseApproval:
    """Record the customer's approval. Approving twice is idempotent."""
    return await _decide(
        session_id, checkout_id, user_id, request, decision, approve=True
    )


@router.post(
    "/sessions/{session_id}/link-checkouts/{checkout_id}/decline",
    summary="Decline a Link purchase",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "No such purchase in the caller's chat"},
        409: {"description": "Already approved, expired, or since raised"},
    },
)
async def decline_link_purchase(
    session_id: str,
    checkout_id: CheckoutId,
    request: Request,
    user_id: Annotated[str, Security(auth.get_user_id)],
    decision: Annotated[LinkPurchaseDecision | None, Body()] = None,
) -> LinkPurchaseApproval:
    return await _decide(
        session_id, checkout_id, user_id, request, decision, approve=False
    )


async def _decide(
    session_id: str,
    checkout_id: str,
    user_id: str,
    request: Request,
    decision: LinkPurchaseDecision | None,
    *,
    approve: bool,
) -> LinkPurchaseApproval:
    try:
        view = await decide(
            checkout_id,
            user_id,
            session_id,
            approve=approve,
            user_agent=request.headers.get("user-agent"),
            revision=decision.revision if decision else 0,
        )
    except ApprovalConflict as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    if view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return _approval(view)


def _approval(view: ApprovalView) -> LinkPurchaseApproval:
    pending = view.pending
    return LinkPurchaseApproval(
        checkout_id=pending.checkout_id,
        state=view.state,
        merchant_name=pending.merchant_name,
        merchant_url=pending.merchant_url,
        context=pending.context,
        amount=pending.amount,
        currency=pending.currency,
        test_mode=pending.test_mode,
        expires_at=pending.expires_at,
        revision=pending.revision,
        previous_amount=pending.previous_amount,
        reason=pending.reason,
    )
