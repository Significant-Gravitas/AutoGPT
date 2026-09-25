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
from fastapi import APIRouter, HTTPException, Path, Request, Security, status
from pydantic import BaseModel

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
        409: {"description": "Already declined, or expired"},
    },
)
async def approve_link_purchase(
    session_id: str,
    checkout_id: CheckoutId,
    request: Request,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> LinkPurchaseApproval:
    """Record the customer's approval. Approving twice is idempotent."""
    return await _decide(session_id, checkout_id, user_id, request, approve=True)


@router.post(
    "/sessions/{session_id}/link-checkouts/{checkout_id}/decline",
    summary="Decline a Link purchase",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "No such purchase in the caller's chat"},
        409: {"description": "Already approved, or expired"},
    },
)
async def decline_link_purchase(
    session_id: str,
    checkout_id: CheckoutId,
    request: Request,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> LinkPurchaseApproval:
    return await _decide(session_id, checkout_id, user_id, request, approve=False)


async def _decide(
    session_id: str, checkout_id: str, user_id: str, request: Request, approve: bool
) -> LinkPurchaseApproval:
    try:
        view = await decide(
            checkout_id,
            user_id,
            session_id,
            approve=approve,
            user_agent=request.headers.get("user-agent"),
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
    )
