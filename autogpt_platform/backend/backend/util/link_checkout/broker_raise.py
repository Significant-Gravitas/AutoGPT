"""Raising an unpaid checkout's total when the final price came out higher
(Link's incremental authorization), in the broker beside ``broker_checkout``.

The customer approves every new total afresh, so a raise never pays anything:
it changes what the next approval is for, and the deadline restarts with it.
https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online
"""

import time
from pathlib import Path

from backend.util.link_checkout import broker_link
from backend.util.link_checkout.broker_checkout import (
    CHECKOUT_TTL_SECONDS,
    first_spend,
    require_payable,
)
from backend.util.link_checkout.broker_protocol import (
    CheckoutView,
    RaiseCheckout,
    session_key,
)
from backend.util.link_checkout.checkout_record import read_intent, replace_intent
from backend.util.link_checkout.checkout_view import view
from backend.util.link_checkout.link import link_action_url, validate_spend
from backend.util.link_checkout.models import CheckoutIntent
from backend.util.link_checkout.refusals import (
    RAISE_CLOSED,
    RAISE_NOT_HIGHER,
    CheckoutRefused,
)
from backend.util.link_checkout.runtime import browser_operation
from backend.util.link_checkout.status import TERMINAL_STATUSES


async def raise_checkout(request: RaiseCheckout) -> CheckoutView:
    """Raise an unpaid checkout's total when the final price came out higher.
    The customer approves the new total afresh: in Link (``_raise_in_link``),
    or in the chat while nothing exists in Link yet. The deadline restarts
    with it."""
    key = session_key(request)
    async with browser_operation(key) as directory:
        intent = read_intent(directory, request.checkout_id, request.user_id, key)
        require_payable(intent.plan.test_mode)
        if request.amount <= intent.plan.amount:
            raise CheckoutRefused(RAISE_NOT_HIGHER)
        raised = intent.model_copy(
            update={
                "plan": intent.plan.model_copy(update={"amount": request.amount}),
                "revision": intent.revision + 1,
                "expires_at": time.time() + CHECKOUT_TTL_SECONDS,
                "approval_mode": (
                    "link" if intent.spend_request_id else request.approval_mode
                ),
            }
        )
        if raised.spend_request_id is not None:
            return await _raise_in_link(directory, intent, raised, request)
        replace_intent(directory, raised)
        if raised.approval_mode == "in_app":
            return view(raised)
        spend = await first_spend(directory, raised, request.access_token, None)
        return view(raised, spend)


async def _raise_in_link(
    directory: Path,
    intent: CheckoutIntent,
    raised: CheckoutIntent,
    request: RaiseCheckout,
) -> CheckoutView:
    """Link's guide: only an approved request is raised (incremental
    authorization). One still waiting for approval, or one Link won't raise,
    is canceled and asked for again at the new total; one that is over can't
    change."""
    token = request.access_token
    current = await broker_link.status(intent, token)
    if current.status in TERMINAL_STATUSES:
        raise CheckoutRefused(RAISE_CLOSED)
    spend = (
        await broker_link.raise_total(raised, token)
        if current.status == "approved"
        else None
    )
    if spend is None:
        await broker_link.cancel(intent, token)
        raised.spend_request_id = None
        raised.approval_url = ""
        return view(raised, await first_spend(directory, raised, token, None))
    raised.approval_url = link_action_url(spend.approval_url) or raised.approval_url
    validate_spend(raised, spend)
    replace_intent(directory, raised)
    return view(raised, spend)
