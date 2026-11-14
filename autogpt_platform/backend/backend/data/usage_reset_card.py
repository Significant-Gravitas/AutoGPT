"""Paying for a usage reset by card: Stripe Checkout, granted once.

The purchase is stored as *pending* when its Checkout session is opened and
grants nothing. It becomes the window's next reset exactly once, when Stripe
says the session is paid: its webhook, or the buyer returning from Checkout,
whichever comes first, through one conditional update that only a pending row
passes. See ``usage_reset`` for what a reset is.
"""

import logging

import stripe
from prisma.enums import SubscriptionTier
from prisma.errors import UniqueViolationError
from prisma.models import UsageReset
from pydantic import BaseModel

from backend.copilot.usage_clock import clock_from_state
from backend.data.credit import get_stripe_customer_id
from backend.data.pro_activation import get_usage_activation_state
from backend.data.stripe_client import stripe_call
from backend.data.stripe_invoice_payments import stripe_id
from backend.data.usage_reset import ResetPurchase, announce_reset, find_reset_purchase
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# How long a Checkout session can be paid: a little over the shortest Stripe
# allows (30 minutes), and under ``rate_limit.RESET_CUTOFF_SECONDS``, so a
# session expires before the weekly window it was opened in ends.
CHECKOUT_LIFETIME_SECONDS = 31 * 60
RETURN_PATH = "/settings/billing"

_GRANT_ATTEMPTS = 5
# How many of the buyer's latest unpaid purchases are looked at for a
# checkout that is still open.
_OTHER_CHECKOUTS_CHECKED = 3


class CardGrant(BaseModel):
    purchase: ResetPurchase
    # True for the one call that granted it; False when it already was.
    granted_now: bool


class CheckoutUnsettled(Exception):
    """The purchase's Checkout session is neither open, expired nor paid in
    full: a payment may be in flight. Nothing is opened in its place."""


class CardCheckout(BaseModel):
    purchase: ResetPurchase
    # Where the buyer pays; None when the purchase is already granted.
    url: str | None


async def start_card_reset(
    user_id: str,
    purchase_key: str,
    *,
    tier: SubscriptionTier,
    price_credits: int,
    now: int,
) -> CardCheckout:
    """Open (or reopen) the Checkout session that pays for the reset by card.

    Nothing is granted here. A key whose earlier session is still open gets
    that session back; one that was paid meanwhile is granted now; one that
    expired gets a new session on the same purchase. A purchase's session is
    only ever replaced once Stripe says it expired (so it can't be paid any
    more): a session a customer could still pay never loses its purchase.

    Raises ``CheckoutUnsettled`` when the earlier session is in none of those
    states.
    """
    if price_credits <= 0:
        raise ValueError("A usage reset needs a price")
    pending = await find_reset_purchase(user_id, purchase_key)
    if pending is not None:
        # Granted (by credits, too), open or paid: nothing new is opened.
        resumed = await _resume_checkout(pending, now, price_credits)
        if resumed is not None:
            return resumed
    else:
        # Another tab's checkout for the same reset is this one's too: two
        # open checkouts would be two payments for one reset.
        shared = await _open_checkout_elsewhere(user_id, tier, price_credits)
        if shared is not None:
            return shared
    session = await _create_checkout(user_id, purchase_key, tier, price_credits, now)
    if not session.url:
        raise ValueError("Stripe did not return a checkout session URL")
    if pending is None:
        return await _store_pending(user_id, purchase_key, tier, price_credits, session)
    # The row moves to the new session only if it is still pending on the
    # expired one this request saw: of two requests replacing it at once, one
    # wins and the other is handed the winner's session, so the one session
    # the purchase points at is the only one a customer is ever shown.
    moved = await UsageReset.prisma().update_many(
        where={
            "id": pending.id,
            "status": "pending",
            "stripeCheckoutSessionId": pending.checkout_session_id,
        },
        data={
            "stripeCheckoutSessionId": session.id,
            "priceCredits": price_credits,
            "tier": tier,
        },
    )
    current = await find_reset_purchase(user_id, purchase_key)
    if current is None:
        raise CheckoutUnsettled("The purchase is gone")
    if moved:
        return CardCheckout(purchase=current, url=session.url)
    resumed = await _resume_checkout(current, now, price_credits)
    if resumed is None:
        raise CheckoutUnsettled("The purchase's session changed meanwhile")
    return resumed


async def fulfill_card_reset(
    session_id: str, now: int, *, user_id: str | None = None
) -> CardGrant | None:
    """Grant the reset a paid Checkout session bought, once.

    Called from Stripe's webhook for a completed Checkout session and when
    the buyer returns from Checkout (then with ``user_id``: only their own
    session counts). Returns the purchase when the session is one of ours
    and paid, granted now or earlier; None when no purchase has that session
    or it isn't paid. Safe to call any number of times.
    """
    row = await UsageReset.prisma().find_first(
        where={"stripeCheckoutSessionId": session_id}
    )
    if row is None or (user_id is not None and row.userId != user_id):
        return None
    if row.status == "granted":
        return CardGrant(purchase=ResetPurchase.from_db(row), granted_now=False)
    session = await stripe_call(stripe.checkout.Session.retrieve_async, session_id)
    if not _pays_for(session, row):
        return None
    state = await get_usage_activation_state(row.userId)
    if state.tier.value != SubscriptionTier(row.tier).value:
        # Paid at one plan's price after moving to another plan: that price
        # doesn't buy the new plan's weekly limit. Not granted; it stays
        # pending for support to refund. ERROR reaches Sentry.
        logger.error(
            "A paid usage reset was not granted: the plan changed between "
            "opening its checkout and paying it"
        )
        return None
    clock = clock_from_state(state, now)
    if clock is None:
        raise RuntimeError("No billing clock to grant a paid usage reset in")
    windows = clock.windows
    granted_now = bool(
        await _grant_paid(row, session, windows.week_start, windows.week_end, now)
    )
    if granted_now:
        logger.info("A usage reset paid by card was granted")
        await announce_reset(row.userId)
    current = await UsageReset.prisma().find_unique_or_raise(where={"id": row.id})
    return CardGrant(purchase=ResetPurchase.from_db(current), granted_now=granted_now)


async def _grant_paid(
    row: UsageReset,
    session: stripe.checkout.Session,
    week_start: int,
    week_end: int,
    now: int,
) -> int:
    """Turn the paid pending row into the window's next reset. Returns 1 when
    this call granted it, 0 when another delivery already had.

    One conditional update: of any number of deliveries, only the first finds
    the row pending. It is paid for, so it is always granted: if another
    purchase takes the number first, it takes the next one.
    """
    for _ in range(_GRANT_ATTEMPTS):
        taken = await UsageReset.prisma().count(
            where={"userId": row.userId, "status": "granted", "weekStart": week_start}
        )
        try:
            return await UsageReset.prisma().update_many(
                where={"id": row.id, "status": "pending"},
                data={
                    "status": "granted",
                    "weekStart": week_start,
                    "weekEnd": week_end,
                    "sequence": taken + 1,
                    "grantedAt": now,
                    "stripePaymentIntentId": stripe_id(session.get("payment_intent"))
                    or None,
                },
            )
        except UniqueViolationError:
            continue
    raise RuntimeError("Could not number a paid usage reset in its window")


async def _store_pending(
    user_id: str,
    purchase_key: str,
    tier: SubscriptionTier,
    price_credits: int,
    session: stripe.checkout.Session,
) -> CardCheckout:
    try:
        row = await UsageReset.prisma().create(
            data={
                "userId": user_id,
                "purchaseKey": purchase_key,
                "tier": tier,
                "priceCredits": price_credits,
                "paidWith": "card",
                "status": "pending",
                "stripeCheckoutSessionId": session.id,
            }
        )
    except UniqueViolationError:
        # The same key opened its session a moment ago: that one is the
        # purchase's. The session made here is never shown, so never paid.
        first = await find_reset_purchase(user_id, purchase_key)
        if first is None:
            raise
        if first.status == "granted":
            # Credits paid for this key at the same moment.
            return CardCheckout(purchase=first, url=None)
        opened = await stripe_call(
            stripe.checkout.Session.retrieve_async, first.checkout_session_id or ""
        )
        return CardCheckout(purchase=first, url=opened.get("url"))
    return CardCheckout(purchase=ResetPurchase.from_db(row), url=session.url)


async def _open_checkout_elsewhere(
    user_id: str, tier: SubscriptionTier, price_credits: int
) -> CardCheckout | None:
    """A checkout the buyer already has open under another key, for the same
    plan and price: returned instead of opening a second one."""
    rows = await UsageReset.prisma().find_many(
        where={"userId": user_id, "status": "pending", "paidWith": "card"},
        order={"createdAt": "desc"},
        take=_OTHER_CHECKOUTS_CHECKED,
    )
    for row in rows:
        same_offer = row.priceCredits == price_credits and row.tier == tier
        if not same_offer or not row.stripeCheckoutSessionId:
            continue
        session = await stripe_call(
            stripe.checkout.Session.retrieve_async, row.stripeCheckoutSessionId
        )
        if session.get("status") == "open" and session.get("url"):
            return CardCheckout(
                purchase=ResetPurchase.from_db(row), url=session.get("url")
            )
    return None


async def _resume_checkout(
    pending: ResetPurchase, now: int, price_credits: int
) -> CardCheckout | None:
    """The purchase's earlier session, if it can still be used or was paid.
    None when a new session is needed."""
    if pending.status == "granted":
        return CardCheckout(purchase=pending, url=None)
    session_id = pending.checkout_session_id or ""
    session = await stripe_call(stripe.checkout.Session.retrieve_async, session_id)
    if session.get("status") == "open" and session.get("url"):
        if pending.price_credits == price_credits:
            # Still payable: if it is paid a moment from now, it is still
            # the purchase's session and its webhook grants it.
            return CardCheckout(purchase=pending, url=session.get("url"))
        # Open at a price the buyer no longer agreed to. Closed at Stripe
        # first, so it can't be paid, and only then replaced. If Stripe
        # won't close it (it was just paid), nothing is replaced.
        try:
            await stripe_call(stripe.checkout.Session.expire_async, session_id)
        except stripe.StripeError as error:
            raise CheckoutUnsettled("An open session could not be closed") from error
        return None
    if session.get("status") == "expired":
        return None
    # Completed: paid, unless Stripe says otherwise.
    granted = await fulfill_card_reset(session_id, now)
    if granted is not None:
        return CardCheckout(purchase=granted.purchase, url=None)
    raise CheckoutUnsettled("The purchase's session is neither open nor paid")


async def _create_checkout(
    user_id: str,
    purchase_key: str,
    tier: SubscriptionTier,
    price_credits: int,
    now: int,
) -> stripe.checkout.Session:
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    metadata = {
        "usage_reset": "1",
        "user_id": user_id,
        "purchase_key": purchase_key,
        "tier": tier.value,
    }
    return await stripe_call(
        stripe.checkout.Session.create_async,
        customer=await get_stripe_customer_id(user_id),
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "AutoGPT usage reset"},
                    "unit_amount": price_credits,
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        ui_mode="hosted",
        # Cards confirm while the buyer is there (3-D Secure included): no
        # payment that settles days later, after the window is gone.
        payment_method_types=["card"],
        # The card on file is offered first, so paying is one confirmation.
        saved_payment_method_options={
            "payment_method_save": "enabled",
            "allow_redisplay_filters": ["always", "limited"],
        },
        payment_intent_data={
            "description": "AutoGPT usage reset",
            "metadata": metadata,
        },
        success_url=f"{base_url}{RETURN_PATH}"
        "?usage_reset=success&session_id={CHECKOUT_SESSION_ID}",
        cancel_url=f"{base_url}{RETURN_PATH}?usage_reset=cancel",
        expires_at=now + CHECKOUT_LIFETIME_SECONDS,
        automatic_tax={"enabled": True},
        billing_address_collection="auto",
        customer_update={"address": "auto"},
        metadata=metadata,
    )


def _pays_for(session: stripe.checkout.Session, row: UsageReset) -> bool:
    """Whether the session is this purchase's, and paid in full."""
    metadata = session.get("metadata") or {}
    return (
        session.get("mode") == "payment"
        and session.get("payment_status") == "paid"
        and metadata.get("usage_reset") == "1"
        and metadata.get("user_id") == row.userId
        and metadata.get("purchase_key") == row.purchaseKey
        and session.get("amount_subtotal") == row.priceCredits
    )
