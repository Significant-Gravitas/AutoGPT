"""Start a full billing period from a late payment.

A renewal or a trial's first charge that fails cuts access at once while
Stripe keeps retrying the invoice (see ``subscription_payment_failure``).
When the invoice is paid later, by a retry, the payment link, a bank that
confirms days later or the wallet, the customer must get a full period from
that payment and renew on its date, not pay for the days they had no access.

Stripe resets a classic subscription's anchor only to "now", and the reset
invoices a whole new period at once. With prorations off that would charge
the customer twice; with them on it would charge for the unavailable days.
So the late invoice is first credited to the customer's Stripe balance with
a credit note, and then the anchor is reset with prorations off: the reset
invoice, a full period from now, is paid from that balance. The customer pays
once, the invoice that is paid states the new period's dates, and every later
renewal follows it. Stripe works out month ends and February 29 from the new
anchor. ``error_if_incomplete`` keeps a subscription whose reset invoice the
balance did not cover (a price change in between) on its old period, instead
of putting it past due again.

Both steps can be repeated safely. A ``SubscriptionBillingRestart`` row per
invoice records how far the restart got, the credit note is also looked up
by its metadata before another is created, and the reset happens only while
the late invoice is still the subscription's latest. So a repeated or
delayed event resumes or finds the work done, even after Stripe's 24-hour
idempotency window. A lock per subscription keeps concurrent deliveries of
one payment (``invoice.paid``, ``invoice.payment_succeeded``,
``invoice_payment.paid``) from running the steps side by side.

The new period starts when the payment's event is handled, normally seconds
after the payment. Stripe cannot anchor a classic subscription to a past
time, so a notification held up for hours gives the customer those hours
free; it never charges them for time without access.
"""

import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import stripe
from prisma import Prisma
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionBillingRestart, User
from pydantic import BaseModel

from backend.data.credit import (
    _invoice_subscription_id,
    invalidate_billing_lookups,
    replaced_by_newer_plan,
)
from backend.data.db import transaction
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.stripe_invoice_payments import stripe_id
from backend.data.subscription_billing_restart_invoice import (
    bills_only_the_plan,
    credit_covers_new_period,
    current_period_start,
    line_period_start,
)

logger = logging.getLogger(__name__)

# Stripe makes the first automatic attempt about an hour after it creates a
# renewal invoice at the period start, so an on-time payment lands just after
# an hour. One paid later than this, or after a failed attempt, was late.
LATE_PAYMENT_AFTER = timedelta(hours=3)
# The invoices that open a billing period. An upgrade's proration invoice
# (``subscription_update``) or a one-off charge never moves the renewal date.
PERIOD_OPENING_REASONS = ("subscription_cycle", "subscription_create")
RESTARTED_FROM_INVOICE_KEY = "billing_restarted_from_invoice"


class RestartOutcome(StrEnum):
    RESTARTED = "restarted"
    NOT_LATE = "not_late"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_MANUAL_FIX = "needs_manual_fix"


class LatePayment(BaseModel):
    period_start: int
    paid_at: int

    @property
    def seconds_late(self) -> int:
        return self.paid_at - self.period_start


def late_payment(invoice: dict, period_start: int | None = None) -> LatePayment | None:
    """Whether a paid invoice that opened a billing period was paid late.

    This is the rule for when the period restarts from the payment: paid
    more than ``LATE_PAYMENT_AFTER`` after the period began, or after a
    failed attempt. ``period_start`` defaults to the invoice's own lines.
    """
    if invoice.get("status") != "paid":
        return None
    if invoice.get("billing_reason") not in PERIOD_OPENING_REASONS:
        return None
    paid_at = (invoice.get("status_transitions") or {}).get("paid_at")
    start = period_start if period_start is not None else line_period_start(invoice)
    if paid_at is None or start is None:
        return None
    late = LatePayment(period_start=start, paid_at=paid_at)
    if late.seconds_late > LATE_PAYMENT_AFTER.total_seconds():
        return late
    return late if (invoice.get("attempt_count") or 0) > 1 else None


async def restart_period_after_late_payment(invoice_id: str) -> RestartOutcome:
    """Restart the subscription's billing period from this invoice's payment,
    if it was paid late and still opens the subscription's current period."""
    invoice = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
    sub_id = _invoice_subscription_id(invoice)
    if not sub_id or invoice.get("billing_reason") not in PERIOD_OPENING_REASONS:
        return RestartOutcome.NOT_APPLICABLE
    if invoice.get("status") != "paid":
        return RestartOutcome.NOT_APPLICABLE
    async with transaction(timeout=timedelta(seconds=90)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"billing-restart:{sub_id}",
        )
        return await _restart_locked(invoice, sub_id, tx)


async def _restart_locked(invoice: dict, sub_id: str, tx: Prisma) -> RestartOutcome:
    invoice_id: str = invoice["id"]
    subscription = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
    if subscription.get("status") != "active":
        return RestartOutcome.NOT_APPLICABLE
    if stripe_id(subscription.get("latest_invoice")) != invoice_id:
        # Restarted already, or a newer invoice opened a later period: an old
        # payment never moves the renewal date of the period after it.
        return RestartOutcome.NOT_APPLICABLE
    late = late_payment(invoice, current_period_start(subscription))
    if late is None:
        return RestartOutcome.NOT_LATE
    customer_id = stripe_id(subscription.get("customer"))
    user = await User.prisma(tx).find_first(where={"stripeCustomerId": customer_id})
    if user is None or user.subscriptionTier == SubscriptionTier.ENTERPRISE:
        return RestartOutcome.NOT_APPLICABLE
    if await replaced_by_newer_plan(customer_id, subscription):
        # The revived old plan is being ended; nothing to restart.
        return RestartOutcome.NOT_APPLICABLE
    if not _restartable(subscription) or not bills_only_the_plan(invoice):
        logger.error(
            f"Subscription {sub_id} was paid late (invoice {invoice_id}) but has"
            " a schedule, a pending update or a bill beyond the plan, so its"
            " billing period cannot be restarted automatically; restart it by"
            " hand"
        )
        return RestartOutcome.NEEDS_MANUAL_FIX
    progress = await _record_progress(invoice_id, user.id, sub_id, late, tx)
    if progress.restartedAt is not None:
        return RestartOutcome.NOT_APPLICABLE
    if not await _credit_late_invoice(invoice, late, progress, tx):
        return RestartOutcome.NEEDS_MANUAL_FIX
    if not await _reset_anchor(sub_id, invoice_id):
        return RestartOutcome.NEEDS_MANUAL_FIX
    await SubscriptionBillingRestart.prisma(tx).update(
        where={"stripeInvoiceId": invoice_id},
        data={"restartedAt": datetime.now(UTC)},
    )
    await invalidate_billing_lookups(customer_id)
    logger.info(
        f"Restarted the billing period of subscription {sub_id} from the late"
        f" payment of invoice {invoice_id}, paid {_iso(late.paid_at)}"
    )
    return RestartOutcome.RESTARTED


async def _record_progress(
    invoice_id: str, user_id: str, sub_id: str, late: LatePayment, tx: Prisma
) -> SubscriptionBillingRestart:
    return await SubscriptionBillingRestart.prisma(tx).upsert(
        where={"stripeInvoiceId": invoice_id},
        data={
            "create": {
                "stripeInvoiceId": invoice_id,
                "userId": user_id,
                "stripeSubscriptionId": sub_id,
                "paidAt": datetime.fromtimestamp(late.paid_at, UTC),
                "periodStart": datetime.fromtimestamp(late.period_start, UTC),
            },
            "update": {},
        },
    )


async def _credit_late_invoice(
    invoice: dict,
    late: LatePayment,
    progress: SubscriptionBillingRestart,
    tx: Prisma,
) -> bool:
    """Credit the late invoice to the customer's balance, once, and only
    when that credit pays the whole reset invoice."""
    invoice_id: str = invoice["id"]
    total: int = invoice.get("total") or 0
    credited = bool(progress.stripeCreditNoteId)
    if not credited:
        ours, credited_otherwise = await _existing_credit_notes(invoice_id)
        if credited_otherwise:
            return False
        if ours:
            await _save_credit_note(invoice_id, ours, tx)
            credited = True
    if not await credit_covers_new_period(
        progress.stripeSubscriptionId, invoice_id, total, credited=credited
    ):
        return False
    if credited or total <= 0:
        return True
    paid_on = datetime.fromtimestamp(late.paid_at, UTC).strftime("%-d %B %Y")
    note = await stripe_call(
        stripe.CreditNote.create_async,
        invoice=invoice_id,
        amount=total,
        credit_amount=total,
        memo=(
            f"Your plan restarts from your payment on {paid_on}. This credit"
            " pays the new billing period, so you are not charged for the"
            " days your plan was paused."
        ),
        metadata={RESTARTED_FROM_INVOICE_KEY: invoice_id},
        idempotency_key=f"billing-restart-credit:{invoice_id}",
    )
    await _save_credit_note(invoice_id, note["id"], tx)
    return True


async def _existing_credit_notes(invoice_id: str) -> tuple[str | None, bool]:
    """This restart's credit note on the invoice, if any, and whether another
    credit note already credits it."""
    notes = await stripe_call(
        stripe.CreditNote.list_async, invoice=invoice_id, limit=100
    )
    async for note in stripe_list_items(notes):
        if (note.get("metadata") or {}).get(RESTARTED_FROM_INVOICE_KEY) == invoice_id:
            return note["id"], False
        if note.get("status") != "void":
            logger.error(
                f"Invoice {invoice_id} already has credit note {note.get('id')};"
                " not crediting it again to restart its billing period"
            )
            return None, True
    return None, False


async def _save_credit_note(invoice_id: str, credit_note_id: str, tx: Prisma) -> None:
    await SubscriptionBillingRestart.prisma(tx).update(
        where={"stripeInvoiceId": invoice_id},
        data={"stripeCreditNoteId": credit_note_id},
    )


async def _reset_anchor(sub_id: str, invoice_id: str) -> bool:
    # An upgrade may have changed the plan while the credit was made.
    current = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
    if stripe_id(current.get("latest_invoice")) != invoice_id or not _restartable(
        current
    ):
        logger.error(
            f"Subscription {sub_id} changed while its billing period was being"
            f" restarted from invoice {invoice_id}; the credit is on the"
            " customer's balance and the period was not restarted"
        )
        return False
    try:
        await stripe_call(
            stripe.Subscription.modify_async,
            sub_id,
            billing_cycle_anchor="now",
            proration_behavior="none",
            payment_behavior="error_if_incomplete",
            idempotency_key=f"billing-restart-anchor:{invoice_id}",
        )
    except stripe.CardError:
        logger.error(
            f"Restarting subscription {sub_id} after late invoice {invoice_id}"
            " needed a further charge that failed; the credit stays on the"
            " customer's balance and the period was not restarted"
        )
        return False
    return True


def _restartable(subscription: dict) -> bool:
    return subscription.get("status") == "active" and not (
        subscription.get("schedule") or subscription.get("pending_update")
    )


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()
