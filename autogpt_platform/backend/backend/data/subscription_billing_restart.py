"""Start a full billing period from a late payment.

A renewal or a trial's first charge that fails cuts access at once while
Stripe keeps retrying the invoice (see ``subscription_payment_failure``),
which records the pause here (``record_plan_paused``). When the invoice is
paid later, by a retry, the payment link, a bank that confirms days later or
the wallet, the customer must get a full period from that payment and renew
on its date, not pay for the days they had no access. A payment that is only
slow, such as a bank debit still processing, keeps the plan active and
pauses nothing, so it keeps its date.

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

When the credit is made but the reset does not land (the subscription
changed, or Stripe refused the reset), the credit is taken back from the
balance, since Stripe voids a credit note only on an open invoice, and the
restart is left for a person.

Both steps can be repeated safely. A ``SubscriptionBillingRestart`` row per
paused invoice records how far the restart got, the credit note is also looked up
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

# The invoices that open a billing period. An upgrade's proration invoice
# (``subscription_update``) or a one-off charge never moves the renewal date.
PERIOD_OPENING_REASONS = ("subscription_cycle", "subscription_create")
RESTARTED_FROM_INVOICE_KEY = "billing_restarted_from_invoice"
CREDIT_TAKEN_BACK_KEY = "billing_restart_credit_taken_back"
# Stripe errors that a retry of the same request can get past. Any other is
# returned again for the same idempotency key, so retrying it changes nothing.
TRANSIENT_STRIPE_ERRORS = (
    stripe.APIConnectionError,
    stripe.RateLimitError,
    stripe.APIError,
)


class RestartOutcome(StrEnum):
    RESTARTED = "restarted"
    NOT_LATE = "not_late"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_MANUAL_FIX = "needs_manual_fix"


class LatePayment(BaseModel):
    paused_at: int
    paid_at: int

    @property
    def seconds_late(self) -> int:
        return self.paid_at - self.paused_at


def late_payment(
    invoice: dict, pause: SubscriptionBillingRestart | None
) -> LatePayment | None:
    """Whether a paid invoice that opened a billing period was paid late.

    This is the rule for when the period restarts from the payment: its
    failed payment paused the plan (``pause``), and it was paid after that.
    An invoice paid without a pause kept its plan throughout, however long
    the payment took.
    """
    if pause is None or invoice.get("status") != "paid":
        return None
    if invoice.get("billing_reason") not in PERIOD_OPENING_REASONS:
        return None
    paid_at = (invoice.get("status_transitions") or {}).get("paid_at")
    paused_at = int(pause.pausedAt.timestamp())
    if paid_at is None or paid_at < paused_at:
        return None
    return LatePayment(paused_at=paused_at, paid_at=paid_at)


async def record_plan_paused(user_id: str, invoice: dict, subscription: dict) -> None:
    """Record that this invoice's failed payment cut the customer's access, so
    paying it later restarts the billing period from the payment. The first
    pause counts: Stripe's further failed retries leave the record as it is.
    """
    if invoice.get("billing_reason") not in PERIOD_OPENING_REASONS:
        return
    period_start = current_period_start(subscription) or line_period_start(invoice)
    if period_start is None:
        return
    await SubscriptionBillingRestart.prisma().upsert(
        where={"stripeInvoiceId": invoice["id"]},
        data={
            "create": {
                "stripeInvoiceId": invoice["id"],
                "userId": user_id,
                "stripeSubscriptionId": subscription["id"],
                "periodStart": datetime.fromtimestamp(period_start, UTC),
                "pausedAt": datetime.now(UTC),
            },
            "update": {},
        },
    )


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
        return await _not_applicable(invoice, subscription, tx)
    if stripe_id(subscription.get("latest_invoice")) != invoice_id:
        # Restarted already, or a newer invoice opened a later period: an old
        # payment never moves the renewal date of the period after it.
        await _record_finished_reset(invoice, subscription, tx)
        return await _not_applicable(invoice, subscription, tx)
    pause = await SubscriptionBillingRestart.prisma(tx).find_unique(
        where={"stripeInvoiceId": invoice_id}
    )
    late = late_payment(invoice, pause)
    if pause is None or late is None:
        return RestartOutcome.NOT_LATE
    if pause.restartedAt is not None:
        return RestartOutcome.NOT_APPLICABLE
    if pause.abandonedAt is not None:
        return RestartOutcome.NEEDS_MANUAL_FIX
    customer_id = stripe_id(subscription.get("customer"))
    user = await User.prisma(tx).find_first(where={"stripeCustomerId": customer_id})
    if user is None or user.subscriptionTier == SubscriptionTier.ENTERPRISE:
        return RestartOutcome.NOT_APPLICABLE
    if await replaced_by_newer_plan(customer_id, subscription):
        # The revived old plan is being ended; nothing to restart.
        return await _not_applicable(invoice, subscription, tx)
    if not _restartable(subscription) or not bills_only_the_plan(invoice):
        logger.error(
            f"Subscription {sub_id} was paid late (invoice {invoice_id}) but has"
            " a schedule, a pending update or a bill beyond the plan, so its"
            " billing period cannot be restarted automatically; restart it by"
            " hand"
        )
        return RestartOutcome.NEEDS_MANUAL_FIX
    await SubscriptionBillingRestart.prisma(tx).update(
        where={"stripeInvoiceId": invoice_id},
        data={"paidAt": datetime.fromtimestamp(late.paid_at, UTC)},
    )
    credit_note_id = await _credit_late_invoice(invoice, late, pause, tx)
    if credit_note_id is None:
        return RestartOutcome.NEEDS_MANUAL_FIX
    if not await _reset_anchor(sub_id, invoice_id):
        if credit_note_id:
            await _take_back_credit(invoice, customer_id, credit_note_id, tx)
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


async def _not_applicable(
    invoice: dict, subscription: dict, tx: Prisma
) -> RestartOutcome:
    """Nothing to restart now, but an earlier attempt may have credited the
    invoice before it failed, and its record of the credit rolled back with
    it. Left on the balance, that credit pays the next invoice, so it is
    taken back and the restart left for a person."""
    invoice_id: str = invoice["id"]
    metadata = subscription.get("metadata") or {}
    if metadata.get(RESTARTED_FROM_INVOICE_KEY) == invoice_id:
        return RestartOutcome.NOT_APPLICABLE
    pause = await SubscriptionBillingRestart.prisma(tx).find_unique(
        where={"stripeInvoiceId": invoice_id}
    )
    if pause is None or pause.restartedAt is not None or pause.abandonedAt:
        return RestartOutcome.NOT_APPLICABLE
    ours, _ = await _existing_credit_notes(invoice_id)
    if ours is None:
        return RestartOutcome.NOT_APPLICABLE
    if (ours.get("metadata") or {}).get(CREDIT_TAKEN_BACK_KEY):
        await _abandon(invoice_id, ours["id"], tx)
    else:
        customer_id = stripe_id(invoice.get("customer"))
        await _take_back_credit(invoice, customer_id, ours["id"], tx)
    return RestartOutcome.NEEDS_MANUAL_FIX


async def _record_finished_reset(invoice: dict, subscription: dict, tx: Prisma) -> None:
    """Record a reset Stripe made whose record was lost, when the transaction
    around it rolled back after the reset, so the audit doesn't list it."""
    invoice_id: str = invoice["id"]
    metadata = subscription.get("metadata") or {}
    if metadata.get(RESTARTED_FROM_INVOICE_KEY) != invoice_id:
        return
    pause = await SubscriptionBillingRestart.prisma(tx).find_unique(
        where={"stripeInvoiceId": invoice_id}
    )
    late = late_payment(invoice, pause)
    if pause is None or late is None or pause.restartedAt is not None:
        return
    credit_note_id = pause.stripeCreditNoteId
    if not credit_note_id:
        ours, _ = await _existing_credit_notes(invoice_id)
        credit_note_id = ours["id"] if ours else None
    await SubscriptionBillingRestart.prisma(tx).update(
        where={"stripeInvoiceId": invoice_id},
        data={
            "paidAt": datetime.fromtimestamp(late.paid_at, UTC),
            "restartedAt": datetime.now(UTC),
            "stripeCreditNoteId": credit_note_id,
        },
    )
    await invalidate_billing_lookups(stripe_id(subscription.get("customer")))


async def _credit_late_invoice(
    invoice: dict,
    late: LatePayment,
    progress: SubscriptionBillingRestart,
    tx: Prisma,
) -> str | None:
    """Credit the late invoice to the customer's balance, once, and only
    when that credit pays the whole reset invoice. The credit note's id, or
    "" for an invoice with nothing to credit; None to leave it for a person,
    with any credit an earlier attempt made taken back."""
    invoice_id: str = invoice["id"]
    total: int = invoice.get("total") or 0
    credit_note_id = progress.stripeCreditNoteId
    if not credit_note_id:
        ours, credited_otherwise = await _existing_credit_notes(invoice_id)
        if credited_otherwise:
            return None
        if ours and (ours.get("metadata") or {}).get(CREDIT_TAKEN_BACK_KEY):
            # An earlier attempt gave up, and its record was rolled back.
            await _abandon(invoice_id, ours["id"], tx)
            return None
        if ours:
            credit_note_id = ours["id"]
            await _save_credit_note(invoice_id, credit_note_id, tx)
    if not await credit_covers_new_period(
        progress.stripeSubscriptionId, invoice_id, total, credited=bool(credit_note_id)
    ):
        if credit_note_id:
            customer_id = stripe_id(invoice.get("customer"))
            await _take_back_credit(invoice, customer_id, credit_note_id, tx)
        return None
    if credit_note_id or total <= 0:
        return credit_note_id or ""
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
    return note["id"]


async def _existing_credit_notes(invoice_id: str) -> tuple[dict | None, bool]:
    """This restart's credit note on the invoice, if any, and whether another
    credit note already credits it."""
    notes = await stripe_call(
        stripe.CreditNote.list_async, invoice=invoice_id, limit=100
    )
    async for note in stripe_list_items(notes):
        if (note.get("metadata") or {}).get(RESTARTED_FROM_INVOICE_KEY) == invoice_id:
            return dict(note), False
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


async def _take_back_credit(
    invoice: dict, customer_id: str, credit_note_id: str, tx: Prisma
) -> None:
    """Take the restart's credit back off the customer's balance, where
    Stripe would otherwise spend it on the next renewal at the old date.

    Stripe voids a credit note only on an open invoice, and the late invoice
    is paid, so the balance is debited by the same amount instead. Its
    idempotency key covers a retry within a day; a later one finds the debit
    by its metadata. The credit note's memo is corrected to say the plan was
    not restarted.
    """
    invoice_id: str = invoice["id"]
    total: int = invoice.get("total") or 0
    if total > 0 and not await _credit_taken_back(customer_id, invoice_id):
        await stripe_call(
            stripe.Customer.create_balance_transaction_async,
            customer_id,
            amount=total,
            currency=invoice["currency"],
            description=(
                f"Reverses credit note {credit_note_id}: the billing period"
                " was not restarted"
            ),
            metadata={CREDIT_TAKEN_BACK_KEY: invoice_id},
            idempotency_key=f"billing-restart-credit-back:{invoice_id}",
        )
    await stripe_call(
        stripe.CreditNote.modify_async,
        credit_note_id,
        memo=(
            "This credit was taken back: your plan's billing period could not"
            " be restarted from your payment, so your renewal date is"
            " unchanged."
        ),
        metadata={CREDIT_TAKEN_BACK_KEY: "true"},
    )
    await _abandon(invoice_id, credit_note_id, tx)
    logger.error(
        f"Took back the restart's credit on late invoice {invoice_id}: its"
        " billing period was not restarted; restart it by hand"
    )


async def _credit_taken_back(customer_id: str, invoice_id: str) -> bool:
    transactions = await stripe_call(
        stripe.Customer.list_balance_transactions_async, customer_id, limit=100
    )
    async for entry in stripe_list_items(transactions):
        metadata = entry.get("metadata") or {}
        if metadata.get(CREDIT_TAKEN_BACK_KEY) == invoice_id:
            return True
    return False


async def _abandon(invoice_id: str, credit_note_id: str, tx: Prisma) -> None:
    await SubscriptionBillingRestart.prisma(tx).update(
        where={"stripeInvoiceId": invoice_id},
        data={"abandonedAt": datetime.now(UTC), "stripeCreditNoteId": credit_note_id},
    )


async def _reset_anchor(sub_id: str, invoice_id: str) -> bool:
    # An upgrade may have changed the plan while the credit was made.
    current = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
    if stripe_id(current.get("latest_invoice")) != invoice_id or not _restartable(
        current
    ):
        logger.error(
            f"Subscription {sub_id} changed while its billing period was being"
            f" restarted from invoice {invoice_id}; the period was not restarted"
        )
        return False
    try:
        await stripe_call(
            stripe.Subscription.modify_async,
            sub_id,
            billing_cycle_anchor="now",
            proration_behavior="none",
            payment_behavior="error_if_incomplete",
            # Set with the reset itself, so a retry can tell it happened.
            metadata={RESTARTED_FROM_INVOICE_KEY: invoice_id},
            idempotency_key=f"billing-restart-anchor:{invoice_id}",
        )
    except TRANSIENT_STRIPE_ERRORS:
        # The reset may have landed; the retry finds it by the subscription's
        # latest invoice and metadata, or makes it.
        raise
    except stripe.StripeError:
        logger.exception(
            f"Stripe refused to restart subscription {sub_id} after late"
            f" invoice {invoice_id}; the period was not restarted"
        )
        return False
    return True


def _restartable(subscription: dict) -> bool:
    return subscription.get("status") == "active" and not (
        subscription.get("schedule") or subscription.get("pending_update")
    )


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()
