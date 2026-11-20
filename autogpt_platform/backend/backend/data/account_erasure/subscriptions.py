"""End the Stripe subscriptions of the account and of the orgs that go with
it, so nobody is billed for an account or org that no longer exists.

Closing the account (`cancel_all_subscriptions_now`, SECRT-2877) cancelled
its own customer's subscriptions, and refused to close the account if it
could not, and a Checkout started around the close is expired. One paid all
the same (a Stripe failure in closing's sweep, say) would bill a closed
account, so its own customer is covered again here, with the customers of
its sole orgs. The customers and their invoices and payments
stay in Stripe: they are the billing records accounting keeps.
"""

import stripe

from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.credit import (
    _ACCOUNT_DELETION_STATUSES,
    ACCOUNT_DELETION_CANCELLATION_COMMENT,
    _cancel_customer_subscriptions,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_checkout import expire_all_open_checkouts
from backend.util.settings import Settings


async def cancel_subscriptions(subject: Subject, *, dry_run: bool) -> StepReport:
    """Expire their open Checkouts and cancel every subscription Stripe can
    still cancel, now and with no invoice, the way closing the account does."""
    customers = subject.stripe_customer_ids
    if not customers:
        return _report(0, dry_run)
    if not Settings().secrets.stripe_api_key:
        return StepReport(
            name="subscriptions",
            status=StepStatus.SKIPPED,
            note="STRIPE_API_KEY is not set; check the account's and its orgs' "
            "customers in Stripe by hand",
        )
    if dry_run:
        counts = [await _count_cancellable(c) for c in customers]
        return _report(sum(counts), dry_run)
    cancelled = 0
    for customer_id in customers:
        await expire_all_open_checkouts(customer_id)
        cancelled += await _cancel_customer_subscriptions(
            customer_id,
            at_period_end=False,
            cancellation_comment=ACCOUNT_DELETION_CANCELLATION_COMMENT,
            statuses=_ACCOUNT_DELETION_STATUSES,
        )
    return _report(cancelled, dry_run)


def _report(count: int, dry_run: bool) -> StepReport:
    return StepReport(
        name="subscriptions",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"subscriptions": count},
    )


async def _count_cancellable(customer_id: str) -> int:
    count = 0
    for status in _ACCOUNT_DELETION_STATUSES:
        page = await stripe_call(
            stripe.Subscription.list_async,
            customer=customer_id,
            status=status,
            limit=100,
        )
        count += len([sub async for sub in stripe_list_items(page)])
    return count
