"""End the account's Stripe subscriptions, so a deleted account is never
billed again. The Stripe customer and its invoices and payments stay: they are
the billing records accounting keeps.
"""

import stripe

from backend.data.account_deletion.models import StepReport, StepStatus, Subject
from backend.data.credit import _cancel_customer_subscriptions
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.util.settings import Settings

# Read by the `customer.subscription.deleted` handler, like the payment
# failure comment beside `_cancel_customer_subscriptions`.
ACCOUNT_DELETED_COMMENT = "autogpt:account_deleted"
_BILLABLE_STATUSES = ("active", "trialing")


async def cancel_subscriptions(subject: Subject, *, dry_run: bool) -> StepReport:
    """Cancel every active or trialing subscription now, not at period end:
    nobody is left to use the rest of the period."""
    if not subject.stripe_customer_ids:
        return _report(0, dry_run)
    if not Settings().secrets.stripe_api_key:
        return StepReport(
            name="subscriptions",
            status=StepStatus.SKIPPED,
            note="STRIPE_API_KEY is not set; check the customer in Stripe by hand",
        )
    if dry_run:
        counts = [await _count_billable(c) for c in subject.stripe_customer_ids]
        return _report(sum(counts), dry_run)
    cancelled = [
        await _cancel_customer_subscriptions(
            customer_id, cancellation_comment=ACCOUNT_DELETED_COMMENT
        )
        for customer_id in subject.stripe_customer_ids
    ]
    return _report(sum(cancelled), dry_run)


def _report(count: int, dry_run: bool) -> StepReport:
    return StepReport(
        name="subscriptions",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"subscriptions": count},
    )


async def _count_billable(customer_id: str) -> int:
    count = 0
    for status in _BILLABLE_STATUSES:
        page = await stripe_call(
            stripe.Subscription.list_async,
            customer=customer_id,
            status=status,
            limit=100,
        )
        count += len([sub async for sub in stripe_list_items(page)])
    return count
