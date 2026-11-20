"""Preview and revalidate an owned Pro-to-Max invoice before charging."""

import time
from uuid import uuid4

import stripe
from prisma.enums import SubscriptionTier

from backend.data.credit import _get_active_subscription, get_subscription_price_id
from backend.data.redis_client import get_redis_async
from backend.data.stripe_client import stripe_call
from backend.data.subscription_upgrade_cache import subscription_upgrade_preview_key
from backend.data.subscription_upgrade_models import (
    StoredUpgradePreview,
    SubscriptionUpgradePreview,
    SubscriptionUpgradePreviewRequest,
    UpgradeInvoicePreview,
    UpgradePreviewUnavailable,
    UpgradePrice,
    UpgradeProrationLine,
    UpgradeSubscription,
)
from backend.data.user import get_user_by_id

PREVIEW_TTL = 15 * 60


async def preview_subscription_upgrade(
    user_id: str, request: SubscriptionUpgradePreviewRequest
) -> SubscriptionUpgradePreview:
    quote = await _build_preview(user_id, request, int(time.time()), uuid4().hex)
    redis = await get_redis_async()
    await redis.setex(
        subscription_upgrade_preview_key(user_id, quote.preview.preview_id),
        PREVIEW_TTL,
        quote.model_dump_json(),
    )
    return quote.preview


async def validate_subscription_upgrade_preview(
    user_id: str,
    request: SubscriptionUpgradePreviewRequest,
    preview_id: str,
    proration_date: int,
) -> StoredUpgradePreview:
    redis = await get_redis_async()
    raw = await redis.get(subscription_upgrade_preview_key(user_id, preview_id))
    if not raw:
        raise UpgradePreviewUnavailable("This review expired. Review the charge again.")
    saved = StoredUpgradePreview.model_validate_json(raw)
    if (
        saved.user_id != user_id
        or saved.preview.preview_id != preview_id
        or saved.preview.proration_date != proration_date
        or saved.preview.billing_cycle != request.billing_cycle
        or saved.preview.expires_at <= int(time.time())
    ):
        raise UpgradePreviewUnavailable("This review changed. Review the charge again.")
    current = await _build_preview(user_id, request, proration_date, preview_id)
    if current != saved:
        raise UpgradePreviewUnavailable(
            "Your billing details changed. Review the updated charge before paying."
        )
    return saved


async def _owned_subscription(user_id: str) -> UpgradeSubscription:
    user = await get_user_by_id(user_id)
    if user.subscription_tier != SubscriptionTier.PRO or not user.stripe_customer_id:
        raise UpgradePreviewUnavailable("An active paid Pro plan is required.")
    raw = await _get_active_subscription(user.stripe_customer_id)
    if raw is None:
        raise UpgradePreviewUnavailable("No active paid subscription was found.")
    sub = UpgradeSubscription.model_validate(raw)
    if (
        sub.customer != user.stripe_customer_id
        or sub.status != "active"
        or sub.metadata.get("user_id", user_id) != user_id
        or sub.items.has_more
        or len(sub.items.data) != 1
    ):
        raise UpgradePreviewUnavailable("This subscription needs a billing review.")
    return sub


async def _build_preview(
    user_id: str,
    request: SubscriptionUpgradePreviewRequest,
    proration_date: int,
    preview_id: str,
) -> StoredUpgradePreview:
    sub = await _owned_subscription(user_id)
    price_id = await get_subscription_price_id(
        SubscriptionTier(request.tier), request.billing_cycle
    )
    if not price_id:
        raise UpgradePreviewUnavailable("Max is unavailable for this billing cycle.")
    price = UpgradePrice.model_validate(
        await stripe_call(stripe.Price.retrieve_async, price_id)
    )
    interval = "year" if request.billing_cycle == "yearly" else "month"
    if price.recurring.interval != interval or price.recurring.interval_count != 1:
        raise UpgradePreviewUnavailable("This price needs a billing review.")
    item = sub.items.data[0]
    invoice = UpgradeInvoicePreview.model_validate(
        await stripe_call(
            stripe.Invoice.create_preview_async,
            customer=sub.customer,
            subscription=sub.id,
            subscription_details={
                "items": [
                    {"id": item.id, "price": price_id, "quantity": item.quantity}
                ],
                "proration_behavior": "always_invoice",
                "proration_date": proration_date,
            },
        )
    )
    if invoice.customer != sub.customer or invoice.currency != price.currency:
        raise UpgradePreviewUnavailable("The invoice could not be verified.")
    if invoice.lines.has_more or not invoice.lines.data:
        raise UpgradePreviewUnavailable("The complete invoice could not be loaded.")
    period_end = _renewal_date(sub, price, invoice)
    if period_end <= proration_date:
        raise UpgradePreviewUnavailable("The billing period changed. Try again.")
    return StoredUpgradePreview(
        user_id=user_id,
        subscription_id=sub.id,
        item_id=item.id,
        current_price_id=item.price.id,
        target_price_id=price_id,
        quantity=item.quantity,
        preview=_public_preview(
            invoice,
            price,
            item.quantity,
            request,
            preview_id,
            proration_date,
            period_end,
        ),
    )


def _renewal_date(
    sub: UpgradeSubscription,
    price: UpgradePrice,
    invoice: UpgradeInvoicePreview,
) -> int:
    item = sub.items.data[0]
    current_end = item.current_period_end or sub.current_period_end
    if item.price.recurring == price.recurring and current_end is not None:
        return current_end
    periods = [
        line.period.end
        for line in invoice.lines.data
        if line.amount >= 0
        and line.subscription_item_id == item.id
        and line.price_id == price.id
    ]
    if not periods:
        raise UpgradePreviewUnavailable("The next billing date could not be verified.")
    return max(periods)


def _public_preview(
    invoice: UpgradeInvoicePreview,
    price: UpgradePrice,
    quantity: int,
    request: SubscriptionUpgradePreviewRequest,
    preview_id: str,
    proration_date: int,
    period_end: int,
) -> SubscriptionUpgradePreview:
    taxes = invoice.total_taxes
    if taxes is None:
        taxes = invoice.total_tax_amounts
    exclusive_tax = sum(tax.amount for tax in (taxes or []) if not tax.is_inclusive)
    lines = [
        UpgradeProrationLine(
            description=line.description or "Subscription adjustment",
            amount=line.amount,
            period_start=line.period.start,
            period_end=line.period.end,
            proration=line.is_proration,
        )
        for line in invoice.lines.data
    ]
    return SubscriptionUpgradePreview(
        preview_id=preview_id,
        amount_due=invoice.amount_due,
        tax_amount=sum(tax.amount for tax in taxes) if taxes is not None else None,
        tax_included=bool(taxes) and all(tax.is_inclusive for tax in taxes),
        adjustment_amount=invoice.amount_due
        - sum(line.amount for line in lines)
        - exclusive_tax,
        currency=invoice.currency,
        proration_date=proration_date,
        period_end=period_end,
        expires_at=proration_date + PREVIEW_TTL,
        recurring_amount=price.unit_amount * quantity,
        recurring_tax_included=price.tax_behavior == "inclusive",
        billing_cycle=request.billing_cycle,
        lines=lines,
    )
