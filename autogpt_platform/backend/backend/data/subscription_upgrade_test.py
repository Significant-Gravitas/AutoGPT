"""Financial and ownership invariants of the Max upgrade charge review."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit
from backend.data import subscription_upgrade as billing
from backend.data.subscription_upgrade_models import (
    SubscriptionUpgradePreviewRequest,
    UpgradePreviewUnavailable,
)

NOW = 1791374400
REQUEST = SubscriptionUpgradePreviewRequest(tier="MAX")


@pytest.fixture(autouse=True)
def graph_cleanup():
    """No graph/database resources are used by these Stripe boundary tests."""
    yield


@pytest.fixture
def setup(monkeypatch):
    user = SimpleNamespace(
        stripe_customer_id="cus_owned", subscription_tier=SubscriptionTier.PRO
    )
    sub = {
        "id": "sub_owned",
        "customer": "cus_owned",
        "status": "active",
        "current_period_end": NOW + 2500000,
        "metadata": {"user_id": "user-1"},
        "schedule": None,
        "cancel_at_period_end": False,
        "items": {
            "data": [
                {
                    "id": "si_owned",
                    "price": {"id": "price_pro", "recurring": {"interval": "month"}},
                    "quantity": 1,
                }
            ]
        },
    }
    invoice = {
        "customer": "cus_owned",
        "currency": "usd",
        "amount_due": 27278,
        "total": 27278,
        "total_tax_amounts": [{"amount": 2020, "inclusive": False}],
        "lines": {
            "data": [
                {
                    "description": "Max",
                    "amount": 29935,
                    "proration": True,
                    "period": {"start": NOW, "end": NOW + 2500000},
                },
                {
                    "description": "Unused Pro time",
                    "amount": -4677,
                    "proration": True,
                    "period": {"start": NOW, "end": NOW + 2500000},
                },
            ]
        },
    }
    price = {
        "id": "price_max",
        "unit_amount": 32000,
        "currency": "usd",
        "recurring": {"interval": "month"},
        "tax_behavior": "exclusive",
    }
    cache = {}

    async def setex(key, ttl, value):
        assert ttl == 900
        cache[key] = value

    redis = SimpleNamespace(
        setex=AsyncMock(side_effect=setex),
        get=AsyncMock(side_effect=lambda key: cache.get(key)),
    )
    monkeypatch.setattr(billing.time, "time", lambda: NOW)
    monkeypatch.setattr(billing, "get_user_by_id", AsyncMock(return_value=user))
    monkeypatch.setattr(
        billing,
        "_get_active_subscription",
        AsyncMock(side_effect=lambda _: stripe.Subscription.construct_from(sub, "k")),
    )
    monkeypatch.setattr(
        billing, "get_subscription_price_id", AsyncMock(return_value="price_max")
    )
    monkeypatch.setattr(billing, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(stripe.Price, "retrieve_async", AsyncMock(return_value=price))
    preview = AsyncMock(side_effect=lambda **_: invoice)
    monkeypatch.setattr(stripe.Invoice, "create_preview_async", preview)
    return SimpleNamespace(
        user=user, sub=sub, invoice=invoice, price=price, preview=preview, redis=redis
    )


async def test_preview_and_confirm_use_identical_stripe_proration_date(setup):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    assert preview.amount_due == 27278
    assert preview.tax_amount == 2020
    assert [line.amount for line in preview.lines] == [29935, -4677]
    assert preview.adjustment_amount == 0
    assert preview.recurring_amount == 32000
    quote = await billing.validate_subscription_upgrade_preview(
        "user-1", REQUEST, preview.preview_id, preview.proration_date
    )
    assert quote.preview == preview
    assert setup.preview.await_count == 2
    for call in setup.preview.await_args_list:
        assert call.kwargs["subscription_details"] == {
            "items": [{"id": "si_owned", "price": "price_max", "quantity": 1}],
            "proration_behavior": "always_invoice",
            "proration_date": NOW,
        }


@pytest.mark.parametrize(
    "change", ["amount", "currency", "item", "price", "period", "tax"]
)
async def test_changed_invoice_requires_new_consent(setup, change):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    if change == "amount":
        setup.invoice["amount_due"] += 100
    elif change == "currency":
        setup.invoice["currency"] = "eur"
    elif change == "item":
        setup.sub["items"]["data"][0]["id"] = "si_changed"
    elif change == "price":
        setup.price["unit_amount"] += 100
    elif change == "period":
        setup.invoice["lines"]["data"][0]["period"]["end"] += 3600
    else:
        setup.invoice["total_tax_amounts"][0]["amount"] += 100
    with pytest.raises(UpgradePreviewUnavailable):
        await billing.validate_subscription_upgrade_preview(
            "user-1", REQUEST, preview.preview_id, preview.proration_date
        )


async def test_quote_cannot_be_used_by_another_user_or_backdated(setup):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    with pytest.raises(UpgradePreviewUnavailable):
        await billing.validate_subscription_upgrade_preview(
            "user-2", REQUEST, preview.preview_id, preview.proration_date
        )
    with pytest.raises(UpgradePreviewUnavailable):
        await billing.validate_subscription_upgrade_preview(
            "user-1", REQUEST, preview.preview_id, preview.proration_date - 1
        )
    assert setup.preview.await_count == 1


async def test_expired_preview_never_reaches_stripe(setup, monkeypatch):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    monkeypatch.setattr(billing.time, "time", lambda: NOW + 901)
    with pytest.raises(UpgradePreviewUnavailable, match="Review"):
        await billing.validate_subscription_upgrade_preview(
            "user-1", REQUEST, preview.preview_id, preview.proration_date
        )
    assert setup.preview.await_count == 1


async def test_basil_inclusive_tax_and_account_credit_are_disclosed(setup):
    setup.invoice.pop("total_tax_amounts")
    setup.invoice["total_taxes"] = [{"amount": 2020, "tax_behavior": "inclusive"}]
    setup.invoice["total"] = 25258
    setup.invoice["amount_due"] = 24258
    setup.price["tax_behavior"] = "inclusive"
    for line in setup.invoice["lines"]["data"]:
        line.pop("proration")
        line["parent"] = {"subscription_item_details": {"proration": True}}
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    assert preview.tax_amount == 2020
    assert preview.tax_included is True
    assert preview.recurring_tax_included is True
    assert preview.adjustment_amount == -1000
    assert all(line.proration for line in preview.lines)


async def test_one_off_invoice_items_do_not_move_the_next_renewal(setup):
    setup.invoice["lines"]["data"].append(
        {
            "description": "One-off service",
            "amount": 1000,
            "period": {"start": NOW, "end": NOW + 5000000},
        }
    )
    setup.invoice["amount_due"] += 1000
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    assert preview.period_end == setup.sub["current_period_end"]
    assert preview.amount_due == 28278


@pytest.mark.parametrize("basil", [False, True])
async def test_cycle_change_uses_only_the_owned_target_subscription_line(setup, basil):
    setup.price["recurring"]["interval"] = "year"
    end = NOW + 365 * 86400
    debit = setup.invoice["lines"]["data"][0]
    debit["period"]["end"] = end
    if basil:
        debit["parent"] = {
            "subscription_item_details": {
                "subscription_item": "si_owned",
                "proration": False,
            }
        }
        debit["pricing"] = {"price_details": {"price": "price_max"}}
    else:
        debit["subscription_item"] = "si_owned"
        debit["price"] = {"id": "price_max"}
    setup.invoice["lines"]["data"].append(
        {
            "description": "One-off service",
            "amount": 1000,
            "period": {"start": NOW, "end": end + 86400},
        }
    )
    preview = await billing.preview_subscription_upgrade(
        "user-1", SubscriptionUpgradePreviewRequest(tier="MAX", billing_cycle="yearly")
    )
    assert preview.period_end == end


@pytest.mark.parametrize("change", ["customer", "metadata", "trial", "pagination"])
async def test_foreign_or_incomplete_subscription_is_not_previewed(setup, change):
    if change == "customer":
        setup.sub["customer"] = "cus_other"
    elif change == "metadata":
        setup.sub["metadata"]["user_id"] = "other"
    elif change == "trial":
        setup.sub["status"] = "trialing"
    else:
        setup.invoice["lines"]["has_more"] = True
    with pytest.raises(UpgradePreviewUnavailable):
        await billing.preview_subscription_upgrade("user-1", REQUEST)


async def test_modify_passes_reviewed_date_and_idempotency_key_to_stripe(
    setup, monkeypatch
):
    setup.sub["items"]["data"][0]["quantity"] = 2
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    quote = await billing.validate_subscription_upgrade_preview(
        "user-1", REQUEST, preview.preview_id, preview.proration_date
    )
    monkeypatch.setattr(
        credit, "get_subscription_price_id", AsyncMock(return_value="price_max")
    )
    monkeypatch.setattr(credit, "get_user_by_id", AsyncMock(return_value=setup.user))
    monkeypatch.setattr(
        credit,
        "_get_active_subscription",
        AsyncMock(return_value=stripe.Subscription.construct_from(setup.sub, "k")),
    )
    monkeypatch.setattr(credit, "set_subscription_tier", AsyncMock())
    monkeypatch.setattr(credit, "_track_billing_event", Mock())
    monkeypatch.setattr(credit.get_pending_subscription_change, "cache_delete", Mock())
    modify = AsyncMock()
    monkeypatch.setattr(stripe.Subscription, "modify_async", modify)
    await credit.modify_stripe_subscription_for_tier(
        "user-1",
        SubscriptionTier.MAX,
        proration_date=preview.proration_date,
        upgrade_preview=quote,
    )
    assert modify.await_args.kwargs["proration_date"] == NOW
    assert (
        modify.await_args.kwargs["idempotency_key"]
        == f"subscription-upgrade:{preview.preview_id}"
    )
    assert modify.await_args.kwargs["proration_behavior"] == "always_invoice"
    assert modify.await_args.kwargs["payment_behavior"] == "error_if_incomplete"
    assert modify.await_args.kwargs["items"] == [
        {"id": "si_owned", "price": "price_max", "quantity": 2}
    ]
    assert quote.preview.recurring_amount == 64000
    assert setup.preview.await_args.kwargs["subscription_details"]["items"] == [
        {"id": "si_owned", "price": "price_max", "quantity": 2}
    ]
