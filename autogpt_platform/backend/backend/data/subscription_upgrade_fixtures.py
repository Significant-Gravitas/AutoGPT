"""Stripe and Redis boundaries for reviewed subscription upgrades."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_upgrade as billing

NOW = 1791374400


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
        delete=AsyncMock(side_effect=lambda key: int(cache.pop(key, None) is not None)),
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
