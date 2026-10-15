"""Upgrade previews expose the charge before an immediate paid-tier change."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import pytest
import stripe
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient
from prisma.enums import SubscriptionTier

from backend.api.features.billing.subscriptions import routes
from backend.data.subscription_upgrade_models import UpgradePreviewUnavailable

USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
PREVIEW_ID = "a" * 32
DATE = 1791374400
PREVIEW = {
    "preview_id": PREVIEW_ID,
    "amount_due": 25258,
    "tax_amount": 0,
    "tax_included": False,
    "adjustment_amount": 0,
    "currency": "usd",
    "proration_date": DATE,
    "period_end": DATE + 2500000,
    "expires_at": DATE + 900,
    "recurring_amount": 32000,
    "recurring_tax_included": False,
    "billing_cycle": "monthly",
    "lines": [],
}


@pytest.fixture(autouse=True)
def graph_cleanup():
    """These ASGI tests mock data boundaries and never create graphs."""
    yield


@pytest.fixture
def client(monkeypatch):
    app = fastapi.FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.enforce_subscription_status_rate_limit] = (
        lambda: None
    )
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": USER_ID,
        "role": "user",
        "email": "test@example.com",
    }
    monkeypatch.setattr(
        routes, "evaluate_feature_flag", AsyncMock(return_value=(True, True))
    )
    monkeypatch.setattr(
        routes,
        "get_user_by_id",
        AsyncMock(return_value=SimpleNamespace(subscription_tier=SubscriptionTier.PRO)),
    )
    monkeypatch.setattr(
        routes, "get_user_billing_cycle", AsyncMock(return_value="monthly")
    )
    monkeypatch.setattr(
        routes,
        "get_active_subscription_period_end",
        AsyncMock(return_value=DATE + 2500000),
    )
    monkeypatch.setattr(
        routes, "get_subscription_price_id", AsyncMock(return_value="price_max")
    )
    monkeypatch.setattr(
        routes,
        "get_subscription_status",
        AsyncMock(
            return_value=routes.SubscriptionStatusResponse(
                tier="MAX",
                monthly_cost=32000,
                tier_costs={"PRO": 5000, "MAX": 32000},
                proration_credit_cents=0,
            )
        ),
    )
    with TestClient(app) as client:
        yield client


def test_preview_uses_authenticated_user_and_requested_cycle(client, monkeypatch):
    preview = AsyncMock(return_value={**PREVIEW, "billing_cycle": "yearly"})
    monkeypatch.setattr(routes, "preview_subscription_upgrade", preview)
    response = client.post(
        "/credits/subscription/preview",
        json={"tier": "MAX", "billing_cycle": "yearly", "user_id": "other-user"},
    )
    assert response.status_code == 200
    assert response.json()["amount_due"] == 25258
    assert preview.await_args.args[0] == USER_ID
    assert preview.await_args.args[1].billing_cycle == "yearly"


def test_failed_preview_does_not_mutate_subscription(client, monkeypatch):
    preview = AsyncMock(side_effect=UpgradePreviewUnavailable("Review unavailable"))
    modify = AsyncMock()
    monkeypatch.setattr(routes, "preview_subscription_upgrade", preview)
    monkeypatch.setattr(routes, "modify_stripe_subscription_for_tier", modify)
    response = client.post("/credits/subscription/preview", json={"tier": "MAX"})
    assert response.status_code == 409
    modify.assert_not_awaited()


def test_confirmation_validates_review_and_passes_same_date(client, monkeypatch):
    quote = object()
    validate = AsyncMock(return_value=quote)
    modify = AsyncMock(return_value=True)
    monkeypatch.setattr(routes, "validate_subscription_upgrade_preview", validate)
    monkeypatch.setattr(routes, "modify_stripe_subscription_for_tier", modify)
    response = client.post(
        "/credits/subscription",
        json={"tier": "MAX", "preview_id": PREVIEW_ID, "proration_date": DATE},
    )
    assert response.status_code == 200
    assert validate.await_args.args[0] == USER_ID
    assert validate.await_args.args[2:] == (PREVIEW_ID, DATE)
    modify.assert_awaited_once_with(
        USER_ID,
        SubscriptionTier.MAX,
        "monthly",
        proration_date=DATE,
        upgrade_preview=quote,
    )


def test_changed_quote_requires_review_before_any_charge(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "validate_subscription_upgrade_preview",
        AsyncMock(side_effect=UpgradePreviewUnavailable("Review again")),
    )
    modify = AsyncMock()
    monkeypatch.setattr(routes, "modify_stripe_subscription_for_tier", modify)
    response = client.post(
        "/credits/subscription",
        json={"tier": "MAX", "preview_id": PREVIEW_ID, "proration_date": DATE},
    )
    assert response.status_code == 409
    modify.assert_not_awaited()


@pytest.mark.parametrize(
    "partial", [{"preview_id": PREVIEW_ID}, {"proration_date": DATE}]
)
def test_partial_review_cannot_backdate_a_charge(client, monkeypatch, partial):
    modify = AsyncMock()
    monkeypatch.setattr(routes, "modify_stripe_subscription_for_tier", modify)
    response = client.post("/credits/subscription", json={"tier": "MAX", **partial})
    assert response.status_code == 422
    modify.assert_not_awaited()


@pytest.mark.parametrize(
    "failure,status",
    [
        (stripe.CardError("Declined", "payment_method", "card_declined"), 402),
        (
            stripe.CardError(
                "Authentication required",
                "payment_method",
                "subscription_payment_intent_requires_action",
            ),
            402,
        ),
        (
            stripe.InvalidRequestError(
                "No payment method",
                "default_payment_method",
                code="resource_missing",
                http_status=400,
            ),
            402,
        ),
        (stripe.APIConnectionError("Connection lost"), 502),
        (stripe.APIError("Internal server error", http_status=500), 502),
        (stripe.RateLimitError("Rate limited", http_status=429), 502),
        (stripe.IdempotencyError("Request in progress", http_status=409), 502),
    ],
)
def test_only_definite_payment_failures_return_402(
    client, monkeypatch, failure, status
):
    monkeypatch.setattr(
        routes,
        "validate_subscription_upgrade_preview",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(
        routes,
        "modify_stripe_subscription_for_tier",
        AsyncMock(side_effect=failure),
    )
    response = client.post(
        "/credits/subscription",
        json={"tier": "MAX", "preview_id": PREVIEW_ID, "proration_date": DATE},
    )
    assert response.status_code == status
