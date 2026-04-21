"""Tests for subscription tier API endpoints."""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
import stripe
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import SubscriptionTier

from .v1 import _validate_checkout_redirect_url, v1_router

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
TEST_FRONTEND_ORIGIN = "https://app.example.com"


@pytest.fixture()
def client() -> fastapi.testclient.TestClient:
    """Fresh FastAPI app + client per test with auth override applied.

    Using a fixture avoids the leaky global-app + try/finally teardown pattern:
    if a test body raises before teardown_auth runs, dependency overrides were
    previously leaking into subsequent tests.
    """
    app = fastapi.FastAPI()
    app.include_router(v1_router)

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": TEST_USER_ID, "role": "user", "email": "test@example.com"}

    app.dependency_overrides[get_jwt_payload] = override_get_jwt_payload
    try:
        yield fastapi.testclient.TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _configure_frontend_origin(mocker: pytest_mock.MockFixture) -> None:
    """Pin the configured frontend origin used by the open-redirect guard."""
    from backend.api.features import v1 as v1_mod

    mocker.patch.object(
        v1_mod.settings.config, "frontend_base_url", TEST_FRONTEND_ORIGIN
    )


@pytest.fixture(autouse=True)
def _stub_pending_subscription_change(mocker: pytest_mock.MockFixture) -> None:
    """Default pending-change lookup to None so tests don't hit Stripe/DB.

    Individual tests can override via their own mocker.patch call.
    """
    mocker.patch(
        "backend.api.features.v1.get_pending_subscription_change",
        new_callable=AsyncMock,
        return_value=None,
    )


@pytest.fixture(autouse=True)
def _stub_subscription_status_lookups(mocker: pytest_mock.MockFixture) -> None:
    """Stub Stripe price + proration lookups used by get_subscription_status.

    The POST /credits/subscription handler now returns the full subscription
    status payload from every branch (same-tier, FREE downgrade, paid→paid
    modify, checkout creation), so every POST test implicitly hits these
    helpers.  Individual tests can override via their own mocker.patch call.
    """
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        # Valid URLs matching the configured frontend origin
        (f"{TEST_FRONTEND_ORIGIN}/success", True),
        (f"{TEST_FRONTEND_ORIGIN}/cancel?ref=abc", True),
        # Wrong origin
        ("https://evil.example.org/phish", False),
        ("https://evil.example.org", False),
        # @ in URL (user:pass@host attack)
        (f"https://attacker.example.com@{TEST_FRONTEND_ORIGIN}/ok", False),
        # Backslash normalisation attack
        (f"https:{TEST_FRONTEND_ORIGIN}\\@attacker.example.com/ok", False),
        # javascript: scheme
        ("javascript:alert(1)", False),
        # Empty string
        ("", False),
        # Control character (U+0000) in URL
        (f"{TEST_FRONTEND_ORIGIN}/ok\x00evil", False),
        # Non-http scheme
        (f"ftp://{TEST_FRONTEND_ORIGIN}/ok", False),
    ],
)
def test_validate_checkout_redirect_url(
    url: str,
    expected: bool,
    mocker: pytest_mock.MockFixture,
) -> None:
    """_validate_checkout_redirect_url rejects adversarial inputs."""
    from backend.api.features import v1 as v1_mod

    mocker.patch.object(
        v1_mod.settings.config, "frontend_base_url", TEST_FRONTEND_ORIGIN
    )
    assert _validate_checkout_redirect_url(url) is expected


def test_get_subscription_status_pro(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription returns PRO tier with Stripe price for a PRO user."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return "price_pro" if tier == SubscriptionTier.PRO else None

    async def mock_stripe_price_amount(price_id: str) -> int:
        return 1999 if price_id == "price_pro" else 0

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=mock_price_id,
    )
    mocker.patch(
        "backend.api.features.v1._get_stripe_price_amount",
        side_effect=mock_stripe_price_amount,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=500,
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "PRO"
    assert data["monthly_cost"] == 1999
    assert data["tier_costs"]["PRO"] == 1999
    assert data["tier_costs"]["BUSINESS"] == 0
    assert data["tier_costs"]["FREE"] == 0
    assert data["proration_credit_cents"] == 500


def test_get_subscription_status_defaults_to_free(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription when subscription_tier is None defaults to FREE."""
    mock_user = Mock()
    mock_user.subscription_tier = None

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == SubscriptionTier.FREE.value
    assert data["monthly_cost"] == 0
    assert data["tier_costs"] == {
        "FREE": 0,
        "PRO": 0,
        "BUSINESS": 0,
        "ENTERPRISE": 0,
    }
    assert data["proration_credit_cents"] == 0


def test_get_subscription_status_stripe_error_falls_back_to_zero(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription returns cost=0 when Stripe price fetch fails (returns None).

    _get_stripe_price_amount returns None on StripeError so the error state is
    not cached.  The endpoint must treat None as 0 — not raise or return invalid data.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return "price_pro" if tier == SubscriptionTier.PRO else None

    async def mock_stripe_price_amount_none(price_id: str) -> None:
        return None

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=mock_price_id,
    )
    mocker.patch(
        "backend.api.features.v1._get_stripe_price_amount",
        side_effect=mock_stripe_price_amount_none,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "PRO"
    # When Stripe returns None, cost falls back to 0
    assert data["monthly_cost"] == 0
    assert data["tier_costs"]["PRO"] == 0


def test_update_subscription_tier_free_no_payment(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription to FREE tier when payment disabled skips Stripe."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def mock_feature_disabled(*args, **kwargs):
        return False

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_disabled,
    )
    mocker.patch(
        "backend.api.features.v1.set_subscription_tier",
        new_callable=AsyncMock,
    )

    response = client.post("/credits/subscription", json={"tier": "FREE"})

    assert response.status_code == 200
    assert response.json()["url"] == ""


def test_update_subscription_tier_paid_beta_user(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for paid tier when payment disabled returns 422."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.FREE

    async def mock_feature_disabled(*args, **kwargs):
        return False

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_disabled,
    )

    response = client.post("/credits/subscription", json={"tier": "PRO"})

    assert response.status_code == 422
    assert "not available" in response.json()["detail"]


def test_update_subscription_tier_paid_requires_urls(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for paid tier without success/cancel URLs returns 422."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.FREE

    async def mock_feature_enabled(*args, **kwargs):
        return True

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_enabled,
    )

    response = client.post("/credits/subscription", json={"tier": "PRO"})

    assert response.status_code == 422


def test_update_subscription_tier_creates_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription creates Stripe Checkout Session for paid upgrade."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.FREE

    async def mock_feature_enabled(*args, **kwargs):
        return True

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_enabled,
    )
    mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
        return_value="https://checkout.stripe.com/pay/cs_test_abc",
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == "https://checkout.stripe.com/pay/cs_test_abc"


def test_update_subscription_tier_rejects_open_redirect(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription rejects success/cancel URLs outside the frontend origin."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.FREE

    async def mock_feature_enabled(*args, **kwargs):
        return True

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_enabled,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": "https://evil.example.org/phish",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 422
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_enterprise_blocked(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """ENTERPRISE users cannot self-service change tiers — must get 403."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.ENTERPRISE

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    set_tier_mock = mocker.patch(
        "backend.api.features.v1.set_subscription_tier",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 403
    set_tier_mock.assert_not_awaited()


def test_update_subscription_tier_same_tier_releases_pending_change(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for the user's current tier releases any pending change.

    "Stay on my current tier" — the collapsed replacement for the old
    /credits/subscription/cancel-pending route. Always calls
    release_pending_subscription_schedule (idempotent when nothing is pending)
    and returns the refreshed status with url="". Never creates a Checkout
    Session — that would double-charge a user who double-clicks their own tier.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    release_mock = mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        new_callable=AsyncMock,
        return_value=True,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )
    feature_mock = mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "BUSINESS",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "BUSINESS"
    assert data["url"] == ""
    release_mock.assert_awaited_once_with(TEST_USER_ID)
    checkout_mock.assert_not_awaited()
    # Same-tier branch short-circuits before the payment-flag check.
    feature_mock.assert_not_awaited()


def test_update_subscription_tier_same_tier_no_pending_change_returns_status(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Same-tier request when nothing is pending still returns status with url=""."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    release_mock = mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        new_callable=AsyncMock,
        return_value=False,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "PRO"
    assert data["url"] == ""
    assert data["pending_tier"] is None
    release_mock.assert_awaited_once_with(TEST_USER_ID)
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_same_tier_stripe_error_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Same-tier request surfaces a 502 when Stripe release fails.

    Carries forward the error contract from the removed
    /credits/subscription/cancel-pending route so clients keep seeing 502 for
    transient Stripe failures.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        side_effect=stripe.StripeError("network"),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "BUSINESS",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 502
    assert "contact support" in response.json()["detail"].lower()


def test_update_subscription_tier_free_with_payment_schedules_cancel_and_does_not_update_db(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Downgrading to FREE schedules Stripe cancellation at period end.

    The DB tier must NOT be updated immediately — the customer.subscription.deleted
    webhook fires at period end and downgrades to FREE then.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def mock_feature_enabled(*args, **kwargs):
        return True

    mock_cancel = mocker.patch(
        "backend.api.features.v1.cancel_stripe_subscription",
        new_callable=AsyncMock,
    )
    mock_set_tier = mocker.patch(
        "backend.api.features.v1.set_subscription_tier",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_enabled,
    )

    response = client.post("/credits/subscription", json={"tier": "FREE"})

    assert response.status_code == 200
    mock_cancel.assert_awaited_once()
    mock_set_tier.assert_not_awaited()


def test_update_subscription_tier_free_cancel_failure_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Downgrading to FREE returns 502 with a generic error (no Stripe detail leakage)."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def mock_feature_enabled(*args, **kwargs):
        return True

    mocker.patch(
        "backend.api.features.v1.cancel_stripe_subscription",
        side_effect=stripe.StripeError(
            "You did not provide an API key — internal detail that must not leak"
        ),
    )
    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        side_effect=mock_feature_enabled,
    )

    response = client.post("/credits/subscription", json={"tier": "FREE"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    # The raw Stripe error message must not appear in the client-facing detail.
    assert "API key" not in detail
    assert "contact support" in detail.lower()


def test_stripe_webhook_unconfigured_secret_returns_503(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Stripe webhook endpoint returns 503 when STRIPE_WEBHOOK_SECRET is not set.

    An empty webhook secret allows HMAC forgery: an attacker can compute a valid
    HMAC signature over the same empty key. The handler must reject all requests
    when the secret is unconfigured rather than proceeding with signature verification.
    """
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="",
    )
    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=fake"},
    )
    assert response.status_code == 503


def test_stripe_webhook_dispatches_subscription_events(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/stripe_webhook routes customer.subscription.created to sync handler."""
    stripe_sub_obj = {
        "id": "sub_test",
        "customer": "cus_test",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro"}}]},
    }
    event = {
        "type": "customer.subscription.created",
        "data": {"object": stripe_sub_obj},
    }

    # Ensure the webhook secret guard passes (non-empty secret required).
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.stripe.Webhook.construct_event",
        return_value=event,
    )
    sync_mock = mocker.patch(
        "backend.api.features.v1.sync_subscription_from_stripe",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=abc"},
    )

    assert response.status_code == 200
    sync_mock.assert_awaited_once_with(stripe_sub_obj)


def test_stripe_webhook_dispatches_invoice_payment_failed(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/stripe_webhook routes invoice.payment_failed to the failure handler."""
    invoice_obj = {
        "customer": "cus_test",
        "subscription": "sub_test",
        "amount_due": 1999,
    }
    event = {
        "type": "invoice.payment_failed",
        "data": {"object": invoice_obj},
    }

    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.stripe.Webhook.construct_event",
        return_value=event,
    )
    failure_mock = mocker.patch(
        "backend.api.features.v1.handle_subscription_payment_failure",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=abc"},
    )

    assert response.status_code == 200
    failure_mock.assert_awaited_once_with(invoice_obj)


def test_update_subscription_tier_paid_to_paid_modifies_subscription(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription modifies existing subscription for paid→paid changes."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=True,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "BUSINESS",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == ""
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.BUSINESS)
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_admin_granted_paid_to_paid_updates_db_directly(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Admin-granted paid tier users are NOT sent to Stripe checkout for paid→paid changes.

    When modify_stripe_subscription_for_tier returns False (no Stripe subscription
    found — admin-granted tier), the endpoint must update the DB tier directly and
    return 200 with url="", rather than falling through to Checkout Session creation.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    # Return False = no Stripe subscription (admin-granted tier)
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=False,
    )
    set_tier_mock = mocker.patch(
        "backend.api.features.v1.set_subscription_tier",
        new_callable=AsyncMock,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "BUSINESS",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == ""
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.BUSINESS)
    # DB tier updated directly — no Stripe Checkout Session created
    set_tier_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.BUSINESS)
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_paid_to_paid_stripe_error_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription returns 502 when Stripe modification fails."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        side_effect=stripe.StripeError("connection error"),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "BUSINESS",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 502


def test_update_subscription_tier_free_no_stripe_subscription(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Downgrading to FREE when no Stripe subscription exists updates DB tier directly.

    Admin-granted paid tiers have no associated Stripe subscription.  When such a
    user requests a self-service downgrade, cancel_stripe_subscription returns False
    (nothing to cancel), so the endpoint must immediately call set_subscription_tier
    rather than waiting for a webhook that will never arrive.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    # Simulate no active Stripe subscriptions — returns False
    cancel_mock = mocker.patch(
        "backend.api.features.v1.cancel_stripe_subscription",
        new_callable=AsyncMock,
        return_value=False,
    )
    set_tier_mock = mocker.patch(
        "backend.api.features.v1.set_subscription_tier",
        new_callable=AsyncMock,
    )

    response = client.post("/credits/subscription", json={"tier": "FREE"})

    assert response.status_code == 200
    assert response.json()["url"] == ""
    cancel_mock.assert_awaited_once_with(TEST_USER_ID)
    # DB tier must be updated immediately — no webhook will fire for a missing sub
    set_tier_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.FREE)


def test_get_subscription_status_includes_pending_tier(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription exposes pending_tier and pending_tier_effective_at."""
    import datetime as dt

    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    effective_at = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return None

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=mock_price_id,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.v1.get_pending_subscription_change",
        new_callable=AsyncMock,
        return_value=(SubscriptionTier.PRO, effective_at),
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["pending_tier"] == "PRO"
    assert data["pending_tier_effective_at"] is not None


def test_get_subscription_status_no_pending_tier(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When no pending change exists the response omits pending_tier."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.v1.get_pending_subscription_change",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["pending_tier"] is None
    assert data["pending_tier_effective_at"] is None


def test_update_subscription_tier_downgrade_paid_to_paid_schedules(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """A BUSINESS→PRO downgrade request dispatches to modify_stripe_subscription_for_tier."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=True,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == ""
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.PRO)
    checkout_mock.assert_not_awaited()


def test_stripe_webhook_dispatches_subscription_schedule_released(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """subscription_schedule.released routes to sync_subscription_schedule_from_stripe."""
    schedule_obj = {"id": "sub_sched_1", "subscription": "sub_pro"}
    event = {
        "type": "subscription_schedule.released",
        "data": {"object": schedule_obj},
    }
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.stripe.Webhook.construct_event",
        return_value=event,
    )
    sync_mock = mocker.patch(
        "backend.api.features.v1.sync_subscription_schedule_from_stripe",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=abc"},
    )

    assert response.status_code == 200
    sync_mock.assert_awaited_once_with(schedule_obj)


def test_stripe_webhook_ignores_subscription_schedule_updated(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """subscription_schedule.updated must NOT dispatch: our own
    SubscriptionSchedule.create/.modify calls fire this event and would
    otherwise loop redundant traffic through the sync handler. State
    transitions we care about surface via .released/.completed, and phase
    advance to a new price is already covered by customer.subscription.updated.
    """
    schedule_obj = {"id": "sub_sched_1", "subscription": "sub_pro"}
    event = {
        "type": "subscription_schedule.updated",
        "data": {"object": schedule_obj},
    }
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.stripe.Webhook.construct_event",
        return_value=event,
    )
    sync_mock = mocker.patch(
        "backend.api.features.v1.sync_subscription_schedule_from_stripe",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=abc"},
    )

    assert response.status_code == 200
    sync_mock.assert_not_awaited()
