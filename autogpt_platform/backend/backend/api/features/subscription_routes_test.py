"""Tests for subscription tier API endpoints."""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import SubscriptionTier

from .v1 import v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)

client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def setup_auth(app: fastapi.FastAPI):
    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": TEST_USER_ID, "role": "user", "email": "test@example.com"}

    app.dependency_overrides[get_jwt_payload] = override_get_jwt_payload


def teardown_auth(app: fastapi.FastAPI):
    app.dependency_overrides.clear()


def test_get_subscription_status_pro(
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription returns PRO tier with Stripe price for a PRO user."""
    setup_auth(app)
    try:
        mock_user = Mock()
        mock_user.subscription_tier = SubscriptionTier.PRO

        mock_price = Mock()
        mock_price.unit_amount = 1999  # $19.99

        async def mock_price_id(tier: SubscriptionTier) -> str | None:
            return "price_pro" if tier == SubscriptionTier.PRO else None

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
            "backend.api.features.v1.stripe.Price.retrieve",
            return_value=mock_price,
        )

        response = client.get("/credits/subscription")

        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "PRO"
        assert data["monthly_cost"] == 1999
        assert data["tier_costs"]["PRO"] == 1999
        assert data["tier_costs"]["BUSINESS"] == 0
        assert data["tier_costs"]["FREE"] == 0
    finally:
        teardown_auth(app)


def test_get_subscription_status_defaults_to_free(
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription when subscription_tier is None defaults to FREE."""
    setup_auth(app)
    try:
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
    finally:
        teardown_auth(app)


def test_update_subscription_tier_free_no_payment(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription to FREE tier when payment disabled skips Stripe."""
    setup_auth(app)
    try:
        mock_user = Mock()
        mock_user.subscription_tier = SubscriptionTier.PRO

        async def mock_feature_disabled(*args, **kwargs):
            return False

        async def mock_set_tier(*args, **kwargs):
            pass

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
            side_effect=mock_set_tier,
        )

        response = client.post("/credits/subscription", json={"tier": "FREE"})

        assert response.status_code == 200
        assert response.json()["url"] == ""
    finally:
        teardown_auth(app)


def test_update_subscription_tier_paid_beta_user(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for paid tier when payment disabled sets tier directly."""
    setup_auth(app)
    try:
        mock_user = Mock()
        mock_user.subscription_tier = SubscriptionTier.FREE

        async def mock_feature_disabled(*args, **kwargs):
            return False

        async def mock_set_tier(*args, **kwargs):
            pass

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
            side_effect=mock_set_tier,
        )

        response = client.post("/credits/subscription", json={"tier": "PRO"})

        assert response.status_code == 200
        assert response.json()["url"] == ""
    finally:
        teardown_auth(app)


def test_update_subscription_tier_paid_requires_urls(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for paid tier without success/cancel URLs returns 422."""
    setup_auth(app)
    try:
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
    finally:
        teardown_auth(app)


def test_update_subscription_tier_creates_checkout(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription creates Stripe Checkout Session for paid upgrade."""
    setup_auth(app)
    try:
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
                "success_url": "https://app.example.com/success",
                "cancel_url": "https://app.example.com/cancel",
            },
        )

        assert response.status_code == 200
        assert response.json()["url"] == "https://checkout.stripe.com/pay/cs_test_abc"
    finally:
        teardown_auth(app)


def test_update_subscription_tier_free_with_payment_cancels_stripe(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Downgrading to FREE cancels active Stripe subscription when payment is enabled."""
    setup_auth(app)
    try:
        mock_user = Mock()
        mock_user.subscription_tier = SubscriptionTier.PRO

        async def mock_feature_enabled(*args, **kwargs):
            return True

        mock_cancel = mocker.patch(
            "backend.api.features.v1.cancel_stripe_subscription",
            new_callable=AsyncMock,
        )

        async def mock_set_tier(*args, **kwargs):
            pass

        mocker.patch(
            "backend.api.features.v1.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        )
        mocker.patch(
            "backend.api.features.v1.set_subscription_tier",
            side_effect=mock_set_tier,
        )
        mocker.patch(
            "backend.api.features.v1.is_feature_enabled",
            side_effect=mock_feature_enabled,
        )

        response = client.post("/credits/subscription", json={"tier": "FREE"})

        assert response.status_code == 200
        mock_cancel.assert_awaited_once()
    finally:
        teardown_auth(app)
