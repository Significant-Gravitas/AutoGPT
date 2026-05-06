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


_DEFAULT_TIER_PRICES: dict[SubscriptionTier, str | None] = {
    SubscriptionTier.BASIC: None,  # Legacy: stripe-price-id-basic unset by default.
    SubscriptionTier.PRO: "price_pro",
    SubscriptionTier.MAX: "price_max",
    SubscriptionTier.BUSINESS: None,  # Reserved: Business card hidden by default.
}
# Distinct yearly stubs so a routing bug leaking the wrong cycle into a Stripe
# call surfaces as an assertion diff rather than silently passing because both
# cycles share the same stub price.
_DEFAULT_TIER_PRICES_YEARLY: dict[SubscriptionTier, str | None] = {
    SubscriptionTier.BASIC: None,
    SubscriptionTier.PRO: "price_pro_yearly",
    SubscriptionTier.MAX: "price_max_yearly",
    SubscriptionTier.BUSINESS: None,
}


@pytest.fixture(autouse=True)
def _stub_subscription_status_lookups(mocker: pytest_mock.MockFixture) -> None:
    """Stub Stripe price + proration + tier-multiplier lookups used by
    get_subscription_status.

    The POST /credits/subscription handler now returns the full subscription
    status payload from every branch (same-tier, BASIC downgrade, paid→paid
    modify, checkout creation), so every POST test implicitly hits these
    helpers.  Individual tests can override via their own mocker.patch call.
    """

    async def default_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle == "yearly":
            return _DEFAULT_TIER_PRICES_YEARLY.get(tier)
        return _DEFAULT_TIER_PRICES.get(tier)

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=default_price_id,
    )
    mocker.patch(
        "backend.api.features.v1.get_proration_credit_cents",
        new_callable=AsyncMock,
        return_value=0,
    )
    # Default tier-multiplier resolver to the backend defaults so the endpoint
    # never reaches LaunchDarkly during tests.  Individual tests override for
    # LD-override scenarios. get_tier_multipliers returns a string-keyed dict
    # (per its docstring) so we mirror that shape here — passing the
    # enum-keyed _DEFAULT_TIER_MULTIPLIERS directly would silently mismatch
    # the real lookup and let bugs through.
    from backend.copilot.rate_limit import _DEFAULT_TIER_MULTIPLIERS

    mocker.patch(
        "backend.api.features.v1.get_tier_multipliers",
        new_callable=AsyncMock,
        return_value={t.value: v for t, v in _DEFAULT_TIER_MULTIPLIERS.items()},
    )
    # Default billing-cycle resolver to None (treated as monthly) so existing
    # tests don't have to opt into the yearly-aware code path.
    mocker.patch(
        "backend.api.features.v1.get_user_billing_cycle",
        new_callable=AsyncMock,
        return_value=None,
    )
    # Default to a non-None period_end so same-tier short-circuit tests still
    # fire (they assume an active Stripe subscription).  Tests that exercise
    # the admin-granted "no Stripe sub" fall-through override this to None.
    mocker.patch(
        "backend.api.features.v1.get_active_subscription_period_end",
        new_callable=AsyncMock,
        return_value=1_900_000_000,
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
    """GET /credits/subscription returns PRO tier with Stripe prices for all priced tiers."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    prices = {
        SubscriptionTier.BASIC: "price_basic",
        SubscriptionTier.PRO: "price_pro",
        SubscriptionTier.MAX: "price_max",
        SubscriptionTier.BUSINESS: "price_business",
    }
    amounts = {
        "price_basic": 0,
        "price_pro": 1999,
        "price_max": 4999,
        "price_business": 14999,
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return prices.get(tier)

    async def mock_stripe_price_amount(price_id: str) -> int:
        return amounts.get(price_id, 0)

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
    assert data["tier_costs"]["MAX"] == 4999
    assert data["tier_costs"]["BUSINESS"] == 14999
    assert data["tier_costs"]["BASIC"] == 0
    assert "ENTERPRISE" not in data["tier_costs"]
    assert data["proration_credit_cents"] == 500
    # tier_multipliers mirrors the same set of tiers that land in tier_costs,
    # so the frontend never renders a multiplier badge for a hidden row.
    assert set(data["tier_multipliers"].keys()) == set(data["tier_costs"].keys())
    assert data["tier_multipliers"]["BASIC"] == 1.0
    assert data["tier_multipliers"]["PRO"] == 5.0
    assert data["tier_multipliers"]["MAX"] == 20.0
    assert data["tier_multipliers"]["BUSINESS"] == 60.0


def test_get_subscription_status_tier_multipliers_ld_override(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """A LaunchDarkly-overridden tier multiplier flows through the response."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )

    # LD says PRO is 7.5× (instead of the 5× default); other tiers unchanged.
    # Keys are tier enum string values to match get_tier_multipliers'
    # documented return shape (dict[str, float]).
    mocker.patch(
        "backend.api.features.v1.get_tier_multipliers",
        new_callable=AsyncMock,
        return_value={
            "BASIC": 1.0,
            "PRO": 7.5,
            "MAX": 20.0,
            "BUSINESS": 60.0,
            "ENTERPRISE": 60.0,
        },
    )

    response = client.get("/credits/subscription")
    assert response.status_code == 200
    data = response.json()
    # Only tiers that made it into tier_costs get a multiplier (default stub
    # exposes PRO + MAX via _DEFAULT_TIER_PRICES).
    assert data["tier_multipliers"]["PRO"] == 7.5
    assert data["tier_multipliers"]["MAX"] == 20.0
    # BUSINESS has no price configured → hidden from both maps.
    assert "BUSINESS" not in data["tier_multipliers"]


def test_get_subscription_status_defaults_to_no_tier(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When user has no subscription_tier, defaults to NO_TIER (the explicit
    no-active-subscription state)."""
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
    assert data["tier"] == SubscriptionTier.NO_TIER.value
    assert data["monthly_cost"] == 0
    assert data["tier_costs"] == {}
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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


def test_update_subscription_tier_no_tier_no_payment(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription to NO_TIER (cancel) when payment disabled skips Stripe."""
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

    response = client.post("/credits/subscription", json={"tier": "NO_TIER"})

    assert response.status_code == 200
    assert response.json()["url"] == ""


def test_update_subscription_tier_paid_beta_user(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription for paid tier when payment disabled returns 422."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=False,
    )

    response = client.post("/credits/subscription", json={"tier": "PRO"})

    assert response.status_code == 422


def test_update_subscription_tier_currency_mismatch_returns_422(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Stripe rejects a SubscriptionSchedule whose phases mix currencies (e.g.
    GBP-checkout sub trying to schedule a USD-only target Price). The handler
    must convert that into a specific 422 instead of the generic 502 so the
    caller can tell the difference between a currency-config bug and a Stripe
    outage."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.MAX

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
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        side_effect=stripe.InvalidRequestError(
            "The price specified only supports `usd`. This doesn't match the"
            " expected currency: `gbp`.",
            param="currency",
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "billing currency" in detail.lower()
    assert "contact support" in detail.lower()


def test_update_subscription_tier_non_currency_invalid_request_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Locks the contract that *only* currency-mismatch InvalidRequestErrors
    translate to 422 — every other Stripe InvalidRequestError must still
    surface as the generic 502 so that widening the conditional later is
    caught by the suite."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.MAX

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
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        side_effect=stripe.InvalidRequestError(
            "No such price: 'price_does_not_exist'",
            param="items[0][price]",
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 502
    assert "billing currency" not in response.json()["detail"].lower()


def test_update_subscription_tier_creates_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription creates Stripe Checkout Session for paid upgrade."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=False,
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


def test_update_subscription_tier_forwards_yearly_billing_cycle(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """billing_cycle=yearly is forwarded to modify + checkout helpers and the
    target_price_id lookup so the 422 fail-closed branch fires correctly when
    yearly is unconfigured for the tier."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=True,
    )
    mocker.patch(
        "backend.api.features.v1._get_stripe_price_amount",
        new_callable=AsyncMock,
        return_value=1999,
    )
    price_lookup_calls: list[tuple] = []

    async def price_lookup(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        price_lookup_calls.append((tier, billing_cycle))
        return "price_pro_yearly"

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "billing_cycle": "yearly",
        },
    )

    assert response.status_code == 200
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.PRO, "yearly")
    assert (SubscriptionTier.PRO, "yearly") in price_lookup_calls


def test_update_subscription_tier_yearly_unconfigured_returns_422(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When yearly price is not configured for the tier, the route fails closed
    with a 422 instead of silently using the monthly price."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
        "backend.api.features.v1._get_stripe_price_amount",
        new_callable=AsyncMock,
        return_value=1999,
    )

    async def price_lookup(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        if billing_cycle == "yearly":
            return None
        return "price_pro_monthly"

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "billing_cycle": "yearly",
        },
    )

    assert response.status_code == 422
    modify_mock.assert_not_awaited()


def test_update_subscription_tier_creates_checkout_with_yearly_billing(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When no active sub, create_subscription_checkout receives billing_cycle=yearly."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
        "backend.api.features.v1._get_stripe_price_amount",
        new_callable=AsyncMock,
        return_value=1999,
    )

    async def price_lookup(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        return "price_pro_yearly"

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )
    mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=False,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
        return_value="https://checkout.stripe.com/pay/cs_test_yearly",
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "billing_cycle": "yearly",
        },
    )

    assert response.status_code == 200
    checkout_mock.assert_awaited_once()
    kwargs = checkout_mock.await_args.kwargs
    assert kwargs["billing_cycle"] == "yearly"
    assert kwargs["tier"] == SubscriptionTier.PRO


def test_update_subscription_tier_rejects_open_redirect(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription rejects success/cancel URLs outside the frontend origin."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
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


def test_update_subscription_tier_no_tier_with_payment_schedules_cancel_and_does_not_update_db(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Cancelling to NO_TIER schedules Stripe cancellation at period end.

    The DB tier must NOT be updated immediately — the customer.subscription.deleted
    webhook fires at period end and downgrades to NO_TIER then.
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

    response = client.post("/credits/subscription", json={"tier": "NO_TIER"})

    assert response.status_code == 200
    mock_cancel.assert_awaited_once()
    mock_set_tier.assert_not_awaited()


def test_update_subscription_tier_no_tier_cancel_failure_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Cancelling to NO_TIER returns 502 with a generic error (no Stripe detail leakage)."""
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

    response = client.post("/credits/subscription", json={"tier": "NO_TIER"})

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

    async def price_id_with_business(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return {
            **_DEFAULT_TIER_PRICES,
            SubscriptionTier.BUSINESS: "price_business",
        }.get(tier)

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_id_with_business,
    )
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
    modify_mock.assert_awaited_once_with(
        TEST_USER_ID, SubscriptionTier.BUSINESS, "monthly"
    )
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_max_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription from PRO→MAX modifies the existing subscription."""
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
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == ""
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.MAX, "monthly")
    checkout_mock.assert_not_awaited()


def test_update_subscription_tier_no_active_sub_falls_through_to_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Any tier change from a user with no active Stripe sub goes through Checkout.

    Admin-granted users (no Stripe sub yet) and never-paid users follow the
    exact same path: modify returns False → Checkout to set up payment. The
    endpoint has no admin-specific branch — admin tier grants happen out-of-band
    via the admin portal, not this user-facing route.
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
        return_value="https://checkout.stripe.com/pay/cs_test_no_sub",
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 200
    assert response.json()["url"] == "https://checkout.stripe.com/pay/cs_test_no_sub"
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.MAX, "monthly")
    # No DB-flip — payment must be collected via Checkout regardless of direction.
    set_tier_mock.assert_not_awaited()
    checkout_mock.assert_awaited_once()


def test_update_subscription_tier_priced_basic_no_sub_falls_through_to_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Once stripe-price-id-basic is configured, a BASIC user without an active sub
    must hit Stripe Checkout rather than being silently set_subscription_tier'd."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return {
            SubscriptionTier.BASIC: "price_basic",
            SubscriptionTier.PRO: "price_pro",
            SubscriptionTier.MAX: "price_max",
            SubscriptionTier.BUSINESS: "price_business",
        }.get(tier)

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
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
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
        return_value="https://checkout.stripe.com/pay/cs_test_priced_basic",
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
    assert (
        response.json()["url"] == "https://checkout.stripe.com/pay/cs_test_priced_basic"
    )
    # Priced-BASIC user without an active sub: must NOT silently flip DB tier —
    # they need to set up payment via Checkout.
    set_tier_mock.assert_not_awaited()
    checkout_mock.assert_awaited_once()
    # modify is still called first; returning False just means "no active sub".
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.PRO, "monthly")


def test_update_subscription_tier_target_without_ld_price_returns_422(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Paid target with no LD-configured Stripe price must fail fast with 422.

    Matches the UI hiding: if `stripe-price-id-pro` resolves to None we can't
    start a Checkout Session anyway, and we don't want to surface an opaque
    Stripe error mid-flow. The handler rejects the request before touching
    Stripe at all.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BASIC

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return None  # Neither BASIC nor PRO have an LD price.

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
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
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

    assert response.status_code == 422
    assert "not available" in response.json()["detail"].lower()
    checkout_mock.assert_not_awaited()
    modify_mock.assert_not_awaited()


def test_update_subscription_tier_pro_to_max_card_declined_returns_402(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Pro→Max upgrade where Stripe raises CardError must return HTTP 402.

    The "tier stays on Pro after CardError" invariant is verified at the
    credit.py layer (see test_modify_stripe_subscription_for_tier_pro_to_max
    _card_decline_does_not_flip_tier); this test covers the route's HTTP
    surface only.
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
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        side_effect=stripe.CardError(
            "Your card was declined.", param="card", code="card_declined"
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 402
    assert "card was declined" in response.json()["detail"].lower()
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.MAX, "monthly")


def test_update_subscription_tier_pro_to_max_authentication_required_returns_402(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """SCA-required cards raise CardError(code='authentication_required'). The
    handler must surface a different message than the plain decline path so EU
    users don't try a different card when their existing one only needs 3DS."""
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
        side_effect=stripe.CardError(
            "Your card was declined.",
            param="card",
            code="authentication_required",
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 402
    detail = response.json()["detail"].lower()
    assert "authentication" in detail
    # Must NOT use the generic decline copy — that would tell the user to
    # change cards when the card itself is fine.
    assert "card was declined" not in detail


def test_update_subscription_tier_pro_to_max_subscription_payment_intent_requires_action_returns_402(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Subscription.modify under error_if_incomplete raises CardError with
    code='subscription_payment_intent_requires_action' (not the raw
    authentication_required from PaymentIntent.confirm). The SCA branch must
    cover both codes, otherwise the user gets the generic "card was declined"
    copy and would re-enter a card that's already fine."""
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
        side_effect=stripe.CardError(
            "Payment for this subscription requires additional user action"
            " before it can be completed successfully.",
            param=None,
            code="subscription_payment_intent_requires_action",
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 402
    detail = response.json()["detail"].lower()
    assert "authentication" in detail
    assert "card was declined" not in detail


def test_update_subscription_tier_pro_to_max_no_payment_method_returns_402(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Customer with no default payment method: error_if_incomplete makes Stripe
    raise InvalidRequestError (not CardError). The handler must map that to 402
    so the UI prompts to add a card instead of the generic 502 outage copy."""
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
        side_effect=stripe.InvalidRequestError(
            "This customer has no attached payment source or default payment"
            " method. Please consider adding a default payment method.",
            param="default_payment_method",
            code="resource_missing",
        ),
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "MAX",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
        },
    )

    assert response.status_code == 402
    detail = response.json()["detail"].lower()
    assert "no payment method" in detail
    assert "add a payment method" in detail


def test_update_subscription_tier_paid_to_paid_stripe_error_returns_502(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /credits/subscription returns 502 when Stripe modification fails."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def price_id_with_business(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return {
            **_DEFAULT_TIER_PRICES,
            SubscriptionTier.BUSINESS: "price_business",
        }.get(tier)

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_id_with_business,
    )
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


def test_update_subscription_tier_no_tier_no_stripe_subscription(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Cancelling to NO_TIER when no Stripe subscription exists updates DB tier directly.

    Admin-granted paid tiers have no associated Stripe subscription.  When such a
    user requests a self-service cancel, cancel_stripe_subscription returns False
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

    response = client.post("/credits/subscription", json={"tier": "NO_TIER"})

    assert response.status_code == 200
    assert response.json()["url"] == ""
    cancel_mock.assert_awaited_once_with(TEST_USER_ID)
    # DB tier must be updated immediately — no webhook will fire for a missing sub
    set_tier_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.NO_TIER)


def test_get_subscription_status_includes_pending_tier(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /credits/subscription exposes pending_tier and pending_tier_effective_at."""
    import datetime as dt

    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    effective_at = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
        return_value=(SubscriptionTier.PRO, effective_at, "monthly"),
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["pending_tier"] == "PRO"
    assert data["pending_tier_effective_at"] is not None
    assert data["pending_billing_cycle"] == "monthly"


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

    async def price_id_with_business(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return {
            **_DEFAULT_TIER_PRICES,
            SubscriptionTier.BUSINESS: "price_business",
        }.get(tier)

    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_id_with_business,
    )
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
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.PRO, "monthly")
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


def test_get_subscription_status_yearly_only_tier_visible(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When LD configures only the yearly price for a tier, the row must still
    appear in tier_costs (with monthly cost 0) and the yearly cost surfaces
    via tier_costs_yearly so the frontend can render it."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def price_lookup(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        # PRO has only a yearly price configured; MAX has both; others none.
        if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
            return "price_pro_yearly"
        if tier == SubscriptionTier.MAX and billing_cycle == "monthly":
            return "price_max_monthly"
        if tier == SubscriptionTier.MAX and billing_cycle == "yearly":
            return "price_max_yearly"
        return None

    amounts = {
        "price_pro_yearly": 19_999,
        "price_max_monthly": 4999,
        "price_max_yearly": 49_999,
    }

    async def stripe_amount(price_id: str) -> int:
        return amounts.get(price_id, 0)

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )
    mocker.patch(
        "backend.api.features.v1._get_stripe_price_amount",
        side_effect=stripe_amount,
    )
    mocker.patch(
        "backend.api.features.v1.get_user_billing_cycle",
        new_callable=AsyncMock,
        return_value="yearly",
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    # PRO row still visible despite no monthly price.
    assert "PRO" in data["tier_costs"]
    assert data["tier_costs"]["PRO"] == 0
    assert data["tier_costs_yearly"]["PRO"] == 19_999
    # MAX has both cycles.
    assert data["tier_costs"]["MAX"] == 4999
    assert data["tier_costs_yearly"]["MAX"] == 49_999
    # User is on yearly Pro → monthly_cost reflects the yearly price.
    assert data["billing_cycle"] == "yearly"
    assert data["monthly_cost"] == 19_999


def test_get_subscription_status_yearly_user_both_cycles_uses_yearly_cost(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """When LD has both monthly and yearly prices and the user is on yearly,
    monthly_cost in the response reflects the yearly price (the user's actual
    cost), not the monthly equivalent."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def price_lookup(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if tier == SubscriptionTier.PRO:
            return (
                "price_pro_yearly" if billing_cycle == "yearly" else "price_pro_monthly"
            )
        return None

    amounts = {
        "price_pro_monthly": 1999,
        "price_pro_yearly": 19_999,
    }

    async def stripe_amount(price_id: str) -> int:
        return amounts.get(price_id, 0)

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )
    mocker.patch(
        "backend.api.features.v1._get_stripe_price_amount",
        side_effect=stripe_amount,
    )
    mocker.patch(
        "backend.api.features.v1.get_user_billing_cycle",
        new_callable=AsyncMock,
        return_value="yearly",
    )

    response = client.get("/credits/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["billing_cycle"] == "yearly"
    assert data["tier_costs"]["PRO"] == 1999
    assert data["tier_costs_yearly"]["PRO"] == 19_999
    # User pays the yearly price — surface it via monthly_cost so the UI shows
    # the user's real recurring cost rather than the unrelated monthly equivalent.
    assert data["monthly_cost"] == 19_999


def test_update_subscription_tier_same_tier_cycle_change_routes_to_modify(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """A monthly Pro user posting {tier:'PRO', billing_cycle:'yearly'} must
    NOT short-circuit through release_pending_subscription_schedule — that
    would no-op the cycle change. Route through modify_stripe_subscription_for_tier
    so Stripe swaps to the yearly price."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def price_lookup(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if tier == SubscriptionTier.PRO:
            return (
                "price_pro_yearly" if billing_cycle == "yearly" else "price_pro_monthly"
            )
        return None

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
        "backend.api.features.v1.get_subscription_price_id",
        side_effect=price_lookup,
    )
    # User is currently on monthly Pro.
    mocker.patch(
        "backend.api.features.v1.get_user_billing_cycle",
        new_callable=AsyncMock,
        return_value="monthly",
    )
    release_mock = mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        new_callable=AsyncMock,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=True,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "billing_cycle": "yearly",
        },
    )

    assert response.status_code == 200
    release_mock.assert_not_awaited()
    modify_mock.assert_awaited_once_with(TEST_USER_ID, SubscriptionTier.PRO, "yearly")


def test_update_subscription_tier_same_tier_same_cycle_still_releases_pending(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """A yearly Pro user posting {tier:'PRO', billing_cycle:'yearly'} keeps the
    "stay on my current tier + cycle" semantics — release any pending change."""
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    mocker.patch(
        "backend.api.features.v1.get_user_billing_cycle",
        new_callable=AsyncMock,
        return_value="yearly",
    )
    release_mock = mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        new_callable=AsyncMock,
        return_value=True,
    )
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/subscription",
        json={
            "tier": "PRO",
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "billing_cycle": "yearly",
        },
    )

    assert response.status_code == 200
    release_mock.assert_awaited_once_with(TEST_USER_ID)
    modify_mock.assert_not_awaited()


def test_update_subscription_tier_same_tier_no_stripe_sub_falls_through_to_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
) -> None:
    """Admin-granted tier (DB tier set, no active Stripe subscription) posting
    their *current* tier must fall through to the Checkout flow, not short-
    circuit through release_pending_subscription_schedule. Otherwise "start
    paying for my current tier" silently no-ops for these users.
    """
    mock_user = Mock()
    mock_user.subscription_tier = SubscriptionTier.PRO

    mocker.patch(
        "backend.api.features.v1.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    )
    # No active Stripe subscription — period_end is None.
    mocker.patch(
        "backend.api.features.v1.get_active_subscription_period_end",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.v1.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )
    release_mock = mocker.patch(
        "backend.api.features.v1.release_pending_subscription_schedule",
        new_callable=AsyncMock,
    )
    # Admin-granted user has no active Stripe sub, so the modify path returns
    # False (no sub to mutate). Mock explicitly so the test doesn't reach the
    # real backend.data.credit.modify_stripe_subscription_for_tier (which
    # would try to read from Prisma + call Stripe).
    modify_mock = mocker.patch(
        "backend.api.features.v1.modify_stripe_subscription_for_tier",
        new_callable=AsyncMock,
        return_value=False,
    )
    checkout_mock = mocker.patch(
        "backend.api.features.v1.create_subscription_checkout",
        new_callable=AsyncMock,
        return_value="https://checkout.example.com/sess_admingranted",
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
    assert data["url"] == "https://checkout.example.com/sess_admingranted"
    release_mock.assert_not_awaited()
    modify_mock.assert_awaited_once()
    checkout_mock.assert_awaited_once()
