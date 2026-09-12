from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

from backend.api.features import subscription_trial_routes as routes
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer


@pytest.fixture(autouse=True)
def billing_return_origin(monkeypatch):
    settings = routes.Settings()
    settings.config.frontend_base_url = "https://platform.example.com"
    monkeypatch.setattr(routes, "Settings", lambda: settings)


@pytest.fixture
def trial() -> TrialState:
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="api-v1",
            new_users_from=now - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id=None,
        checkout_attempt=0,
        success_url="https://example.com/ok",
        cancel_url="https://example.com/no",
        checkout_metadata={},
        status="checkout_pending",
        card_verified_at=None,
        started_at=None,
        ends_at=None,
        consumed_at=None,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


@pytest.mark.asyncio
async def test_disabled_flag_hides_pending_enrollment(trial):
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(routes, "get_trial_offer", AsyncMock(return_value=None)),
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=False)
        ),
    ):
        status = await routes.get_trial_status(trial.user_id)
    assert not status.eligible
    assert status.status == "checkout_pending"


@pytest.mark.asyncio
async def test_status_never_exposes_internal_spend_or_stripe_identifiers(trial):
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(routes, "get_trial_offer", AsyncMock(return_value=trial.offer)),
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=True)
        ),
    ):
        status = await routes.get_trial_status(trial.user_id)
    public = status.model_dump_json()
    for private in (
        "customer_id",
        "price_id",
        "daily_cost_limit",
        "total_cost_limit",
        "user_id",
    ):
        assert private not in public
    assert status.onboarding_credits_previously_received


@pytest.mark.asyncio
async def test_status_exposes_safe_rejection_reason(trial):
    trial = trial.model_copy(
        update={"status": "canceled", "rejection_reason": "intro_offer_already_used"}
    )
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=False)
        ),
    ):
        status = await routes.get_trial_status(trial.user_id)
    assert status.model_dump().get("rejection_reason") == "intro_offer_already_used"


@pytest.mark.asyncio
async def test_checkout_uses_authenticated_identity_and_server_return_urls(trial):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_user_id] = lambda: trial.user_id
    app.dependency_overrides[routes.enforce_subscription_status_rate_limit] = (
        lambda: None
    )
    with patch.object(
        routes,
        "create_trial_checkout",
        AsyncMock(return_value="https://checkout.stripe.com/test"),
    ) as checkout:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="https://example.com"
        ) as client:
            response = await client.post(
                "/credits/trial",
                json={"offer_token": trial.offer.token, "return_to": "onboarding"},
                headers={"X-Datafast-Visitor-Id": "visitor-1"},
            )
    assert response.status_code == 200
    params = checkout.await_args.kwargs
    assert params["user_id"] == trial.user_id
    assert (
        params["success_url"] == "https://platform.example.com/onboarding?trial=success"
    )
    assert (
        params["cancel_url"]
        == "https://platform.example.com/onboarding?trial=cancelled"
    )
    assert params["metadata"]["datafast_visitor_id"] == "visitor-1"


@pytest.mark.asyncio
async def test_unavailable_offer_is_conflict_not_success(trial):
    with patch.object(
        routes,
        "create_trial_checkout",
        AsyncMock(side_effect=routes.TrialUnavailable("Offer changed")),
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.start_trial_checkout(
                routes.TrialCheckoutRequest(offer_token=trial.offer.token),
                trial.user_id,
            )
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_cancellation_ends_trial_immediately(trial):
    trial = trial.model_copy(
        update={"subscription_id": "sub_1", "consumed_at": datetime.now(UTC)}
    )
    canceled = {"id": "sub_1", "customer": trial.customer_id, "status": "canceled"}
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            routes.stripe.Subscription,
            "retrieve_async",
            AsyncMock(
                return_value=MagicMock(customer=trial.customer_id, status="trialing")
            ),
        ),
        patch.object(
            routes.stripe.Subscription, "cancel_async", AsyncMock(return_value=canceled)
        ) as cancel,
        patch.object(
            routes.stripe.Subscription, "modify_async", AsyncMock(return_value=canceled)
        ) as modify,
        patch.object(routes, "sync_subscription_from_stripe", AsyncMock()) as sync,
        patch.object(
            routes,
            "get_trial_status",
            AsyncMock(return_value=routes.TrialStatusResponse(status="canceled")),
        ),
    ):
        result = await routes.cancel_trial(trial.user_id)
    cancel.assert_awaited_once_with("sub_1", invoice_now=False, prorate=False)
    modify.assert_not_awaited()
    sync.assert_awaited_once_with(canceled)
    assert not result.active


@pytest.mark.asyncio
async def test_cancellation_retry_reconciles_an_already_canceled_trial(trial):
    trial = trial.model_copy(
        update={"subscription_id": "sub_1", "consumed_at": datetime.now(UTC)}
    )
    canceled = routes.stripe.Subscription.construct_from(
        {"id": "sub_1", "customer": trial.customer_id, "status": "canceled"}, "test-key"
    )
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            routes.stripe.Subscription,
            "retrieve_async",
            AsyncMock(return_value=canceled),
        ),
        patch.object(routes.stripe.Subscription, "cancel_async", AsyncMock()) as cancel,
        patch.object(routes, "sync_subscription_from_stripe", AsyncMock()) as sync,
        patch.object(
            routes,
            "get_trial_status",
            AsyncMock(return_value=routes.TrialStatusResponse(status="canceled")),
        ),
    ):
        assert not (await routes.cancel_trial(trial.user_id)).active
    cancel.assert_not_awaited()
    sync.assert_awaited_once_with(dict(canceled))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "customer,status", [("cus_other", "trialing"), ("cus_1", "active")]
)
async def test_cancellation_cannot_cancel_another_customer_or_paid_plan(
    trial, customer, status
):
    trial = trial.model_copy(
        update={"subscription_id": "sub_1", "consumed_at": datetime.now(UTC)}
    )
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            routes.stripe.Subscription,
            "retrieve_async",
            AsyncMock(return_value=MagicMock(customer=customer, status=status)),
        ),
        patch.object(routes.stripe.Subscription, "cancel_async", AsyncMock()) as cancel,
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(trial.user_id)
    assert error.value.status_code == 409
    cancel.assert_not_awaited()
