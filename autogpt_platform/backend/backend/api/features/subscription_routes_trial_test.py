"""Choosing a plan while a trial is cancel-pending: the same plan converts the
trial's own subscription in place instead of opening a second checkout."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import fastapi.testclient
import pytest
import pytest_mock
import stripe
from prisma.enums import SubscriptionTier

from backend.api.features import subscription_routes_test as routes_test
from backend.api.features.subscription_routes_test import (
    TEST_FRONTEND_ORIGIN,
    TEST_USER_ID,
    _patch_payment_flag,
)
from backend.data import subscription_trial_conversion as conversion
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

client = routes_test.client
_configure_frontend_origin = routes_test._configure_frontend_origin
_stub_lifecycle_emails = routes_test._stub_lifecycle_emails
_stub_pending_subscription_change = routes_test._stub_pending_subscription_change
track_checkout_started = routes_test.track_checkout_started
_stub_subscription_status_lookups = routes_test._stub_subscription_status_lookups


TRIAL_RUNNING_DETAIL = (
    "Your accepted plan starts after your trial. Manage the trial in billing."
)


@pytest.fixture
def cancel_pending_trial(trial: TrialState) -> TrialState:
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "user_id": TEST_USER_ID,
            "subscription_id": "sub_1",
            "status": "trialing",
            "card_verified_at": now - timedelta(days=2),
            "started_at": now - timedelta(days=2),
            "ends_at": now + timedelta(days=5),
            "consumed_at": now - timedelta(days=2),
            "cancel_at_period_end": True,
        }
    )


def _subscription_list(*subscriptions: dict) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"data": list(subscriptions), "has_more": False}, "test-key"
    )


@pytest.fixture
def trial_conversion(
    mocker: pytest_mock.MockFixture, cancel_pending_trial: TrialState
) -> MagicMock:
    """A TRIAL-tier user whose trial is cancel-pending, with every Stripe and
    database boundary of the in-place conversion recorded in call order."""
    calls: list[str] = []
    assert cancel_pending_trial.ends_at is not None
    live = {
        "id": "sub_1",
        "customer": cancel_pending_trial.customer_id,
        "status": "trialing",
        "cancel_at_period_end": True,
        "trial_end": int(cancel_pending_trial.ends_at.timestamp()),
        "metadata": {
            "user_id": TEST_USER_ID,
            "trial_enrollment_id": cancel_pending_trial.id,
        },
    }
    converted = stripe.Subscription.construct_from(
        {**live, "status": "active", "cancel_at_period_end": False}, "test-key"
    )

    def recorded(name: str, result: object = None) -> AsyncMock:
        async def effect(*args, **kwargs):
            calls.append(name)
            return result

        return AsyncMock(side_effect=effect)

    @asynccontextmanager
    async def lock(user_id: str):
        calls.append(f"lock:{user_id}")
        try:
            yield
        finally:
            calls.append("unlock")

    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_user_by_id",
        new_callable=AsyncMock,
        return_value=Mock(subscription_tier=SubscriptionTier.TRIAL),
    )
    _patch_payment_flag(mocker)
    mocker.patch.object(
        conversion,
        "get_subscription_trial",
        AsyncMock(return_value=cancel_pending_trial),
    )
    mocker.patch.object(conversion, "subscription_checkout_lock", lock)
    card = mocker.patch.object(
        conversion, "subscription_card_can_be_charged", AsyncMock(return_value=True)
    )
    retrieve = recorded(
        "retrieve", stripe.Subscription.construct_from(live, "test-key")
    )
    mocker.patch.object(stripe.Subscription, "retrieve_async", retrieve)
    return MagicMock(
        calls=calls,
        live=live,
        retrieve=retrieve,
        card=card,
        converted=converted,
        others=mocker.patch.object(
            stripe.Subscription,
            "list_async",
            recorded("others", _subscription_list(live)),
        ),
        modify=mocker.patch.object(
            stripe.Subscription, "modify_async", recorded("modify", converted)
        ),
        expire=mocker.patch.object(
            conversion, "expire_other_subscription_checkouts", recorded("expire")
        ),
        sync=mocker.patch.object(
            conversion, "sync_subscription_from_stripe", recorded("sync")
        ),
        modify_for_tier=mocker.patch(
            "backend.api.features.billing.subscriptions.routes.modify_stripe_subscription_for_tier",
            new_callable=AsyncMock,
        ),
        checkout=mocker.patch(
            "backend.api.features.billing.subscriptions.routes.create_subscription_checkout",
            new_callable=AsyncMock,
            return_value="https://checkout.stripe.com/pay/cs_test_max",
        ),
    )


def _post_plan(
    client: fastapi.testclient.TestClient, tier: str, billing_cycle: str = "monthly"
):
    return client.post(
        "/credits/subscription",
        json={
            "tier": tier,
            "billing_cycle": billing_cycle,
            "success_url": f"{TEST_FRONTEND_ORIGIN}/success",
            "cancel_url": f"{TEST_FRONTEND_ORIGIN}/cancel",
            "surface": "billing",
        },
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"cancel_at_period_end": False},
        {"converted_at": datetime.now(UTC)},
        {"ends_at": datetime.now(UTC) - timedelta(minutes=1)},
    ],
)
def test_update_subscription_tier_trial_still_waits_for_the_trial_to_end(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
    cancel_pending_trial: TrialState,
    change: dict | None,
) -> None:
    """A TRIAL-tier user whose trial is not cancel-pending keeps today's 409."""
    mocker.patch.object(
        conversion,
        "get_subscription_trial",
        AsyncMock(
            return_value=change and cancel_pending_trial.model_copy(update=change)
        ),
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 409
    assert response.json()["detail"] == TRIAL_RUNNING_DETAIL
    assert trial_conversion.calls == []
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


def test_update_subscription_tier_cancel_pending_trial_converts_its_own_plan(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    track_checkout_started: AsyncMock,
) -> None:
    response = _post_plan(client, "PRO")

    assert response.status_code == 200
    assert response.json()["url"] == ""
    assert trial_conversion.calls == [
        f"lock:{TEST_USER_ID}",
        "retrieve",
        "expire",
        "others",
        "modify",
        "sync",
        "unlock",
    ]
    trial_conversion.retrieve.assert_awaited_once_with("sub_1")
    trial_conversion.expire.assert_awaited_once_with("cus_1")
    trial_conversion.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        trial_end="now",
        payment_behavior="error_if_incomplete",
    )
    trial_conversion.sync.assert_awaited_once_with(dict(trial_conversion.converted))
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()
    track_checkout_started.assert_not_awaited()
