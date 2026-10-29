from unittest.mock import AsyncMock, call, patch

import pytest
import stripe
from pydantic import ValidationError

from backend.data import subscription_trial_billing as billing


@pytest.mark.parametrize(
    "event_type,data,expected",
    [
        ("customer.updated", {"object": {"id": "cus_1"}}, ["cus_1"]),
        ("customer.deleted", {"object": {"id": "cus_1", "deleted": True}}, ["cus_1"]),
        (
            "payment_method.attached",
            {"object": {"id": "pm_1", "customer": "cus_1"}},
            ["cus_1"],
        ),
        (
            "payment_method.updated",
            {
                "object": {"id": "pm_1", "customer": "cus_1"},
                "previous_attributes": {"customer": "cus_1"},
            },
            ["cus_1"],
        ),
        (
            "payment_method.detached",
            {
                "object": {"id": "pm_1", "customer": None},
                "previous_attributes": {"customer": "cus_old"},
            },
            ["cus_old"],
        ),
        (
            "payment_method.updated",
            {
                "object": {"id": "pm_1", "customer": "cus_new"},
                "previous_attributes": {"customer": "cus_old"},
            },
            ["cus_new", "cus_old"],
        ),
        (
            "payment_method.automatically_updated",
            {"object": {"id": "pm_1", "customer": "cus_1"}},
            ["cus_1"],
        ),
        (
            "setup_intent.succeeded",
            {"object": {"id": "seti_1", "customer": "cus_1"}},
            ["cus_1"],
        ),
        ("payment_method.detached", {"object": {"id": "pm_1", "customer": None}}, []),
        ("unrelated.event", None, []),
    ],
)
def test_billing_event_customer_identity(event_type, data, expected):
    assert billing.billing_event_customer_ids(event_type, data) == expected


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"object": {"id": ""}},
        {"object": {"id": "pm_1", "customer": {"id": "cus_1"}}},
    ],
)
def test_malformed_event_does_not_invent_customer_identity(data):
    with pytest.raises(ValidationError):
        billing.billing_event_customer_ids("payment_method.updated", data)


def target(number):
    return billing.TrialBillingTarget(
        id=f"trial_{number}",
        user_id=f"user_{number}",
        customer_id="cus_1",
        subscription_id=f"sub_{number}",
    )


def subscription(number):
    return {
        "id": f"sub_{number}",
        "customer": "cus_1",
        "status": "trialing",
        "metadata": {
            "trial_enrollment_id": f"trial_{number}",
            "user_id": f"user_{number}",
        },
    }


@pytest.fixture
def boundaries():
    with (
        patch.object(
            billing,
            "get_trial_billing_targets",
            AsyncMock(side_effect=[[target(1)], []]),
        ) as targets,
        patch.object(
            stripe.Subscription,
            "retrieve_async",
            AsyncMock(return_value=subscription(1)),
        ) as retrieve,
        patch.object(billing, "sync_subscription_from_stripe", AsyncMock()) as sync,
    ):
        yield targets, retrieve, sync


@pytest.mark.asyncio
async def test_event_payload_only_routes_current_stripe_reconciliation(boundaries):
    targets, retrieve, sync = boundaries
    await billing.sync_trials_for_billing_event(
        "customer.updated",
        {
            "object": {
                "id": "cus_1",
                "default_payment_method": None,
                "status": "canceled",
            },
        },
    )
    retrieve.assert_awaited_once_with("sub_1")
    sync.assert_awaited_once_with(subscription(1))
    assert targets.await_args_list == [call(["cus_1"], ""), call(["cus_1"], "trial_1")]


@pytest.mark.asyncio
async def test_unattached_method_does_not_query_trials(boundaries):
    targets, retrieve, sync = boundaries
    await billing.sync_trials_for_billing_event(
        "payment_method.updated", {"object": {"id": "pm_unattached", "customer": None}}
    )
    targets.assert_not_awaited()
    retrieve.assert_not_awaited()
    sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"id": "sub_other"},
        {"customer": "cus_other"},
        {"metadata": {"trial_enrollment_id": "trial_other", "user_id": "user_1"}},
        {"metadata": {"trial_enrollment_id": "trial_1", "user_id": "user_other"}},
    ],
)
async def test_foreign_subscription_cannot_change_trial(boundaries, changed):
    _, retrieve, sync = boundaries
    retrieve.return_value.update(changed)
    with pytest.raises(RuntimeError, match="retry the event"):
        await billing.sync_trials_for_billing_event(
            "customer.updated", {"object": {"id": "cus_1"}}
        )
    sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_target_does_not_starve_other_pages_and_event_is_retryable(
    boundaries,
):
    targets, retrieve, sync = boundaries
    targets.side_effect = [[target(1)], [target(2)], []]
    retrieve.side_effect = [
        stripe.APIConnectionError("temporary failure"),
        subscription(2),
    ]
    with pytest.raises(RuntimeError, match="retry the event"):
        await billing.sync_trials_for_billing_event(
            "customer.updated", {"object": {"id": "cus_1"}}
        )
    sync.assert_awaited_once_with(subscription(2))
    assert targets.await_args_list == [
        call(["cus_1"], ""),
        call(["cus_1"], "trial_1"),
        call(["cus_1"], "trial_2"),
    ]


@pytest.mark.asyncio
async def test_database_failure_is_not_acknowledged(boundaries):
    targets, retrieve, sync = boundaries
    targets.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await billing.sync_trials_for_billing_event(
            "customer.updated", {"object": {"id": "cus_1"}}
        )
    retrieve.assert_not_awaited()
    sync.assert_not_awaited()
