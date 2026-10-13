"""A trial converted onto a plan the customer chose in place of the accepted
one is announced as that plan, at its price; every other notice keeps the
accepted terms."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import NotificationResult
from backend.data.subscription_trial import TrialState
from backend.notifications import trial as notices
from backend.notifications.trial_plan import notice_plan

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.fixture
def converted(trial: TrialState) -> TrialState:
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "subscription_id": "sub_1",
            "status": "active",
            "ends_at": now - timedelta(minutes=1),
            "consumed_at": now - timedelta(days=2),
            "converted_at": now,
            "conversion_invoice_id": "in_max",
        }
    )


def _switched(interval: str = "month") -> dict:
    return {
        "id": "sub_1",
        "customer": "cus_1",
        "status": "active",
        "metadata": {
            "user_id": "user-1",
            "trial_enrollment_id": "trial-1",
            "trial_checkout_attempt": "0",
            "trial_conversion_price_id": "price_max",
            "trial_conversion_tier": "MAX",
        },
        "items": {
            "data": [
                {
                    "id": "si_1",
                    "quantity": 1,
                    "price": {
                        "id": "price_max",
                        "unit_amount": 32000,
                        "currency": "usd",
                        "recurring": {"interval": interval},
                    },
                }
            ],
            "has_more": False,
        },
    }


def test_a_conversion_onto_the_chosen_plan_is_announced_as_that_plan(converted):
    data = notices.trial_notice_data(converted, "converted", "Sam", _switched())

    assert data.plan.name == "Max"
    assert data.plan.label == "Max · monthly"
    assert data.plan.price_display == "$320.00 / month"


def test_a_yearly_choice_is_announced_as_yearly(converted):
    plan = notice_plan(converted, "converted", _switched("year"))

    assert (plan.tier, plan.cycle) == ("MAX", "yearly")


@pytest.mark.parametrize("interval,noun", [("month", "month"), ("year", "year")])
@pytest.mark.parametrize(
    "currency,unit_amount,amount",
    [
        ("cad", None, "Not available"),
        ("usd", None, "Not available"),
        ("cad", 0, "0.00 CAD"),
    ],
)
def test_conversion_formats_missing_and_zero_amounts(
    converted, interval, noun, currency, unit_amount, amount
):
    current = _switched(interval)
    current["items"]["data"][0]["price"].update(
        currency=currency, unit_amount=unit_amount
    )
    data = notices.trial_notice_data(converted, "converted", "Sam", current)

    assert data.plan.price_display == f"{amount} / {noun}"


@pytest.mark.parametrize("kind", ["started", "canceled", "ended", "payment_failed"])
def test_other_notices_keep_the_accepted_terms(converted, kind):
    data = notices.trial_notice_data(converted, kind, "Sam", _switched())

    assert data.plan.name == "Pro"
    assert data.plan.price_display == "$50.00 / month"


def test_a_conversion_on_the_accepted_price_keeps_its_terms(converted, subscription):
    data = notices.trial_notice_data(converted, "converted", "Sam", subscription)

    assert data.plan.name == "Pro"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,price,expected",
    [
        ("Max · monthly", "$320.00 / month", "current"),
        ("Max — monthly", "$320.00 / month", "current"),
        ("Max · monthly", "$320.00 USD / month", "current"),
        ("Max — monthly", "$320.00 USD / month", "current"),
        ("Max — monthly", "$50.00 USD / month", "obsolete"),
        ("Max — monthly", "$100.00 USD / month", "obsolete"),
    ],
)
async def test_conversion_freshness_matches_only_the_current_plan(
    converted, label, price, expected
):
    current = _switched()
    data = notices.trial_notice_data(converted, "converted", "Sam", current)
    data.notice_key = notices.trial_notice_key(converted, "converted")
    data.plan = data.plan.model_copy(update={"label": label, "price_display": price})
    original = data.model_copy(deep=True)
    database = MagicMock(
        get_subscription_trial=AsyncMock(return_value=converted),
        sync_subscription_from_stripe=AsyncMock(),
    )
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(notices, "stripe_call", AsyncMock(return_value=current)),
    ):
        assert (
            await notices.trial_notice_disposition(converted.user_id, data) == expected
        )
    assert data == original


@pytest.mark.asyncio
async def test_the_conversion_event_reports_the_chosen_plan(converted):
    current = _switched()
    with (
        patch.object(notices, "stripe_call", AsyncMock(return_value=current)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=converted)
            ),
        ),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(
                get_user_by_id=AsyncMock(
                    return_value=SimpleNamespace(id="user-1", name="Sam")
                )
            ),
        ),
        patch.object(notices, "claim_once", AsyncMock(return_value=True)),
        patch.object(notices, "queue_trial_audience_change", AsyncMock()),
        patch.object(notices, "join_paying_audience", AsyncMock()),
        patch.object(
            notices,
            "queue_notification_async",
            AsyncMock(return_value=NotificationResult(success=True)),
        ) as queue,
        patch.object(notices, "_track_billing_event") as track,
    ):
        assert await notices.notify_trial(current, "converted")

    assert queue.await_args.args[0].data.plan.name == "Max"
    properties = track.call_args.args[2]
    assert properties["subscription_tier"] == "MAX"
    assert properties["billing_cycle"] == "monthly"
