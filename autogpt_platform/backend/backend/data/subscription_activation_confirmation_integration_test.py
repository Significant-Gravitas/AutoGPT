"""An explicit Max choice binds payment and reset to durable confirmed terms."""

import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from functools import partial

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import PaidUsageActivation, SubscriptionTrial, User

from backend.data import credit
from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation import reconcile_paid_activation
from backend.data.subscription_activation_attempt import get_attempt
from backend.data.subscription_activation_models import ActivationConfirmRequest

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("trial_cost", [25, 100])
@pytest.mark.parametrize("source_tier", ["PRO", "BASIC", "BUSINESS"])
async def test_confirmed_trial_to_max_uses_confirmed_price_and_one_reset(
    activation_case, mocker, trial_cost, source_tier
):
    case = activation_case
    trial = await prepare_trial_checkout(case, trial_cost, source_tier)
    max_price = {
        "id": "price_max",
        "unit_amount": 6000,
        "currency": "usd",
        "recurring": {"interval": "month", "interval_count": 1},
    }
    mock_billing_boundaries(case, mocker, max_price)
    mutation = mocker.patch(
        "stripe.Subscription.modify_async",
        side_effect=partial(settle_conversion, case, max_price),
    )

    preview = await checkout.preview_activation(case.user_id, "/chat/kept", "MAX")
    assert preview.status == "confirmation_required" and preview.terms
    assert preview.terms.plan == "MAX" and preview.terms.price_id == "price_max"
    assert preview.terms.amount_due == 6000 and not preview.usage_reset
    mutation.assert_not_called()
    assert await case.counters(None) == (trial_cost, trial_cost)
    assert preview.id
    body = ActivationConfirmRequest(confirmed=True, terms_token=preview.terms.token)
    result = await checkout.confirm_activation(case.user_id, preview.id, body)
    assert result.status == "ready" and result.usage_reset and result.activation_id
    assert result.return_to == "/chat/kept"
    assert await case.counters(result.activation_id) == (0, 0)
    await case.add_cost(result.activation_id, 197)
    assert await checkout.get_activation(case.user_id, preview.id) == result
    assert await checkout.confirm_activation(case.user_id, preview.id, body) == result
    assert await case.counters(result.activation_id) == (197, 197)
    mutation.assert_awaited_once()
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.convertedAt and row.costMicrodollars == trial_cost
    assert row.offer == trial.offer.model_dump(mode="json")
    assert row.consumedAt == trial.consumed_at
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 1
    await assert_retired_max_recovery(case, mocker, result.activation_id)


async def assert_retired_max_recovery(case, mocker, generation):
    mocker.patch.object(credit, "build_price_to_tier_map", return_value={})
    for renewal in (False, True):
        if renewal:
            invoice = deepcopy(case.invoices[-1])
            invoice.update(
                id=f"renewal_{invoice['id']}", billing_reason="subscription_cycle"
            )
            invoice["created"] += 30 * 86400
            invoice["status_transitions"]["paid_at"] += 30 * 86400
            invoice["lines"]["data"][0]["period"]["start"] += 30 * 86400
            invoice["lines"]["data"][0]["period"]["end"] += 30 * 86400
            case.invoices.append(invoice)
            case.subscription["latest_invoice"] = invoice["id"]
        await User.prisma().update(
            where={"id": case.user_id}, data={"subscriptionTier": "NO_TIER"}
        )
        result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
        assert result is not None
        if renewal:
            assert not result.usage_reset and result.activation_id is None
        state = await get_usage_activation_state(case.user_id)
        assert state.ready and state.tier == "MAX" and state.generation == generation
        assert await case.counters(generation) == (197, 197)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 1


async def test_unpaid_confirmed_max_conversion_keeps_trial_usage(
    activation_case, mocker
):
    case = activation_case
    trial = await prepare_trial_checkout(case, 100)
    max_price = {
        "id": "price_max",
        "unit_amount": 6000,
        "currency": "usd",
        "recurring": {"interval": "month", "interval_count": 1},
    }
    mock_billing_boundaries(case, mocker, max_price)
    mutation = mocker.patch(
        "stripe.Subscription.modify_async",
        side_effect=partial(settle_conversion, case, max_price, invoice_status="open"),
    )
    preview = await checkout.preview_activation(case.user_id, "/chat/kept", "MAX")
    assert preview.id and preview.terms
    body = ActivationConfirmRequest(confirmed=True, terms_token=preview.terms.token)

    response = await checkout.confirm_activation(case.user_id, preview.id, body)

    assert response.status == "payment_required" and not response.usage_reset
    assert response.activation_id is None
    assert await checkout.get_activation(case.user_id, preview.id) == response
    assert (await get_usage_activation_state(case.user_id)).generation is None
    assert await case.counters(None) == (100, 100)
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.convertedAt is None and row.costMicrodollars == 100
    assert row.offer == trial.offer.model_dump(mode="json")
    mutation.assert_awaited_once()


async def prepare_trial_checkout(case, cost, tier="PRO"):
    trial = await case.add_trial(cost, tier)
    ends_at = datetime.now(UTC) + timedelta(days=4)
    await SubscriptionTrial.prisma().update(
        where={"id": trial.id}, data={"endsAt": ends_at}
    )
    case.subscription.update(status="trialing", trial_end=int(ends_at.timestamp()))
    case.subscription["items"]["data"][0].update(id="si_owned")
    case.subscription["items"]["data"][0]["price"].update(
        unit_amount=2000,
        currency="usd",
        recurring={"interval": "month", "interval_count": 1},
    )
    case.invoice["amount_paid"] = 0
    case.invoice["lines"]["data"][0]["amount"] = 0
    await case.add_cost(None, cost)
    return trial


def mock_billing_boundaries(case, mocker, max_price):
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={
            "price_pro": SubscriptionTier.PRO,
            "price_max": SubscriptionTier.MAX,
        },
    )
    mocker.patch.object(credit, "get_subscription_price_id", return_value="price_max")
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")
    mocker.patch(
        "stripe.Price.retrieve_async",
        return_value=stripe.Price.construct_from(max_price, None),
    )
    mocker.patch(
        "stripe.Invoice.create_preview_async",
        return_value=stripe.Invoice.construct_from(
            {
                "customer": case.subscription["customer"],
                "currency": "usd",
                "amount_due": 6000,
            },
            None,
        ),
    )


async def settle_conversion(
    case, max_price, subscription_id, *, invoice_status="paid", **kwargs
):
    attempt = await get_attempt(case.user_id)
    assert attempt and attempt.confirmed_at and attempt.terms.plan == "MAX"
    assert subscription_id == case.subscription["id"]
    assert kwargs["items"] == [{"id": "si_owned", "price": "price_max", "quantity": 1}]
    assert kwargs["metadata"]["pro_activation_attempt_id"] == attempt.id
    assert kwargs["idempotency_key"] == f"pro-activation:{attempt.id}"
    now = int(datetime.now(UTC).timestamp())
    case.subscription["items"]["data"][0]["price"] = max_price
    case.subscription.update(status="active", trial_end=now)
    case.subscription["metadata"].update(kwargs["metadata"])
    invoice = deepcopy(case.invoice)
    invoice.update(
        id=f"paid_{case.invoice['id']}",
        created=now,
        status=invoice_status,
        amount_paid=6000 if invoice_status == "paid" else 0,
        amount_remaining=0 if invoice_status == "paid" else 6000,
        payments={"data": [], "has_more": False},
    )
    invoice["status_transitions"]["paid_at"] = now if invoice_status == "paid" else None
    invoice["lines"]["data"][0].update(
        amount=6000, price=max_price, period={"start": now, "end": now + 30 * 86400}
    )
    case.invoices.append(invoice)
    case.subscription["latest_invoice"] = invoice["id"]
    return stripe.Subscription.construct_from(case.subscription, None)
