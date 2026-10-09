"""Explicit target selection binds preview, consent, mutation, and recovery."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import stripe
from prisma.enums import SubscriptionTier
from pydantic import ValidationError

from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_checkout_fixtures as fixtures
from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import (
    ActivationConfirmRequest,
    ActivationPreviewRequest,
    ActivationResponse,
    ActivationUnavailable,
    PaidActivationResult,
)

pytest_plugins = ("backend.data.subscription_trial_fixtures",)
attempt = fixtures.attempt
max_attempt = fixtures.max_attempt
boundaries = fixtures.boundaries
live_subscription = fixtures.live_subscription


def test_preview_selects_max_and_rejects_unrelated_plans():
    assert ActivationPreviewRequest(plan="MAX").plan == "MAX"
    assert ActivationPreviewRequest().plan is None
    for value in ("BASIC", "BUSINESS", "max"):
        with pytest.raises(ValidationError):
            ActivationPreviewRequest(plan=value)


@pytest.mark.asyncio
async def test_default_quote_retains_original_max_offer(
    trial, live_subscription, monkeypatch
):
    trial.offer = trial.offer.model_copy(
        update={"tier": "MAX", "price_id": "price_retired_max"}
    )
    live_subscription.items.data[0].price.id = trial.offer.price_id
    prices = AsyncMock()
    monkeypatch.setattr(billing.credit, "get_subscription_price_id", prices)
    monkeypatch.setattr(
        billing,
        "stripe_call",
        AsyncMock(
            return_value={
                "customer": "cus_1",
                "currency": "usd",
                "amount_due": 4800,
            }
        ),
    )
    terms = await billing.quote_terms(trial, live_subscription)
    assert terms.plan == "MAX" and terms.price_id == "price_retired_max"
    assert terms.accepted_offer_token == trial.offer.token
    prices.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cycle,interval", [("monthly", "month"), ("yearly", "year")])
@pytest.mark.parametrize(
    "source,target", [("PRO", "MAX"), ("MAX", "PRO"), ("BASIC", "MAX")]
)
async def test_alternate_plan_quote_matches_exact_owned_item_mutation(
    trial, live_subscription, monkeypatch, cycle, interval, source, target
):
    trial.offer = trial.offer.model_copy(
        update={"tier": source, "billing_cycle": cycle}
    )
    live_subscription.items.data[0].price.recurring.interval = interval
    live_subscription.items.data[0] = billing.ActivationItem.model_validate(
        {
            **live_subscription.items.data[0].model_dump(),
            "id": "si_owned",
        }
    )
    original = trial.offer.model_dump()
    target_price = {
        "id": "price_target",
        "unit_amount": 9000,
        "currency": "usd",
        "recurring": {"interval": interval, "interval_count": 1},
    }
    calls = AsyncMock(
        side_effect=[
            target_price,
            {
                "customer": "cus_1",
                "currency": "usd",
                "amount_due": 8700,
            },
        ]
    )
    prices = AsyncMock(return_value="price_target")
    monkeypatch.setattr(billing, "stripe_call", calls)
    monkeypatch.setattr(billing.credit, "get_subscription_price_id", prices)
    terms = await billing.quote_terms(trial, live_subscription, target)
    assert terms.plan == target and terms.price_id == "price_target"
    assert terms.amount_due == 8700 and terms.renewal_unit_amount == 9000
    assert terms.accepted_offer_token == trial.offer.token
    assert trial.offer.model_dump() == original
    prices.assert_awaited_once_with(SubscriptionTier(target), cycle)
    calls.assert_any_await(stripe.Price.retrieve_async, "price_target")
    assert calls.await_args.kwargs["subscription_details"] == {
        "trial_end": "now",
        "proration_behavior": "none",
        "items": [{"id": "si_owned", "price": "price_target", "quantity": 1}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "missing_price",
        "currency",
        "interval",
        "price_id",
        "missing_item",
        "trial_ending",
    ],
)
async def test_alternate_quote_fails_closed(
    trial, live_subscription, monkeypatch, invalid
):
    live_subscription.items.data[0] = billing.ActivationItem.model_validate(
        {
            **live_subscription.items.data[0].model_dump(),
            "id": "si_owned" if invalid != "missing_item" else None,
        }
    )
    if invalid == "trial_ending":
        live_subscription.trial_end = int(datetime.now(UTC).timestamp()) + 30
    target_price = {
        "id": "wrong" if invalid == "price_id" else "price_max",
        "unit_amount": 9000,
        "currency": "eur" if invalid == "currency" else "usd",
        "recurring": {
            "interval": "year" if invalid == "interval" else "month",
            "interval_count": 1,
        },
    }
    calls = AsyncMock(return_value=target_price)
    monkeypatch.setattr(billing, "stripe_call", calls)
    monkeypatch.setattr(
        billing.credit,
        "get_subscription_price_id",
        AsyncMock(
            return_value=None if invalid == "missing_price" else "price_max",
        ),
    )
    with pytest.raises(ActivationUnavailable):
        await billing.quote_terms(trial, live_subscription, "MAX")
    assert all(
        call.args[0] != stripe.Invoice.create_preview_async
        for call in calls.await_args_list
    )


@pytest.mark.asyncio
async def test_preview_passes_selected_target(
    max_attempt, boundaries, trial, live_subscription
):
    response = await checkout.preview_activation(max_attempt.user_id, "/chat", "MAX")
    assert response.terms.plan == "MAX"
    boundaries.quote_terms.assert_awaited_once_with(trial, live_subscription, "MAX")
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_confirmation_and_retry_use_same_item_price_and_idempotency_key(
    max_attempt, boundaries
):
    boundaries.stripe_call.side_effect = stripe.APIConnectionError("unknown outcome")
    body = ActivationConfirmRequest(confirmed=True, terms_token=max_attempt.terms.token)
    for _ in range(2):
        response = await checkout.confirm_activation(
            max_attempt.user_id, max_attempt.id, body
        )
        assert response.status == "processing"
        boundaries.get_attempt.return_value = boundaries.save_confirmation.return_value
    boundaries.save_confirmation.assert_awaited_once()
    assert boundaries.stripe_call.await_count == 2
    for call in boundaries.stripe_call.await_args_list:
        assert call.args == (stripe.Subscription.modify_async, "sub_1")
        assert call.kwargs["items"] == [
            {"id": "si_owned", "price": "price_max", "quantity": 1}
        ]
        assert call.kwargs["idempotency_key"] == "pro-activation:operation-1"
        assert call.kwargs["trial_end"] == "now"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_max_recovery_accepts_original_trial_before_mutation(
    max_attempt, boundaries, confirmed
):
    if confirmed:
        max_attempt.confirmed_at = datetime.now(UTC)
    result = await checkout.get_activation(max_attempt.user_id, max_attempt.id)
    assert result.status == ("processing" if confirmed else "confirmation_required")
    assert result.error_code is None
    assert result.terms.plan == "MAX" and not result.usage_reset
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_paid_max_ready_does_not_imply_reset(
    attempt, boundaries, live_subscription, monkeypatch
):
    live_subscription.status = "active"
    live_subscription.items.data[0].price.id = "price_max"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_paid", customer="cus_1", status="paid"
    )
    monkeypatch.setattr(
        billing.credit,
        "build_price_to_tier_map",
        AsyncMock(return_value={"price_max": SubscriptionTier.MAX}),
    )
    boundaries.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_paid", usage_reset=False
    )
    response = await checkout._payment_status(
        attempt.user_id, live_subscription, ActivationResponse(status="processing")
    )
    assert response.status == "ready" and not response.usage_reset
    assert response.activation_id is None


@pytest.mark.asyncio
async def test_non_target_trial_requires_explicit_paid_plan(
    trial, live_subscription, monkeypatch
):
    trial.offer = trial.offer.model_copy(update={"tier": "BASIC"})
    calls = AsyncMock()
    monkeypatch.setattr(billing, "stripe_call", calls)
    with pytest.raises(ActivationUnavailable, match="convert to Pro or Max"):
        await billing.quote_terms(trial, live_subscription)
    calls.assert_not_awaited()
