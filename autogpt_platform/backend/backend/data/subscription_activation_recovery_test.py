"""Recovery distinguishes paid access, initial resets, and unrelated plans."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_checkout_fixtures as fixtures
from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import PaidActivationResult

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

attempt = fixtures.attempt
boundaries = fixtures.boundaries
live_subscription = fixtures.live_subscription


@pytest.mark.asyncio
async def test_confirmed_recovery_uses_invoice_settled_after_initial_snapshot(
    recovery, live_subscription, attempt
):
    recovery.get_attempt.return_value = attempt.model_copy(
        update={"confirmed_at": datetime.now(UTC)}
    )
    live_subscription.status = "trialing"
    recovery.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_conversion", usage_reset=True, activation_id="generation-1"
    )
    response = await checkout.get_activation(attempt.user_id, attempt.id)
    assert response.status == "ready" and response.usage_reset
    assert response.invoice_id == "in_conversion"
    recovery.stripe_call.assert_not_awaited()


@pytest.fixture
def recovery(monkeypatch, boundaries, live_subscription):
    live_subscription.status = "active"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_paid", customer=live_subscription.customer, status="paid"
    )
    monkeypatch.setattr(
        checkout.User,
        "prisma",
        lambda: MagicMock(
            find_unique_or_raise=AsyncMock(
                return_value=SimpleNamespace(stripeCustomerId="cus_1")
            )
        ),
    )
    boundaries.get_attempt.return_value = None
    boundaries.stripe_call.return_value = SimpleNamespace(
        data=[live_subscription], has_more=False
    )
    monkeypatch.setattr(
        checkout, "activation_return_to", AsyncMock(return_value="/chat/resume")
    )
    return boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("activation_id", [None, "generation-1"])
async def test_current_distinguishes_paid_readiness_from_initial_reset(
    recovery, activation_id
):
    recovery.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_paid",
        usage_reset=activation_id is not None,
        activation_id=activation_id,
    )
    for _ in range(2):
        response = await checkout.current_activation("user-1")
        assert response.status == "ready"
        assert response.usage_reset is (activation_id is not None)
        assert response.activation_id == activation_id
        assert response.return_to == "/chat/resume"
    assert all(
        call.args[0].__name__ == "list_async"
        for call in recovery.stripe_call.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", [SubscriptionTier.BASIC, SubscriptionTier.MAX])
@pytest.mark.parametrize("has_attempt", [False, True])
async def test_non_pro_recovery_is_terminal(
    recovery, live_subscription, attempt, monkeypatch, tier, has_attempt
):
    live_subscription.items.data[0].price.id = "price_other"
    monkeypatch.setattr(
        "backend.data.credit.build_price_to_tier_map",
        AsyncMock(return_value={"price_other": tier}),
    )
    monkeypatch.setattr(billing, "get_subscription_trial", AsyncMock(return_value=None))
    if has_attempt:
        recovery.get_attempt.return_value = attempt
    for _ in range(2):
        response = await checkout.current_activation("user-1")
        assert response.status == "not_applicable"
        assert response.retry_after_seconds is None
        assert response.error_code == "not_pro_subscription"
        assert not response.usage_reset and response.activation_id is None
        assert response.return_to == (
            attempt.return_to if has_attempt else "/chat/resume"
        )
    recovery.reconcile_paid_activation.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_price_is_processing_without_claiming_failure_or_reset(
    recovery, live_subscription, monkeypatch
):
    live_subscription.items.data[0].price.id = "price_unknown"
    monkeypatch.setattr(
        "backend.data.credit.build_price_to_tier_map", AsyncMock(return_value={})
    )
    monkeypatch.setattr(billing, "get_subscription_trial", AsyncMock(return_value=None))
    response = await checkout.current_activation("user-1")
    assert response.status == "processing"
    assert response.error_code == "plan_unavailable"
    assert response.retry_after_seconds == 3
    recovery.reconcile_paid_activation.assert_not_awaited()


@pytest.mark.asyncio
async def test_current_skips_non_pro_to_recover_paid_signup(
    recovery, live_subscription, monkeypatch
):
    other = live_subscription.model_copy(deep=True)
    other.id = "sub_max"
    other.items.data[0].price.id = "price_max"
    recovery.stripe_call.return_value.data = [other, live_subscription]
    recovery.owned_subscription.side_effect = [other, live_subscription]
    recovery.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_paid", usage_reset=True, activation_id="generation-1"
    )
    monkeypatch.setattr(
        "backend.data.credit.build_price_to_tier_map",
        AsyncMock(
            return_value={
                "price_max": SubscriptionTier.MAX,
                "price_pro": SubscriptionTier.PRO,
            }
        ),
    )
    response = await checkout.current_activation("user-1")
    assert response.status == "ready" and response.activation_id == "generation-1"
    recovery.reconcile_paid_activation.assert_awaited_once_with("user-1", "sub_1")


@pytest.mark.asyncio
async def test_current_recognizes_original_trial_price_after_offer_changes(
    recovery, live_subscription, trial, monkeypatch
):
    trial.subscription_id = live_subscription.id
    live_subscription.items.data[0].price.id = trial.offer.price_id
    prices = AsyncMock(return_value={})
    monkeypatch.setattr("backend.data.credit.build_price_to_tier_map", prices)
    recovery.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_paid", usage_reset=True, activation_id="generation-1"
    )
    response = await checkout.current_activation("user-1")
    assert response.status == "ready" and response.usage_reset
    prices.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", ["quantity", "multiple", "paginated"])
async def test_ambiguous_plan_remains_processing(
    recovery, live_subscription, ambiguous
):
    if ambiguous == "quantity":
        live_subscription.items.data[0].quantity = 2
    elif ambiguous == "multiple":
        live_subscription.items.data.append(live_subscription.items.data[0])
    else:
        live_subscription.items.has_more = True
    response = await checkout.current_activation("user-1")
    assert response.status == "processing" and response.error_code == "plan_unavailable"
    recovery.reconcile_paid_activation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan,status,error_code",
    [
        ("non_pro", "not_applicable", "not_pro_subscription"),
        ("unknown", "processing", "plan_unavailable"),
        ("quantity", "processing", "plan_unavailable"),
        ("multiple", "processing", "plan_unavailable"),
        ("paginated", "processing", "plan_unavailable"),
        ("changed_pro", "processing", "terms_changed"),
    ],
)
async def test_unconfirmed_trial_classifies_changed_plan_before_confirmation(
    recovery, live_subscription, attempt, monkeypatch, plan, status, error_code
):
    recovery.get_attempt.return_value = attempt
    live_subscription.status = "trialing"
    prices = AsyncMock(
        return_value={
            "price_other": SubscriptionTier.MAX,
            "price_changed": SubscriptionTier.PRO,
        }
    )
    monkeypatch.setattr(billing, "get_subscription_trial", AsyncMock(return_value=None))
    monkeypatch.setattr("backend.data.credit.build_price_to_tier_map", prices)
    if plan == "non_pro":
        live_subscription.items.data[0].price.id = "price_other"
    elif plan == "unknown":
        live_subscription.items.data[0].price.id = "price_unknown"
    elif plan == "quantity":
        live_subscription.items.data[0].quantity = 2
    elif plan == "multiple":
        live_subscription.items.data.append(live_subscription.items.data[0])
    elif plan == "paginated":
        live_subscription.items.has_more = True
    else:
        live_subscription.items.data[0].price.id = "price_changed"

    response = await checkout.current_activation("user-1")

    assert response.status == status
    assert response.error_code == error_code
    assert response.return_to == attempt.return_to
    assert not response.usage_reset and response.activation_id is None
    assert prices.await_count <= 1
    recovery.reconcile_paid_activation.assert_not_awaited()
    assert all(
        call.args[0].__name__ == "list_async"
        for call in recovery.stripe_call.await_args_list
    )


@pytest.mark.asyncio
async def test_unconfirmed_original_trial_keeps_confirmation_without_reconciliation(
    recovery, live_subscription, attempt, monkeypatch
):
    recovery.get_attempt.return_value = attempt
    live_subscription.status = "trialing"
    prices = AsyncMock(return_value={})
    monkeypatch.setattr("backend.data.credit.build_price_to_tier_map", prices)

    response = await checkout.current_activation("user-1")

    assert response.status == "confirmation_required"
    assert response.terms == attempt.terms
    assert response.retry_after_seconds is None
    prices.assert_not_awaited()
    recovery.reconcile_paid_activation.assert_not_awaited()
    recovery.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_current_non_pro_subscription_supersedes_a_failed_old_attempt(
    recovery, live_subscription, attempt, monkeypatch
):
    attempt.subscription_id = "sub_old"
    old = live_subscription.model_copy(
        update={"id": attempt.subscription_id, "status": "canceled"}, deep=True
    )
    live_subscription.id = "sub_new"
    live_subscription.items.data[0].price.id = "price_max"
    recovery.get_attempt.return_value = attempt
    recovery.owned_subscription.side_effect = [old, live_subscription]
    monkeypatch.setattr(billing, "get_subscription_trial", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "backend.data.credit.build_price_to_tier_map",
        AsyncMock(return_value={"price_max": SubscriptionTier.MAX}),
    )

    response = await checkout.current_activation("user-1")

    assert response.status == "not_applicable"
    assert response.error_code == "not_pro_subscription"
    assert response.return_to == "/chat/resume"
    assert not response.usage_reset and response.activation_id is None
    recovery.reconcile_paid_activation.assert_not_awaited()
