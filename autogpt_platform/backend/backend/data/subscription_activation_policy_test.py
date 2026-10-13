"""Only the initial paid conversion of a consumed trial creates a generation."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_activation as activation
from backend.data import subscription_activation_target as target
from backend.data import subscription_trial_fixtures as fixtures
from backend.data.subscription_activation_models import (
    ActivationAttempt,
    ActivationTerms,
)

trial = fixtures.trial
subscription = fixtures.subscription


@pytest.fixture
def policy_case(monkeypatch, trial, subscription):
    now = datetime.now(UTC)
    trial.subscription_id = subscription["id"]
    trial.consumed_at = now - timedelta(days=7)
    subscription.update(
        status="active", trial_end=int(now.timestamp()), latest_invoice="in_first"
    )
    invoice = {
        "id": "in_first",
        "customer": trial.customer_id,
        "subscription": subscription["id"],
        "status": "paid",
        "amount_remaining": 0,
        "created": int(now.timestamp()),
        "status_transitions": {"paid_at": int(now.timestamp())},
        "billing_reason": "subscription_cycle",
        "lines": {
            "data": [
                {
                    "type": "subscription",
                    "subscription": subscription["id"],
                    "quantity": 1,
                    "price": {"id": trial.offer.price_id},
                    "amount": 2000,
                    "period": {
                        "start": int(now.timestamp()),
                        "end": int(now.timestamp()) + 86400,
                    },
                }
            ]
        },
    }
    tx = MagicMock()
    tx.paidusageactivation.find_unique = AsyncMock(return_value=None)
    tx.paidusageactivation.create = AsyncMock()
    tx.paidusageactivation.update_many = AsyncMock()
    tx.paidusageactivationpolicy.find_unique_or_raise = AsyncMock(
        return_value=SimpleNamespace(startsAt=now - timedelta(days=30))
    )
    tx.subscriptiontrial.update = AsyncMock()
    user = SimpleNamespace(
        id=trial.user_id,
        stripeCustomerId=trial.customer_id,
        subscriptionTier=SubscriptionTier.TRIAL,
    )
    monkeypatch.setattr(
        activation, "_retrieve_invoice", AsyncMock(return_value=invoice)
    )
    monkeypatch.setattr(
        activation, "first_settled_invoice", AsyncMock(return_value=invoice)
    )
    monkeypatch.setattr(activation, "verify_activation_usage", AsyncMock())
    return SimpleNamespace(
        user=user, trial=trial, subscription=subscription, invoice=invoice, tx=tx
    )


@pytest.mark.asyncio
async def test_initial_paid_signup_preserves_usage(policy_case):
    case = policy_case
    case.subscription["trial_end"] = None
    case.invoice["billing_reason"] = "subscription_create"
    activation.first_settled_invoice.side_effect = AssertionError("No history needed")
    assert await activation.publish_initial_pro_activation(
        case.user, case.subscription, "price_pro", case.tx
    )
    case.tx.paidusageactivation.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_converted_trial_preserves_usage_on_original_invoice(policy_case):
    case = policy_case
    case.trial.converted_at = datetime.now(UTC)
    case.trial.conversion_invoice_id = case.invoice["id"]
    assert await activation.publish_initial_pro_activation(
        case.user, case.subscription, "price_pro", case.tx, case.trial
    )
    case.tx.paidusageactivation.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconsumed_trial_cannot_reset(policy_case):
    case = policy_case
    case.trial.consumed_at = None
    assert not await activation.publish_initial_pro_activation(
        case.user, case.subscription, "price_pro", case.tx, case.trial
    )
    case.tx.paidusageactivation.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"user_id": "foreign"},
        {"customer_id": "foreign"},
        {"subscription_id": "foreign"},
    ],
)
async def test_unrelated_trial_ledger_cannot_reset(policy_case, changed):
    case = policy_case
    trial = case.trial.model_copy(update=changed)
    assert not await activation.publish_initial_pro_activation(
        case.user, case.subscription, "price_pro", case.tx, trial
    )
    case.tx.paidusageactivation.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", ["PRO", "MAX"])
async def test_consumed_initial_trial_conversion_creates_one_generation(
    policy_case, plan
):
    case = policy_case
    price_id = f"price_{plan.lower()}"
    case.trial.offer = case.trial.offer.model_copy(
        update={"tier": plan, "price_id": price_id}
    )
    case.subscription["items"]["data"][0]["price"]["id"] = price_id
    case.invoice["lines"]["data"][0]["price"]["id"] = price_id
    assert await activation.publish_initial_pro_activation(
        case.user, case.subscription, price_id, case.tx, case.trial
    )
    case.tx.paidusageactivation.create.assert_awaited_once()
    saved = case.tx.paidusageactivation.create.await_args.kwargs["data"]
    assert saved["stripeInvoiceId"] == case.invoice["id"]
    assert saved["stripeSubscriptionId"] == case.subscription["id"]
    case.tx.paidusageactivation.find_unique.return_value = SimpleNamespace(
        id=saved["id"], stripeSubscriptionId=case.subscription["id"]
    )
    assert await activation.publish_initial_pro_activation(
        case.user, case.subscription, price_id, case.tx, case.trial
    )
    case.tx.paidusageactivation.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {},
        {"user_id": "other"},
        {"trial_enrollment_id": "other"},
        {"trial_checkout_attempt": "1"},
    ],
)
async def test_trial_metadata_cannot_invent_reset_ownership(policy_case, changed):
    case = policy_case
    if changed:
        case.subscription["metadata"].update(changed)
    else:
        case.subscription["metadata"] = {}
    if changed.get("user_id"):
        with pytest.raises(ValueError, match="belong"):
            await activation.publish_initial_pro_activation(
                case.user, case.subscription, "price_pro", case.tx, case.trial
            )
    else:
        assert not await activation.publish_initial_pro_activation(
            case.user, case.subscription, "price_pro", case.tx, case.trial
        )
    case.tx.paidusageactivation.create.assert_not_awaited()


@pytest.fixture
def confirmed_target(monkeypatch, policy_case):
    case = policy_case
    case.subscription["items"]["data"][0]["price"]["id"] = "price_max"
    case.subscription["metadata"]["pro_activation_attempt_id"] = "attempt-1"
    attempt = ActivationAttempt(
        id="attempt-1",
        user_id=case.trial.user_id,
        customer_id=case.trial.customer_id,
        subscription_id=case.subscription["id"],
        return_to="/chat/resume",
        confirmed_at=datetime.now(UTC),
        terms=ActivationTerms(
            plan="MAX",
            price_id="price_max",
            accepted_offer_token=case.trial.offer.token,
            amount_due=10000,
            currency="usd",
            billing_interval="month",
            renewal_unit_amount=10000,
            renewal_terms="Renews monthly",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        ),
    )
    lookup = AsyncMock(return_value=[attempt])
    monkeypatch.setattr(target, "query_raw_with_schema", lookup)
    return attempt, lookup


@pytest.mark.asyncio
async def test_confirmed_selected_max_binds_target_under_transaction(
    policy_case, confirmed_target
):
    attempt, lookup = confirmed_target
    case = policy_case
    assert await activation.accepted_conversion_target(
        case.trial, case.subscription, case.tx
    ) == (SubscriptionTier.MAX, "price_max")
    assert lookup.await_args.kwargs["client"] is case.tx
    case.invoice["lines"]["data"][0]["price"]["id"] = "price_max"
    assert await activation.publish_initial_pro_activation(
        case.user, case.subscription, "price_max", case.tx, case.trial
    )
    case.tx.paidusageactivation.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("confirmed_at", None),
        ("user_id", "other"),
        ("customer_id", "other"),
        ("subscription_id", "other"),
        ("id", "other"),
    ],
)
async def test_foreign_or_unconfirmed_attempt_cannot_authorize_target(
    policy_case, confirmed_target, field, value
):
    attempt, _ = confirmed_target
    setattr(attempt, field, value)
    case = policy_case
    assert (
        await activation.accepted_conversion_target(
            case.trial, case.subscription, case.tx
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("price_id", "price_other"), ("accepted_offer_token", "b" * 64)]
)
async def test_changed_accepted_terms_cannot_authorize_target(
    policy_case, confirmed_target, field, value
):
    attempt, _ = confirmed_target
    setattr(attempt.terms, field, value)
    case = policy_case
    assert (
        await activation.accepted_conversion_target(
            case.trial, case.subscription, case.tx
        )
        is None
    )


@pytest.mark.asyncio
async def test_unknown_attempt_cannot_authorize_target(policy_case, confirmed_target):
    _, lookup = confirmed_target
    lookup.return_value = []
    case = policy_case
    assert (
        await activation.accepted_conversion_target(
            case.trial, case.subscription, case.tx
        )
        is None
    )
