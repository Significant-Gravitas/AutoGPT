from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import urllib3.connectionpool
from posthog import Posthog
from prisma.enums import SubscriptionTier
from prisma.errors import PrismaError

import backend.util.feature_flag.posthog as ph
from backend.data import subscription_trial_checkout as checkout
from backend.data.experiments import Experiment
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_stripe import (
    SubscriptionSnapshot,
    trial_subscription_tier,
)


@pytest.fixture
def checkout_guard():
    with patch.object(checkout, "subscription_checkout_lock", return_value=AsyncMock()):
        yield


@pytest.fixture
def trial() -> TrialState:
    return TrialState(
        id="trial-1",
        user_id="user-1",
        offer=AcceptedTrialOffer(
            version="offer-a",
            new_users_from=datetime(2026, 9, 10, tzinfo=UTC),
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
        customer_id="cus_1",
        checkout_session_id=None,
        subscription_id=None,
        checkout_attempt=0,
        success_url="https://platform.agpt.co/settings/billing?trial=success",
        cancel_url="https://platform.agpt.co/settings/billing",
        checkout_metadata={"datafast_visitor_id": "visitor-1"},
        status="checkout_pending",
        card_verified_at=None,
        started_at=None,
        ends_at=None,
        consumed_at=None,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


def test_card_required_and_cancel_if_card_disappears(trial):
    params = checkout.trial_checkout_params(trial, no_charge_copy=False)
    assert params["mode"] == "subscription"
    assert params["payment_method_types"] == ["card"]
    assert params["payment_method_collection"] == "always"
    assert params["subscription_data"]["trial_period_days"] == 7
    assert params["subscription_data"]["trial_settings"] == {
        "end_behavior": {"missing_payment_method": "cancel"}
    }
    assert params["allow_promotion_codes"] is False


def test_checkout_says_no_charge_today_for_the_offers_length(trial):
    trial.offer = AcceptedTrialOffer.model_validate(
        {**trial.offer.model_dump(), "duration_days": 14}
    )
    params = checkout.trial_checkout_params(trial, no_charge_copy=True)
    message = params["custom_text"]["after_submit"]["message"]
    assert message.startswith("No charge today.")
    assert "14-day trial" in message
    assert "Settings > Billing" in message


def test_uses_accepted_price_and_preserves_attribution(trial):
    params = checkout.trial_checkout_params(trial, no_charge_copy=False)
    assert params["line_items"] == [{"price": "price_pro", "quantity": 1}]
    assert params["metadata"]["datafast_visitor_id"] == "visitor-1"
    assert params["subscription_data"]["metadata"] == params["metadata"]
    assert params["metadata"]["trial_enrollment_id"] == trial.id
    assert params["metadata"]["trial_offer_version"] == trial.offer.version


def test_attribution_cannot_overwrite_server_owned_identity(trial):
    trial.checkout_metadata.update(user_id="other-user", trial_enrollment_id="other")
    params = checkout.trial_checkout_params(trial, no_charge_copy=False)
    assert params["metadata"]["user_id"] == trial.user_id
    assert params["metadata"]["trial_enrollment_id"] == trial.id


@pytest.mark.parametrize(
    "change",
    [
        {"duration_days": 14},
        {"price_id": "price_other"},
        {"unit_amount": 3000},
        {"total_cost_limit": 2_000_000},
    ],
)
def test_terms_token_changes_even_if_operator_reuses_offer_version(trial, change):
    changed = AcceptedTrialOffer.model_validate({**trial.offer.model_dump(), **change})
    assert changed.token != trial.offer.token


@pytest.mark.asyncio
async def test_disabled_offer_does_not_create_stripe_customer_or_session(
    trial, checkout_guard
):
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=None)),
        patch.object(checkout, "get_stripe_customer_id", AsyncMock()) as customer,
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="not available"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
    customer.assert_not_awaited()
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_previously_consumed_trial_is_never_reopened(trial, checkout_guard):
    trial.consumed_at = datetime(2026, 9, 10, tzinfo=UTC)
    trial.status = "canceled"
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=trial.offer)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="already been used"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_checkout_reuses_accepted_offer(trial, checkout_guard):
    future_offer = trial.offer.model_copy(update={"duration_days": 14})
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=future_offer)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            checkout, "_resume_checkout", AsyncMock(return_value="checkout")
        ) as resume,
    ):
        result = await checkout.create_trial_checkout(
            trial.user_id, trial.offer.token, trial.success_url, trial.cancel_url, {}
        )
    assert result == "checkout"
    resume.assert_awaited_once_with(trial)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, SubscriptionTier.TRIAL),
        ({"cancel_at_period_end": True}, SubscriptionTier.TRIAL),
        ({"default_payment_method": None}, SubscriptionTier.NO_TIER),
        ({"pending_setup_intent": "seti_needs_auth"}, SubscriptionTier.NO_TIER),
        ({"status": "past_due"}, SubscriptionTier.NO_TIER),
        ({"status": "canceled"}, SubscriptionTier.NO_TIER),
        ({"trial_end": 0}, SubscriptionTier.NO_TIER),
        ({"status": "active"}, SubscriptionTier.NO_TIER),
    ],
)
def test_entitlement_requires_completed_card_setup_and_unexpired_trial(
    trial, overrides, expected
):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    subscription = SubscriptionSnapshot.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "trialing",
            "trial_end": int(now.timestamp()) + 86400,
            "default_payment_method": {
                "id": "pm_1",
                "type": "card",
                "card": {"exp_month": 12, "exp_year": 2030},
            },
            **overrides,
        }
    )
    assert trial_subscription_tier(trial, subscription, now) == expected


@pytest.mark.parametrize(
    "invoice_status,reason,created_delta,expected",
    [
        ("paid", "subscription_cycle", 0, SubscriptionTier.PRO),
        ("open", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("draft", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("uncollectible", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("paid", "subscription_create", 0, SubscriptionTier.NO_TIER),
        ("paid", "subscription_cycle", -1, SubscriptionTier.NO_TIER),
    ],
)
def test_paid_conversion_requires_paid_post_trial_invoice(
    trial, invoice_status, reason, created_delta, expected
):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    end = int(now.timestamp())
    subscription = SubscriptionSnapshot.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "active",
            "trial_end": end,
            "latest_invoice": {
                "id": "in_1",
                "status": invoice_status,
                "billing_reason": reason,
                "created": end + created_delta,
            },
        }
    )
    assert trial_subscription_tier(trial, subscription, now) == expected


@pytest.mark.asyncio
async def test_full_trial_surfaces_as_a_retryable_checkout_failure(
    trial, checkout_guard
):
    """A full trial must reach the client as the 409 the route maps, not a 500."""
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=trial.offer)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=None)),
        patch.object(
            checkout, "get_stripe_customer_id", AsyncMock(return_value="cus_1")
        ),
        patch.object(checkout, "_verify_eligibility", AsyncMock()),
        patch.object(
            checkout, "resolve_trial_price", AsyncMock(return_value=trial.offer)
        ),
        patch.object(
            checkout,
            "reserve_subscription_trial",
            AsyncMock(side_effect=checkout.TrialCapacityReached("The trial is full")),
        ),
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="full"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkout_asks_the_flag_with_the_visitors_country(trial, checkout_guard):
    with (
        patch.object(
            checkout, "get_trial_offer", AsyncMock(return_value=None)
        ) as offer,
        patch.object(checkout, "get_stripe_customer_id", AsyncMock()) as customer,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="not available"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
                country="IN",
            )
    assert offer.await_args.kwargs["country"] == "IN"
    customer.assert_not_awaited()


@pytest.mark.asyncio
async def test_excluded_visitor_cannot_resume_a_pending_checkout(trial, checkout_guard):
    """The current flag decides, even for someone who reserved while eligible."""
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=None)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="not available"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
                country="IN",
            )
    resume.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("seat,resumed", [(True, True), (False, False)])
async def test_resume_is_checked_against_the_current_cap(
    trial, checkout_guard, seat, resumed
):
    """The live cap applies, not the one pinned on the enrolment."""
    capped = trial.offer.model_copy(update={"max_active_trials": 10})
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=capped)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            checkout, "renew_trial_seat", AsyncMock(return_value=seat)
        ) as check,
        patch.object(
            checkout, "_resume_checkout", AsyncMock(return_value="https://x")
        ) as resume,
    ):
        if resumed:
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
        else:
            with pytest.raises(checkout.TrialUnavailable, match="full"):
                await checkout.create_trial_checkout(
                    trial.user_id,
                    trial.offer.token,
                    trial.success_url,
                    trial.cancel_url,
                    {},
                )
    assert check.await_args.args == (capped, trial.id)
    assert resume.await_count == int(resumed)


NO_CHARGE_EXPERIMENT = Experiment.TRIAL_CHECKOUT_NO_CHARGE_COPY.value


@pytest.mark.parametrize("arm", ["no-charge-copy", "control"])
async def test_a_new_session_shows_the_no_charge_line_only_in_its_arm(
    trial, stripe_session_create, stored_arms, posthog_flags, arm
):
    posthog_flags.feature_flags = [_no_charge_experiment(everyone_in=arm)]

    await checkout._resume_checkout(trial)

    params = stripe_session_create.await_args.kwargs
    assert ("custom_text" in params) is (arm == "no-charge-copy")
    stored = stored_arms.create.await_args.kwargs["data"]
    assert (stored["experimentKey"], stored["variant"], stored["source"]) == (
        NO_CHARGE_EXPERIMENT,
        arm,
        "backend",
    )


@pytest.mark.parametrize(
    "definitions",
    [None, [], "other-flags-only"],
    ids=["no-secret-key", "no-definitions", "flag-not-created"],
)
async def test_without_the_experiment_flag_nobody_sees_the_line_or_is_enrolled(
    mocker, trial, stripe_session_create, stored_arms, posthog_flags, definitions
):
    if definitions is None:
        mocker.patch.object(ph, "get_flag_client", return_value=None)
    elif definitions == "other-flags-only":
        other = _no_charge_experiment(everyone_in="no-charge-copy")
        posthog_flags.feature_flags = [{**other, "key": "some-other-experiment"}]
    else:
        posthog_flags.feature_flags = definitions

    await checkout._resume_checkout(trial)

    assert "custom_text" not in stripe_session_create.await_args.kwargs
    stored_arms.create.assert_not_awaited()


async def test_a_users_first_arm_is_kept_when_posthog_buckets_them_elsewhere(
    trial, stripe_session_create, stored_arms, posthog_flags
):
    """A replaced, expired session must show what the first one showed."""
    stored_arms.find_unique.return_value = _assignment_row("no-charge-copy")
    posthog_flags.feature_flags = [_no_charge_experiment(everyone_in="control")]

    await checkout._resume_checkout(trial)

    assert "custom_text" in stripe_session_create.await_args.kwargs
    stored_arms.create.assert_not_awaited()


async def test_a_failed_assignment_still_opens_todays_checkout(
    trial, stripe_session_create, stored_arms, posthog_flags
):
    stored_arms.find_unique.side_effect = PrismaError("assignments unavailable")
    posthog_flags.feature_flags = [_no_charge_experiment(everyone_in="no-charge-copy")]

    url = await checkout._resume_checkout(trial)

    assert url == "https://checkout.test/cs_1"
    assert "custom_text" not in stripe_session_create.await_args.kwargs


@pytest.fixture
def stripe_session_create(mocker) -> AsyncMock:
    """Leads `_resume_checkout` to create a session; awaited with its params."""
    mocker.patch.object(checkout, "_find_checkout", AsyncMock(return_value=None))
    mocker.patch.object(checkout, "expire_other_subscription_checkouts", AsyncMock())
    mocker.patch.object(checkout, "_verify_eligibility", AsyncMock())
    trials = mocker.patch.object(checkout.SubscriptionTrial, "prisma")
    trials.return_value.update_many = AsyncMock()
    session = mocker.Mock(id="cs_1", status="open", url="https://checkout.test/cs_1")
    create = AsyncMock(return_value=session)
    mocker.patch.object(checkout, "stripe_call", create)
    return create


@pytest.fixture
def stored_arms(mocker):
    """The ExperimentAssignment table, holding no arm until one is created."""
    table = mocker.patch("prisma.models.ExperimentAssignment.prisma").return_value
    table.find_unique = AsyncMock(return_value=None)
    table.create = AsyncMock(
        side_effect=lambda data: _assignment_row(data["variant"], data["source"])
    )
    return table


@pytest.fixture
def posthog_flags(mocker, monkeypatch) -> Posthog:
    """A real PostHog client with no network; its `feature_flags` are the definitions."""
    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "urlopen", _refuse_network
    )
    client = Posthog(
        "phc_test",
        host="http://127.0.0.1:9",
        secret_key="phx_test",
        enable_local_evaluation=False,
        send=False,
        sync_mode=True,
    )
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    return client


def _no_charge_experiment(everyone_in: str) -> dict:
    return {
        "id": 1,
        "key": NO_CHARGE_EXPERIMENT,
        "active": True,
        "filters": {
            "groups": [{"properties": [], "rollout_percentage": 100}],
            "multivariate": {
                "variants": [
                    {"key": arm, "rollout_percentage": 100 if arm == everyone_in else 0}
                    for arm in ("control", "no-charge-copy")
                ]
            },
        },
    }


def _assignment_row(variant: str, source: str = "backend"):
    return type(
        "Row",
        (),
        {
            "experimentKey": NO_CHARGE_EXPERIMENT,
            "variant": variant,
            "source": source,
            "createdAt": datetime(2026, 10, 24, tzinfo=UTC),
        },
    )()


def _refuse_network(pool, method, url, *args, **kwargs):
    raise urllib3.exceptions.ProtocolError(f"no network in tests: {method} {url}")
