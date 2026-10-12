import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from ldclient import Context, LDClient
from ldclient.config import Config
from ldclient.integrations.test_data import TestData
from pydantic import ValidationError

from backend.data import subscription_trial_config as trials


@contextmanager
def captured_logs():
    """Records emitted by the module logger; caplog sees nothing under the app's config."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(trials.__name__)
    handler = Collector()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)


def offer_data() -> dict:
    return {
        "version": "trial-experiment-a-v1",
        "new_users_from": "2026-09-10T00:00:00Z",
        "duration_days": 7,
        "tier": "PRO",
        "billing_cycle": "monthly",
        "daily_cost_limit": 250_000,
        "weekly_cost_limit": 1_000_000,
        "total_cost_limit": 1_000_000,
        "onboarding_credit_amount": 300,
    }


def test_offer_has_no_implicit_existing_user_eligibility():
    offer = trials.TrialOffer.model_validate(offer_data())
    assert offer.allow_existing_beta_users is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration_days": 0},
        {"duration_days": 731},
        {"duration_days": True},
        {"daily_cost_limit": -1},
        {"weekly_cost_limit": 0},
        {"total_cost_limit": "1000000"},
        {"total_cost_limit": True},
        {"daily_cost_limit": 2_000_000},
        {"weekly_cost_limit": 2_000_000},
        {"new_users_from": "2026-09-10T00:00:00"},
        {"tier": "TRIAL"},
        {"tier": "ENTERPRISE"},
        {"billing_cycle": "weekly"},
        {"allow_existing_beta_users": "true"},
        {"version": ""},
        {"unknown_setting": 1},
        {"max_active_trials": -1},
        {"max_active_trials": "5"},
        {"max_active_trials": True},
    ],
)
def test_rejects_invalid_or_ambiguous_offer(overrides):
    with pytest.raises(ValidationError):
        trials.TrialOffer.model_validate({**offer_data(), **overrides})


@pytest.mark.parametrize(
    "days_from_cutoff,beta_allowed,has_history,tier,expected",
    [
        (0, False, False, "NO_TIER", True),
        (1, False, False, "NO_TIER", True),
        (-1, False, False, "NO_TIER", False),
        (-1, True, False, "NO_TIER", True),
        (-1, True, True, "NO_TIER", False),
        (1, False, True, "NO_TIER", False),
        (1, True, False, "PRO", False),
        (1, True, False, "TRIAL", False),
        (1, True, False, "ENTERPRISE", False),
    ],
)
def test_eligibility_never_retrials_or_overwrites_paid_access(
    days_from_cutoff, beta_allowed, has_history, tier, expected
):
    offer = trials.TrialOffer.model_validate(
        {**offer_data(), "allow_existing_beta_users": beta_allowed}
    )
    assert (
        offer.is_eligible(
            created_at=datetime(2026, 9, 10, tzinfo=UTC)
            + timedelta(days=days_from_cutoff),
            current_tier=tier,
            has_subscription_history=has_history,
        )
        is expected
    )


def test_accepted_terms_survive_future_offer_changes():
    original = offer_data()
    accepted = trials.TrialOffer.model_validate(original)
    original.update(duration_days=14, total_cost_limit=9_000_000)
    assert accepted.duration_days == 7
    assert accepted.total_cost_limit == 1_000_000
    with pytest.raises(ValidationError):
        accepted.duration_days = 14


@pytest.mark.parametrize(
    "seconds_remaining,expected", [(1, True), (0, False), (-1, False)]
)
def test_trial_expires_at_exact_deadline(seconds_remaining, expected):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    assert (
        trials.trial_is_active(
            status="trialing",
            trial_end=now + timedelta(seconds=seconds_remaining),
            card_verified=True,
            now=now,
        )
        is expected
    )


@pytest.mark.parametrize("status", ["trialing", "active", "past_due", "canceled"])
def test_no_trial_entitlement_without_verified_card(status):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    assert not trials.trial_is_active(
        status=status,
        trial_end=now + timedelta(days=7),
        card_verified=False,
        now=now,
    )


@pytest.mark.asyncio
async def test_platform_payment_disabled_hides_trial_offer():
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=False)
    ), patch.object(
        trials,
        "get_feature_flag_value",
        AsyncMock(
            side_effect=lambda flag, *args, **kwargs: (
                offer_data() if flag == trials.Flag.CARD_REQUIRED_TRIAL_OFFER else False
            )
        ),
    ):
        assert await trials.get_trial_offer("user-1") is None


@pytest.mark.asyncio
async def test_invalid_remote_offer_fails_closed():
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ), patch.object(
        trials, "get_feature_flag_value", AsyncMock(return_value={"duration_days": "7"})
    ):
        assert await trials.get_trial_offer("user-1") is None


@pytest.mark.asyncio
async def test_payment_enabled_and_valid_offer_is_available():
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ) as enabled, patch.object(
        trials, "get_feature_flag_value", AsyncMock(return_value=offer_data())
    ):
        assert await trials.get_trial_offer(
            "user-1"
        ) == trials.TrialOffer.model_validate(offer_data())
    enabled.assert_awaited_once_with(
        trials.Flag.ENABLE_PLATFORM_PAYMENT, "user-1", default=False
    )


def test_offer_without_a_cap_keeps_its_old_meaning():
    """Offers written before the cap existed validate and stay uncapped."""
    assert trials.TrialOffer.model_validate(offer_data()).max_active_trials is None


def test_zero_cap_is_expressible_and_distinct_from_absent():
    """0 pauses enrolment; absent means uncapped. They must not collapse."""
    paused = trials.TrialOffer.model_validate({**offer_data(), "max_active_trials": 0})
    assert paused.max_active_trials == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "country,attributes",
    [
        ("IN", {"country": "IN"}),
        (" in ", {"country": "IN"}),
        (None, None),
        # a blank header is an unknown country, not a known non-excluded one
        ("", None),
        ("   ", None),
    ],
)
async def test_country_is_handed_to_the_flag_not_decided_here(country, attributes):
    """Who sees a trial is the flag's targeting; the code only supplies the fact."""
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ), patch.object(
        trials, "get_feature_flag_value", AsyncMock(return_value=offer_data())
    ) as flag:
        await trials.get_trial_offer("user-1", country=country)
    assert flag.await_args.kwargs["attributes"] == attributes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "country",
    # a duplicated header comma-joined, unassigned and reserved codes, a name
    ["IN, US", "US,IN", "XX", "ZZ", "EU", "T1", "INDIA", "U S", "us\u200b"],
)
async def test_a_country_that_is_not_an_iso_code_never_reaches_the_flag(country):
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ), patch.object(
        trials, "get_feature_flag_value", AsyncMock(return_value=offer_data())
    ) as flag:
        assert await trials.get_trial_offer("user-1", country=country) is None
    flag.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "country,offered",
    [
        ("US", True),
        (" gb ", True),
        ("IN", False),
        (None, False),
        # junk must not satisfy "country is not one of IN"
        ("IN, US", False),
        ("XX", False),
        ("India", False),
    ],
)
async def test_the_recommended_rule_fails_closed_through_the_real_evaluator(
    country, offered
):
    """The production rule shape, evaluated by LaunchDarkly's own SDK."""
    td = TestData.data_source()
    td.update(
        td.flag(trials.Flag.CARD_REQUIRED_TRIAL_OFFER.value)
        .variations({"enabled": False}, offer_data())
        .fallthrough_variation(0)
        .if_not_match("country", "IN")
        .then_return(1)
    )
    client = LDClient(Config("sdk-test", update_processor_class=td, send_events=False))
    try:
        with patch.object(
            trials, "is_feature_enabled", AsyncMock(return_value=True)
        ), patch("backend.util.feature_flag.ldclient.get", return_value=client), patch(
            "backend.util.feature_flag._fetch_user_context_status",
            AsyncMock(
                return_value=(Context.builder("user-1").kind("user").build(), True)
            ),
        ):
            offer = await trials.get_trial_offer("user-1", country=country)
    finally:
        client.close()
    assert (offer is not None) is offered


@pytest.mark.parametrize(
    "flag_value",
    [{"enabled": False}, {"enabled": True}, {}, None, False],
)
@pytest.mark.asyncio
async def test_flag_value_that_is_not_an_offer_is_silent(flag_value):
    """The disabled variation is an object, and an absent offer is not a fault."""
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ), patch.object(
        trials, "get_feature_flag_value", AsyncMock(return_value=flag_value)
    ), captured_logs() as records:
        assert await trials.get_trial_offer("user-1", country="IN") is None

    assert [r for r in records if r.levelno >= logging.WARNING] == []
    assert [r.getMessage() for r in records] == [
        "No card-required trial offer configured"
    ]


@pytest.mark.asyncio
async def test_offer_that_fails_validation_is_still_an_error():
    with patch.object(
        trials, "is_feature_enabled", AsyncMock(return_value=True)
    ), patch.object(
        trials,
        "get_feature_flag_value",
        AsyncMock(return_value={**offer_data(), "tier": "ENTERPRISE"}),
    ), captured_logs() as records:
        assert await trials.get_trial_offer("user-1") is None

    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "Invalid card-required-trial-offer" in errors[0].getMessage()


# The live production offer variation, verbatim from LaunchDarkly, and the
# token dev computes for it before the cap existed.
LIVE_OFFER = {
    "allow_existing_beta_users": False,
    "billing_cycle": "monthly",
    "daily_cost_limit": 3125000,
    "duration_days": 7,
    "new_users_from": "2026-09-06T00:00:00Z",
    "onboarding_credit_amount": 300,
    "tier": "PRO",
    "total_cost_limit": 31250000,
    "version": "pro-equivalent-v1",
    "weekly_cost_limit": 15625000,
    "price_id": "price_live",
    "unit_amount": 5000,
    "currency": "usd",
}
DEV_TOKEN = "8e7acb63affd0d9ecc34e906a6c4f0fee8c139a1e57431881f83de455aa6a49f"


@pytest.mark.parametrize("cap", [None, 0, 500])
def test_the_cap_never_moves_the_offer_token(cap):
    """Deploying the cap, or changing it, must not invalidate a shown offer.

    The token gates checkout: a mismatch refuses it with "the offer changed".
    """
    offer = {**LIVE_OFFER, **({} if cap is None else {"max_active_trials": cap})}
    assert trials.AcceptedTrialOffer.model_validate(offer).token == DEV_TOKEN


def test_a_real_term_still_moves_the_offer_token():
    changed = {**LIVE_OFFER, "duration_days": 14}
    assert trials.AcceptedTrialOffer.model_validate(changed).token != DEV_TOKEN
