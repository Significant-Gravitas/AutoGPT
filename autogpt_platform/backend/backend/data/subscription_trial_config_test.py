from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from backend.data import subscription_trial_config as trials


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
