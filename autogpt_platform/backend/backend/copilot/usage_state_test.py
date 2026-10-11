"""The usage API's verdict: ok, low or exhausted, the limit behind it and when
it refills, computed from the same status the meters show."""

from datetime import UTC, datetime

from backend.copilot.rate_limit import (
    CoPilotUsagePublic,
    CoPilotUsageStatus,
    SubscriptionTier,
    UsageWindow,
)
from backend.copilot.usage_state import UsageVerdict, WindowReading, usage_verdict

DAILY_RESET = datetime(2026, 10, 15, tzinfo=UTC)
WEEKLY_RESET = datetime(2026, 10, 19, tzinfo=UTC)
TRIAL_END = datetime(2026, 10, 25, tzinfo=UTC)


def _usage(
    daily_used: int,
    weekly_used: int,
    *,
    daily_limit: int = 1_000_000,
    weekly_limit: int = 5_000_000,
    tier: SubscriptionTier = SubscriptionTier.PRO,
    weekly_reset: datetime = WEEKLY_RESET,
    trial_spent: bool = False,
    weekly_is_trial_allowance: bool = False,
) -> CoPilotUsagePublic:
    return CoPilotUsagePublic.from_status(
        CoPilotUsageStatus(
            daily=UsageWindow(
                used=daily_used, limit=daily_limit, resets_at=DAILY_RESET
            ),
            weekly=UsageWindow(
                used=weekly_used, limit=weekly_limit, resets_at=weekly_reset
            ),
            tier=tier,
            trial_spent=trial_spent,
            weekly_is_trial_allowance=weekly_is_trial_allowance,
        )
    )


def _verdict(usage: CoPilotUsagePublic) -> tuple:
    return usage.state, usage.limiting_window, usage.resets_at


def test_room_in_every_window_is_ok():
    assert _verdict(_usage(100_000, 1_000_000)) == ("ok", None, None)


def test_low_from_where_the_meters_turn_amber():
    assert _verdict(_usage(800_000, 1_000_000)) == ("low", "daily", DAILY_RESET)
    assert _verdict(_usage(799_000, 1_000_000)) == ("ok", None, None)


def test_low_names_the_fullest_window():
    # 81% of the day, 90% of the week.
    usage = _usage(810_000, 4_500_000)
    assert _verdict(usage) == ("low", "weekly", WEEKLY_RESET)


def test_a_used_up_window_is_exhausted_until_it_refills():
    assert _verdict(_usage(1_000_000, 2_000_000)) == (
        "exhausted",
        "daily",
        DAILY_RESET,
    )


def test_with_both_used_up_the_window_that_refills_last_decides():
    assert _verdict(_usage(1_000_000, 5_000_000)) == (
        "exhausted",
        "weekly",
        WEEKLY_RESET,
    )


def test_a_tie_on_refill_time_goes_to_the_weekly_window():
    # On a Sunday both windows refill at the same midnight.
    readings = [
        WindowReading(
            window=name, percent_used=100.0, exhausted=True, resets_at=WEEKLY_RESET
        )
        for name in ("daily", "weekly")
    ]
    verdict = usage_verdict(
        readings, weekly_is_trial_allowance=False, trial_spent=False
    )
    assert verdict == UsageVerdict(
        state="exhausted", limiting_window="weekly", resets_at=WEEKLY_RESET
    )


def test_a_zero_limit_is_exhausted_not_unlimited():
    # No plan, or a trial that is over: the window's reset brings nothing
    # back, so there is no reset to wait for.
    usage = _usage(0, 0, daily_limit=0, weekly_limit=0)
    assert _verdict(usage) == ("exhausted", "weekly", None)


def test_a_window_that_allows_nothing_outlasts_one_that_refills():
    usage = _usage(1_000_000, 0, weekly_limit=0)
    assert _verdict(usage) == ("exhausted", "weekly", None)


def test_uncapped_windows_are_ok():
    usage = _usage(9_000_000, 9_000_000, daily_limit=-1, weekly_limit=-1)
    assert (usage.daily, usage.weekly) == (None, None)
    assert _verdict(usage) == ("ok", None, None)


def test_a_meter_rounded_up_to_100_is_low_while_a_turn_can_still_start():
    # 99.96% displays as 100.0, but the cap isn't reached, so the turn runs.
    usage = _usage(999_600, 0)
    assert usage.daily is not None and usage.daily.percent_used == 100.0
    assert _verdict(usage) == ("low", "daily", DAILY_RESET)


def test_a_trials_weekly_window_is_its_allowance_without_a_reset():
    # A lifetime trial's allowance is folded into the weekly window, which
    # "resets" when the trial ends.
    folded = {
        "tier": SubscriptionTier.TRIAL,
        "weekly_reset": TRIAL_END,
        "weekly_is_trial_allowance": True,
    }
    assert _verdict(_usage(0, 4_500_000, **folded)) == ("low", "trial", None)
    spent = _usage(0, 5_000_000, trial_spent=True, **folded)
    assert _verdict(spent) == ("exhausted", "trial", None)


def test_a_trials_own_weekly_cap_refills_like_anyones():
    # A trial whose period caps are tighter than its allowance keeps a real
    # weekly window: it refills on Monday, and Send at reset can wait for it.
    trial = SubscriptionTier.TRIAL
    assert _verdict(_usage(0, 4_500_000, tier=trial)) == (
        "low",
        "weekly",
        WEEKLY_RESET,
    )
    assert _verdict(_usage(0, 5_000_000, tier=trial)) == (
        "exhausted",
        "weekly",
        WEEKLY_RESET,
    )


def test_a_spent_trial_allowance_is_exhausted_with_room_in_the_windows():
    # A trial whose period caps are tighter than its allowance doesn't fold
    # the allowance into its windows, so the status flags it.
    usage = _usage(100_000, 1_000_000, tier=SubscriptionTier.TRIAL, trial_spent=True)
    assert _verdict(usage) == ("exhausted", "trial", None)


def test_a_trials_used_up_day_still_refills_at_midnight():
    usage = _usage(1_000_000, 1_000_000, tier=SubscriptionTier.TRIAL)
    assert _verdict(usage) == ("exhausted", "daily", DAILY_RESET)


def test_the_verdict_leaves_the_trial_meter_clamp_alone():
    usage = _usage(
        0,
        4_999_999,
        tier=SubscriptionTier.TRIAL,
        weekly_reset=TRIAL_END,
        weekly_is_trial_allowance=True,
    )
    assert usage.weekly is not None and usage.weekly.percent_used == 99.9
    assert _verdict(usage) == ("low", "trial", None)
