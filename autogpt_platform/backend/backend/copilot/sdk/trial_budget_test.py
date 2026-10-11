import pytest

from backend.copilot.sdk.trial_budget import resolve_trial_sdk_budget


def test_trial_budget_is_never_rounded_up_to_paid_floor():
    assert resolve_trial_sdk_budget(10.0, 0.05) == 0.05


@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan")])
def test_unavailable_trial_budget_refuses_sdk_dispatch(remaining):
    with pytest.raises(ValueError, match="Trial budget"):
        resolve_trial_sdk_budget(10.0, remaining)


def test_trial_still_respects_static_per_turn_cap():
    assert resolve_trial_sdk_budget(10.0, 20.0) == 10.0
