import pytest
from pydantic import ValidationError

from backend.copilot.gate.policy import DEFAULT_MODE

from .delegation_settings import (
    DEFAULT_DELEGATION_MODE,
    DelegationSettings,
    DelegationSettingsUpdate,
)


def test_the_default_mode_is_the_gates_default():
    assert DEFAULT_DELEGATION_MODE == DEFAULT_MODE


def test_defaults_match_the_settings_tab():
    assert DelegationSettings().model_dump() == {
        "mode": "auto",
        "per_delegation_cap_usd": 2.0,
        "daily_budget_usd": 10.0,
        "ask_before_external": True,
        "ask_before_over_cap": True,
        "new_experts_ask_first": False,
    }


def test_a_negative_cap_is_refused():
    with pytest.raises(ValidationError):
        DelegationSettingsUpdate(
            mode="auto",
            per_delegation_cap_usd=-1,
            daily_budget_usd=10,
            ask_before_external=True,
            ask_before_over_cap=True,
            new_experts_ask_first=False,
        )
