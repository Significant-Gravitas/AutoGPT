"""Recover only the native CLI spend and tokens not reported in earlier results."""

import logging

import pytest

from backend.copilot.sdk.cost_tracking import CLISessionUsage, TokenUsage

from .turn_cost_test import _CALL_COST, _CALL_USAGE, _KIMI, _kimi_rate_card_cost


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", _KIMI])
def test_missing_result_recovery_excludes_reported_usage(model):
    usage = TokenUsage()
    usage.start_cli_session(
        CLISessionUsage(cost_usd=_CALL_COST, tokens=dict(_CALL_USAGE))
    )
    usage.record_result(_CALL_USAGE, 2 * _CALL_COST, model, "")
    snapshot = CLISessionUsage(
        cost_usd=3 * _CALL_COST,
        tokens={key: 3 * value for key, value in _CALL_USAGE.items()},
    )
    usage.record_unreported(snapshot, model, "")
    usage.record_unreported(snapshot, model, "")

    per_call = _kimi_rate_card_cost() if model == _KIMI else _CALL_COST
    assert usage.cost_usd == pytest.approx(2 * per_call)
    assert usage.prompt_tokens == 2 * _CALL_USAGE["input_tokens"]
    assert usage.completion_tokens == 2 * _CALL_USAGE["output_tokens"]
    assert usage.cache_read_tokens == 2 * _CALL_USAGE["cache_read_input_tokens"]
    assert usage.cache_creation_tokens == 2 * _CALL_USAGE["cache_creation_input_tokens"]


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", _KIMI])
def test_incomplete_query_does_not_reprice_previous_model(model):
    first_model = _KIMI if model != _KIMI else "claude-sonnet-4-6"
    usage = TokenUsage()
    usage.start_cli_session(CLISessionUsage())
    usage.record_result(_CALL_USAGE, _CALL_COST, first_model, "")
    usage.record_unreported(
        CLISessionUsage(
            cost_usd=2 * _CALL_COST,
            tokens={key: 2 * value for key, value in _CALL_USAGE.items()},
        ),
        model,
        "",
    )
    assert usage.cost_usd == pytest.approx(_CALL_COST + _kimi_rate_card_cost())


def test_missing_moonshot_tokens_alerts_and_uses_cli_spend(caplog):
    usage = TokenUsage()
    usage.start_cli_session(CLISessionUsage(cost_usd=_CALL_COST))
    with caplog.at_level(logging.ERROR):
        usage.record_unreported(CLISessionUsage(cost_usd=2 * _CALL_COST), _KIMI, "")

    assert usage.cost_usd == pytest.approx(_CALL_COST)
    assert [record.levelno for record in caplog.records] == [logging.ERROR]


def test_recovery_after_cli_counter_reset_uses_new_session_tokens():
    usage = TokenUsage()
    usage.start_cli_session(
        CLISessionUsage(
            cost_usd=2 * _CALL_COST,
            tokens={key: 2 * value for key, value in _CALL_USAGE.items()},
        )
    )
    usage.record_result(_CALL_USAGE, _CALL_COST, _KIMI, "")
    usage.record_unreported(
        CLISessionUsage(
            cost_usd=2 * _CALL_COST,
            tokens={key: 2 * value for key, value in _CALL_USAGE.items()},
        ),
        _KIMI,
        "",
    )
    assert usage.cost_usd == pytest.approx(2 * _kimi_rate_card_cost())
