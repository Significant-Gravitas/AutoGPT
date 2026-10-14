"""Compaction budgets over a launch's window and trigger, and the Codex account
limits they come from."""

import pytest

from backend.copilot.sdk.context_window import (
    BARE_MESSAGE_TOKEN_FLOOR,
    CodexEngineWindow,
    cli_autocompact_threshold,
    compaction_target_tokens,
    retry_target_tokens,
    seed_target_tokens,
)


@pytest.mark.parametrize(
    "window, limit, expected",
    [
        (272_000, 244_800, 90),
        (400_000, 200_000, 50),
        (400_000, 399_000, 90),  # held at the CLI's usable ceiling
        (400_000, None, None),
        (400_000, 400_000, None),  # not a trigger
        (400_000, 0, None),
    ],
)
def test_codex_trigger_from_the_account_limit(window, limit, expected):
    engine = CodexEngineWindow(context_window=window, auto_compact_token_limit=limit)
    assert engine.trigger_pct == expected


class TestThreshold:
    def test_percentage_trigger(self):
        assert cli_autocompact_threshold(200_000, 50) == 100_000

    def test_buffer_caps_a_high_trigger(self):
        assert cli_autocompact_threshold(200_000, 95) == 200_000 - 13_000

    def test_no_trigger_keeps_only_the_buffer(self):
        assert cli_autocompact_threshold(200_000, None) == 187_000


class TestBudgets:
    def test_compaction_target_sits_below_the_threshold(self):
        assert compaction_target_tokens(400_000, 90) == 360_000 - 20_000

    def test_compaction_target_has_a_floor(self):
        assert compaction_target_tokens(20_000, 10) == 10_000

    def test_retry_budgets_end_on_the_bare_message_floor(self):
        assert retry_target_tokens(200_000) == (50_000, BARE_MESSAGE_TOKEN_FLOOR)

    def test_seed_budget_is_fifteen_percent(self):
        assert seed_target_tokens(200_000) == 30_000
