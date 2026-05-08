"""Unit tests for ClaudeCodeBlock COST_USD billing migration.

Verifies:
- Block emits provider_cost / cost_usd when Claude Code CLI returns
  total_cost_usd.
- block_usage_cost resolves the COST_USD entry to the expected ceil(usd *
  cost_amount) credit charge.
- Missing total_cost_usd gracefully produces provider_cost=None (no bill).
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.blocks._base import BlockCostType
from backend.blocks.claude_code import ClaudeCodeBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost


def test_claude_code_registered_as_cost_usd_150():
    """Sanity: BLOCK_COSTS holds the COST_USD, 150 cr/$ entry."""
    entries = BLOCK_COSTS[ClaudeCodeBlock]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.cost_type == BlockCostType.COST_USD
    assert entry.cost_amount == 150


@pytest.mark.parametrize(
    "total_cost_usd, expected_credits",
    [
        (0.50, 75),  # $0.50 × 150 = 75 cr
        (1.00, 150),  # $1.00 × 150 = 150 cr
        (0.0134, 3),  # ceil(0.0134 × 150) = ceil(2.01) = 3
        (2.00, 300),  # $2 × 150 = 300 cr
        (0.001, 1),  # ceil(0.001 × 150) = ceil(0.15) = 1 — no 0-cr leak on
        # sub-cent runs
    ],
)
def test_cost_usd_resolver_applies_150_multiplier(total_cost_usd, expected_credits):
    """block_usage_cost with cost_usd stats returns ceil(usd * 150)."""
    block = ClaudeCodeBlock()
    # cost_filter requires matching e2b_credentials; supply the ones the
    # registration uses so _is_cost_filter_match accepts the input.
    entry = BLOCK_COSTS[ClaudeCodeBlock][0]
    input_data = {"e2b_credentials": entry.cost_filter["e2b_credentials"]}
    stats = NodeExecutionStats(
        provider_cost=total_cost_usd,
        provider_cost_type="cost_usd",
    )
    cost, matching_filter = block_usage_cost(
        block=block, input_data=input_data, stats=stats
    )
    assert cost == expected_credits
    assert matching_filter == entry.cost_filter


def test_cost_usd_resolver_returns_zero_when_stats_missing_cost():
    """Pre-flight (no stats) or unbilled run (provider_cost None) → 0."""
    block = ClaudeCodeBlock()
    entry = BLOCK_COSTS[ClaudeCodeBlock][0]
    input_data = {"e2b_credentials": entry.cost_filter["e2b_credentials"]}
    # No stats at all → pre-flight path, returns 0.
    pre_cost, _ = block_usage_cost(block=block, input_data=input_data)
    assert pre_cost == 0
    # Stats present but no provider_cost → resolver can't bill.
    stats = NodeExecutionStats()
    post_cost, _ = block_usage_cost(block=block, input_data=input_data, stats=stats)
    assert post_cost == 0


def test_record_cli_cost_emits_provider_cost_when_total_cost_present():
    """``_record_cli_cost`` (the helper called from ``execute_claude_code``)
    must emit a single ``merge_stats`` with provider_cost + cost_usd tag
    when the CLI JSON payload carries ``total_cost_usd``.
    """
    block = ClaudeCodeBlock()
    captured: list[NodeExecutionStats] = []
    with patch.object(block, "merge_stats", side_effect=captured.append):
        block._record_cli_cost(
            {
                "result": "hello from claude",
                "total_cost_usd": 0.0421,
                "usage": {"input_tokens": 1234, "output_tokens": 56},
            }
        )

    assert len(captured) == 1
    stats = captured[0]
    assert stats.provider_cost == pytest.approx(0.0421)
    assert stats.provider_cost_type == "cost_usd"


def test_record_cli_cost_skips_merge_when_total_cost_absent():
    """If the CLI payload lacks ``total_cost_usd`` (legacy / non-JSON
    output), ``_record_cli_cost`` must not call ``merge_stats`` — otherwise
    we'd pollute telemetry with a ``cost_usd`` emission that has no real
    cost attached.
    """
    block = ClaudeCodeBlock()
    mock = MagicMock()
    with patch.object(block, "merge_stats", mock):
        block._record_cli_cost({"result": "hello"})
    mock.assert_not_called()
