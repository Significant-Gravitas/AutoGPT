"""Unit tests for exa/helpers cost-extraction + merge helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.blocks.exa.helpers import extract_exa_cost_usd, merge_exa_cost
from backend.data.model import NodeExecutionStats


@pytest.mark.parametrize(
    "response, expected",
    [
        # Dataclass / SimpleNamespace with cost_dollars.total
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=0.05)), 0.05),
        # Dict camelCase
        ({"costDollars": {"total": 0.10}}, 0.10),
        # Dict snake_case
        ({"cost_dollars": {"total": 0.07}}, 0.07),
        # code_context endpoint shape: plain numeric string
        (SimpleNamespace(cost_dollars="0.005"), 0.005),
        # Scalar float on cost_dollars directly
        (SimpleNamespace(cost_dollars=0.02), 0.02),
        # Scalar int on cost_dollars
        (SimpleNamespace(cost_dollars=3), 3.0),
        # Missing cost info — returns None
        ({}, None),
        (SimpleNamespace(other="foo"), None),
        (None, None),
        # Nested total=None
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=None)), None),
        # Invalid numeric string
        (SimpleNamespace(cost_dollars="not-a-number"), None),
        # Negative values clamp to 0
        (SimpleNamespace(cost_dollars=SimpleNamespace(total=-1.0)), 0.0),
    ],
)
def test_extract_exa_cost_usd_handles_all_shapes(response, expected):
    assert extract_exa_cost_usd(response) == expected


def test_merge_exa_cost_emits_stats_when_cost_present():
    block = MagicMock()
    response = SimpleNamespace(cost_dollars=SimpleNamespace(total=0.0421))
    merge_exa_cost(block, response)

    block.merge_stats.assert_called_once()
    stats: NodeExecutionStats = block.merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(0.0421)
    assert stats.provider_cost_type == "cost_usd"


def test_merge_exa_cost_noops_when_no_cost():
    """Webset CRUD endpoints don't surface cost_dollars today — the helper
    must silently skip instead of emitting a 0-cost telemetry record."""
    block = MagicMock()
    merge_exa_cost(block, SimpleNamespace(other_field="nothing"))
    block.merge_stats.assert_not_called()


def test_merge_exa_cost_noops_when_response_is_none():
    block = MagicMock()
    merge_exa_cost(block, None)
    block.merge_stats.assert_not_called()
