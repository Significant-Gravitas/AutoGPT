"""A dry run's block that runs for real on the platform's key is checked and metered."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.manager import execute_node

_MOD = "backend.executor.manager"


@pytest.mark.asyncio
async def test_a_spent_allowance_stops_the_block_before_it_runs(dry_run_boundaries):
    refusal, track = dry_run_boundaries
    refusal.return_value = "You've reached your daily usage limit."
    block = _block()

    outputs = [item async for item in execute_node(*_args(block))]

    assert outputs == [("error", "You've reached your daily usage limit.")]
    block.ran.assert_not_called()
    refusal.assert_awaited_once_with("user-1")
    track.assert_not_awaited()


@pytest.mark.asyncio
async def test_what_the_block_spent_is_recorded_against_the_user(dry_run_boundaries):
    _, track = dry_run_boundaries
    block = _block()

    outputs = [item async for item in execute_node(*_args(block))]

    assert outputs == [("response", "ok")]
    track.assert_awaited_once()
    assert track.await_args.args == (block.execution_stats,)
    assert track.await_args.kwargs["user_id"] == "user-1"
    assert track.await_args.kwargs["model"] == "sim-model"


@pytest.fixture
def dry_run_boundaries():
    scope = MagicMock(_user=None, _tags={})
    refusal = AsyncMock(return_value=None)
    track = AsyncMock()
    with (
        patch(
            f"{_MOD}.validate_exec",
            side_effect=lambda _node, data, **_kwargs: (data, None),
        ),
        patch(f"{_MOD}._sentry_get_current_scope", return_value=scope),
        patch(
            f"{_MOD}.prepare_dry_run",
            side_effect=lambda _block, data: {**data, "model": "sim-model"},
        ),
        patch(f"{_MOD}.get_dry_run_credentials", return_value=MagicMock()),
        patch(f"{_MOD}.platform_spend_refusal", refusal),
        patch(f"{_MOD}.track_dry_run_block_cost", track),
    ):
        yield refusal, track


def _block() -> MagicMock:
    input_schema = MagicMock()
    input_schema.get_credentials_fields.return_value = {}
    input_schema.get_credentials_fields_info.return_value = {}
    input_schema.get_auto_credentials_fields.return_value = {}
    block = MagicMock()
    block.id = "block-1"
    block.name = "PlatformKeyBlock"
    block.disabled = False
    block.input_schema = input_schema
    block.execution_stats = NodeExecutionStats(input_token_count=10)
    block.ran = MagicMock()

    async def execute(_input_data, **_kwargs):
        block.ran()
        yield "response", "ok"

    block.execute = execute
    return block


def _args(block: MagicMock):
    node = MagicMock()
    node.block = block
    node.input_default = {}
    entry = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(dry_run=True),
    )
    return node, entry, SimpleNamespace(creds_manager=MagicMock())
