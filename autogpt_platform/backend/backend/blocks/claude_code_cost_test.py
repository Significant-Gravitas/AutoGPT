"""ClaudeCodeBlock is charged for the platform's E2B sandbox by the walltime the
executor measures."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.blocks.claude_code import (
    TEST_ANTHROPIC_CREDENTIALS,
    TEST_ANTHROPIC_CREDENTIALS_INPUT,
    ClaudeCodeBlock,
)
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import charge_reconciled_usage
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import e2b_credentials

WALLTIME_SECONDS = 95.0
WALLTIME_CREDITS = 10  # ceil(95 / 10) at 1 credit per 10 s

PLATFORM_E2B_INPUT = {
    "id": e2b_credentials.id,
    "provider": e2b_credentials.provider,
    "type": e2b_credentials.type,
    "title": e2b_credentials.title,
}
OWN_E2B_INPUT = {"id": "user-own-e2b", "provider": "e2b", "type": "api_key"}


@pytest.mark.parametrize(
    "cli_stdout",
    [
        '{"result": "done", "total_cost_usd": 2.5}',
        '{"result": "done", "total_cost_usd": 0}',
        '{"result": "done"}',
        "done",
    ],
    ids=["reports-a-cost", "reports-zero", "reports-no-cost", "not-json"],
)
async def test_a_platform_sandbox_run_is_charged_its_walltime(cli_stdout: str):
    block = ClaudeCodeBlock()
    inputs = {
        "e2b_credentials": PLATFORM_E2B_INPUT,
        "anthropic_credentials": TEST_ANTHROPIC_CREDENTIALS_INPUT,
        "prompt": "hi",
    }
    await _run_block(block, inputs, cli_stdout)

    # The executor's order: the block's own stats, then the walltime it measured.
    stats = NodeExecutionStats()
    stats += block.execution_stats
    stats.walltime = WALLTIME_SECONDS

    db_client = MagicMock()
    db_client.spend_credits = AsyncMock(return_value=500)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        delta, _ = await charge_reconciled_usage(
            _node_exec(block.id, inputs), stats, pre_flight_charge=0
        )

    assert delta == WALLTIME_CREDITS
    db_client.spend_credits.assert_awaited_once()
    assert db_client.spend_credits.await_args.kwargs["cost"] == WALLTIME_CREDITS


def test_a_run_on_the_users_own_sandbox_is_not_charged():
    cost, _ = block_usage_cost(
        block=ClaudeCodeBlock(),
        input_data={
            "e2b_credentials": OWN_E2B_INPUT,
            "anthropic_credentials": TEST_ANTHROPIC_CREDENTIALS_INPUT,
        },
        stats=NodeExecutionStats(walltime=WALLTIME_SECONDS),
    )
    assert cost == 0


async def _run_block(block: ClaudeCodeBlock, inputs: dict, cli_stdout: str) -> None:
    box = AsyncMock()
    box.sandbox_id = "sb-new"

    async def run(command: str, **kwargs):
        if "claude --session-id" in command:
            stdout = cli_stdout
        elif command.startswith("date"):
            stdout = "2026-10-31T00:00:00"
        else:
            stdout = ""
        return AsyncMock(exit_code=0, stdout=stdout, stderr="")

    box.commands.run = AsyncMock(side_effect=run)
    with (
        patch("backend.blocks.claude_code.create_sandbox", AsyncMock(return_value=box)),
        patch("backend.blocks.claude_code.kill_sandbox", AsyncMock()),
        patch(
            "backend.blocks.claude_code.extract_and_store_sandbox_files",
            AsyncMock(return_value=[]),
        ),
    ):
        outputs = [
            output
            async for output in block.run(
                ClaudeCodeBlock.Input(**inputs),
                e2b_credentials=e2b_credentials,
                anthropic_credentials=TEST_ANTHROPIC_CREDENTIALS,
                execution_context=ExecutionContext(user_id="user-a"),
            )
        ]
    assert ("response", "done") in outputs


def _node_exec(block_id: str, inputs: dict) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-a",
        graph_exec_id=str(uuid4()),
        graph_id=str(uuid4()),
        graph_version=1,
        node_exec_id=str(uuid4()),
        node_id=str(uuid4()),
        block_id=block_id,
        inputs=inputs,
        execution_context=ExecutionContext(user_id="user-a"),
    )
