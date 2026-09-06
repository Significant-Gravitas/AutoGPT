from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.autopilot import AutoPilotBlock
from backend.data.execution import ExecutionStatus, update_graph_execution_stats


@pytest.fixture(autouse=True)
def mock_score_completed_run() -> Iterator[AsyncMock]:
    with patch(
        "backend.data.execution.score_completed_run", new_callable=AsyncMock
    ) as mock:
        yield mock


def test_default_block_has_execution_timeout():
    """Leaf compute blocks inherit the default wall-clock cap."""

    from backend.blocks.basic import StoreValueBlock

    assert (
        StoreValueBlock().execution_timeout_seconds
        == DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    )


def test_coordination_blocks_opt_out_of_timeout():
    """AgentExecutorBlock and AutoPilotBlock must not have a wall-clock cap;
    they wait on sub-graphs that have their own per-node timeouts."""
    assert AgentExecutorBlock().execution_timeout_seconds is None
    assert AutoPilotBlock().execution_timeout_seconds is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [
        ExecutionStatus.FAILED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.TERMINATED,
    ],
)
async def test_terminal_transition_cascades_running_children(terminal_status):
    """When a graph_exec moves to a terminal state, its still-RUNNING child
    node_execs must be batch-updated to FAILED in the same call so we never
    leak the 'parent terminal, child running' invariant."""
    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node:
        mock_graph.prisma.return_value.update_many = AsyncMock()
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock(
            return_value=MagicMock()
        )
        mock_node.prisma.return_value.update_many = AsyncMock()

        with patch(
            "backend.data.execution.GraphExecution.from_db",
            return_value=MagicMock(),
        ):
            await update_graph_execution_stats(
                graph_exec_id="ge-1",
                status=terminal_status,
            )

    mock_node.prisma.return_value.update_many.assert_awaited_once()
    where = mock_node.prisma.return_value.update_many.await_args.kwargs["where"]
    assert where["agentGraphExecutionId"] == "ge-1"
    assert where["executionStatus"] == ExecutionStatus.RUNNING.value


@pytest.mark.asyncio
async def test_non_terminal_transition_does_not_cascade():
    """Mid-flight status changes (RUNNING/QUEUED/REVIEW) must leave child rows alone."""
    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node:
        mock_graph.prisma.return_value.update_many = AsyncMock()
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock(
            return_value=MagicMock()
        )
        mock_node.prisma.return_value.update_many = AsyncMock()

        with patch(
            "backend.data.execution.GraphExecution.from_db",
            return_value=MagicMock(),
        ):
            await update_graph_execution_stats(
                graph_exec_id="ge-1",
                status=ExecutionStatus.RUNNING,
            )

    mock_node.prisma.return_value.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_cascade_can_be_disabled_explicitly():
    """`cascade_running_children=False` is the escape hatch for callers that
    need to mark a graph terminal without touching children (e.g. resume flows)."""
    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node:
        mock_graph.prisma.return_value.update_many = AsyncMock()
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock(
            return_value=MagicMock()
        )
        mock_node.prisma.return_value.update_many = AsyncMock()

        with patch(
            "backend.data.execution.GraphExecution.from_db",
            return_value=MagicMock(),
        ):
            await update_graph_execution_stats(
                graph_exec_id="ge-1",
                status=ExecutionStatus.FAILED,
                cascade_running_children=False,
            )

    mock_node.prisma.return_value.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_cascade_records_terminal_status_in_node_error():
    """The child error stamp should reference which terminal status caused it
    so we can tell deploy-time cancellations from billing failures from manual stops."""
    captured: dict = {}

    def fake_get_update_status_data(status, execution_data, stats):
        captured["status"] = status
        captured["stats"] = stats
        return {"executionStatus": status}

    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node, patch(
        "backend.data.execution._get_update_status_data",
        side_effect=fake_get_update_status_data,
    ):
        mock_graph.prisma.return_value.update_many = AsyncMock()
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock(
            return_value=MagicMock()
        )
        mock_node.prisma.return_value.update_many = AsyncMock()

        with patch(
            "backend.data.execution.GraphExecution.from_db",
            return_value=MagicMock(),
        ):
            await update_graph_execution_stats(
                graph_exec_id="ge-1",
                status=ExecutionStatus.TERMINATED,
            )

    assert captured.get("status") == ExecutionStatus.FAILED
    assert "terminated" in captured.get("stats", {}).get("error", "").lower()


@pytest.mark.asyncio
async def test_rejected_terminal_transition_returns_none_and_does_nothing(
    mock_score_completed_run,
):
    """A terminal write that VALID_STATUS_TRANSITIONS filters out (update_many
    matches zero rows, e.g. the run already finished) must return None and must
    not score the run or cascade its children. Returning the unchanged row here
    made the executor count a completion that never happened."""
    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node:
        mock_graph.prisma.return_value.update_many = AsyncMock(return_value=0)
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock()
        mock_node.prisma.return_value.update_many = AsyncMock()

        result = await update_graph_execution_stats(
            graph_exec_id="ge-1", status=ExecutionStatus.COMPLETED
        )

    assert result is None
    mock_score_completed_run.assert_not_awaited()
    mock_node.prisma.return_value.update_many.assert_not_awaited()
    mock_graph.prisma.return_value.find_unique_or_raise.assert_not_awaited()


@pytest.mark.asyncio
async def test_applied_terminal_transition_returns_row(mock_score_completed_run):
    """The one-row case is unchanged: score, cascade, return the fresh row."""
    with patch("backend.data.execution.AgentGraphExecution") as mock_graph, patch(
        "backend.data.execution.AgentNodeExecution"
    ) as mock_node, patch(
        "backend.data.execution.GraphExecution.from_db", return_value="row"
    ):
        mock_graph.prisma.return_value.update_many = AsyncMock(return_value=1)
        mock_graph.prisma.return_value.find_unique_or_raise = AsyncMock(
            return_value=MagicMock()
        )
        mock_node.prisma.return_value.update_many = AsyncMock()

        result = await update_graph_execution_stats(
            graph_exec_id="ge-1", status=ExecutionStatus.COMPLETED
        )

    assert result == "row"
    mock_score_completed_run.assert_awaited_once()
    mock_node.prisma.return_value.update_many.assert_awaited_once()
