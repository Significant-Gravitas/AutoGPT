from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionStatus, update_node_execution_status


async def _write_status(
    current: ExecutionStatus, requested: ExecutionStatus
) -> MagicMock:
    """Write `requested` to a node execution that is `current`, with the update
    matching no row (VALID_STATUS_TRANSITIONS doesn't allow it), and return the
    module logger's mock."""
    row = MagicMock(executionStatus=current)
    with patch("backend.data.execution.AgentNodeExecution") as mock_node, patch(
        "backend.data.execution.NodeExecutionResult.from_db",
        side_effect=lambda db_row: db_row,
    ), patch("backend.data.execution.logger") as mock_logger:
        mock_node.prisma.return_value.update = AsyncMock(return_value=None)
        mock_node.prisma.return_value.find_unique = AsyncMock(return_value=row)
        result = await update_node_execution_status("node-exec-1", requested)
    assert result is row
    return mock_logger


@pytest.mark.asyncio
async def test_a_rejected_node_transition_is_logged():
    """A write that VALID_STATUS_TRANSITIONS rejects leaves the row as it was,
    and the caller gets that row back without knowing. Skipped nodes were
    written straight from QUEUED to COMPLETED this way and sat QUEUED unnoticed,
    so a rejection must at least be logged."""
    mock_logger = await _write_status(ExecutionStatus.QUEUED, ExecutionStatus.COMPLETED)

    mock_logger.warning.assert_called_once()
    message = mock_logger.warning.call_args.args[0]
    assert "node-exec-1" in message
    assert "QUEUED" in message and "COMPLETED" in message


@pytest.mark.asyncio
async def test_writing_the_status_a_node_already_has_is_not_logged():
    """Writing the status a node already has changes nothing and isn't a
    rejection worth a warning, e.g. a node resumed while RUNNING that passes
    through RUNNING again when it's skipped."""
    mock_logger = await _write_status(ExecutionStatus.RUNNING, ExecutionStatus.RUNNING)

    mock_logger.warning.assert_not_called()
