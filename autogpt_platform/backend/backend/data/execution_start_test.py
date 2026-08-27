from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.models import AgentGraphExecution

from backend.data.execution import ExecutionStatus, update_graph_execution_start_time


@pytest.mark.asyncio
async def test_start_transition_only_claims_queued_or_incomplete_execution():
    client = MagicMock()
    client.update_many = AsyncMock(return_value=0)
    client.find_unique = AsyncMock()

    with patch.object(AgentGraphExecution, "prisma", return_value=client):
        result = await update_graph_execution_start_time("exec-1")

    assert result is None
    where = client.update_many.await_args.kwargs["where"]
    assert where == {
        "id": "exec-1",
        "executionStatus": {"in": [ExecutionStatus.QUEUED, ExecutionStatus.INCOMPLETE]},
    }
    client.find_unique.assert_not_awaited()
