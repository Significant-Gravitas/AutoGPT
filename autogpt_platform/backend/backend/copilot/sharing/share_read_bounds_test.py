"""The chat-share file allowlist reads the same bounded window as the viewer.

`_build_shared_execution_file_allowlist` authorises exactly the files the
shared-run page can show. It filtered the run's node executions in Python
after loading all of them with their nodes and input rows, so its cost grew
with the run while the page it serves stayed the same size.
"""

from unittest.mock import AsyncMock, patch

import prisma.models
import pytest
from prisma.enums import AgentExecutionStatus

from backend.blocks.io import AgentOutputBlock
from backend.copilot.sharing.db import _build_shared_execution_file_allowlist
from backend.data.includes import MAX_NODE_EXECUTIONS_FETCH

OUTPUT_BLOCK_ID = AgentOutputBlock().id


@pytest.fixture
def execution_client():
    client = AsyncMock()
    client.find_unique.return_value = None
    with patch.object(prisma.models.AgentGraphExecution, "prisma", return_value=client):
        yield client


@pytest.mark.asyncio
async def test_the_allowlist_scan_is_bounded(execution_client):
    await _build_shared_execution_file_allowlist(
        execution_id="ge-1", share_token="tok-1", user_id="user-1"
    )

    node_execs = execution_client.find_unique.call_args.kwargs["include"][
        "NodeExecutions"
    ]
    assert node_execs["take"] == MAX_NODE_EXECUTIONS_FETCH
    assert AgentExecutionStatus.INCOMPLETE in [
        clause["executionStatus"] for clause in node_execs["where"]["NOT"]
    ]
    assert OUTPUT_BLOCK_ID in (
        node_execs["where"]["Node"]["is"]["AgentBlock"]["is"]["id"]["in"]
    )
    # Still the rows the outputs are built from.
    assert "Input" in node_execs["include"]
