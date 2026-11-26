"""The chat-share file allowlist reads the same bounded window as the viewer.

`_build_shared_execution_file_allowlist` authorises exactly the files the
shared-run page can show, so it reads that page's window: output and input
blocks, a bounded number of rows, and the node read once for the run.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest
from prisma.enums import AgentExecutionStatus

from backend.blocks.io import AgentOutputBlock
from backend.copilot.sharing.db import _build_shared_execution_file_allowlist
from backend.data.includes import MAX_NODE_EXECUTIONS_FETCH

OUTPUT_BLOCK_ID = AgentOutputBlock().id
NODE_ID = "node-1"


@pytest.fixture
def execution_client():
    client = AsyncMock()
    client.find_unique.return_value = None
    with patch.object(prisma.models.AgentGraphExecution, "prisma", return_value=client):
        yield client


@pytest.mark.asyncio
async def test_the_allowlist_holds_the_files_an_output_block_named(mocker):
    """The rows arrive without their node, so the scan must attach them.

    Without that, every row looks like a node-less one, the scan finds no
    output block, and a shared chat stops authorising its own run's files.
    """
    file_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    execution = SimpleNamespace(
        NodeExecutions=[
            SimpleNamespace(
                agentNodeId=NODE_ID,
                Node=None,
                executionData={
                    "name": "Final report",
                    "value": f"workspace://{file_id}#text/markdown",
                },
                Input=None,
                Output=None,
                stats=None,
            )
        ]
    )
    _patch_prisma(mocker, prisma.models.AgentGraphExecution, find_unique=execution)
    _patch_prisma(
        mocker,
        prisma.models.AgentNode,
        find_many=[SimpleNamespace(id=NODE_ID, agentBlockId=OUTPUT_BLOCK_ID)],
    )
    _patch_prisma(
        mocker, prisma.models.UserWorkspace, find_unique=SimpleNamespace(id="ws-1")
    )
    _patch_prisma(
        mocker,
        prisma.models.UserWorkspaceFile,
        find_many=[SimpleNamespace(id=file_id)],
    )
    shared_file = _patch_prisma(mocker, prisma.models.SharedExecutionFile, create=None)

    created = await _build_shared_execution_file_allowlist(
        execution_id="ge-1", share_token="tok-1", user_id="user-1"
    )

    assert created == 1
    assert shared_file.create.await_args.kwargs["data"]["fileId"] == file_id


def _patch_prisma(mocker, model, **returns):
    client = AsyncMock()
    for name, value in returns.items():
        getattr(client, name).return_value = value
    mocker.patch.object(model, "prisma", return_value=client)
    return client


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
    # Still the rows the outputs are built from, and the node read once for
    # the whole run rather than joined to each row.
    assert "Input" in node_execs["include"]
    assert "Node" not in node_execs["include"]
