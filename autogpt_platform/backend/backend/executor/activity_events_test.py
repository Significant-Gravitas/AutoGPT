"""Tests for the executor's activity-event emission."""

from unittest.mock import AsyncMock

from backend.blocks.airtable.schema import AirtableCreateTableBlock
from backend.data.execution import NodeExecutionEntry
from backend.executor.activity_events import log_node_integration_activity


async def test_an_integration_action_is_titled_in_words() -> None:
    block = AirtableCreateTableBlock()
    node_exec = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="run-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id=block.id,
        inputs={"credentials": {"id": "cred-1", "provider": "airtable"}},
    )
    db_client = AsyncMock()

    await log_node_integration_activity(node_exec, block, db_client)

    draft = db_client.create_activity_event.await_args.kwargs["draft"]
    assert draft.title == "Airtable Create Table"
    assert draft.provider == "airtable"
